"""IntradayFeatureEngine: technical features on CLOSED 4-hour BTC bars.

Phase 8 Part 8. Deterministic pandas/``ta`` only — NEVER an LLM (project rule
since 2026-07-10). Bitcoin only; the 20 altcoins stay daily.

Reads ``crypto_prices`` at ``timeframe='4h'`` and writes ``btc_4h_features``,
one row per bar:

* ``rsi_14``        — RSI(14) over 4h bars (~2.3 days of tape, NOT comparable
                      to the daily RSI(14) in ``daily_features``)
* ``macd`` / ``macd_signal`` / ``macd_hist`` — MACD(12,26,9) over 4h bars
* ``volume`` / ``volume_ma_30`` — bar volume and its trailing 30-bar mean
                      (~5 days), the reference for "above/below average"

Defence layer 2: closed bars only
---------------------------------
:class:`~src.collectors.binance_fetcher.BinanceFetcher` already drops the
forming bar before saving (layer 1). This module re-checks anyway, because
the engine can be run directly against a warehouse populated by some other
path, and because a still-forming bar reaching the prompt is precisely the
failure this layer exists to prevent. Same two-layer shape as Part 1's
``_warn_if_base_stale`` sitting behind the fetcher's abort policy.

Warm-up honesty
---------------
MACD(12,26,9) needs 26+9 bars before ``macd_signal`` means anything and
``volume_ma_30`` needs 30. Leading rows are therefore written with NULLs
rather than partial values — the same convention as ``daily_features``, and
the reason :meth:`recent_closed_bars` exists for readers that need a window
guaranteed to be fully warmed up.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from ta.momentum import RSIIndicator
from ta.trend import MACD

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import (
    BASE_SYMBOL,
    INTRADAY_TIMEFRAME,
    Btc4hFeature,
    CryptoPrice,
)
from src.timeframes import bar_label, floor_to_bar, is_bar_closed

logger = logging.getLogger(__name__)

#: Feature columns written by this engine (excludes PK + lineage).
INTRADAY_FEATURE_COLUMNS: tuple[str, ...] = (
    "close",
    "volume",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "volume_ma_30",
)

#: Trailing bars for the volume average. 30 bars = 5 days: long enough to be
#: stable, short enough that "above average" still describes this week rather
#: than this quarter.
VOLUME_MA_BARS = 30

#: Bars needed before MACD(12,26,9) and the volume MA are all meaningful.
#: 26 (slow EMA) + 9 (signal) = 35; the volume MA needs 30. Rows before this
#: are written with NULLs, not partial numbers.
WARMUP_BARS = 35

#: Default bars loaded for a feature run. 1000 is Binance's per-call cap and
#: ~166 days of 4h history — far beyond warm-up, one request.
DEFAULT_LOOKBACK_BARS = 1000

#: Rows per upsert batch (mirrors ``features/engine.py``).
_UPSERT_CHUNK_SIZE = 500


class IntradayFeatureError(RuntimeError):
    """Raised when 4H features cannot be computed at all."""


@dataclass(slots=True)
class IntradayBar:
    """One closed 4h bar with its features — what readers actually consume."""

    bar_open: datetime
    close: float | None
    volume: float | None
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    volume_ma_30: float | None

    @property
    def label(self) -> str:
        """``2026-07-29 04:00-08:00 UTC`` — date included deliberately."""
        return bar_label(self.bar_open)

    @property
    def macd_state(self) -> str:
        """``bullish`` / ``bearish`` / ``unknown`` from MACD vs its signal."""
        if self.macd is None or self.macd_signal is None:
            return "unknown"
        return "bullish" if self.macd > self.macd_signal else "bearish"

    @property
    def volume_state(self) -> str:
        """``above_avg`` / ``below_avg`` / ``unknown`` vs the 30-bar mean."""
        if self.volume is None or not self.volume_ma_30:
            return "unknown"
        return "above_avg" if self.volume > self.volume_ma_30 else "below_avg"


class IntradayFeatureEngine:
    """Computes and persists 4H technical features for Bitcoin."""

    def __init__(
        self,
        *,
        symbol: str = BASE_SYMBOL,
        lookback_bars: int = DEFAULT_LOOKBACK_BARS,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.symbol = symbol
        self.lookback_bars = lookback_bars

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    async def load_closed_bars(self, now: datetime | None = None) -> pd.DataFrame:
        """Closed 4h bars, ascending, indexed by bar open time.

        The ``timestamp < cutoff`` predicate is defence layer 2. It is applied
        in SQL rather than after loading so a forming bar can never occupy a
        slot in the lookback window either.
        """
        reference = now or datetime.now(tz=UTC)
        cutoff = floor_to_bar(reference)

        query = (
            select(
                CryptoPrice.timestamp,
                CryptoPrice.close,
                CryptoPrice.volume,
            )
            .where(
                CryptoPrice.exchange == "binance",
                CryptoPrice.symbol == self.symbol,
                CryptoPrice.timeframe == INTRADAY_TIMEFRAME,
                CryptoPrice.timestamp < cutoff,
            )
            .order_by(CryptoPrice.timestamp.desc())
            .limit(self.lookback_bars)
        )
        async with session_scope() as session:
            rows = (await session.execute(query)).all()

        if not rows:
            raise IntradayFeatureError(
                f"no CLOSED {INTRADAY_TIMEFRAME} candles for {self.symbol} in "
                f"crypto_prices before {bar_label(cutoff)} — run the intraday "
                f"collector first"
            )

        frame = pd.DataFrame(
            {
                "close": [float(row.close) for row in rows],
                "volume": [float(row.volume) for row in rows],
            },
            index=pd.DatetimeIndex([row.timestamp for row in rows], name="bar_open"),
        ).sort_index()

        # Layer-2 assertion. The SQL cutoff already guarantees this; if it
        # ever fires, something upstream changed and silence would be worse
        # than a loud failure.
        leaked = [ts for ts in frame.index if not is_bar_closed(ts, now=reference)]
        if leaked:
            raise IntradayFeatureError(
                f"{len(leaked)} still-forming bar(s) reached the feature "
                f"engine despite the cutoff (first: {bar_label(leaked[0])}) — "
                f"refusing to compute features on non-final data"
            )

        logger.info(
            "[4h-features] loaded %d closed bar(s) for %s: %s -> %s",
            len(frame), self.symbol,
            bar_label(frame.index[0]), bar_label(frame.index[-1]),
        )
        return frame

    # ------------------------------------------------------------------
    # Computation (pure — testable with a synthetic frame)
    # ------------------------------------------------------------------
    @staticmethod
    def compute(frame: pd.DataFrame) -> pd.DataFrame:
        """Add the indicator columns to a closed-bar frame. Pure function."""
        if frame.empty:
            raise IntradayFeatureError("cannot compute features on an empty frame")

        close = frame["close"]
        volume = frame["volume"]

        features = pd.DataFrame(index=frame.index)
        features["close"] = close
        features["volume"] = volume
        features["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        features["macd"] = macd.macd()
        features["macd_signal"] = macd.macd_signal()
        features["macd_hist"] = macd.macd_diff()

        # min_periods = the full window: a "30-bar average" computed from 4
        # bars is not a 30-bar average, and calling it one would make
        # above/below-average judgements on essentially one bar of data.
        features["volume_ma_30"] = volume.rolling(
            window=VOLUME_MA_BARS, min_periods=VOLUME_MA_BARS
        ).mean()

        return features.replace([float("inf"), float("-inf")], pd.NA)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    async def save(self, features: pd.DataFrame) -> int:
        """Idempotent upsert into ``btc_4h_features`` on ``(timestamp, symbol)``."""
        if features.empty:
            logger.warning("[4h-features] nothing to save — empty frame")
            return 0

        payload: list[dict] = []
        for bar_open, row in features.iterrows():
            record: dict = {
                "timestamp": bar_open.to_pydatetime(),
                "symbol": self.symbol,
            }
            for column in INTRADAY_FEATURE_COLUMNS:
                record[column] = self._decimal(row.get(column))
            payload.append(record)

        table = Btc4hFeature.__table__
        written = 0
        async with session_scope() as session:
            for start in range(0, len(payload), _UPSERT_CHUNK_SIZE):
                chunk = payload[start : start + _UPSERT_CHUNK_SIZE]
                statement = pg_insert(table).values(chunk)
                update_set = {
                    column: statement.excluded[column]
                    for column in INTRADAY_FEATURE_COLUMNS
                }
                update_set["ingested_at"] = func.now()
                statement = statement.on_conflict_do_update(
                    index_elements=["timestamp", "symbol"], set_=update_set
                )
                await session.execute(statement)
                written += len(chunk)

        logger.info(
            "[4h-features] upserted %d row(s) into 'btc_4h_features'", written
        )
        return written

    # ------------------------------------------------------------------
    async def run(self, now: datetime | None = None) -> int:
        """Load closed bars -> compute -> persist. Returns rows written."""
        frame = await self.load_closed_bars(now=now)
        features = self.compute(frame)
        if len(features) < WARMUP_BARS:
            logger.warning(
                "[4h-features] only %d bar(s) available (<%d warm-up) — the "
                "leading MACD/volume-MA values will be NULL",
                len(features), WARMUP_BARS,
            )
        return await self.save(features)

    # ------------------------------------------------------------------
    # Reader API
    # ------------------------------------------------------------------
    async def recent_closed_bars(
        self, count: int = 6, *, now: datetime | None = None
    ) -> list[IntradayBar]:
        """The last ``count`` CLOSED bars from ``btc_4h_features``, oldest first.

        Used by the prompt builder. Reads the persisted table rather than
        recomputing so that what the model sees is exactly what was stored and
        can be audited later from SQL.
        """
        reference = now or datetime.now(tz=UTC)
        cutoff = floor_to_bar(reference)

        query = (
            select(Btc4hFeature)
            .where(
                Btc4hFeature.symbol == self.symbol,
                Btc4hFeature.timestamp < cutoff,
            )
            .order_by(Btc4hFeature.timestamp.desc())
            .limit(count)
        )
        async with session_scope() as session:
            rows = (await session.execute(query)).scalars().all()

        bars = [
            IntradayBar(
                bar_open=row.timestamp,
                close=self._float(row.close),
                volume=self._float(row.volume),
                rsi_14=self._float(row.rsi_14),
                macd=self._float(row.macd),
                macd_signal=self._float(row.macd_signal),
                macd_hist=self._float(row.macd_hist),
                volume_ma_30=self._float(row.volume_ma_30),
            )
            for row in reversed(rows)
        ]
        if bars:
            logger.info(
                "[4h-features] %d closed bar(s) for context: %s -> %s",
                len(bars), bars[0].label, bars[-1].label,
            )
        else:
            logger.warning(
                "[4h-features] no closed bars in 'btc_4h_features' before %s",
                bar_label(cutoff),
            )
        return bars

    # ------------------------------------------------------------------
    @staticmethod
    def _decimal(value: object) -> Decimal | None:
        """Float -> Decimal for NUMERIC columns; NaN/inf/None -> NULL."""
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return Decimal(str(number))

    @staticmethod
    def _float(value: object) -> float | None:
        """Decimal -> float for readers; None stays None."""
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(number) or math.isinf(number) else number

"""FeatureEngine: warehouse tables -> engineered features -> ``daily_features``.

Pipeline (deterministic pandas/numpy/`ta` — NO LLM, per project rule):

1. **Load** the six source hypertables into pandas DataFrames. Since Phase 8
   the OHLCV load covers **every active symbol** in ``coins_registry``, not
   just Bitcoin.
2. **Align** everything onto a daily spine (that symbol's candle dates —
   crypto trades every calendar day) and **forward-fill** sources that skip
   weekends/holidays (FRED, market indices) or lag a day (on-chain,
   funding). ffill is capped at :data:`FFILL_LIMIT_DAYS` so stale values
   are never dragged across long outages.
3. **Compute** per-pillar features, once per symbol:

   * Technical  — RSI(14), MACD(12/26/9), Bollinger(20,2), ATR(14) via the
     ``ta`` library (pandas-ta cannot install on Python 3.14), plus
     realized volatility (7d/30d annualized std of log returns, 365d year).
     Computed from **that symbol's own** OHLCV.
   * Relative   — ``price_btc_ratio`` and 7d/30d relative strength vs BTC.
     THE altcoin feature: a coin up 5% on a day BTC is up 9% is losing, and
     only the BTC-denominated series says so.
   * Macro      — 7d/30d rate-of-change of DXY (DTWEXBGS) and US10Y (DGS10).
   * On-chain   — 7d/30d moving averages of hash-rate and tx count.
   * Sentiment  — Fear & Greed level and 7d rate-of-change.
   * Liquidity  — funding-rate daily change and OI/volume ratio
     (both legs in BTC units, so the ratio is dimensionless).
   * Indices    — S&P 500 / Nasdaq 100 closes + 7d RoC (research-grounded
     extra: BTC's Nasdaq beta dominates daily returns, Krause 2026).

4. **Upsert** one wide row per (day, symbol) into ``daily_features`` so
   Phase 4 agents need exactly one simple query per day and asset.

Multi-asset semantics (Phase 8)
-------------------------------
Only the technical block and the relative block are per-asset. The macro,
on-chain, sentiment, liquidity and index pillars are **market-wide / BTC
context** and are replicated verbatim onto every symbol's row — DXY does
not become a different number because you are looking at Solana. That
keeps every symbol's row complete and lets Part 2 decide, per asset, which
pillars deserve weight.

``btc_close``/``btc_volume`` therefore stay literally Bitcoin's, on every
row; the row's own price lives in ``asset_close``/``asset_volume``. On a BTC
row the two pairs are identical, which is exactly why the Bitcoin pipeline's
output is unchanged by this phase.

Leading rows keep NULLs during indicator warm-up windows (e.g. the first
29 days of a 30-day MA) — recent rows are complete, which is what the
agents consume and what the Phase 3 test enforces.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import (
    BASE_SYMBOL,
    DAILY_FEATURE_CATEGORICAL_COLUMNS,
    DAILY_FEATURE_COLUMNS,
    CoinRegistry,
    CryptoPrice,
    DailyFeature,
    DerivativesMetric,
    MacroIndicator,
    MarketCapTier,
    MarketIndex,
    OnChainMetric,
    SentimentData,
    SupplyMetric,
    TokenomicsEvent,
    TokenomicsEventType,
    VolRegime,
)

logger = logging.getLogger(__name__)

#: Cap forward-fill so stale values never bridge long source outages.
FFILL_LIMIT_DAYS = 7

#: Per-column overrides for sources with structural publication lag:
#: * funding_rate — MONTHLY Binance Vision archives: the running month has
#:   no settled values yet; the last settled rate is carried up to 35 days
#:   and is automatically overwritten by official values (upsert) once the
#:   month's archive is published.
#: * dxy_value — FRED DTWEXBGS ships with the weekly H.10 release
#:   (~3-8 day lag); 14 days bridges the lag plus holidays.
FFILL_LIMIT_OVERRIDES: dict[str, int] = {"funding_rate": 35, "dxy_value": 14}

#: Crypto trades 365 days/year — used to annualize realized volatility.
TRADING_DAYS_PER_YEAR = 365

#: Decay for the EWMA volatility estimate (2026-08-02). Higher = slower.
#:
#: Chosen by measurement over 34.8k asset-days across all 21 assets, not by
#: convention. Band coverage at a nominal 90%, by lambda:
#:     lambda   +1d    +7d   +30d
#:     0.90    89.0%  86.5%  79.2%
#:     0.94    89.9%  87.7%  81.1%
#:     0.97    91.0%  89.8%  87.0%   <-- best at every horizon
#: RiskMetrics' canonical 0.94 is tuned for equities and under-covers here.
#: Do not "correct" this to 0.94 without re-running that comparison.
EWMA_LAMBDA = 0.97

#: Cap on how far BTC's close may be carried forward when aligning it onto
#: an altcoin's date spine. Both series are daily crypto candles so they
#: should match exactly; this only bridges an isolated missing BTC candle.
#: Deliberately short — a BTC price a week stale would silently corrupt
#: every relative-strength number derived from it.
BTC_ALIGN_FFILL_DAYS = 3

#: Upsert chunk size. 49 feature columns + timestamp + symbol = 51 bind
#: params per row; 500 x 51 = 25.5k, still under asyncpg's 32767 cap.
#: Recompute this when adding columns — 640 rows would blow the limit.
_UPSERT_CHUNK_SIZE = 500

#: Volatility-percentile boundaries for :class:`VolRegime`.
VOL_REGIME_LOW_PCT = 30.0
VOL_REGIME_HIGH_PCT = 70.0

#: Market-cap tier boundaries, in USD.
MARKET_CAP_LARGE_USD = 10_000_000_000
MARKET_CAP_MID_USD = 1_000_000_000

#: Trailing window for the volatility percentile, and the minimum sample
#: below which the percentile is meaningless (a "percentile" over 20 days
#: says nothing about a yearly distribution).
VOL_PERCENTILE_WINDOW = 365
VOL_PERCENTILE_MIN_PERIODS = 180

#: Rolling window for beta/correlation vs BTC. Below this the estimate is
#: too noisy to publish, so it stays NULL.
BETA_WINDOW = 30

#: Lookahead used by ``supply_shock_score_30d``.
SUPPLY_SHOCK_HORIZON_DAYS = 30

#: Event types whose ``supply_impact_pct`` is expressed as a % of
#: CIRCULATING SUPPLY and may therefore be summed.
#:
#: ``halving`` is deliberately EXCLUDED: its impact is a change to the
#: EMISSION RATE (-50), a different unit entirely. Summing it with unlock
#: percentages would report Bitcoin as shedding half its circulating supply
#: on halving day. ``mainnet_launch`` is excluded for the same reason — its
#: meaning varies per project.
SUPPLY_STOCK_EVENT_TYPES = (
    TokenomicsEventType.UNLOCK.value,
    TokenomicsEventType.BURN.value,
)

#: Event types that count as "supply arriving" for days_to_next_unlock.
SUPPLY_INCREASING_EVENT_TYPES = (TokenomicsEventType.UNLOCK.value,)

#: Momentum normalisation: RSI points of 7-day change treated as a full
#: signal, and MACD-histogram 7-day change as a % of price likewise.
_MOMENTUM_RSI_FULL_SCALE = 30.0
_MOMENTUM_MACD_FULL_SCALE = 1.0

#: Columns forward-filled on the daily spine (sources with gaps/lag).
_FFILL_COLUMNS = (
    "dxy_value",
    "us10y_value",
    "hash_rate",
    "tx_count",
    "fear_greed_value",
    "funding_rate",
    "open_interest",
    "sp500_close",
    "nasdaq_close",
)


class FeatureEngineError(RuntimeError):
    """Raised when feature computation cannot proceed (e.g. no price data)."""


class FeatureEngine:
    """Computes the ``daily_features`` table from the six source tables.

    Multi-asset since Phase 8: one feature row per (day, symbol) for every
    ``is_active`` entry in ``coins_registry``. Pass ``symbols=[...]`` to
    restrict a run (useful for tests and backfills).
    """

    def __init__(
        self,
        *,
        symbols: list[str] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        #: Explicit symbol override; None means "read coins_registry".
        self._symbols_override = symbols

    # ------------------------------------------------------------------
    async def resolve_symbols(self) -> list[str]:
        """Active symbols to featurise — BTC always included, always first.

        BTC is force-included for the same reason the collector does it: it
        is the denominator of every relative-strength feature and Phases 1-7
        are built on its row existing.
        """
        if self._symbols_override is not None:
            symbols = list(self._symbols_override)
        else:
            async with session_scope() as session:
                symbols = list(
                    (
                        await session.execute(
                            select(CoinRegistry.symbol)
                            .where(CoinRegistry.is_active.is_(True))
                            .order_by(CoinRegistry.symbol)
                        )
                    ).scalars().all()
                )
            if not symbols:
                logger.warning(
                    "[features] coins_registry has no active symbols — falling "
                    "back to %s only (run scripts/migrate_phase8.py to seed it)",
                    BASE_SYMBOL,
                )
        ordered = [BASE_SYMBOL] + sorted(s for s in set(symbols) if s != BASE_SYMBOL)
        logger.info("[features] symbols to featurise: %s", ", ".join(ordered))
        return ordered

    # ------------------------------------------------------------------
    # 1) LOAD
    # ------------------------------------------------------------------
    async def load_source_data(self) -> dict:
        """Read all six source tables into DataFrames indexed by day.

        ``sources["ohlcv_by_symbol"]`` is a ``{symbol: DataFrame}`` mapping;
        every other key is a single market-wide DataFrame.
        """
        symbols = await self.resolve_symbols()
        sources: dict = {}
        async with session_scope() as session:
            # --- OHLCV for every active symbol (ONE query, not N) ---
            rows = (
                await session.execute(
                    select(
                        CryptoPrice.symbol,
                        CryptoPrice.timestamp,
                        CryptoPrice.open,
                        CryptoPrice.high,
                        CryptoPrice.low,
                        CryptoPrice.close,
                        CryptoPrice.volume,
                    )
                    .where(
                        CryptoPrice.exchange == "binance",
                        CryptoPrice.symbol.in_(symbols),
                        CryptoPrice.timeframe == "1d",
                    )
                    .order_by(CryptoPrice.symbol, CryptoPrice.timestamp)
                )
            ).all()
            # One DataFrame, one type conversion pass, then split by symbol —
            # cheaper than N round-trips or N re-parses as the asset count grows.
            all_candles = pd.DataFrame(
                rows,
                columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"],
            )
            ohlcv_by_symbol: dict[str, pd.DataFrame] = {}
            if not all_candles.empty:
                all_candles["timestamp"] = pd.to_datetime(
                    all_candles["timestamp"], utc=True
                ).dt.tz_localize(None)
                for column in ("open", "high", "low", "close", "volume"):
                    all_candles[column] = all_candles[column].astype(float)
                for symbol, group in all_candles.groupby("symbol", sort=False):
                    ohlcv_by_symbol[symbol] = group.drop(columns=["symbol"]).set_index(
                        "timestamp"
                    )
            for symbol in symbols:  # keep requested-but-missing symbols visible
                ohlcv_by_symbol.setdefault(symbol, pd.DataFrame())
            sources["ohlcv_by_symbol"] = ohlcv_by_symbol

            # --- FRED macro (long -> wide) ---
            rows = (
                await session.execute(
                    select(
                        MacroIndicator.timestamp,
                        MacroIndicator.indicator,
                        MacroIndicator.value,
                    )
                    .where(
                        MacroIndicator.source == "FRED",
                        MacroIndicator.indicator.in_(["DTWEXBGS", "DGS10"]),
                    )
                    .order_by(MacroIndicator.timestamp)
                )
            ).all()
            macro = self._frame(rows, ["timestamp", "indicator", "value"])
            sources["macro"] = (
                macro.pivot(columns="indicator", values="value")
                .rename(columns={"DTWEXBGS": "dxy_value", "DGS10": "us10y_value"})
                if not macro.empty
                else pd.DataFrame(columns=["dxy_value", "us10y_value"])
            )

            # --- On-chain (long -> wide) ---
            rows = (
                await session.execute(
                    select(
                        OnChainMetric.timestamp,
                        OnChainMetric.metric,
                        OnChainMetric.value,
                    )
                    .where(
                        OnChainMetric.source == "BLOCKCHAIN_INFO",
                        OnChainMetric.asset == "BTC",
                        OnChainMetric.metric.in_(["hash-rate", "n-transactions"]),
                    )
                    .order_by(OnChainMetric.timestamp)
                )
            ).all()
            onchain = self._frame(rows, ["timestamp", "metric", "value"])
            sources["onchain"] = (
                onchain.pivot(columns="metric", values="value")
                .rename(columns={"hash-rate": "hash_rate", "n-transactions": "tx_count"})
                if not onchain.empty
                else pd.DataFrame(columns=["hash_rate", "tx_count"])
            )

            # --- Sentiment ---
            rows = (
                await session.execute(
                    select(SentimentData.timestamp, SentimentData.value)
                    .where(
                        SentimentData.source == "ALTERNATIVE_ME",
                        SentimentData.metric == "fear_greed_index",
                    )
                    .order_by(SentimentData.timestamp)
                )
            ).all()
            sentiment = self._frame(rows, ["timestamp", "fear_greed_value"])
            sources["sentiment"] = sentiment

            # --- Derivatives (long -> wide) ---
            rows = (
                await session.execute(
                    select(
                        DerivativesMetric.timestamp,
                        DerivativesMetric.metric,
                        DerivativesMetric.value,
                    )
                    .where(
                        DerivativesMetric.exchange == "binance",
                        DerivativesMetric.symbol == "BTCUSDT",
                        DerivativesMetric.metric.in_(["funding_rate", "open_interest"]),
                    )
                    .order_by(DerivativesMetric.timestamp)
                )
            ).all()
            derivatives = self._frame(rows, ["timestamp", "metric", "value"])
            sources["derivatives"] = (
                derivatives.pivot(columns="metric", values="value")
                if not derivatives.empty
                else pd.DataFrame(columns=["funding_rate", "open_interest"])
            )

            # --- Supply metrics per symbol (Part 4; often EMPTY) ---
            rows = (
                await session.execute(
                    select(
                        SupplyMetric.symbol,
                        SupplyMetric.timestamp,
                        SupplyMetric.circulating_supply,
                        SupplyMetric.total_supply,
                    )
                    .where(SupplyMetric.symbol.in_(symbols))
                    .order_by(SupplyMetric.symbol, SupplyMetric.timestamp)
                )
            ).all()
            supply_all = pd.DataFrame(
                rows,
                columns=["symbol", "timestamp", "circulating_supply", "total_supply"],
            )
            supply_by_symbol: dict[str, pd.DataFrame] = {}
            if not supply_all.empty:
                supply_all["timestamp"] = pd.to_datetime(
                    supply_all["timestamp"], utc=True
                ).dt.tz_localize(None)
                for column in ("circulating_supply", "total_supply"):
                    supply_all[column] = pd.to_numeric(
                        supply_all[column], errors="coerce"
                    ).astype(float)
                for symbol, group in supply_all.groupby("symbol", sort=False):
                    supply_by_symbol[symbol] = group.drop(columns=["symbol"]).set_index(
                        "timestamp"
                    )
            sources["supply_by_symbol"] = supply_by_symbol

            # --- Tokenomics events per symbol (Part 4; often EMPTY) ---
            rows = (
                await session.execute(
                    select(
                        TokenomicsEvent.symbol,
                        TokenomicsEvent.event_date,
                        TokenomicsEvent.event_type,
                        TokenomicsEvent.supply_impact_pct,
                    )
                    .where(TokenomicsEvent.symbol.in_(symbols))
                    .order_by(TokenomicsEvent.symbol, TokenomicsEvent.event_date)
                )
            ).all()
            events_all = pd.DataFrame(
                rows,
                columns=["symbol", "event_date", "event_type", "supply_impact_pct"],
            )
            events_by_symbol: dict[str, pd.DataFrame] = {}
            if not events_all.empty:
                events_all["event_date"] = pd.to_datetime(
                    events_all["event_date"], utc=True
                ).dt.tz_localize(None)
                events_all["event_type"] = events_all["event_type"].map(
                    lambda value: value.value if hasattr(value, "value") else str(value)
                )
                events_all["supply_impact_pct"] = pd.to_numeric(
                    events_all["supply_impact_pct"], errors="coerce"
                ).astype(float)
                for symbol, group in events_all.groupby("symbol", sort=False):
                    events_by_symbol[symbol] = group.drop(columns=["symbol"])
            sources["events_by_symbol"] = events_by_symbol
            logger.info(
                "[features] Part 4 inputs: supply for %d symbol(s), tokenomics "
                "events for %d symbol(s)%s",
                len(supply_by_symbol), len(events_by_symbol),
                "" if supply_by_symbol or events_by_symbol
                else " — both empty, DNA supply features will be NULL/0 (expected)",
            )

            # --- Market indices (already wide) ---
            rows = (
                await session.execute(
                    select(
                        MarketIndex.timestamp,
                        MarketIndex.sp500_close,
                        MarketIndex.nasdaq_close,
                    ).order_by(MarketIndex.timestamp)
                )
            ).all()
            sources["indices"] = self._frame(
                rows, ["timestamp", "sp500_close", "nasdaq_close"]
            )

        for name, frame in sources.items():
            if name == "ohlcv_by_symbol":
                for symbol, candles in frame.items():
                    logger.info(
                        "[features] loaded ohlcv '%s': %d rows%s",
                        symbol, len(candles), self._range_label(candles),
                    )
                continue
            if isinstance(frame, dict):  # per-symbol Part 4 inputs
                logger.info(
                    "[features] loaded '%s': %d symbol(s), %d row(s) total",
                    name, len(frame), sum(len(f) for f in frame.values()),
                )
                continue
            logger.info(
                "[features] loaded '%s': %d rows%s",
                name, len(frame), self._range_label(frame),
            )
        self._warn_if_base_stale(sources["ohlcv_by_symbol"])
        return sources

    @staticmethod
    def _warn_if_base_stale(ohlcv_by_symbol: dict[str, pd.DataFrame]) -> None:
        """Flag a Bitcoin history that has fallen behind the other assets.

        The collector already refuses to run with an unknown-age denominator
        (``BinanceMultiFetcher.abort_if_head_fails``). This is the second
        line of defence: the engine can also be invoked directly, and a BTC
        series lagging the altcoins would silently distort every
        ``price_btc_ratio`` through the capped forward-fill. Warn, do not
        raise — a stale run is still better than no features at all, and the
        BTC-only path from Phases 1-7 must keep working.
        """
        base = ohlcv_by_symbol.get(BASE_SYMBOL, pd.DataFrame())
        if base.empty:
            return
        base_last = base.index.max()
        others = {
            symbol: frame.index.max()
            for symbol, frame in ohlcv_by_symbol.items()
            if symbol != BASE_SYMBOL and not frame.empty
        }
        if not others:
            return
        newest_symbol = max(others, key=others.get)
        lag_days = (others[newest_symbol] - base_last).days
        if lag_days > 0:
            logger.warning(
                "[features] %s history ends %s but %s reaches %s (%d day(s) "
                "behind) — relative features will divide by a forward-filled "
                "Bitcoin price. Check the collector logs for a fallback/abort.",
                BASE_SYMBOL, base_last.date(), newest_symbol,
                others[newest_symbol].date(), lag_days,
            )

    @staticmethod
    def _range_label(frame: pd.DataFrame) -> str:
        """`` (first -> last)`` suffix for log lines; empty when no rows."""
        if frame.empty:
            return ""
        return f" ({frame.index.min().date()} -> {frame.index.max().date()})"

    @staticmethod
    def _frame(rows: list, columns: list[str]) -> pd.DataFrame:
        """Rows -> float DataFrame indexed by tz-naive daily timestamp."""
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None)
        frame = frame.set_index("timestamp")
        for column in frame.columns:
            if column not in ("indicator", "metric"):
                frame[column] = frame[column].astype(float)
        return frame

    # ------------------------------------------------------------------
    # 2 + 3) ALIGN & COMPUTE (pure pandas — unit-testable, no I/O)
    # ------------------------------------------------------------------
    def compute_features(self, sources: dict) -> pd.DataFrame:
        """Build the daily feature matrix for EVERY symbol (long format).

        Returns one block per symbol, concatenated: index = day, plus a
        ``symbol`` column and all :data:`DAILY_FEATURE_COLUMNS`.
        """
        ohlcv_by_symbol: dict[str, pd.DataFrame] = sources.get("ohlcv_by_symbol", {})
        btc = ohlcv_by_symbol.get(BASE_SYMBOL, pd.DataFrame())
        if btc.empty:
            raise FeatureEngineError(
                f"no {BASE_SYMBOL} OHLCV data in crypto_prices — run the "
                "collectors first (it is the denominator of every relative "
                "feature, so nothing can be computed without it)"
            )

        blocks: list[pd.DataFrame] = []
        # BTC first: it is the reference series every other block divides by.
        ordered = [BASE_SYMBOL] + [s for s in ohlcv_by_symbol if s != BASE_SYMBOL]
        for symbol in ordered:
            candles = ohlcv_by_symbol.get(symbol, pd.DataFrame())
            if candles.empty:
                logger.warning(
                    "[features] no candles for %s — skipping (collector may have "
                    "failed for this symbol; existing rows are left untouched)",
                    symbol,
                )
                continue
            blocks.append(
                self._compute_symbol_features(symbol, candles, sources, btc)
            )

        features = pd.concat(blocks, axis=0)
        logger.info(
            "[features] computed %d rows across %d symbol(s) x %d feature columns",
            len(features), len(blocks), len(DAILY_FEATURE_COLUMNS),
        )
        return features

    # ------------------------------------------------------------------
    def _compute_symbol_features(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        sources: dict,
        btc: pd.DataFrame,
    ) -> pd.DataFrame:
        """Full feature block for ONE symbol, on that symbol's date spine."""
        # Daily spine = this symbol's candle dates (crypto trades every day).
        features = pd.DataFrame(index=ohlcv.index.copy())
        features["asset_close"] = ohlcv["close"]
        features["asset_volume"] = ohlcv["volume"]

        # Bitcoin reference on THIS spine. For BTC itself the alignment is
        # the identity, so btc_close/btc_volume stay exactly what Phase 3
        # always wrote — that is what keeps the BTC pipeline unchanged.
        features["btc_close"] = self._align_to_spine(btc["close"], features.index)
        features["btc_volume"] = self._align_to_spine(btc["volume"], features.index)

        # --- join non-daily sources, then capped forward-fill ---
        for key in ("macro", "onchain", "sentiment", "derivatives", "indices"):
            frame = sources.get(key, pd.DataFrame())
            if not frame.empty:
                features = features.join(frame, how="left")
        # Which rows were genuinely OBSERVED, captured before the fill hides
        # the difference. A carried value is a fine answer for a LEVEL ("the
        # last funding rate we know of"), but it is not an observation, and
        # any CHANGE derived from two carried copies of one observation is
        # not a measurement at all — see funding_rate_change_1d below.
        observed = {
            column: features[column].notna()
            for column in _FFILL_COLUMNS
            if column in features.columns
        }
        for column in _FFILL_COLUMNS:
            if column not in features.columns:
                features[column] = np.nan
                observed[column] = pd.Series(False, index=features.index)
            limit = FFILL_LIMIT_OVERRIDES.get(column, FFILL_LIMIT_DAYS)
            features[column] = features[column].ffill(limit=limit)
        gap_report = {
            column: int(features[column].isna().sum()) for column in _FFILL_COLUMNS
        }
        logger.info(
            "[features] %s: NaN remaining after capped ffill: %s", symbol, gap_report
        )

        # --- Technical (ta library) ---
        close, high, low = ohlcv["close"], ohlcv["high"], ohlcv["low"]
        features["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        features["macd"] = macd.macd()
        features["macd_signal"] = macd.macd_signal()
        features["macd_hist"] = macd.macd_diff()
        bollinger = BollingerBands(close=close, window=20, window_dev=2)
        features["bb_upper"] = bollinger.bollinger_hband()
        features["bb_middle"] = bollinger.bollinger_mavg()
        features["bb_lower"] = bollinger.bollinger_lband()
        features["atr_14"] = AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()
        log_returns = np.log(close).diff()
        annualize = math.sqrt(TRADING_DAYS_PER_YEAR)
        features["realized_vol_7d"] = log_returns.rolling(7).std() * annualize
        features["realized_vol_30d"] = log_returns.rolling(30).std() * annualize

        # --- EWMA volatility (2026-08-02) ---
        # RiskMetrics-style exponentially weighted variance:
        #     var_t = lambda*var_{t-1} + (1-lambda)*r_{t-1}^2
        #
        # Added because the equally-weighted 30-day estimate above treats a
        # return from 30 days ago exactly like yesterday's, so it reacts to a
        # volatility regime change only after the change has half-filled the
        # window. Measured over 34.8k asset-days across all 21 assets, at a
        # nominal 90% band:
        #     horizon      +1d    +7d   +30d
        #     realized_30  89.2%  87.5%  84.3%
        #     EWMA 0.97    91.0%  89.8%  87.0%
        # Better at every horizon, and essentially calibrated out to a week.
        #
        # lambda=0.97 was chosen by that measurement, not by convention.
        # RiskMetrics' own 0.94 is tuned for equities; on this crypto sample
        # 0.94 over-reacts (89.9% at +1d but 81.1% at +30d) and 0.90 is worse
        # still. Do not "correct" it to 0.94 without re-running the test.
        #
        # `shift(1)` is load-bearing: the variance at row t must be built from
        # returns STRICTLY BEFORE t, or today's own move leaks into the
        # estimate that is used to judge today's decision.
        features["ewma_vol_30d"] = (
            log_returns.shift(1)
            .pow(2)
            .ewm(alpha=1.0 - EWMA_LAMBDA, min_periods=30, adjust=False)
            .mean()
            .pow(0.5)
            * annualize
        )

        # --- Macro RoC (%) ---
        features["dxy_roc_7d"] = features["dxy_value"].pct_change(7) * 100
        features["dxy_roc_30d"] = features["dxy_value"].pct_change(30) * 100
        features["us10y_roc_7d"] = features["us10y_value"].pct_change(7) * 100
        features["us10y_roc_30d"] = features["us10y_value"].pct_change(30) * 100

        # --- On-chain moving averages ---
        features["hash_rate_ma_7d"] = features["hash_rate"].rolling(7).mean()
        features["hash_rate_ma_30d"] = features["hash_rate"].rolling(30).mean()
        features["tx_count_ma_7d"] = features["tx_count"].rolling(7).mean()
        features["tx_count_ma_30d"] = features["tx_count"].rolling(30).mean()

        # --- Sentiment RoC (%) ---
        features["fear_greed_roc_7d"] = features["fear_greed_value"].pct_change(7) * 100

        # --- Liquidity ---
        # A one-day change only exists if funding was OBSERVED on both days.
        #
        # Before this guard, `diff()` ran over the forward-filled series, so
        # two carried copies of one July reading produced a change of exactly
        # 0.0 — and `funding_rate` is capped at a 35-day fill, so that zero
        # could repeat for a MONTH. Measured 2026-08-08: eight consecutive
        # days of `change_1d = 0` while the newest real observation was
        # 2026-07-31. The pillar reading that is "funding is flat, no leverage
        # stress" was a statement about days nobody looked at.
        #
        # This is the same defect the BTC-proxy on-chain columns and the
        # unlock fallback had: ABSENCE NARRATED AS INFORMATION. NULL is the
        # honest answer, and it is the one every downstream consumer already
        # knows how to handle.
        #
        # Note the rule is "observed at t AND at t-1", not merely "observed at
        # t": if the previous real reading was three days back, the difference
        # is a three-day change wearing a one-day label.
        funding_observed = observed.get("funding_rate")
        features["funding_rate_change_1d"] = features["funding_rate"].diff()
        if funding_observed is not None:
            real_change = funding_observed & funding_observed.shift(1, fill_value=False)
            features.loc[~real_change, "funding_rate_change_1d"] = np.nan
        volume = features["btc_volume"].replace(0, np.nan)  # avoid divide-by-zero
        features["oi_volume_ratio"] = features["open_interest"] / volume

        # --- Market indices RoC (%) ---
        features["sp500_roc_7d"] = features["sp500_close"].pct_change(7) * 100
        features["nasdaq_roc_7d"] = features["nasdaq_close"].pct_change(7) * 100

        # --- Relative strength vs Bitcoin (Phase 8) ---
        # The BTC-denominated price series. A zero BTC close would be a data
        # bug rather than a real quote, so it is nulled instead of producing
        # an infinity that later silently becomes NULL.
        btc_close = features["btc_close"].replace(0, np.nan)
        features["price_btc_ratio"] = features["asset_close"] / btc_close
        # % change OF THE RATIO = compounded out/under-performance vs BTC
        # over the window. pandas 3 does not pad across NaNs, so a gap
        # propagates instead of inventing a number.
        features["rs_7d"] = features["price_btc_ratio"].pct_change(7) * 100
        features["rs_30d"] = features["price_btc_ratio"].pct_change(30) * 100

        # --- Coin DNA (Phase 8 Part 4) ---
        self._add_dna_features(features, symbol, sources, ohlcv, btc)

        # Keep exactly the persisted schema, in schema order, + the key.
        features["symbol"] = symbol
        features = features[["symbol", *DAILY_FEATURE_COLUMNS]]
        # inf/-inf -> NaN, but ONLY on numeric columns: the categorical DNA
        # columns hold label strings and `replace` on them is meaningless.
        numeric = [
            column
            for column in features.columns
            if column not in DAILY_FEATURE_CATEGORICAL_COLUMNS
            and column != "symbol"
        ]
        features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
        logger.info(
            "[features] %s: %d feature columns over %d days (%s -> %s)",
            symbol, len(DAILY_FEATURE_COLUMNS), len(features),
            features.index.min().date(), features.index.max().date(),
        )
        return features

    # ------------------------------------------------------------------
    # Coin DNA (Phase 8 Part 4)
    # ------------------------------------------------------------------
    def _add_dna_features(
        self,
        features: pd.DataFrame,
        symbol: str,
        sources: dict,
        ohlcv: pd.DataFrame,
        btc: pd.DataFrame,
    ) -> None:
        """Attach the per-coin "DNA" block, in place.

        Every branch here degrades to NULL (or 0 where 0 is a real answer)
        rather than raising: ``tokenomics_events`` and ``supply_metrics``
        are empty for most coins, and a coin nobody has filed unlocks for
        must not take the pipeline down.
        """
        index = features.index
        close = features["asset_close"]

        # --- Supply & tokenomics ---------------------------------------
        events = sources.get("events_by_symbol", {}).get(symbol, pd.DataFrame())
        days_to_unlock, shock_30d = self._unlock_features(events, index)
        features["days_to_next_unlock"] = days_to_unlock
        features["supply_shock_score_30d"] = shock_30d

        supply = sources.get("supply_by_symbol", {}).get(symbol, pd.DataFrame())
        if supply.empty:
            circulating = pd.Series(np.nan, index=index)
            total = pd.Series(np.nan, index=index)
        else:
            # Supply changes slowly and is polled once a day, so a capped
            # ffill is safe. NEVER bfill: that would let tomorrow's supply
            # explain today's price.
            circulating = self._align_to_spine(
                supply["circulating_supply"], index, limit=FFILL_LIMIT_DAYS
            )
            total = self._align_to_spine(
                supply["total_supply"], index, limit=FFILL_LIMIT_DAYS
            )

        features["inflation_rate_7d"] = circulating.pct_change(7) * 100
        features["supply_utilization"] = circulating / total.replace(0, np.nan)

        # --- Volatility regime -----------------------------------------
        realized_vol = features["realized_vol_30d"]
        percentile = (
            realized_vol.rolling(
                VOL_PERCENTILE_WINDOW, min_periods=VOL_PERCENTILE_MIN_PERIODS
            ).rank(pct=True)
            * 100
        )
        features["vol_percentile_30d"] = percentile
        features["vol_regime"] = percentile.map(self._vol_regime)

        # --- Liquidity & market structure -------------------------------
        # Binance base-asset volume over circulating supply: both in coin
        # units, so the turnover ratio is dimensionless and comparable
        # across assets priced anywhere from 1e-8 to 1e5 dollars.
        features["liquidity_score"] = features["asset_volume"] / circulating.replace(
            0, np.nan
        )
        market_cap = close * circulating
        features["market_cap_tier"] = market_cap.map(self._market_cap_tier)

        asset_returns = np.log(close.replace(0, np.nan)).diff()
        btc_returns = np.log(
            self._align_to_spine(btc["close"], index).replace(0, np.nan)
        ).diff()
        covariance = asset_returns.rolling(BETA_WINDOW).cov(btc_returns)
        btc_variance = btc_returns.rolling(BETA_WINDOW).var()
        features["btc_beta_30d"] = covariance / btc_variance.replace(0, np.nan)
        features["btc_corr_30d"] = asset_returns.rolling(BETA_WINDOW).corr(btc_returns)

        # --- Momentum ---------------------------------------------------
        # Two 7-day trends, each squashed to [-1, 1], averaged and scaled to
        # [-100, +100] to match the sub-agents' conviction scale.
        rsi_trend = np.clip(
            (features["rsi_14"] - features["rsi_14"].shift(7))
            / _MOMENTUM_RSI_FULL_SCALE,
            -1.0,
            1.0,
        )
        macd_hist_pct = features["macd_hist"] / close.replace(0, np.nan) * 100
        macd_trend = np.clip(
            (macd_hist_pct - macd_hist_pct.shift(7)) / _MOMENTUM_MACD_FULL_SCALE,
            -1.0,
            1.0,
        )
        features["momentum_score"] = (rsi_trend + macd_trend) * 50

        # Signal-to-noise over the SAME horizon on both legs: a 30-day move
        # compared against a 30-day volatility. realized_vol_30d is
        # ANNUALISED, so it is scaled down by sqrt(30/365) first — dividing
        # a monthly return by an annual sigma would understate the ratio by
        # roughly 3.5x and make the number uninterpretable.
        return_30d = close.pct_change(30)
        vol_30d_period = realized_vol * math.sqrt(30.0 / TRADING_DAYS_PER_YEAR)
        features["trend_strength"] = return_30d.abs() / vol_30d_period.replace(
            0, np.nan
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _unlock_features(
        events: pd.DataFrame, index: pd.DatetimeIndex
    ) -> tuple[pd.Series, pd.Series]:
        """``(days_to_next_unlock, supply_shock_score_30d)`` for one symbol.

        With no events filed: days -> NaN (unknown) and shock -> 0.0 (a real
        answer: nothing is scheduled). The asymmetry is deliberate and is
        exactly what the brief asks for.
        """
        days = pd.Series(np.nan, index=index, dtype="float64")
        shock = pd.Series(0.0, index=index, dtype="float64")
        if events.empty:
            return days, shock

        increasing = events[
            events["event_type"].isin(SUPPLY_INCREASING_EVENT_TYPES)
        ]["event_date"]
        stock = events[events["event_type"].isin(SUPPLY_STOCK_EVENT_TYPES)]

        unlock_dates = np.sort(increasing.unique()) if not increasing.empty else None
        for position, day in enumerate(index):
            if unlock_dates is not None:
                future = unlock_dates[unlock_dates >= np.datetime64(day)]
                if future.size:
                    days.iloc[position] = (
                        pd.Timestamp(future[0]) - pd.Timestamp(day)
                    ).days
            if not stock.empty:
                window = stock[
                    (stock["event_date"] >= day)
                    & (
                        stock["event_date"]
                        < day + pd.Timedelta(days=SUPPLY_SHOCK_HORIZON_DAYS)
                    )
                ]
                if not window.empty:
                    shock.iloc[position] = float(
                        window["supply_impact_pct"].fillna(0.0).sum()
                    )
        return days, shock

    @staticmethod
    def _vol_regime(percentile: float) -> str | None:
        if percentile is None or pd.isna(percentile):
            return None
        if percentile < VOL_REGIME_LOW_PCT:
            return VolRegime.LOW.value
        if percentile > VOL_REGIME_HIGH_PCT:
            return VolRegime.HIGH.value
        return VolRegime.NORMAL.value

    @staticmethod
    def _market_cap_tier(market_cap: float) -> str | None:
        if market_cap is None or pd.isna(market_cap) or market_cap <= 0:
            return None
        if market_cap > MARKET_CAP_LARGE_USD:
            return MarketCapTier.LARGE.value
        if market_cap >= MARKET_CAP_MID_USD:
            return MarketCapTier.MID.value
        return MarketCapTier.SMALL.value

    # ------------------------------------------------------------------
    @staticmethod
    def _align_to_spine(
        series: pd.Series, spine: pd.DatetimeIndex, limit: int | None = None
    ) -> pd.Series:
        """Project a reference series onto another symbol's date spine.

        Safe forward-fill, in the precise sense the task requires:

        * reindex onto the UNION first, so the fill sees the reference's own
          observations before any gap is filled;
        * ``ffill`` only — never ``bfill``. Filling backwards would leak a
          future BTC price into a past row (look-ahead bias), which would
          quietly flatter every backtest built on these features;
        * capped at :data:`BTC_ALIGN_FFILL_DAYS`, so a genuine outage leaves
          an honest NULL rather than a stale price pretending to be current;
        * dates before the reference series begins stay NaN — correct, since
          there is no BTC price to divide by yet.
        """
        if series.empty:
            return pd.Series(np.nan, index=spine)
        combined = series.reindex(series.index.union(spine)).ffill(
            limit=BTC_ALIGN_FFILL_DAYS if limit is None else limit
        )
        return combined.reindex(spine)

    # ------------------------------------------------------------------
    # 4) SAVE
    # ------------------------------------------------------------------
    async def save_features(self, features: pd.DataFrame) -> int:
        """Idempotent batched upsert of the matrix into ``daily_features``.

        Conflict target is the composite PK ``(timestamp, symbol)``: one
        symbol's rows can never overwrite another's, and re-running the
        engine refreshes values in place instead of duplicating them.
        """
        if features.empty:
            logger.warning("[features] nothing to save — empty feature matrix")
            return 0

        payload: list[dict] = []
        for day, row in features.iterrows():
            record: dict = {
                "timestamp": datetime(day.year, day.month, day.day, tzinfo=UTC),
                "symbol": row["symbol"],
            }
            for column in DAILY_FEATURE_COLUMNS:
                value = row[column]
                if column in DAILY_FEATURE_CATEGORICAL_COLUMNS:
                    # Label strings, not numbers — never send these through
                    # the Decimal conversion (float('HIGH') would explode).
                    record[column] = None if pd.isna(value) else str(value)
                elif column == "days_to_next_unlock":
                    # INTEGER column: asyncpg rejects a Decimal here.
                    decimal_value = self._decimal(value)
                    record[column] = (
                        None if decimal_value is None else int(decimal_value)
                    )
                else:
                    record[column] = self._decimal(value)
            payload.append(record)

        table = DailyFeature.__table__
        written = 0
        async with session_scope() as session:
            for start in range(0, len(payload), _UPSERT_CHUNK_SIZE):
                chunk = payload[start : start + _UPSERT_CHUNK_SIZE]
                statement = pg_insert(table).values(chunk)
                update_set = {
                    column: statement.excluded[column]
                    for column in DAILY_FEATURE_COLUMNS
                }
                update_set["ingested_at"] = func.now()
                statement = statement.on_conflict_do_update(
                    index_elements=["timestamp", "symbol"], set_=update_set
                )
                await session.execute(statement)
                written += len(chunk)

        by_symbol = features["symbol"].value_counts().to_dict()
        logger.info(
            "[features] upserted %d rows into 'daily_features' %s",
            written, by_symbol,
        )
        return written

    @staticmethod
    def _decimal(value: object) -> Decimal | None:
        """Float -> Decimal for NUMERIC columns; NaN/inf -> NULL."""
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return Decimal(repr(number))

    # ------------------------------------------------------------------
    async def run(self) -> int:
        """Load -> compute -> save; returns the number of rows persisted."""
        logger.info("[features] FeatureEngine run started")
        sources = await self.load_source_data()
        features = self.compute_features(sources)
        written = await self.save_features(features)
        logger.info("[features] FeatureEngine run finished: %d rows", written)
        return written

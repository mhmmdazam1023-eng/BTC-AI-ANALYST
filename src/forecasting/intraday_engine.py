"""IntradayForecastEngine: near-term BTC forecasts over 4-hour bars (Part 8).

Horizons: 1 bar (4h), 3 bars (12h), 6 bars (24h). Bitcoin only.

Reuses, not reimplements
------------------------
The Monte Carlo core is :meth:`ForecastEngine.simulate_horizon` **verbatim**.
That function turned out to be entirely step-agnostic: its ``horizon_days``
parameter is really "number of steps", ``block_size`` is "steps per block",
and ``daily_drift`` is "drift per step". Nothing in it knows about calendars.
So this module supplies a 4h return pool and step counts, and the bootstrap
machinery — including the Phase 7 affine calibration transform — is shared.

The parameter names still say "days" because renaming them would touch the
daily engine and its tests for no functional gain. Every call site here
passes BARS, and says so.

Anchoring — why it is bar-aligned
---------------------------------
``base_at`` is the open time of the last CLOSED 4h bar and ``base_price`` is
that bar's close. ``target_at = base_at + horizon_bars * 4h``, also exactly
on a bar boundary. That is what makes evaluation an exact ``crypto_prices``
lookup rather than an interpolation — the property whose absence caused the
2026-07-25 evaluator bug.

A live mid-price would be "fresher" but would leave ``target_at`` between
bars and make the realized comparison approximate. The prompt layer can and
does show the live price separately; the *forecast* stays anchored to a
final bar.

Calibration for the 4h scale — deliberately NOT scaled-down daily numbers
------------------------------------------------------------------------
* ``BLOCK_SIZE_BARS = 12`` (2 days). Intraday returns cluster on two cycles
  the daily engine cannot see: the 8-hour funding-rate cycle (2 bars) and the
  Asia/Europe/US session rotation (6 bars). A 12-bar block spans whole
  multiples of both, so a sampled block carries an intact daily shape instead
  of slicing one in half.
* ``LOOKBACK_BARS = 2190`` (~1 year). Not 3 years: intraday microstructure
  drifts (fee tiers, venue mix, market-maker behaviour), and 2190 samples is
  already a large bootstrap pool. Longer history here buys staleness, not
  precision.

Expectation, stated honestly
---------------------------
Crypto at a 4-24h horizon is far closer to a random walk than at 7-180 days.
A directional hit-rate near 50% is the realistic prior, not the ~100% seen in
the daily backtest. The evaluation/learning loop exists to measure that
rather than assume it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import (
    BASE_SYMBOL,
    INTRADAY_TIMEFRAME,
    CryptoPrice,
    IntradayForecast,
    IntradayLearningMetric,
)
from src.forecasting.engine import (
    COVERAGE_NARROW_ABOVE,
    COVERAGE_WIDEN_BELOW,
    MAX_BIAS_CORRECTION,
    VOL_NARROW_FACTOR,
    VOL_WIDEN_FACTOR,
    ForecastEngine,
    ForecastError,
)
from src.timeframes import bar_label, describe_horizon, floor_to_bar, target_bar_open

logger = logging.getLogger(__name__)

#: Horizons in 4-HOUR BARS: 4h, 12h, 24h.
HORIZONS_BARS: tuple[int, ...] = (1, 3, 6)

#: Bars per bootstrap block — 2 days. Spans whole multiples of both the
#: 8h funding cycle (2 bars) and the 6-bar session rotation.
BLOCK_SIZE_BARS = 12

#: Target bootstrap pool, ~1 year of 4h bars. The engine uses whatever exists
#: and warns when short (Binance's 1000-candle cap gives ~166 days per call;
#: deeper history needs the fetcher's ``since`` pagination).
LOOKBACK_BARS = 2190

#: Simulated paths per horizon. Same as the daily engine.
N_PATHS = 20_000

#: Below this many bars the block bootstrap is not worth running.
MIN_POOL_BARS = BLOCK_SIZE_BARS + 20


class IntradayForecastEngine:
    """Block-bootstrap forecasts over 4h bars, anchored to a closed bar."""

    def __init__(
        self,
        *,
        symbol: str = BASE_SYMBOL,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.symbol = symbol

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    async def _load_closed_closes(
        self, now: datetime | None = None
    ) -> tuple[list[datetime], np.ndarray]:
        """Closed 4h bar opens + closes, ascending.

        Only CLOSED bars: the training distribution must never contain a
        partial bar, whose range is "so far" rather than final.
        """
        reference = now or datetime.now(tz=UTC)
        cutoff = floor_to_bar(reference)

        query = (
            select(CryptoPrice.timestamp, CryptoPrice.close)
            .where(
                CryptoPrice.exchange == "binance",
                CryptoPrice.symbol == self.symbol,
                CryptoPrice.timeframe == INTRADAY_TIMEFRAME,
                CryptoPrice.timestamp < cutoff,
            )
            .order_by(CryptoPrice.timestamp.desc())
            .limit(LOOKBACK_BARS)
        )
        async with session_scope() as session:
            rows = (await session.execute(query)).all()

        if not rows:
            raise ForecastError(
                f"no CLOSED {INTRADAY_TIMEFRAME} candles for {self.symbol} "
                f"before {bar_label(cutoff)}"
            )

        ordered = list(reversed(rows))
        timestamps = [row.timestamp for row in ordered]
        closes = np.array([float(row.close) for row in ordered], dtype=np.float64)
        return timestamps, closes

    # ------------------------------------------------------------------
    async def _load_corrections(self) -> dict[int, dict[str, float]]:
        """Latest intraday learning metrics per horizon: {bars: {...}}.

        Reads ``intraday_learning_metrics`` — NOT ``learning_metrics``. Those
        rows key on ``horizon_days``; applying a 30-day bias correction to a
        4-hour forecast would be a silent unit error.
        """
        corrections: dict[int, dict[str, float]] = {}
        async with session_scope() as session:
            for horizon_bars in HORIZONS_BARS:
                row = (
                    await session.execute(
                        select(
                            IntradayLearningMetric.magnitude_bias,
                            IntradayLearningMetric.coverage_rate,
                        )
                        .where(IntradayLearningMetric.horizon_bars == horizon_bars)
                        .order_by(IntradayLearningMetric.evaluated_at.desc())
                        .limit(1)
                    )
                ).first()
                if row is None:
                    continue
                corrections[horizon_bars] = {
                    "magnitude_bias": float(row.magnitude_bias or 0.0),
                    "coverage_rate": (
                        float(row.coverage_rate)
                        if row.coverage_rate is not None
                        else None
                    ),
                }
        return corrections

    # ------------------------------------------------------------------
    @staticmethod
    def _derive_adjustment(
        horizon_bars: int, correction: dict[str, float] | None
    ) -> tuple[float, float]:
        """Map a horizon's learning metrics to (per_bar_drift, vol_factor).

        Same shape as ``ForecastEngine._derive_adjustment`` but the drift is
        spread across BARS, not days: ``ln(1+bias) / horizon_bars`` so the
        terminal median shifts by ~bias regardless of horizon length.
        """
        if not correction:
            return 0.0, 1.0

        bias = correction.get("magnitude_bias", 0.0) or 0.0
        bias = max(-MAX_BIAS_CORRECTION, min(MAX_BIAS_CORRECTION, bias))
        per_bar_drift = float(np.log1p(bias) / horizon_bars) if bias != 0.0 else 0.0

        coverage = correction.get("coverage_rate")
        vol_factor = 1.0
        if coverage is not None:
            if coverage < COVERAGE_WIDEN_BELOW:
                vol_factor = VOL_WIDEN_FACTOR
            elif coverage > COVERAGE_NARROW_ABOVE:
                vol_factor = VOL_NARROW_FACTOR
        return per_bar_drift, vol_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate_and_save(self, now: datetime | None = None) -> int:
        """Forecast every horizon from the last closed bar; upsert. Returns rows."""
        reference = now or datetime.now(tz=UTC)
        timestamps, closes = await self._load_closed_closes(now=reference)

        log_returns = np.diff(np.log(closes))
        pool = log_returns[-LOOKBACK_BARS:]
        if pool.shape[0] < MIN_POOL_BARS:
            raise ForecastError(
                f"intraday pool too short: {pool.shape[0]} return(s) < "
                f"{MIN_POOL_BARS} minimum (block size {BLOCK_SIZE_BARS})"
            )
        # LOOKBACK_BARS is a count of BARS; differencing consumes one, so a
        # fully-stocked pool holds LOOKBACK_BARS - 1 returns. Comparing
        # against LOOKBACK_BARS itself would make this warning fire forever,
        # which trains readers to ignore it.
        if pool.shape[0] < LOOKBACK_BARS - 1:
            logger.warning(
                "[4h-forecast] pool is %d return(s), below the %d target "
                "(~1 year of 4h bars) — usable, but backfill deeper 4h "
                "history via the fetcher's `since` pagination for a fuller "
                "distribution",
                pool.shape[0], LOOKBACK_BARS - 1,
            )

        base_at = timestamps[-1]
        base_price = float(closes[-1])

        # Deterministic per (anchor bar, symbol): re-running the same day
        # reproduces identical numbers, which is what makes the stored row
        # auditable months later.
        seed = int(base_at.timestamp()) ^ hash(self.symbol) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        corrections = await self._load_corrections()
        logger.info(
            "[4h-forecast] anchor %s close=%.2f | pool=%d bars | block=%d | "
            "paths=%d | horizons=%s",
            bar_label(base_at), base_price, pool.shape[0], BLOCK_SIZE_BARS,
            N_PATHS, ", ".join(describe_horizon(h) for h in HORIZONS_BARS),
        )

        payload: list[dict] = []
        for horizon_bars in HORIZONS_BARS:
            per_bar_drift, vol_factor = self._derive_adjustment(
                horizon_bars, corrections.get(horizon_bars)
            )
            # simulate_horizon is step-based: horizon_days == number of BARS,
            # block_size == bars per block, daily_drift == drift per BAR.
            stats = ForecastEngine.simulate_horizon(
                pool,
                base_price,
                horizon_bars,
                rng,
                n_paths=N_PATHS,
                block_size=BLOCK_SIZE_BARS,
                daily_drift=per_bar_drift,
                vol_factor=vol_factor,
            )
            target_at = target_bar_open(base_at, horizon_bars)

            if per_bar_drift != 0.0 or vol_factor != 1.0:
                logger.info(
                    "[4h-forecast] +%s: calibration applied "
                    "(drift/bar=%+.6f, vol x%.2f)",
                    describe_horizon(horizon_bars), per_bar_drift, vol_factor,
                )
            logger.info(
                "[4h-forecast] +%-14s target %s | p5=%.0f p50=%.0f p95=%.0f "
                "| P(up)=%.1f%%",
                describe_horizon(horizon_bars), bar_label(target_at),
                stats["p5"], stats["p50"], stats["p95"], stats["prob_up"] * 100,
            )

            payload.append(
                {
                    "symbol": self.symbol,
                    "base_at": base_at,
                    "target_at": target_at,
                    "horizon_bars": horizon_bars,
                    "predicted_price_p5": Decimal(str(round(stats["p5"], 10))),
                    "predicted_price_p50": Decimal(str(round(stats["p50"], 10))),
                    "predicted_price_p95": Decimal(str(round(stats["p95"], 10))),
                    "prob_price_up": stats["prob_up"],
                    "base_price": Decimal(str(round(base_price, 10))),
                    "pool_bars": int(pool.shape[0]),
                }
            )

        table = IntradayForecast.__table__
        async with session_scope() as session:
            statement = pg_insert(table).values(payload)
            statement = statement.on_conflict_do_update(
                constraint="uq_intraday_forecast_bar_horizon",
                set_={
                    column: statement.excluded[column]
                    for column in (
                        "target_at",
                        "predicted_price_p5",
                        "predicted_price_p50",
                        "predicted_price_p95",
                        "prob_price_up",
                        "base_price",
                        "pool_bars",
                    )
                },
            )
            await session.execute(statement)

        logger.info(
            "[4h-forecast] upserted %d forecast(s) into 'intraday_forecasts'",
            len(payload),
        )
        return len(payload)

"""ForecastEngine: Monte Carlo BLOCK-bootstrap price forecasts (Phase 6).

Method (pure numpy — no compiled dependencies):

1. Load daily closes of BTC/USDT and compute log-returns over the last
   :data:`LOOKBACK_DAYS` (~3 years; gracefully uses whatever history exists
   above :data:`MIN_HISTORY_DAYS`).
2. **Moving-block bootstrap**: sample overlapping 10-day blocks of ACTUAL
   historical returns (block size preserves volatility clustering — the
   defining stylized fact of BTC returns) and concatenate them into
   :data:`N_PATHS` simulated paths per horizon (7/30/60/90/180 days).
3. Final simulated price = base_price * exp(sum of sampled log-returns).
   Report the 5th/50th/95th percentiles and P(price > base).
4. Upsert one row per (forecast_date, horizon) into ``forecasts`` —
   re-running the same day updates, never duplicates.

Reproducibility: the RNG is seeded from the forecast date, so the same day's
forecast is identical across re-runs (and changes day to day).

Backtesting: ``generate_and_save_forecasts(as_of="YYYY-MM-DD")`` builds the
forecast using ONLY data up to that date — used by the evaluator tests and
for filling historical forecasts.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import BASE_SYMBOL, CryptoPrice, Decision, Forecast

logger = logging.getLogger(__name__)

#: Forecast horizons in days.
HORIZONS: tuple[int, ...] = (7, 30, 60, 90, 180)
#: Block length (days) — preserves volatility clustering.
BLOCK_SIZE = 10
#: Simulated paths per horizon.
N_PATHS = 20_000
#: Return history window (~3 years).
LOOKBACK_DAYS = 3 * 365
#: Refuse to forecast on flimsy history.
MIN_HISTORY_DAYS = 180

# --- Continuous-learning feedback loop (Phase 7) ---
#: Coverage below this ⇒ bands too narrow (overconfident) ⇒ widen volatility.
COVERAGE_WIDEN_BELOW = 0.85
#: Coverage above this ⇒ bands too wide (underconfident) ⇒ narrow volatility.
COVERAGE_NARROW_ABOVE = 0.95
VOL_WIDEN_FACTOR = 1.1
VOL_NARROW_FACTOR = 0.95
#: Clamp the applied magnitude-bias correction (fraction) for stability, so a
#: noisy/extreme learning estimate can never blow up the simulation.
MAX_BIAS_CORRECTION = 0.25


class ForecastError(RuntimeError):
    """Raised when a forecast cannot be produced (insufficient data, ...)."""


class ForecastEngine:
    """Monte Carlo block-bootstrap forecaster over the BTC daily closes."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    async def _load_closes(
        self, as_of: datetime | None
    ) -> tuple[list[datetime], np.ndarray]:
        """Daily close series (ascending) up to ``as_of`` (inclusive)."""
        query = (
            select(CryptoPrice.timestamp, CryptoPrice.close)
            .where(
                CryptoPrice.exchange == "binance",
                CryptoPrice.symbol == "BTC/USDT",
                CryptoPrice.timeframe == "1d",
            )
            .order_by(CryptoPrice.timestamp)
        )
        if as_of is not None:
            query = query.where(CryptoPrice.timestamp <= as_of)
        async with session_scope() as session:
            rows = (await session.execute(query)).all()
        if not rows:
            raise ForecastError("no BTC/USDT candles in crypto_prices")
        timestamps = [row.timestamp for row in rows]
        closes = np.array([float(row.close) for row in rows], dtype=np.float64)
        return timestamps, closes

    async def _latest_decision_snapshot(self, forecast_date: datetime) -> dict | None:
        """Most recent BITCOIN decision at/before the forecast date.

        The ``symbol`` filter is mandatory since Phase 8 Part 6: ``decisions``
        now holds one row per (day, asset), so an unfiltered
        ``ORDER BY timestamp DESC LIMIT 1`` would staple an arbitrary
        altcoin's BUY/SELL onto Bitcoin's forecast snapshot. This engine is
        BTC-only by design (see the module docstring).
        """
        async with session_scope() as session:
            row = (
                await session.execute(
                    select(Decision)
                    .where(
                        Decision.symbol == BASE_SYMBOL,
                        Decision.timestamp <= forecast_date,
                    )
                    .order_by(Decision.timestamp.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        return {
            "decision_date": row.timestamp.date().isoformat(),
            "action": row.action,
            "confidence_score": row.confidence_score,
        }

    # ------------------------------------------------------------------
    # Simulation core (pure numpy, deterministic per seed)
    # ------------------------------------------------------------------
    @staticmethod
    def simulate_horizon(
        log_returns: np.ndarray,
        base_price: float,
        horizon_days: int,
        rng: np.random.Generator,
        *,
        n_paths: int = N_PATHS,
        block_size: int = BLOCK_SIZE,
        daily_drift: float = 0.0,
        vol_factor: float = 1.0,
    ) -> dict[str, float]:
        """Moving-block bootstrap simulation for one horizon.

        Continuous-learning corrections (Phase 7), applied as an affine
        transform to the return POOL before block sampling (block/clustering
        structure preserved):

        * ``vol_factor`` scales each return's deviation from the pool mean —
          widening (>1) or narrowing (<1) the terminal distribution.
        * ``daily_drift`` shifts every return up/down — moving the median.
          The caller derives it from the horizon-level magnitude bias as
          ``ln(1 + bias) / horizon`` so the terminal median shifts by ~bias.

        Returns dict with p5/p50/p95 final prices and prob_up.
        """
        n_returns = log_returns.shape[0]
        if n_returns < block_size + 1:
            raise ForecastError(
                f"history too short for block bootstrap "
                f"({n_returns} returns < block size {block_size})"
            )

        # Affine calibration transform on the return pool.
        if vol_factor != 1.0 or daily_drift != 0.0:
            pool_mean = float(np.mean(log_returns))
            log_returns = pool_mean + (log_returns - pool_mean) * vol_factor + daily_drift

        n_blocks = int(np.ceil(horizon_days / block_size))
        max_start = n_returns - block_size  # inclusive upper bound for starts
        # (n_paths, n_blocks) random block starts over the (adjusted) history.
        starts = rng.integers(0, max_start + 1, size=(n_paths, n_blocks))
        # Gather blocks -> (n_paths, n_blocks, block_size) -> flatten & trim.
        offsets = np.arange(block_size)
        sampled = log_returns[starts[:, :, None] + offsets[None, None, :]]
        paths = sampled.reshape(n_paths, n_blocks * block_size)[:, :horizon_days]

        final_prices = base_price * np.exp(paths.sum(axis=1))
        p5, p50, p95 = np.percentile(final_prices, [5.0, 50.0, 95.0])
        prob_up = float(np.mean(final_prices > base_price))
        return {
            "p5": float(p5),
            "p50": float(p50),
            "p95": float(p95),
            "prob_up": prob_up,
        }

    # ------------------------------------------------------------------
    # Continuous-learning corrections (Phase 7)
    # ------------------------------------------------------------------
    async def _load_corrections(self) -> dict[int, dict[str, float]]:
        """Latest learning metrics per horizon: {horizon: {bias, coverage}}.

        Reads one row per horizon (the most recent by ``evaluated_at``).
        Returns an empty dict when the learning layer hasn't run yet, so the
        forecast simply runs uncorrected.
        """
        # Local import avoids a circular import (learning imports this module).
        from src.db.models import LearningMetric

        corrections: dict[int, dict[str, float]] = {}
        async with session_scope() as session:
            for horizon in HORIZONS:
                row = (
                    await session.execute(
                        select(
                            LearningMetric.magnitude_bias,
                            LearningMetric.coverage_rate,
                        )
                        .where(LearningMetric.horizon_days == horizon)
                        .order_by(LearningMetric.evaluated_at.desc())
                        .limit(1)
                    )
                ).first()
                if row is None:
                    continue
                corrections[horizon] = {
                    "magnitude_bias": (
                        float(row.magnitude_bias)
                        if row.magnitude_bias is not None
                        else 0.0
                    ),
                    "coverage_rate": (
                        float(row.coverage_rate)
                        if row.coverage_rate is not None
                        else None
                    ),
                }
        return corrections

    @staticmethod
    def _derive_adjustment(
        horizon: int, correction: dict[str, float] | None
    ) -> tuple[float, float]:
        """Map a horizon's learning metrics to (daily_drift, vol_factor)."""
        if not correction:
            return 0.0, 1.0

        # Bias correction: clamp, then spread ln(1+bias) evenly across days.
        bias = correction.get("magnitude_bias", 0.0) or 0.0
        bias = max(-MAX_BIAS_CORRECTION, min(MAX_BIAS_CORRECTION, bias))
        daily_drift = math.log1p(bias) / horizon if bias != 0.0 else 0.0

        # Volatility scaling from coverage calibration.
        coverage = correction.get("coverage_rate")
        vol_factor = 1.0
        if coverage is not None:
            if coverage < COVERAGE_WIDEN_BELOW:
                vol_factor = VOL_WIDEN_FACTOR
            elif coverage > COVERAGE_NARROW_ABOVE:
                vol_factor = VOL_NARROW_FACTOR
        return daily_drift, vol_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate_and_save_forecasts(
        self,
        current_price: float | None = None,
        *,
        as_of: str | None = None,
    ) -> int:
        """Generate forecasts for every horizon and upsert into ``forecasts``.

        ``as_of`` (ISO date) restricts the input data for backtesting;
        default is the full history (live daily use). ``current_price``
        overrides the base price (defaults to the last close used).
        Returns the number of forecast rows written.
        """
        as_of_dt = (
            datetime.fromisoformat(as_of).replace(tzinfo=UTC) if as_of else None
        )
        timestamps, closes = await self._load_closes(as_of_dt)

        log_returns_full = np.diff(np.log(closes))
        log_returns = log_returns_full[-LOOKBACK_DAYS:]
        if log_returns.shape[0] < MIN_HISTORY_DAYS:
            raise ForecastError(
                f"only {log_returns.shape[0]} daily returns available "
                f"(minimum {MIN_HISTORY_DAYS}) — backfill more candles first"
            )
        if log_returns.shape[0] < LOOKBACK_DAYS:
            logger.warning(
                "[forecast] history below the %d-day target: using %d returns "
                "(%s -> %s)",
                LOOKBACK_DAYS, log_returns.shape[0],
                timestamps[-log_returns.shape[0] - 1].date(), timestamps[-1].date(),
            )

        forecast_date = timestamps[-1]
        base_price = float(closes[-1]) if current_price is None else float(current_price)
        # Deterministic per forecast date -> same-day re-runs are identical.
        seed = int(forecast_date.strftime("%Y%m%d"))
        snapshot = await self._latest_decision_snapshot(forecast_date)

        # Continuous-learning feedback: per-horizon bias & volatility correction.
        corrections = await self._load_corrections()
        if corrections:
            logger.info(
                "[forecast] learning corrections available for horizons: %s",
                sorted(corrections),
            )
        else:
            logger.info("[forecast] no learning metrics yet — forecasting uncorrected")

        logger.info(
            "[forecast] simulating %d paths x %s horizons from %s "
            "(base=%.2f, %d returns, block=%d, seed=%d)",
            N_PATHS, list(HORIZONS), forecast_date.date(),
            base_price, log_returns.shape[0], BLOCK_SIZE, seed,
        )

        payload: list[dict] = []
        for horizon in HORIZONS:
            rng = np.random.default_rng(seed + horizon)
            daily_drift, vol_factor = self._derive_adjustment(
                horizon, corrections.get(horizon)
            )
            stats = self.simulate_horizon(
                log_returns, base_price, horizon, rng,
                daily_drift=daily_drift, vol_factor=vol_factor,
            )
            if daily_drift != 0.0 or vol_factor != 1.0:
                logger.info(
                    "[forecast] +%3dd correction applied: bias_drift/day=%+.5f "
                    "(≈%+.2f%% at horizon), vol_factor=%.2f",
                    horizon, daily_drift, (math.exp(daily_drift * horizon) - 1) * 100,
                    vol_factor,
                )
            payload.append(
                {
                    "forecast_date": forecast_date,
                    "target_date": forecast_date + timedelta(days=horizon),
                    "horizon_days": horizon,
                    "predicted_price_p5": Decimal(f"{stats['p5']:.2f}"),
                    "predicted_price_p50": Decimal(f"{stats['p50']:.2f}"),
                    "predicted_price_p95": Decimal(f"{stats['p95']:.2f}"),
                    "prob_price_up": round(stats["prob_up"], 4),
                    "base_price": Decimal(f"{base_price:.2f}"),
                    "ai_decision_snapshot": snapshot,
                }
            )
            logger.info(
                "[forecast] +%3dd: p5=%.0f  p50=%.0f  p95=%.0f  P(up)=%.1f%%",
                horizon, stats["p5"], stats["p50"], stats["p95"],
                stats["prob_up"] * 100,
            )

        statement = pg_insert(Forecast.__table__).values(payload)
        update_set = {
            column: statement.excluded[column]
            for column in (
                "target_date",
                "predicted_price_p5",
                "predicted_price_p50",
                "predicted_price_p95",
                "prob_price_up",
                "base_price",
                "ai_decision_snapshot",
            )
        }
        update_set["ingested_at"] = func.now()
        statement = statement.on_conflict_do_update(
            constraint="uq_forecast_date_horizon", set_=update_set
        )
        async with session_scope() as session:
            await session.execute(statement)

        logger.info(
            "[forecast] %d forecast rows upserted for %s",
            len(payload), forecast_date.date(),
        )
        return len(payload)

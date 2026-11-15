"""LearningEvaluator: continuous-learning calibration metrics (Phase 7).

Turns the raw forecast-vs-actual history in ``forecast_evaluations`` into
per-horizon CALIBRATION diagnostics, persisted to ``learning_metrics``. The
ForecastEngine then reads the latest metrics and self-corrects future
forecasts (bias shift + volatility scaling).

Per horizon, over the trailing :data:`ROLLING_WINDOW` matured evaluations:

* rolling_mape    = mean(|price_error_pct|)                    [percent]
* magnitude_bias  = mean((actual - p50) / p50)                 [fraction]
                    >0 ⇒ model predicts too LOW (too pessimistic) ⇒ shift up.
* coverage_rate   = fraction of actuals within [p5, p95]       [0..1]
                    ~0.90 ideal; <0.85 overconfident; >0.95 underconfident.
* brier_score     = mean((prob_up - outcome)^2)                [0..1]
                    outcome = 1 if actual > base_price else 0; lower = better.

Guards (numerical stability, per CLAUDE.md):
* Global: if ``forecast_evaluations`` has < :data:`MIN_TOTAL_EVALUATIONS`
  rows, skip entirely and log "Not enough data for learning yet".
* Per horizon: need >= :data:`MIN_HORIZON_SAMPLES` matured evaluations to
  emit a row; horizons with fewer are skipped quietly.
* Never divide by zero (p50 <= 0 rows are dropped defensively).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import func, select

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import Forecast, ForecastEvaluation, LearningMetric
from src.forecasting.engine import HORIZONS

logger = logging.getLogger(__name__)

#: Trailing number of matured evaluations per horizon used for the metrics.
ROLLING_WINDOW = 30
#: Global gate — below this many total evaluations, learning is skipped.
MIN_TOTAL_EVALUATIONS = 10
#: Per-horizon minimum to emit a metrics row.
MIN_HORIZON_SAMPLES = 3


class LearningEvaluator:
    """Computes per-horizon calibration metrics and persists them."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    async def evaluate_and_save(self) -> int:
        """Compute metrics for every horizon; persist. Returns rows saved."""
        async with session_scope() as session:
            total = (
                await session.execute(select(func.count(ForecastEvaluation.id)))
            ).scalar_one()

            if total < MIN_TOTAL_EVALUATIONS:
                logger.warning(
                    "[learning] Not enough data for learning yet "
                    "(%d/%d evaluations) — skipping.",
                    total, MIN_TOTAL_EVALUATIONS,
                )
                return 0

            now = datetime.now(tz=UTC)
            saved = 0
            for horizon in HORIZONS:
                metrics = await self._horizon_metrics(session, horizon)
                if metrics is None:
                    continue
                session.add(
                    LearningMetric(
                        evaluated_at=now,
                        horizon_days=horizon,
                        sample_size=metrics["n"],
                        rolling_mape=metrics["rolling_mape"],
                        magnitude_bias=metrics["magnitude_bias"],
                        coverage_rate=metrics["coverage_rate"],
                        brier_score=metrics["brier_score"],
                    )
                )
                saved += 1
                logger.info(
                    "[learning] +%3dd (n=%d): MAPE=%.2f%%  bias=%+.2f%%  "
                    "coverage=%.1f%%  brier=%.3f",
                    horizon, metrics["n"], metrics["rolling_mape"],
                    metrics["magnitude_bias"] * 100, metrics["coverage_rate"] * 100,
                    metrics["brier_score"],
                )

        if saved == 0:
            logger.info(
                "[learning] enough total evaluations, but no horizon reached "
                "the %d-sample minimum yet — no metrics saved.",
                MIN_HORIZON_SAMPLES,
            )
        else:
            logger.info("[learning] saved calibration metrics for %d horizon(s)", saved)
        return saved

    # ------------------------------------------------------------------
    async def _horizon_metrics(self, session, horizon: int) -> dict | None:
        """Compute the four metrics for one horizon, or None if too few rows."""
        rows = (
            await session.execute(
                select(
                    ForecastEvaluation.actual_price,
                    Forecast.predicted_price_p5,
                    Forecast.predicted_price_p50,
                    Forecast.predicted_price_p95,
                    Forecast.prob_price_up,
                    Forecast.base_price,
                )
                .join(Forecast, Forecast.id == ForecastEvaluation.forecast_id)
                .where(Forecast.horizon_days == horizon)
                .order_by(ForecastEvaluation.target_date.desc())
                .limit(ROLLING_WINDOW)
            )
        ).all()

        # Drop any defensive-bad rows (p50 <= 0) to avoid divide-by-zero.
        actual, p5, p50, p95, prob_up, base = [], [], [], [], [], []
        for row in rows:
            if row.predicted_price_p50 is None or float(row.predicted_price_p50) <= 0:
                continue
            actual.append(float(row.actual_price))
            p5.append(float(row.predicted_price_p5))
            p50.append(float(row.predicted_price_p50))
            p95.append(float(row.predicted_price_p95))
            prob_up.append(float(row.prob_price_up))
            base.append(float(row.base_price))

        n = len(actual)
        if n < MIN_HORIZON_SAMPLES:
            logger.debug(
                "[learning] +%dd: only %d matured evaluation(s) (<%d) — skipping",
                horizon, n, MIN_HORIZON_SAMPLES,
            )
            return None

        actual = np.array(actual)
        p5 = np.array(p5)
        p50 = np.array(p50)
        p95 = np.array(p95)
        prob_up = np.array(prob_up)
        base = np.array(base)

        rel_error = (actual - p50) / p50  # fraction
        rolling_mape = float(np.mean(np.abs(rel_error)) * 100)  # percent
        magnitude_bias = float(np.mean(rel_error))  # fraction
        coverage_rate = float(np.mean((actual >= p5) & (actual <= p95)))  # 0..1
        outcome = (actual > base).astype(float)  # 1 if up else 0
        brier_score = float(np.mean((prob_up - outcome) ** 2))

        return {
            "n": n,
            "rolling_mape": rolling_mape,
            "magnitude_bias": magnitude_bias,
            "coverage_rate": coverage_rate,
            "brier_score": brier_score,
        }

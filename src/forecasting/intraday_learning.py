"""IntradayLearningEvaluator: calibration metrics per 4H horizon (Part 8).

The intraday twin of :mod:`src.forecasting.learning`. Turns the raw
forecast-vs-actual history in ``intraday_forecast_evaluations`` into per-horizon
calibration diagnostics in ``intraday_learning_metrics``, which
:class:`~src.forecasting.intraday_engine.IntradayForecastEngine` then reads to
self-correct future 4H forecasts.

Metric definitions are IDENTICAL to Phase 7 so the two scoreboards are
directly comparable (see :class:`~src.db.models.LearningMetric`):

* ``rolling_mape``   = mean(|price_error_pct|)                     [percent]
* ``magnitude_bias`` = mean((actual - p50) / p50)                  [fraction]
                       >0 => model predicts too LOW => shift up.
* ``coverage_rate``  = fraction of actuals inside [p5, p95]        [0..1]
* ``brier_score``    = mean((prob_up - outcome)^2)                 [0..1]

Why the sample gates are LOWER than Phase 7
-------------------------------------------
The daily layer needs 10 total / 3 per horizon. The intraday layer produces 3
forecasts per day and matures all three within 24h, so it accumulates
evaluations roughly 6x faster than the daily layer at the 7-day horizon. The
gates here are therefore expressed in the same spirit — enough samples for a
mean to mean something — but reached in days rather than months:
:data:`MIN_TOTAL_EVALUATIONS` 12 and :data:`MIN_HORIZON_SAMPLES` 5.

``ROLLING_WINDOW`` is 60 (~20 days at 3 forecasts/day). Longer than Phase 7's
30 because intraday error is noisier per observation, so a mean over fewer
samples would chase noise into the bias correction.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import func, select

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import (
    IntradayForecast,
    IntradayForecastEvaluation,
    IntradayLearningMetric,
)
from src.forecasting.intraday_engine import HORIZONS_BARS
from src.timeframes import describe_horizon

logger = logging.getLogger(__name__)

#: Trailing matured evaluations per horizon used for the metrics.
ROLLING_WINDOW = 60
#: Global gate — below this many total intraday evaluations, learning is skipped.
MIN_TOTAL_EVALUATIONS = 12
#: Per-horizon minimum to emit a metrics row.
MIN_HORIZON_SAMPLES = 5


class IntradayLearningEvaluator:
    """Computes per-horizon 4H calibration metrics and persists them."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    async def evaluate_and_save(self) -> int:
        """Compute metrics for every 4H horizon; persist. Returns rows saved."""
        async with session_scope() as session:
            total = (
                await session.execute(
                    select(func.count(IntradayForecastEvaluation.id))
                )
            ).scalar_one()

            if total < MIN_TOTAL_EVALUATIONS:
                logger.warning(
                    "[4h-learning] Not enough data for intraday learning yet "
                    "(%d/%d evaluations) — skipping.",
                    total, MIN_TOTAL_EVALUATIONS,
                )
                return 0

            now = datetime.now(tz=UTC)
            saved = 0
            for horizon_bars in HORIZONS_BARS:
                metrics = await self._horizon_metrics(session, horizon_bars)
                if metrics is None:
                    continue
                session.add(
                    IntradayLearningMetric(
                        evaluated_at=now,
                        horizon_bars=horizon_bars,
                        sample_size=metrics["n"],
                        rolling_mape=metrics["rolling_mape"],
                        magnitude_bias=metrics["magnitude_bias"],
                        coverage_rate=metrics["coverage_rate"],
                        brier_score=metrics["brier_score"],
                    )
                )
                saved += 1
                logger.info(
                    "[4h-learning] +%-14s (n=%d): MAPE=%.2f%%  bias=%+.2f%%  "
                    "coverage=%.1f%%  brier=%.3f",
                    describe_horizon(horizon_bars), metrics["n"],
                    metrics["rolling_mape"], metrics["magnitude_bias"] * 100,
                    metrics["coverage_rate"] * 100, metrics["brier_score"],
                )

        if saved == 0:
            logger.info(
                "[4h-learning] enough total evaluations, but no horizon reached "
                "the %d-sample minimum yet — no metrics saved.",
                MIN_HORIZON_SAMPLES,
            )
        else:
            logger.info(
                "[4h-learning] saved calibration metrics for %d horizon(s)", saved
            )
        return saved

    # ------------------------------------------------------------------
    async def _horizon_metrics(self, session, horizon_bars: int) -> dict | None:
        """Compute the four metrics for one 4H horizon, or None if too few rows."""
        rows = (
            await session.execute(
                select(
                    IntradayForecastEvaluation.actual_price,
                    IntradayForecast.predicted_price_p5,
                    IntradayForecast.predicted_price_p50,
                    IntradayForecast.predicted_price_p95,
                    IntradayForecast.prob_price_up,
                    IntradayForecast.base_price,
                )
                .join(
                    IntradayForecast,
                    IntradayForecast.id == IntradayForecastEvaluation.forecast_id,
                )
                .where(IntradayForecast.horizon_bars == horizon_bars)
                .order_by(IntradayForecastEvaluation.target_at.desc())
                .limit(ROLLING_WINDOW)
            )
        ).all()

        # Drop defensive-bad rows (p50 <= 0) to avoid divide-by-zero.
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
                "[4h-learning] +%s: only %d matured evaluation(s) (<%d) — skipping",
                describe_horizon(horizon_bars), n, MIN_HORIZON_SAMPLES,
            )
            return None

        actual_a = np.array(actual)
        p5_a = np.array(p5)
        p50_a = np.array(p50)
        p95_a = np.array(p95)
        prob_up_a = np.array(prob_up)
        base_a = np.array(base)

        rel_error = (actual_a - p50_a) / p50_a  # fraction
        return {
            "n": n,
            "rolling_mape": float(np.mean(np.abs(rel_error)) * 100),
            "magnitude_bias": float(np.mean(rel_error)),
            "coverage_rate": float(
                np.mean((actual_a >= p5_a) & (actual_a <= p95_a))
            ),
            "brier_score": float(
                np.mean((prob_up_a - (actual_a > base_a).astype(float)) ** 2)
            ),
        }

"""Phase 7 test: continuous-learning detection + forecast self-correction.

Scenario: inject synthetic forecasts+evaluations with a KNOWN positive bias
(the model always predicts ~10% too LOW) and overconfident bands (actual
falls outside [p5, p95]). Then verify:

1. The global gate skips when < 10 evaluations exist.
2. LearningEvaluator detects magnitude_bias ≈ +0.10, coverage ≈ 0, and
   persists the metrics.
3. ForecastEngine, reading those metrics, self-corrects: the corrected
   median is ~10% higher (bias removed) and the P5-P95 band is wider
   (volatility scaled up) — shown as a before/after report.

All synthetic rows (forecasts, evaluations, learning_metrics) are cleaned up.
Exit code 0 on success, 1 on failure.

Run from the project root::

    python scripts/test_learning_phase7.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sqlalchemy import delete, select  # noqa: E402

from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import Forecast, ForecastEvaluation, LearningMetric  # noqa: E402
from src.forecasting.engine import LOOKBACK_DAYS, ForecastEngine  # noqa: E402
from src.forecasting.learning import LearningEvaluator  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_learning_phase7")

SENTINEL_YEAR = 2098
N_SYNTH = 30  # fills the full rolling window with a pure signal
HORIZON = 7
TRUE_BIAS = 0.10  # model predicts 10% too low
P50 = 60000.0
BASE = P50 * 0.95      # base < actual -> realized direction = UP
ACTUAL = P50 * (1 + TRUE_BIAS)   # 66000
P5 = P50 * 0.99        # deliberately narrow band -> actual lands outside
P95 = P50 * 1.01
PROB_UP = 0.6          # predicted up (correct) -> brier = (0.6-1)^2 = 0.16


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


async def _cleanup() -> None:
    async with session_scope() as s:
        start = datetime(SENTINEL_YEAR, 1, 1, tzinfo=UTC)
        end = datetime(SENTINEL_YEAR + 1, 1, 1, tzinfo=UTC)
        ids = (
            await s.execute(
                select(Forecast.id).where(
                    Forecast.forecast_date >= start, Forecast.forecast_date < end
                )
            )
        ).scalars().all()
        if ids:
            await s.execute(
                delete(ForecastEvaluation).where(ForecastEvaluation.forecast_id.in_(ids))
            )
            await s.execute(delete(Forecast).where(Forecast.id.in_(ids)))


async def _inject_biased_history() -> None:
    async with session_scope() as s:
        for i in range(N_SYNTH):
            fdate = datetime(SENTINEL_YEAR, 1, 1, tzinfo=UTC) + timedelta(days=i)
            tdate = fdate + timedelta(days=HORIZON)
            fc = Forecast(
                forecast_date=fdate,
                target_date=tdate,
                horizon_days=HORIZON,
                predicted_price_p5=Decimal(f"{P5:.2f}"),
                predicted_price_p50=Decimal(f"{P50:.2f}"),
                predicted_price_p95=Decimal(f"{P95:.2f}"),
                prob_price_up=PROB_UP,
                base_price=Decimal(f"{BASE:.2f}"),
                ai_decision_snapshot=None,
            )
            s.add(fc)
            await s.flush()  # get fc.id
            err_pct = (Decimal(f"{ACTUAL:.2f}") - fc.predicted_price_p50) / fc.predicted_price_p50 * 100
            s.add(
                ForecastEvaluation(
                    forecast_id=fc.id,
                    target_date=tdate,
                    actual_price=Decimal(f"{ACTUAL:.2f}"),
                    price_error_pct=err_pct.quantize(Decimal("0.00000001")),
                    directional_correct=True,
                )
            )


async def main() -> int:
    setup_logging("WARNING")
    learner = LearningEvaluator()
    engine = ForecastEngine()
    ok = True
    try:
        await _cleanup()

        # -- Snapshot pre-existing learning rows so we only clean up ours --
        async with session_scope() as s:
            pre_ids = set((await s.execute(select(LearningMetric.id))).scalars().all())
            total_evals = (await s.execute(select(ForecastEvaluation.id))).scalars().all()

        # ---- (1) Global gate when < 10 evaluations ----
        if len(total_evals) < 10:
            saved = await learner.evaluate_and_save()
            _require(saved == 0, "gate should skip when <10 evaluations")
            logger.info("[TEST] gate OK: skipped with %d (<10) evaluations", len(total_evals))
        else:
            logger.info("[TEST] gate check skipped (DB already has >=10 evaluations)")

        # ---- inject biased synthetic history ----
        await _inject_biased_history()
        logger.info("[TEST] injected %d biased synthetic evaluations (bias=+%.0f%%)",
                    N_SYNTH, TRUE_BIAS * 100)

        # ---- (2) LearningEvaluator detects + persists ----
        saved = await learner.evaluate_and_save()
        _require(saved >= 1, "expected at least the horizon-7 metric to be saved")
        async with session_scope() as s:
            metric = (
                await s.execute(
                    select(LearningMetric)
                    .where(LearningMetric.horizon_days == HORIZON)
                    .order_by(LearningMetric.evaluated_at.desc())
                    .limit(1)
                )
            ).scalar_one()

        print("\n=== DETECTED LEARNING METRICS (horizon 7d) ===")
        print(f"  sample_size    = {metric.sample_size}")
        print(f"  magnitude_bias = {metric.magnitude_bias:+.4f}   [expect ~ +0.1000]")
        print(f"  coverage_rate  = {metric.coverage_rate:.3f}     [expect ~ 0.000 - overconfident]")
        print(f"  rolling_mape   = {metric.rolling_mape:.2f}%     [expect ~ 10.00%]")
        print(f"  brier_score    = {metric.brier_score:.4f}     [expect ~ 0.1600]")
        _require(abs(metric.magnitude_bias - TRUE_BIAS) < 0.01, "bias not detected")
        _require(metric.coverage_rate < 0.85, "coverage should flag overconfidence")
        _require(abs(metric.rolling_mape - 10.0) < 0.5, "MAPE off")
        _require(abs(metric.brier_score - 0.16) < 0.01, "brier off")

        # ---- (3) ForecastEngine self-corrects (before vs after) ----
        _, closes = await engine._load_closes(None)
        log_returns = np.diff(np.log(closes))[-LOOKBACK_DAYS:]
        base_price = float(closes[-1])
        drift, vol = engine._derive_adjustment(
            HORIZON,
            {"magnitude_bias": metric.magnitude_bias, "coverage_rate": metric.coverage_rate},
        )
        # Same seed on both -> identical block samples -> clean A/B.
        baseline = engine.simulate_horizon(
            log_returns, base_price, HORIZON, np.random.default_rng(42)
        )
        corrected = engine.simulate_horizon(
            log_returns, base_price, HORIZON, np.random.default_rng(42),
            daily_drift=drift, vol_factor=vol,
        )
        p50_ratio = corrected["p50"] / baseline["p50"]
        base_range = baseline["p95"] - baseline["p5"]
        corr_range = corrected["p95"] - corrected["p5"]

        print("\n=== FORECAST CORRECTION (before -> after), horizon 7d ===")
        print(f"  derived daily_drift = {drift:+.5f}  vol_factor = {vol:.2f}")
        print(f"  P50 : {baseline['p50']:.0f} -> {corrected['p50']:.0f}  "
              f"(x{p50_ratio:.4f}, expect ~ 1.10 - bias removed)")
        print(f"  P5  : {baseline['p5']:.0f} -> {corrected['p5']:.0f}")
        print(f"  P95 : {baseline['p95']:.0f} -> {corrected['p95']:.0f}")
        print(f"  P5-P95 width : {base_range:.0f} -> {corr_range:.0f}  "
              f"(x{corr_range / base_range:.3f}, expect >1 - bands widened)")
        _require(1.06 < p50_ratio < 1.14, f"median not corrected up ~10% (got {p50_ratio:.3f})")
        _require(corr_range > base_range, "volatility scaling did not widen the band")

        print("\n[TEST] PHASE 7 PASSED — bias detected, persisted, and auto-corrected.")
    except AssertionError as exc:
        logger.error("PHASE 7 TEST FAILED: %s", exc)
        ok = False
    except Exception:
        logger.exception("PHASE 7 TEST crashed")
        ok = False
    finally:
        # Remove synthetic forecasts/evaluations AND any learning rows we added.
        await _cleanup()
        try:
            async with session_scope() as s:
                await s.execute(
                    delete(LearningMetric).where(LearningMetric.id.notin_(pre_ids))
                    if pre_ids
                    else delete(LearningMetric)
                )
        except Exception:
            logger.exception("cleanup of learning_metrics failed")
        await dispose_engine()

    return 0 if ok else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(main()))

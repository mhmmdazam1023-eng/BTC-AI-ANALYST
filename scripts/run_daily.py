"""Bitcoin AI Analyst — THE daily one-button run (Phase 5).

One command does everything:

1. **Data pipeline** — all collectors + Feature Engineering
   (imported from ``run_collectors.run_pipeline``; no duplicated logic).
2. **Gate** — if the pipeline failed TOTALLY (zero successful stages, e.g.
   database down), inference is NOT attempted. If only some fetchers failed
   (e.g. one API timeout), inference proceeds on the freshest features
   already in the database.
3. **AI inference** — 5 sub-agents + Master Agent, decision persisted to the
   ``decisions`` hypertable (imported from ``run_inference.execute_inference``).
4. **Combined summary** — pipeline status + today's AI decision.

Exit codes: 0 = decision produced (warnings possible), 1 = no decision.

Usage (from the project root)::

    python scripts/run_daily.py                 # incremental (daily use)
    python scripts/run_daily.py --mode full     # full backfill + decision
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_collectors import CollectorResult, run_pipeline  # noqa: E402
from run_inference import (  # noqa: E402
    DEFAULT_INFERENCE_MODE,
    DEFAULT_INTER_SYMBOL_DELAY,
    estimate_calls,
    MultiAssetOutcome,
    execute_multi_asset_inference,
    GREEN,
    RED,
    RESET,
    YELLOW,
    BOLD,
    CYAN,
    InferenceError,
    InferenceOutcome,
    execute_inference,
)

from src.agents import DynamicWeightingEngine  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine  # noqa: E402
from src.db.health import assert_tables_accessible  # noqa: E402
from src.collectors.binance_fetcher import BinanceFetcher  # noqa: E402
from src.db.models import BASE_SYMBOL, INTRADAY_TIMEFRAME  # noqa: E402
from src.evaluation import (  # noqa: E402
    DecisionAccuracyEvaluator,
    DecisionEvaluator,
    save_snapshots,
)
from src.features import RegimeDetector, SectorTracker  # noqa: E402
from src.features.intraday_engine import IntradayFeatureEngine  # noqa: E402
from src.forecasting import (  # noqa: E402
    ForecastEngine,
    ForecastEvaluator,
    LearningEvaluator,
    PositionPipeline,
)
from src.forecasting.intraday_engine import IntradayForecastEngine  # noqa: E402
from src.forecasting.intraday_evaluator import (  # noqa: E402
    IntradayForecastEvaluator,
)
from src.forecasting.intraday_learning import (  # noqa: E402
    IntradayLearningEvaluator,
)
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.run_daily")

#: Closed 4h bars fetched per daily run. 1000 is Binance's per-call cap
#: (~166 days) — one request, and far more than the 35-bar warm-up needs.
#: Deeper history for the bootstrap pool is a one-off backfill via `since`.
_INTRADAY_FETCH_LIMIT = 1000


async def _collect_intraday_candles() -> int:
    """Fetch + upsert CLOSED BTC 4h candles. Returns rows written.

    Deliberately its own helper rather than folded into ``run_pipeline``: the
    daily collectors are driven by ``coins_registry`` and are multi-asset,
    while this is Bitcoin-only and timeframe-specific. Keeping it separate
    means the 20 altcoins' collection path is untouched.
    """
    fetcher = BinanceFetcher(
        symbol=BASE_SYMBOL,
        timeframe=INTRADAY_TIMEFRAME,
        limit=_INTRADAY_FETCH_LIMIT,
        drop_unclosed=True,
    )
    try:
        rows = await fetcher.fetch_data()
        return await fetcher.save_to_db(rows) if rows else 0
    finally:
        await fetcher.close()


async def _run_isolated_stage(name: str, coroutine_factory) -> "CollectorResult":
    """Run one Phase 6 stage with the same isolation contract as fetchers."""
    result = CollectorResult(name=name)
    started = time.monotonic()
    try:
        result.rows = await coroutine_factory()
        result.status = "SUCCESS"
        logger.info("<<< %s: SUCCESS (%d rows)", name, result.rows)
    except Exception as exc:  # noqa: BLE001 - stage failure must not kill the run
        result.status = "FAILED"
        result.detail = f"{type(exc).__name__}: {exc}"
        logger.exception("<<< %s: FAILED — daily run continues", name)
    finally:
        result.duration_seconds = time.monotonic() - started
    return result

_STATUS_COLOUR = {"SUCCESS": GREEN, "SKIPPED": YELLOW, "FAILED": RED}


def _print_final_summary(
    results: list[CollectorResult],
    outcome: InferenceOutcome | None,
    inference_detail: str,
    portfolio: "MultiAssetOutcome | None" = None,
) -> None:
    """Compact combined report: data pipeline + AI decision."""
    width = 78
    print(f"\n{BOLD}{CYAN}{'═' * width}{RESET}")
    print(f"{BOLD}{CYAN}{'DAILY RUN SUMMARY'.center(width)}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}")

    print(f"\n{BOLD}1) DATA PIPELINE{RESET}")
    for result in results:
        colour = _STATUS_COLOUR.get(result.status, "")
        print(f"   {colour}{result.line()}{RESET}")

    print(f"\n{BOLD}2) KEPUTUSAN AI HARI INI{RESET}")
    if outcome is not None:
        action = outcome.decision.action.value
        colour = {"BUY": GREEN, "SELL": RED, "HOLD": YELLOW}[action]
        print(
            f"   {colour}{BOLD}{action}{RESET} "
            f"(confidence {outcome.decision.confidence_score}/100) "
            f"— analisis {outcome.analysis_date}, "
            f"{outcome.pillars_reporting}/5 pilar melapor"
        )
        if outcome.fusion is not None:
            regime = outcome.fusion.regime or "UNDETERMINED (bobot setara)"
            print(
                f"   Rezim: {CYAN}{regime}{RESET} · "
                f"skor panel terbobot {outcome.fusion.weighted_score:+.1f}"
            )
        if outcome.sectors is not None:
            print(f"   Sektor: {CYAN}{outcome.sectors.rotation_signal}{RESET}")
        print("   Tersimpan ke tabel 'decisions' (riwayat bisa di-query kapan pun).")
    else:
        print(f"   {RED}{BOLD}TIDAK ADA KEPUTUSAN{RESET} — {inference_detail}")

    # --- Phase 8 Part 6: the full portfolio pass ---
    if portfolio is not None:
        ok, total = portfolio.succeeded, portfolio.requested
        colour = GREEN if not portfolio.failures else YELLOW
        print(f"\n{BOLD}3) PORTOFOLIO MULTI-ASET{RESET}")
        print(
            f"   {colour}{ok}/{total} aset berhasil dianalisis{RESET}"
            + (
                f" · {RED}{len(portfolio.failures)} gagal{RESET}"
                if portfolio.failures
                else ""
            )
        )
        for entry in portfolio.outcomes:
            action = entry.decision.action.value
            action_colour = {"BUY": GREEN, "SELL": RED, "HOLD": YELLOW}[action]
            print(
                f"   {action_colour}{action:<4s}{RESET} {entry.symbol:<13s}"
                f" confidence {entry.decision.confidence_score:>3d}/100"
            )
        for symbol, detail in sorted(portfolio.failures.items()):
            print(f"   {RED}GAGAL{RESET} {symbol:<13s} {detail[:60]}")
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}\n")


async def main(
    mode: str,
    *,
    multi_asset: bool = True,
    symbols: list[str] | None = None,
    delay: float = DEFAULT_INTER_SYMBOL_DELAY,
    inference_mode: str = DEFAULT_INFERENCE_MODE,
) -> int:
    outcome: InferenceOutcome | None = None
    portfolio: MultiAssetOutcome | None = None
    inference_detail = ""
    try:
        # ---- Preflight: can the app role actually use its tables? ----
        # A broken grant set makes several isolated collector stages fail at
        # once with `permission denied`, which reads like a code fault. Fail
        # fast with the fix instead of 30 seconds of tracebacks.
        try:
            await assert_tables_accessible(get_settings().postgres_db)
        except PermissionError as exc:
            logger.error("Preflight FAILED — aborting before any stage runs.")
            print()
            print(f"{RED}{BOLD}HAK AKSES DATABASE BERMASALAH{RESET}")
            print(exc)
            print()
            return 1

        # ---- Stage A: data pipeline (collectors + feature engineering) ----
        _, results = await run_pipeline(mode)
        succeeded = [r for r in results if r.status == "SUCCESS"]
        failed = [r for r in results if r.status == "FAILED"]

        # ---- Gate ----
        if not succeeded:
            inference_detail = (
                "pipeline data gagal total (0 tahap sukses — cek database/"
                "jaringan); inference dibatalkan."
            )
            logger.error("Pipeline failed TOTALLY — skipping inference.")
            _print_final_summary(results, None, inference_detail, portfolio)
            return 1
        if failed:
            logger.warning(
                "%d stage(s) failed (%s) — proceeding with the freshest "
                "features already in the database.",
                len(failed), ", ".join(r.name for r in failed),
            )

        # ---- Stage A2: forecast evaluation → learning → new forecast ----
        # Order matters (Phase 7): score matured forecasts FIRST, recompute
        # calibration metrics, THEN generate today's forecast so it uses the
        # freshest bias/volatility corrections (same-day feedback loop).
        # All isolated: a forecasting problem never blocks the AI decision.
        #
        # (1) Evaluation only scores FINALIZED forecasts (target_date <
        # CURRENT_DATE - 1 day); the finalization guard lives inside
        # ForecastEvaluator (single source of truth).
        results.append(
            await _run_isolated_stage(
                "Forecast Evaluation",
                lambda: ForecastEvaluator().evaluate_expired_forecasts(),
            )
        )
        # (2) Continuous-learning calibration: derive per-horizon bias,
        # coverage, MAPE, Brier from the evaluation history (skips gracefully
        # when there are < 10 evaluations).
        results.append(
            await _run_isolated_stage(
                "Learning Calibration",
                lambda: LearningEvaluator().evaluate_and_save(),
            )
        )
        # (3) Generate today's forecast — reads the metrics just written and
        # self-corrects bias (drift) and calibration (volatility scaling).
        results.append(
            await _run_isolated_stage(
                "Monte Carlo Forecast",
                lambda: ForecastEngine().generate_and_save_forecasts(),
            )
        )

        # ---- Stage A2b: BTC intraday (4H) layer — Phase 8 Part 8 ----
        # Bitcoin only. Same evaluate -> learn -> forecast ORDER as A2 and for
        # the same reason: today's forecast should use today's corrections.
        #
        # Every stage isolated. If all four fail, the daily decision path is
        # byte-identical to Part 7 — the 4H layer is additive context, never a
        # precondition. That is why it sits AFTER the daily forecast and
        # BEFORE inference, rather than replacing anything.
        #
        # (0) Collect closed 4h candles. `drop_unclosed=True` is the whole
        # point: the still-forming bar never enters the warehouse.
        results.append(
            await _run_isolated_stage(
                "BTC 4H Candles",
                lambda: _collect_intraday_candles(),
            )
        )
        # (1) Features on the closed bars.
        results.append(
            await _run_isolated_stage(
                "BTC 4H Features",
                lambda: IntradayFeatureEngine().run(),
            )
        )
        # (2) Score matured 4H forecasts. Since this job runs once a day while
        # horizons are 4h/12h/24h, several normally mature per run.
        results.append(
            await _run_isolated_stage(
                "4H Forecast Evaluation",
                lambda: IntradayForecastEvaluator().evaluate_matured(),
            )
        )
        # (3) Intraday calibration metrics (own table — mixing them with the
        # daily ones would apply a 30-day bias to a 4-hour forecast).
        results.append(
            await _run_isolated_stage(
                "4H Learning Calibration",
                lambda: IntradayLearningEvaluator().evaluate_and_save(),
            )
        )
        # (4) Today's 4H forecast, reading the metrics just written.
        results.append(
            await _run_isolated_stage(
                "4H Monte Carlo Forecast",
                lambda: IntradayForecastEngine().generate_and_save(),
            )
        )

        # ---- Stage A3: regime detection -> dynamic weighting ----
        # Isolated like every other stage: regime detection is an
        # ENHANCEMENT to the decision, never a precondition for it. On
        # failure `regime_snapshot` stays None and execute_inference() fuses
        # the panel with equal weights — exactly the Phase 4-7 behaviour.
        regime_snapshot = None
        regime_result = CollectorResult(name="Regime Detection")
        regime_started = time.monotonic()
        try:
            regime_snapshot = await RegimeDetector().run()
            regime_result.status = "SUCCESS"
            regime_result.rows = 1
            # Same call execute_inference() makes, so the summary can never
            # advertise weights different from the ones actually applied.
            weighting = DynamicWeightingEngine().evaluate(
                regime_snapshot.regime,
                regime_snapshot.previous_regime,
                regime_snapshot.confidence,
            )
            transition = (
                f" [TRANSISI dari {regime_snapshot.previous_regime.value}]"
                if weighting.is_transition
                and regime_snapshot.previous_regime is not None
                else ""
            )
            regime_result.detail = (
                f"{regime_snapshot.regime.value} "
                f"({regime_snapshot.confidence}/100){transition} -> "
                + ", ".join(
                    f"{pillar} {share * 100:.0f}%"
                    for pillar, share in sorted(
                        weighting.weights.items(), key=lambda kv: -kv[1]
                    )
                )
                + (" (setara: rezim belum bertahan)" if weighting.used_default else "")
            )
        except Exception as exc:  # noqa: BLE001 - must never block the decision
            regime_result.status = "FAILED"
            regime_result.detail = (
                f"{type(exc).__name__}: {exc} — using DEFAULT equal weights"
            )
            logger.exception(
                "Regime Detection FAILED — inference continues with equal weights"
            )
        finally:
            regime_result.duration_seconds = time.monotonic() - regime_started
        results.append(regime_result)

        # ---- Stage A4: sector rotation tracking ----
        # Isolated for the same reason as regime detection: cross-asset
        # context enriches the decision, it is never a precondition for it.
        # On failure `sector_snapshot` stays None and the Master Agent
        # simply receives no sector block.
        sector_snapshot = None
        sector_result = CollectorResult(name="Sector Rotation")
        sector_started = time.monotonic()
        try:
            sector_snapshot = await SectorTracker().run()
            sector_result.status = "SUCCESS"
            sector_result.rows = len(sector_snapshot.rankings)
            leaders = ", ".join(
                f"{r.sector} {r.avg_rs_7d:+.1f}%"
                for r in sector_snapshot.top_sectors
                if r.avg_rs_7d is not None
            )
            thin = [r.sector for r in sector_snapshot.rankings if r.is_thin]
            sector_result.detail = f"top: {leaders}" + (
                f" | THIN (<2 coins): {', '.join(thin)}" if thin else ""
            )
        except Exception as exc:  # noqa: BLE001 - must never block the decision
            sector_result.status = "FAILED"
            sector_result.detail = (
                f"{type(exc).__name__}: {exc} — inference continues without "
                "sector context"
            )
            logger.exception(
                "Sector Rotation FAILED — daily run continues without it"
            )
        finally:
            sector_result.duration_seconds = time.monotonic() - sector_started
        results.append(sector_result)

        # ---- Stage A5: decision accuracy — Phase 9 ----
        # Scores PAST decisions (all 21 assets, horizons +1/+7/+30d) and
        # aggregates the per-asset scoreboard. Same evaluate -> learn order
        # as Stage A2, for the same reason: the Kelly sizer in Stage C reads
        # the metrics written here, so they must be fresh when it runs.
        #
        # Placed BEFORE inference deliberately. Nothing here depends on
        # today's decision — today's call cannot mature for at least a day —
        # so running it early means the scoreboard still updates on a day
        # when the model endpoint is down and Stage B fails entirely.
        results.append(
            await _run_isolated_stage(
                "Decision Evaluation",
                lambda: DecisionEvaluator().evaluate_matured(),
            )
        )
        results.append(
            await _run_isolated_stage(
                "Decision Accuracy Metrics",
                lambda: DecisionAccuracyEvaluator().evaluate_and_save(),
            )
        )

        # ---- Stage A6: measurement snapshot ----
        # Composes the null model, the baseline comparator and the pillar
        # ablation into one row per (day, horizon) in `measurement_snapshots`.
        #
        # It exists because all three were report-only scripts somebody had to
        # remember to run, and a measurement nobody looks at cannot change a
        # decision. The value is the TREND: one day's edge on thirteen
        # decisive evaluations is noise, but whether it drifts toward zero or
        # away from it over months is the whole question — and that cannot be
        # reconstructed later from a log nobody kept.
        #
        # Here rather than after Stage B for the same reason as A5: nothing in
        # it depends on today's decision, so the trend keeps growing on days
        # the model endpoint is down.
        results.append(
            await _run_isolated_stage(
                "Measurement Snapshot",
                lambda: save_snapshots(),
            )
        )

        # ---- Stage B: AI inference (+ persistence inside) ----
        # Multi-asset since Phase 8 Part 6. Bitcoin is analysed first, so
        # `outcome` (the headline decision) is available even if a later
        # altcoin fails. Regime and sector context are computed once above
        # and handed in rather than recomputed per asset.
        try:
            if multi_asset:
                portfolio = await execute_multi_asset_inference(
                    symbols=symbols,
                    delay_seconds=delay,
                    mode=inference_mode,
                    print_output=True,
                    regime_snapshot=regime_snapshot,
                    sector_snapshot=sector_snapshot,
                )
                outcome = next(
                    (o for o in portfolio.outcomes if o.symbol == BASE_SYMBOL),
                    portfolio.outcomes[0] if portfolio.outcomes else None,
                )
                if outcome is None:
                    inference_detail = (
                        f"tidak ada aset yang berhasil dianalisis "
                        f"({len(portfolio.failures)} gagal)"
                    )
                    raise InferenceError(inference_detail)
            else:
                outcome = await execute_inference(
                    print_output=True,
                    regime_snapshot=regime_snapshot,
                    sector_snapshot=sector_snapshot,
                )
        except InferenceError as exc:
            inference_detail = str(exc)
            logger.error("Inference failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 - summary must still print
            inference_detail = f"{type(exc).__name__}: {exc}"
            logger.exception("Inference failed unexpectedly")

        # ---- Stage C: scenario engine -> Kelly position sizing ----
        # Runs AFTER inference because it sizes the decisions that were just
        # made. Isolated like every other stage: sizing is an addition to the
        # decision, never a precondition for it.
        if outcome is not None:
            sizing_result = CollectorResult(name="Position Sizing (Kelly)")
            sizing_started = time.monotonic()
            try:
                position_run = await PositionPipeline().run()
                sizing_result.status = "SUCCESS"
                sizing_result.rows = len(position_run.sizings)
                sizing_result.detail = position_run.summary_line()
            except Exception as exc:  # noqa: BLE001 - never blocks the run
                sizing_result.status = "FAILED"
                sizing_result.detail = f"{type(exc).__name__}: {exc}"
                logger.exception("Position Sizing FAILED — daily run continues")
            finally:
                sizing_result.duration_seconds = time.monotonic() - sizing_started
            results.append(sizing_result)

        _print_final_summary(results, outcome, inference_detail, portfolio)
        return 0 if outcome is not None else 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Daily one-button run: data pipeline + AI decision."
    )
    parser.add_argument(
        "--mode",
        choices=("incremental", "full"),
        default="incremental",
        help="Data-collection mode (default: incremental).",
    )
    parser.add_argument(
        "--btc-only",
        action="store_true",
        help="Analyse only Bitcoin (pre-Phase-8-Part-6 behaviour).",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Explicit symbols to analyse; default = all active in coins_registry.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_INTER_SYMBOL_DELAY,
        help="Seconds between assets during inference.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=("consolidated", "panel", "hybrid"),
        default=DEFAULT_INFERENCE_MODE,
        help=(
            "How each asset is scored. NOT the same as --mode, which selects "
            "the DATA collection window. See run_inference.py --help."
        ),
    )
    arguments = parser.parse_args()
    os.system("")  # enable ANSI escape processing on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    setup_logging(get_settings().log_level)
    raise SystemExit(
        asyncio.run(
            main(
                arguments.mode,
                multi_asset=not arguments.btc_only,
                symbols=arguments.symbols,
                delay=arguments.delay,
                inference_mode=arguments.inference_mode,
            )
        )
    )

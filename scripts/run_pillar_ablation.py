"""Report: do the LLM's pillar scores carry information? (Phase 9 ablation)

Read-only. No model calls, no writes — every number comes from
``decisions.sub_agents_json``, which has stored each pillar's score since
2026-07-10.

The five pillar scores are the ONLY place judgement enters this pipeline;
everything downstream is deterministic arithmetic over them. This report is
the prior question to "should we pay for a better model?" — whether the
scores presently relate to what the price did at all.

Read it like this
-----------------
* **IC** is the mean daily cross-sectional rank correlation between a signal
  and the forward return in band units. 0.03-0.05 is a genuinely useful
  factor in liquid markets; 0.10 sustained would be remarkable.
* **n_days is the sample size**, not n_obs. Twenty-one assets on one day are
  not twenty-one observations — they mostly share one market move.
* **The baseline rows are the point.** ``momentum_score``, ``rs_7d`` and
  ``rsi_14`` are read straight from the same ``daily_features`` row the model
  was shown. A pillar that cannot out-rank them is an expensive way to
  compute something free.

Usage::

    python scripts/run_pillar_ablation.py
    python scripts/run_pillar_ablation.py --horizons 1 7 --symbol BTC/USDT
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import BASE_SYMBOL  # noqa: E402
from src.evaluation.attenuation import (  # noqa: E402
    STALE_AFTER_DAYS,
    compute_attenuation,
    load_latest_noise,
)
from src.evaluation.decision_evaluator import DECISION_HORIZONS  # noqa: E402
from src.evaluation.pillar_ablation import (  # noqa: E402
    BASELINE_FEATURES,
    MIN_ASSETS_PER_DAY,
    PANEL_MEAN,
    PILLARS,
    days_needed_for,
    information_coefficient,
    load_ablation_frame,
    timeseries_correlation,
)
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.run_pillar_ablation")

_RULE = "-" * 78


def _fmt(value: float | None, spec: str = "+.3f") -> str:
    return "   n/a" if value is None else format(value, spec)


def _print_noise_provenance(noise, settings) -> None:
    """State where the correction comes from BEFORE showing any corrected number.

    C1 proved the multiplier is not a constant, so a corrected IC without its
    measurement date is a number no one can audit.
    """
    print()
    print("=" * 78)
    print("  NOISE FLOOR USED FOR THE ATTENUATION CORRECTION")
    print("=" * 78)
    if not noise:
        print(
            "  NONE ON RECORD — every IC below is RAW and therefore an\n"
            "  UNDERSTATEMENT of the true value. Noise can only shrink an IC,\n"
            f"  never create one. Measure it:  python scripts/test_determinism.py\n"
            f"      --runs 5   (costs 5 API calls, model {settings.qwen_model})"
        )
        return

    newest = max(m.measured_at for m in noise.values())
    age = max(m.age_days() for m in noise.values())
    stale = age > STALE_AFTER_DAYS
    print(
        f"  measured {newest:%Y-%m-%d} ({age:.0f} day(s) ago) · "
        f"model {settings.qwen_model}"
    )
    sample = next(iter(noise.values()))
    print(
        f"  temperature {sample.temperature} · {sample.n_runs} identical call(s) "
        f"on {sample.symbol}"
    )
    if stale:
        print(
            f"  !! STALE (> {STALE_AFTER_DAYS} days). The floor moves with the\n"
            "     provider's batching; re-measure before quoting corrected ICs."
        )
    if sample.n_runs < 5:
        print(
            f"  !  only {sample.n_runs} runs — the sd behind every multiplier\n"
            "     below is crude. Treat the corrections as indicative."
        )


def _print_attenuation_block(frame, horizon: int, settings, noise) -> None:
    """Raw IC is an understatement; this is by how much, per signal.

    Only the LLM pillars are corrected. The deterministic baselines
    (momentum_score, rs_7d, rsi_14) come straight out of daily_features with
    no model call in the path, so their noise is exactly zero and applying a
    correction to them would flatter the comparison the ablation exists to
    make — which is precisely LLM pillar versus free baseline.
    """
    print()
    print("  ATTENUATION — raw IC is shrunk by the model's own run-to-run noise")
    print(f"  {_RULE}")
    print(
        f"  {'pillar':<18} {'noise sd':>9} {'day sd':>8} {'x':>7} "
        f"{'IC raw':>8} {'IC corr':>8}  note"
    )
    print(f"  {_RULE}")
    for name in (*PILLARS, PANEL_MEAN):
        att = compute_attenuation(
            signal=name,
            frame=frame,
            noise=noise.get(name),
            series_temperature=settings.qwen_temperature,
            min_assets=MIN_ASSETS_PER_DAY,
        )
        ic = information_coefficient(frame, name, horizon).mean_ic
        corrected = att.correct(ic)
        note = att.reason if att.multiplier is None else att.severity
        print(
            f"  {name:<18} {_fmt(att.noise_sd, '.2f'):>9} "
            f"{_fmt(att.observed_sd, '.2f'):>8} "
            f"{_fmt(att.multiplier, '.3f'):>7} "
            f"{_fmt(ic):>8} {_fmt(corrected):>8}  {note}"
        )
    print(f"  {_RULE}")
    print(
        "  x = sqrt(reliability); IC corr = IC raw / x. Noise ATTENUATES only,\n"
        "  so |IC corr| >= |IC raw| always — correction never invents a signal,\n"
        "  it restores one the measurement lost. Baselines are not corrected:\n"
        "  no model call in their path means no noise to remove."
    )


async def report(horizons: tuple[int, ...], symbol: str) -> int:
    signals = (*PILLARS, PANEL_MEAN, *BASELINE_FEATURES)

    # The noise floor is a property of the MODEL, not of a horizon, so it is
    # loaded once. Pinned to the model currently configured: a measurement
    # taken against a different model describes a different process.
    settings = get_settings()
    async with session_scope() as session:
        noise = await load_latest_noise(session, model=settings.qwen_model)
    _print_noise_provenance(noise, settings)

    for horizon in horizons:
        async with session_scope() as session:
            frame = await load_ablation_frame(session, horizon)

        print()
        print("=" * 78)
        print(f"  HORIZON +{horizon}d")
        print("=" * 78)
        if frame.empty:
            print(
                "  No matured (decision, asset) rows at this horizon yet.\n"
                "  Decisions need entry(+1d) + horizon settled candles before "
                "they can be scored."
            )
            continue

        days = frame["decision_ts"].nunique()
        wide = (
            frame.groupby("decision_ts")["symbol"].nunique() >= MIN_ASSETS_PER_DAY
        ).sum()
        print(
            f"  {len(frame)} row(s) · {days} decision day(s) · "
            f"{wide} day(s) with >= {MIN_ASSETS_PER_DAY} assets "
            f"(only those enter the IC)"
        )

        # ---- cross-sectional IC ------------------------------------------
        print()
        print("  CROSS-SECTIONAL IC  (did the signal rank the book correctly?)")
        print(f"  {_RULE}")
        print(
            f"  {'signal':<18} {'IC':>8} {'std':>8} {'t':>7} "
            f"{'days':>5} {'obs':>5}  verdict"
        )
        print(f"  {_RULE}")
        any_ic = False
        for name in signals:
            result = information_coefficient(frame, name, horizon)
            if result.mean_ic is not None:
                any_ic = True
            marker = "  " if name in PILLARS or name == PANEL_MEAN else "* "
            print(
                f"  {marker}{name:<16} {_fmt(result.mean_ic):>8} "
                f"{_fmt(result.std_ic, '.3f'):>8} {_fmt(result.t_stat, '+.2f'):>7} "
                f"{result.n_days:>5} {result.n_obs:>5}  {result.verdict}"
            )
        print(f"  {_RULE}")
        print("  * = deterministic baseline, free, straight from daily_features")

        _print_attenuation_block(frame, horizon, settings, noise)

        if not any_ic:
            print()
            print(
                f"  NOTHING TESTABLE YET at +{horizon}d: no day has "
                f">= {MIN_ASSETS_PER_DAY} assets with a matured return.\n"
                f"  Multi-asset decisions only began 2026-07-28, and each needs "
                f"{horizon + 1} settled days."
            )

        # ---- single-asset fallback ---------------------------------------
        print()
        print(f"  TIME-SERIES CORRELATION — {symbol} only (weaker; see module docs)")
        print(f"  {_RULE}")
        print(
            f"    {'signal':<16} {'rho':>8} {'n':>4} {'eff_n':>6} {'t':>7}  verdict"
        )
        print(f"  {_RULE}")
        for name in signals:
            result = timeseries_correlation(frame, name, symbol, horizon)
            print(
                f"    {name:<16} {_fmt(result.rho):>8} {result.n_obs:>4} "
                f"{result.effective_obs:>6.1f} {_fmt(result.t_stat, '+.2f'):>7}  "
                f"{result.verdict}"
            )
        print(f"  {_RULE}")
        print(
            f"    eff_n = n / horizon: consecutive daily windows at +{horizon}d "
            f"share {horizon - 1}/{horizon} of their span."
        )

    # ---- how much data would settle this ---------------------------------
    print()
    print("=" * 78)
    print("  HOW MUCH DATA WOULD SETTLE THIS")
    print("=" * 78)
    for target in (0.10, 0.05, 0.03):
        needed = days_needed_for(target)
        duration = (
            f" (~{needed / 365:.1f} years of daily runs)" if needed >= 60 else ""
        )
        print(
            f"  detect IC = {target:.2f} at |t| >= 2  ->  ~{needed} independent "
            f"decision days{duration}"
        )
    print(
        "\n  Those are OPTIMISTIC: they assume independent days. Overlapping\n"
        "  return windows and 0.5-0.9 cross-asset correlation both push the\n"
        "  real requirement higher."
    )
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(DECISION_HORIZONS),
        help=f"horizons in days (default: {list(DECISION_HORIZONS)})",
    )
    parser.add_argument(
        "--symbol", default=BASE_SYMBOL,
        help=f"asset for the time-series fallback (default: {BASE_SYMBOL})",
    )
    args = parser.parse_args()
    try:
        return await report(tuple(args.horizons), args.symbol)
    except Exception:
        logger.exception("Ablation report FAILED")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)


if __name__ == "__main__":
    setup_logging("WARNING")  # the report is the output; keep the log quiet
    raise SystemExit(asyncio.run(main()))

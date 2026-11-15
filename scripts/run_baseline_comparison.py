"""Report: does the 5-pillar system beat a dumb strategy?

Read-only. No model calls, no writes.

Two sections, and they answer different questions
--------------------------------------------------
**A — Baselines over full history.** What these strategies score in general,
on years of data. Large samples, so the numbers are stable. This is the
context: it tells you what "good" even looks like on this band.

**B — Head-to-head.** The same baselines restricted to the exact (day, asset)
grid the real system decided on, scored by the same machinery. Tiny sample
today, but it is the only truly apples-to-apples comparison.

``buy_and_hold`` and ``always_hold`` should reproduce the null model's BUY
and HOLD rates exactly — they are the same quantity by two independent
paths. Treat any divergence as a bug, not a finding.

Usage::

    python scripts/run_baseline_comparison.py
    python scripts/run_baseline_comparison.py --horizons 1 7 --symbol BTC/USDT
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import Decision, DecisionEvaluation, DecisionVerdict  # noqa: E402
from src.evaluation.baseline_strategies import (  # noqa: E402
    STRATEGIES,
    compare_baselines,
)
from src.evaluation.decision_evaluator import DECISION_HORIZONS  # noqa: E402
from src.evaluation.null_model import SIGNIFICANCE_Z, edge_z_score  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.run_baseline_comparison")

_RULE = "-" * 88


def _pct(value: float | None) -> str:
    return "   n/a" if value is None else f"{value * 100:5.1f}%"


def _row(label: str, hit, correct, decisive, neutral, eff, note=""):
    print(
        f"  {label:<16} {_pct(hit):>7} {correct:>6}/{decisive:<6} {neutral:>7} "
        f"{eff:>9.1f}  {note}"
    )


async def _system_scoreboard(horizon: int):
    """The real system's own numbers at this horizon, from Phase 9."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(DecisionEvaluation.verdict).where(
                    DecisionEvaluation.horizon_days == horizon
                )
            )
        ).all()
    correct = sum(1 for r in rows if r.verdict is DecisionVerdict.CORRECT)
    wrong = sum(1 for r in rows if r.verdict is DecisionVerdict.WRONG)
    neutral = sum(1 for r in rows if r.verdict is DecisionVerdict.NEUTRAL)
    decisive = correct + wrong
    return correct, wrong, neutral, (correct / decisive if decisive else None)


async def report(horizons: tuple[int, ...], symbols: list[str] | None) -> int:
    scope = ", ".join(symbols) if symbols else "all assets"

    for horizon in horizons:
        print()
        print("=" * 88)
        print(f"  HORIZON +{horizon}d   ({scope})")
        print("=" * 88)

        # ---- A: full history -------------------------------------------
        async with session_scope() as session:
            full = await compare_baselines(session, horizon, symbols=symbols)
        print()
        print("  A) BASELINES OVER FULL HISTORY")
        print(f"  {_RULE}")
        print(
            f"  {'strategy':<16} {'hit':>7} {'correct/dec':>13} {'neutral':>7} "
            f"{'eff_n':>9}"
        )
        print(f"  {_RULE}")
        if not full:
            print("  (no scoreable rows)")
        for name in STRATEGIES:
            result = full.get(name)
            if result is None:
                continue
            _row(
                name, result.hit_rate, result.correct, result.decisive,
                result.neutral, result.effective_n,
            )
        print(f"  {_RULE}")
        if full:
            sample = next(iter(full.values()))
            print(
                f"  {sample.total} (asset, day) row(s) across "
                f"{sample.n_assets} asset(s)"
            )

        # ---- B: head-to-head on the system's own grid -------------------
        async with session_scope() as session:
            days = (
                (
                    await session.execute(
                        select(Decision.timestamp).distinct().order_by(Decision.timestamp)
                    )
                )
                .scalars()
                .all()
            )
            head = await compare_baselines(
                session, horizon, symbols=symbols, days=list(days)
            )
        s_correct, s_wrong, s_neutral, s_hit = await _system_scoreboard(horizon)
        s_decisive = s_correct + s_wrong

        print()
        print("  B) HEAD-TO-HEAD on the system's own decision days")
        print(f"  {_RULE}")
        print(
            f"  {'strategy':<16} {'hit':>7} {'correct/dec':>13} {'neutral':>7} "
            f"{'eff_n':>9}  verdict vs system"
        )
        print(f"  {_RULE}")
        _row(
            "** SYSTEM **", s_hit, s_correct, s_decisive, s_neutral,
            float("nan"), "<- the 5-pillar pipeline",
        )
        if not head:
            print("  (baselines have no scoreable rows on that grid yet)")
        for name in STRATEGIES:
            result = head.get(name)
            if result is None:
                continue
            note = ""
            if result.hit_rate is not None and s_hit is not None:
                gap = (s_hit - result.hit_rate) * 100
                z = edge_z_score(s_hit, result.hit_rate, min(result.effective_n, s_decisive))
                if z is None or abs(z) < SIGNIFICANCE_Z:
                    note = f"system {gap:+.1f}pp — not significant"
                else:
                    note = f"system {gap:+.1f}pp (z={z:.1f})"
            _row(
                name, result.hit_rate, result.correct, result.decisive,
                result.neutral, result.effective_n, note,
            )
        print(f"  {_RULE}")
        if s_decisive == 0:
            print("  The system has no decisive evaluations at this horizon yet.")

        # ---- C: action-for-action, because the aggregate is biased ------
        # HOLD is CORRECT whenever price stays inside the band (~70%), so any
        # strategy emitting more HOLD wins the aggregate without being better.
        # This section removes that advantage by comparing like with like.
        if full:
            print()
            print("  C) PER-ACTION on full history — the aggregate above is")
            print("     biased toward whichever strategy says HOLD most often")
            print(f"  {_RULE}")
            print(
                f"  {'strategy':<16} {'mix B/S/H':>16} {'BUY hit':>12} "
                f"{'SELL hit':>12} {'HOLD hit':>12}"
            )
            print(f"  {_RULE}")
            for name in STRATEGIES:
                result = full.get(name)
                if result is None:
                    continue
                mix = result.action_mix
                mix_text = "/".join(
                    f"{mix.get(a, 0.0) * 100:.0f}" for a in ("BUY", "SELL", "HOLD")
                )
                cells = []
                for action in ("BUY", "SELL", "HOLD"):
                    hit, n = result.action_hit_rate(action)
                    cells.append("     —      " if n == 0 else f"{_pct(hit)} n={n:<5}")
                print(f"  {name:<16} {mix_text:>16} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12}")
            print(f"  {_RULE}")

    print()
    print("=" * 88)
    print("  HOW TO READ THIS")
    print("=" * 88)
    print(
        "  * buy_and_hold and always_hold are CROSS-CHECKS: they must equal the\n"
        "    null model's BUY and HOLD rates exactly. Divergence = bug.\n"
        "  * sma_crossover is the only baseline that looks at anything. It is the\n"
        "    one the pipeline actually has to beat.\n"
        "  * eff_n is overlap-corrected. Section B's samples are tiny; a gap of\n"
        "    10-20pp there is noise, not a result."
    )
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(DECISION_HORIZONS)
    )
    parser.add_argument("--symbol", action="append", dest="symbols")
    args = parser.parse_args()
    try:
        return await report(tuple(args.horizons), args.symbols)
    except Exception:
        logger.exception("Baseline comparison FAILED")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)


if __name__ == "__main__":
    setup_logging("WARNING")
    raise SystemExit(asyncio.run(main()))

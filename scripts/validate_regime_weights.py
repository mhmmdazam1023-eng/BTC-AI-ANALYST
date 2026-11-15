"""Does REGIME_WEIGHTS earn its place? (2026-08-09)

The table was never derived from data
-------------------------------------
``REGIME_WEIGHTS`` came from principles written in the Master Agent's system
prompt (Krause 2026: macro/technical lead in normal regimes, liquidity and
sentiment lead in stressed ones, on-chain is slow and structural). The
original task said "as previously discussed", and no such discussion exists
in this repository. CLAUDE.md has always labelled it a revisable starting
point. This script is what turns that label into a measurement.

The three questions, in the order they can actually be answered
---------------------------------------------------------------
1. **Which rows have ever fired?** A weight row that has never been used
   cannot be validated, and saying so is more useful than pretending the
   table has been tested as a whole.
2. **Does it change any decision?** Recompute every stored decision under the
   regime row and under equal weights. If nothing changes, the table is
   decorative and the other questions are moot.
3. **Does it change them for the better?** Score both action sets through the
   SAME machinery every other comparison here uses — ``classify_verdict``,
   the same band, the same stored return. A comparator that scores itself
   differently is worthless.

Read the per-flip breakdown, not just the totals
------------------------------------------------
Aggregate hit rate is biased toward inaction: HOLD is CORRECT whenever price
stays inside the band (~70% of the time), while BUY/SELL go NEUTRAL in the
same situation. So a change that merely converts SELL into HOLD "wins"
without predicting anything. Section C separates flips that crossed
CORRECT/WRONG (real) from flips that only crossed CORRECT/NEUTRAL (band
geometry). The first measured run had **13 flips, all of them the second
kind**.

Read-only unless ``--backfill-regimes`` is passed.

Usage::

    python scripts/validate_regime_weights.py
    python scripts/validate_regime_weights.py --backfill-regimes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select, text  # noqa: E402

from src.agents.decision_composer import BUY_THRESHOLD, decide_action  # noqa: E402
from src.agents.weighting_engine import (  # noqa: E402
    MIN_CONFIDENCE_TO_APPLY,
    PILLARS,
    REGIME_WEIGHTS,
    DynamicWeightingEngine,
)
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import DecisionVerdict, MarketRegime, MarketRegimeRow  # noqa: E402
from src.evaluation.decision_evaluator import classify_verdict  # noqa: E402
from src.features.regime_detector import RegimeDetector  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.validate_regime_weights")

#: The comparator. Equal weights are the honest null for a weighting scheme,
#: exactly as `always_hold` is the honest null for a strategy: it is what the
#: system does when the table is removed, not a strawman.
EQUAL_WEIGHTS = {pillar: 20 for pillar in PILLARS}

_RULE = "-" * 74


def _pillar_scores(payload) -> dict[str, int] | None:
    panel = payload if isinstance(payload, list) else json.loads(payload)
    scores = {
        item["agent_name"]: item["score"]
        for item in panel
        if item.get("agent_name") in PILLARS and item.get("score") is not None
    }
    return scores if len(scores) == len(PILLARS) else None


# ---------------------------------------------------------------------------
async def backfill_regimes(session) -> int:
    """Classify and store every day the features can support.

    Not cosmetic. ``market_regimes`` held 12 rows when this was written —
    only what daily runs had produced since 2026-07-28 — so five of the six
    weight rows had no history to be judged against and the "regime
    distribution" quoted in CLAUDE.md (1,992 days) existed only inside a
    test, never on disk.

    Safe to run: the detector is pure and deterministic given ``as_of``,
    ``save()`` upserts on the day, and nothing downstream reads historical
    regime rows except this kind of analysis. Transition damping only looks
    back 3 days, so filling older days cannot change a live decision.
    """
    detector = RegimeDetector()
    newest = (
        await session.execute(text("SELECT max(timestamp)::date FROM daily_features"))
    ).scalar_one_or_none()
    oldest = (
        await session.execute(
            text(
                "SELECT min(timestamp)::date FROM daily_features "
                "WHERE symbol = 'BTC/USDT'"
            )
        )
    ).scalar_one_or_none()
    if newest is None or oldest is None:
        logger.warning("no daily_features — nothing to backfill")
        return 0

    # The detector needs a long run-up (SMA200, 1y vol baseline), so start
    # well after the first candle rather than emitting garbage for the warmup.
    start = oldest + timedelta(days=400)
    written = 0
    day = start
    while day <= newest:
        as_of = datetime(day.year, day.month, day.day, tzinfo=UTC)
        try:
            history = await detector.load_history(as_of)
            if not history.empty:
                snapshot = detector.detect(history)
                await detector.save(snapshot)
                written += 1
        except Exception as exc:  # noqa: BLE001 - one bad day must not stop it
            logger.warning("regime backfill skipped %s (%s)", day, exc)
        day += timedelta(days=1)
    logger.info("backfilled %d regime day(s) from %s to %s", written, start, newest)
    return written


# ---------------------------------------------------------------------------
async def section_a_coverage(session) -> None:
    """Which rows of the table have ever been used by a real decision?"""
    print()
    print("=" * 74)
    print("  A. WHICH WEIGHT ROWS HAVE EVER FIRED")
    print("=" * 74)

    rows = (
        await session.execute(
            text(
                """
                SELECT COALESCE(r.regime::text, '(no regime row)') AS regime,
                       count(DISTINCT d.timestamp::date) AS days,
                       count(*) AS decisions,
                       min(r.confidence) AS min_conf
                FROM decisions d
                LEFT JOIN market_regimes r
                       ON r.timestamp::date = d.timestamp::date
                GROUP BY 1 ORDER BY 3 DESC
                """
            )
        )
    ).all()
    used = {r.regime for r in rows}
    print(f"  {'regime':<22}{'days':>6}{'decisions':>11}{'min conf':>10}")
    print("  " + _RULE)
    for r in rows:
        print(f"  {r.regime:<22}{r.days:>6}{r.decisions:>11}"
              f"{(r.min_conf if r.min_conf is not None else '-'):>10}")
    print("  " + _RULE)

    never = [reg.value for reg in REGIME_WEIGHTS if reg.value not in used]
    if never:
        print(
            f"  NEVER EXERCISED ({len(never)}/{len(REGIME_WEIGHTS)} rows): "
            f"{', '.join(never)}"
        )
        print(
            "  Those rows are UNVALIDATED — not wrong, unmeasured. Nothing\n"
            "  below says anything about them."
        )

    hist = (
        await session.execute(
            text(
                "SELECT regime::text AS regime, count(*) AS n "
                "FROM market_regimes GROUP BY 1 ORDER BY 2 DESC"
            )
        )
    ).all()
    total = sum(r.n for r in hist) or 1
    print(f"\n  Regime distribution across all {total} classified day(s):")
    for r in hist:
        print(f"    {r.regime:<22}{r.n:>6}  ({100 * r.n / total:>5.1f}%)")
    print(
        f"\n  Weights only apply above confidence {MIN_CONFIDENCE_TO_APPLY}; "
        f"below it the engine falls back to equal weights anyway."
    )


async def section_b_and_c(session) -> None:
    """Does the table change decisions, and does it change them for the better?"""
    rows = (
        await session.execute(
            text(
                """
                SELECT d.timestamp, d.symbol, d.sub_agents_json,
                       r.regime::text AS regime
                FROM decisions d
                JOIN market_regimes r ON r.timestamp::date = d.timestamp::date
                ORDER BY d.timestamp, d.symbol
                """
            )
        )
    ).all()

    flips = Counter()
    deltas: list[float] = []
    mix_regime, mix_equal = Counter(), Counter()
    for row in rows:
        scores = _pillar_scores(row.sub_agents_json)
        if scores is None:
            continue
        weights = REGIME_WEIGHTS[MarketRegime(row.regime)]
        s_regime, _ = DynamicWeightingEngine.weighted_score(scores, weights)
        s_equal, _ = DynamicWeightingEngine.weighted_score(scores, EQUAL_WEIGHTS)
        a_regime, a_equal = decide_action(s_regime), decide_action(s_equal)
        mix_regime[a_regime.value] += 1
        mix_equal[a_equal.value] += 1
        if a_regime is not a_equal:
            flips[f"{a_equal.value} -> {a_regime.value}"] += 1
        deltas.append(s_regime - s_equal)

    print()
    print("=" * 74)
    print("  B. DOES THE TABLE CHANGE ANY DECISION?")
    print("=" * 74)
    if not deltas:
        print("  No decision has both a full panel and a regime row yet.")
        return
    changed = sum(flips.values())
    print(f"  decisions compared        : {len(deltas)}")
    print(f"  action mix, REGIME weights: {dict(mix_regime)}")
    print(f"  action mix, EQUAL weights : {dict(mix_equal)}")
    print(f"  ACTIONS CHANGED           : {changed} "
          f"({100 * changed / len(deltas):.1f}%)  {dict(flips) or ''}")
    print(
        f"  weighted_score shift      : mean {statistics.mean(deltas):+.2f}  "
        f"sd {statistics.pstdev(deltas):.2f}  "
        f"range {min(deltas):+.2f}..{max(deltas):+.2f}"
    )
    print(
        f"  For reference the BUY/SELL threshold is +/-{BUY_THRESHOLD:.0f}, so a\n"
        f"  shift only matters where it carries a score across that line."
    )
    if changed == 0:
        print("\n  VERDICT: the table is DECORATIVE on this data — it moves the\n"
              "  score but never the decision. Nothing downstream differs.")
        return

    # ---- C: scored head to head -----------------------------------------
    evals = (
        await session.execute(
            text(
                """
                SELECT e.horizon_days AS h, e.return_pct, e.band_pct,
                       d.sub_agents_json, d.symbol, d.timestamp,
                       r.regime::text AS regime
                FROM decision_evaluations e
                JOIN decisions d ON d.timestamp = e.decision_timestamp
                                AND d.symbol   = e.symbol
                JOIN market_regimes r ON r.timestamp::date = d.timestamp::date
                """
            )
        )
    ).all()

    tally: dict[int, Counter] = {}
    decided, band_only, unchanged = [], [], []
    for row in evals:
        scores = _pillar_scores(row.sub_agents_json)
        if scores is None:
            continue
        weights = REGIME_WEIGHTS[MarketRegime(row.regime)]
        a_regime = decide_action(
            DynamicWeightingEngine.weighted_score(scores, weights)[0]
        )
        a_equal = decide_action(
            DynamicWeightingEngine.weighted_score(scores, EQUAL_WEIGHTS)[0]
        )
        ret, band = float(row.return_pct), float(row.band_pct)
        v_regime = classify_verdict(a_regime.value, ret, band)
        v_equal = classify_verdict(a_equal.value, ret, band)
        counter = tally.setdefault(row.h, Counter())
        for tag, verdict in (("reg", v_regime), ("eq", v_equal)):
            if verdict is DecisionVerdict.CORRECT:
                counter[f"{tag}_ok"] += 1
            elif verdict is DecisionVerdict.WRONG:
                counter[f"{tag}_bad"] += 1
        if a_regime is not a_equal:
            record = (row.h, row.symbol, row.timestamp.date(), a_equal.value,
                      a_regime.value, v_equal.value, v_regime.value)
            # Three genuinely different outcomes, and conflating them is how
            # a table with no skill gets credited with some:
            #   CORRECT <-> WRONG    a real difference in being right
            #   CORRECT <-> NEUTRAL  pure band geometry (HOLD is CORRECT
            #                        inside the band; BUY/SELL is NEUTRAL)
            #   same verdict         the action changed, the outcome did not
            pair = {v_equal, v_regime}
            if pair == {DecisionVerdict.CORRECT, DecisionVerdict.WRONG}:
                decided.append(record)
            elif pair == {DecisionVerdict.CORRECT, DecisionVerdict.NEUTRAL}:
                band_only.append(record)
            else:
                unchanged.append(record)

    print()
    print("=" * 74)
    print("  C. DOES IT CHANGE THEM FOR THE BETTER?")
    print("=" * 74)
    print(f"  {'h':>4}{'regime weights':>22}{'equal weights':>22}   winner")
    print("  " + _RULE)
    for horizon in sorted(tally):
        counter = tally[horizon]
        rd = counter["reg_ok"] + counter["reg_bad"]
        ed = counter["eq_ok"] + counter["eq_bad"]
        rr = 100 * counter["reg_ok"] / rd if rd else float("nan")
        er = 100 * counter["eq_ok"] / ed if ed else float("nan")
        winner = "regime" if rr > er else ("equal" if er > rr else "TIE")
        print(f"  +{horizon:>2}d{counter['reg_ok']:>10}/{rd:<4}{rr:>6.1f}%"
              f"{counter['eq_ok']:>12}/{ed:<4}{er:>6.1f}%   {winner}")
    print("  " + _RULE)

    print(f"\n  Flips that crossed CORRECT/WRONG (real skill difference): "
          f"{len(decided)}")
    for rec in decided:
        print(f"    +{rec[0]}d {rec[1]:<11} {rec[2]}  {rec[3]}->{rec[4]}  "
              f"{rec[5]} -> {rec[6]}")
    print(f"\n  Flips that only crossed CORRECT/NEUTRAL (band geometry, NOT "
          f"skill): {len(band_only)}")
    for rec in band_only:
        print(f"    +{rec[0]}d {rec[1]:<11} {rec[2]}  {rec[3]}->{rec[4]}  "
              f"{rec[5]} -> {rec[6]}")
    print(f"\n  Flips where the verdict did not change at all: {len(unchanged)}")
    for rec in unchanged:
        print(f"    +{rec[0]}d {rec[1]:<11} {rec[2]}  {rec[3]}->{rec[4]}  "
              f"{rec[5]} -> {rec[6]}")
    if not decided:
        print(
            "\n  VERDICT: NOT ONE flip turned a CORRECT into a WRONG or back.\n"
            f"  {len(band_only)} were settled purely by band geometry — HOLD is\n"
            "  CORRECT inside the band while BUY/SELL is NEUTRAL there — and\n"
            f"  {len(unchanged)} changed the action without changing the outcome.\n"
            "  The table moved decisions; it did not improve them."
        )


# ---------------------------------------------------------------------------
async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backfill-regimes",
        action="store_true",
        help="Classify and store every historical day first (WRITES).",
    )
    args = parser.parse_args()
    try:
        if args.backfill_regimes:
            async with session_scope() as session:
                before = (
                    await session.execute(
                        select(func.count()).select_from(MarketRegimeRow)
                    )
                ).scalar_one()
            async with session_scope() as session:
                await backfill_regimes(session)
            async with session_scope() as session:
                after = (
                    await session.execute(
                        select(func.count()).select_from(MarketRegimeRow)
                    )
                ).scalar_one()
            print(f"\n  market_regimes: {before} -> {after} row(s)")

        async with session_scope() as session:
            await section_a_coverage(session)
            await section_b_and_c(session)
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)
    print()
    return 0


if __name__ == "__main__":
    setup_logging("WARNING")
    raise SystemExit(asyncio.run(main()))

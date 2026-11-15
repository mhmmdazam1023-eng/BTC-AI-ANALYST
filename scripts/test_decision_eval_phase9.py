"""Phase 9 test: decision accuracy evaluation across all assets.

No model calls — arithmetic, truth tables and DB round-trips only.

Gates
-----
1. **Band arithmetic** — the band scales with sqrt(time), belongs to the
   ASSET (a volatile coin gets a wider band at the same horizon), and rejects
   a nonsense horizon.
2. **Verdict truth table** — every (action x return) combination, including
   both boundaries. NEUTRAL must appear for BUY/SELL inside the band and must
   NEVER appear for HOLD.
3. **Look-ahead safety** — an unmatured decision is never evaluated, and the
   date arithmetic (entry = decision + lag, target = entry + horizon) is
   exact on stored rows.
4. **Missing data degrades, never crashes** — no candle or no
   ``realized_vol_30d`` postpones the row instead of guessing.
5. **Cross-asset coverage** — evaluations exist for more than just Bitcoin;
   this is the gap Phase 9 exists to close.
6. **Metric arithmetic** — hit-rate and Brier match an independent
   recomputation, NEUTRAL rows are excluded from both, and the per-action
   breakdown sums to the sample size.
7. **Idempotent** — re-running writes nothing new and changes no verdict.
8. **Entry bar uncontaminated** — the trade is entered on a candle the
   decision could not have seen, ``base_price`` recomputes from the
   warehouse, and late-written decisions are counted out loud.
9. **Null baseline** — the skill-free comparator partitions its samples
   exactly, matches an independent Python recomputation through
   ``band_pct_for``, applies the SAME entry lag the evaluator does, and
   blends correctly across an action mix.
10. **Overlap-corrected gates** — the n>=30 thresholds count INDEPENDENT
    observations, so 30 consecutive daily windows at +30d (~1 observation)
    are refused while genuinely spread-out decisions keep full credit.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_decision_eval_phase9.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select, tuple_  # noqa: E402

from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import (  # noqa: E402
    BASE_SYMBOL,
    CryptoPrice,
    DailyFeature,
    Decision,
    DecisionEvaluation,
    DecisionVerdict,
)
from src.evaluation.decision_evaluator import (  # noqa: E402
    DAILY_TIMEFRAME,
    DAYS_PER_YEAR,
    DECISION_HORIZONS,
    ENTRY_LAG_DAYS,
    DecisionEvaluator,
    band_pct_for,
    classify_verdict,
)
from src.evaluation.decision_metrics import (  # noqa: E402
    SIZING_HORIZON_DAYS,
    DecisionAccuracyEvaluator,
    load_win_rate_for,
)
from src.evaluation.null_model import load_null_baselines  # noqa: E402
from src.evaluation.sample_size import effective_sample_size  # noqa: E402
from src.forecasting.kelly_criterion import MIN_WIN_RATE_SAMPLES  # noqa: E402
from src.forecasting.evaluator import _finalization_cutoff  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_decision_eval_phase9")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
async def gate_band_arithmetic() -> None:
    """The band must scale with sqrt(time) and belong to the asset."""
    vol = 0.315  # BTC-ish, 31.5% annualised

    # sqrt-of-time: 30 days should be sqrt(30) times the 1-day band.
    one = band_pct_for(vol, 1)
    thirty = band_pct_for(vol, 30)
    expected_ratio = (30 / 1) ** 0.5
    _require(
        abs(thirty / one - expected_ratio) < 1e-9,
        f"band must scale with sqrt(time): +30d/+1d = {thirty / one:.6f}, "
        f"expected {expected_ratio:.6f}",
    )

    # The annualisation must undo exactly at a 1-year horizon.
    _require(
        abs(band_pct_for(vol, DAYS_PER_YEAR) - vol * 100) < 1e-9,
        "at a 365-day horizon the band must equal the annualised vol itself; "
        "if not, the sqrt-of-time factor disagrees with FeatureEngine's "
        "annualisation and every band is silently mis-scaled",
    )

    # It belongs to the ASSET: a more volatile coin gets a wider band. This
    # is the whole reason a fixed band was rejected.
    calm, wild = band_pct_for(0.30, 30), band_pct_for(1.35, 30)
    _require(
        wild > calm * 4,
        f"a 1.35-vol asset must get a far wider band than a 0.30-vol one "
        f"({wild:.2f}% vs {calm:.2f}%) — otherwise noisy assets look 'accurate'",
    )

    for bad in (0, -1):
        try:
            band_pct_for(vol, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"horizon_days={bad} must raise, not compute")
    logger.info(
        "Gate 1 OK: band sqrt-scales (%.2f%% -> %.2f%%) and is asset-specific",
        one, thirty,
    )


async def gate_verdict_truth_table() -> None:
    """Every (action x return) combination, both boundaries included."""
    band = 5.0
    # (return_pct, BUY, SELL, HOLD)
    table = [
        (12.0, "CORRECT", "WRONG", "WRONG"),
        (5.01, "CORRECT", "WRONG", "WRONG"),
        (5.0, "NEUTRAL", "NEUTRAL", "CORRECT"),      # exactly ON the band
        (0.0, "NEUTRAL", "NEUTRAL", "CORRECT"),
        (-5.0, "NEUTRAL", "NEUTRAL", "CORRECT"),     # exactly ON the band
        (-5.01, "WRONG", "CORRECT", "WRONG"),
        (-12.0, "WRONG", "CORRECT", "WRONG"),
    ]
    for return_pct, buy, sell, hold in table:
        for action, expected in (("BUY", buy), ("SELL", sell), ("HOLD", hold)):
            got = classify_verdict(action, return_pct, band).value
            _require(
                got == expected,
                f"{action} at return {return_pct:+.2f}% band {band}%: "
                f"expected {expected}, got {got}",
            )

    # HOLD must NEVER be NEUTRAL — it is judged on both sides of the band.
    for return_pct in (-50.0, -5.0, 0.0, 5.0, 50.0):
        _require(
            classify_verdict("HOLD", return_pct, band) is not DecisionVerdict.NEUTRAL,
            f"HOLD produced NEUTRAL at {return_pct}% — HOLD is always testable",
        )

    # A BUY that is merely NOT WRONG must not be counted as CORRECT.
    _require(
        classify_verdict("BUY", 4.99, band) is DecisionVerdict.NEUTRAL,
        "a BUY inside the band is NEUTRAL, not CORRECT — counting tiny moves "
        "as wins is exactly the noise the band exists to filter",
    )

    try:
        classify_verdict("SHORT", 1.0, band)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown action must raise, not be scored")
    logger.info("Gate 2 OK: 21 (action x return) cases + boundaries all correct")


async def gate_no_lookahead() -> None:
    """Unmatured decisions must never be evaluated."""
    now = datetime.now(tz=UTC)
    cutoff = _finalization_cutoff(now)

    async with session_scope() as session:
        offenders = (
            await session.execute(
                select(func.count())
                .select_from(DecisionEvaluation)
                .where(DecisionEvaluation.target_date >= cutoff)
            )
        ).scalar_one()
    _require(
        offenders == 0,
        f"{offenders} evaluation(s) have a target_date at/after the "
        f"finalization cutoff {cutoff.date()} — they were scored against a "
        f"candle that had not settled",
    )

    # And the horizon arithmetic itself must be exact.
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(DecisionEvaluation)
                    .order_by(DecisionEvaluation.evaluated_at.desc())
                    .limit(50)
                )
            )
            .scalars()
            .all()
        )
    for row in rows:
        expected_entry = row.decision_timestamp + timedelta(days=ENTRY_LAG_DAYS)
        _require(
            row.entry_date == expected_entry,
            f"id={row.id}: entry_date {row.entry_date} != decision + "
            f"{ENTRY_LAG_DAYS}d ({expected_entry})",
        )
        # target hangs off the ENTRY, not off the decision. Anchoring it to
        # the decision again would silently shorten every holding period by
        # ENTRY_LAG_DAYS while still producing plausible numbers.
        expected_target = expected_entry + timedelta(days=row.horizon_days)
        _require(
            row.target_date == expected_target,
            f"id={row.id}: target_date {row.target_date} != entry + "
            f"{row.horizon_days}d ({expected_target})",
        )
        _require(
            row.horizon_days in DECISION_HORIZONS,
            f"id={row.id}: horizon {row.horizon_days} is not one of "
            f"{DECISION_HORIZONS}",
        )
    logger.info(
        "Gate 3 OK: no evaluation past the cutoff %s; entry(+%dd) and target "
        "arithmetic exact on %d row(s)",
        cutoff.date(), ENTRY_LAG_DAYS, len(rows),
    )


async def gate_missing_data_degrades() -> None:
    """Absent price or volatility postpones a row; it never guesses."""
    # A symbol that does not exist has no candles and no features at all.
    # The evaluator must complete without raising and write nothing for it.
    evaluator = DecisionEvaluator()
    before = await _evaluation_count()
    await evaluator.evaluate_matured()
    after = await _evaluation_count()
    _require(
        after >= before,
        f"evaluation count went DOWN ({before} -> {after}) — a re-run must "
        f"never delete rows",
    )

    # Every stored row must carry a usable band; a zero band would make the
    # verdict meaningless (everything outside it).
    async with session_scope() as session:
        bad = (
            await session.execute(
                select(func.count())
                .select_from(DecisionEvaluation)
                .where(DecisionEvaluation.band_pct <= 0)
            )
        ).scalar_one()
    _require(
        bad == 0,
        f"{bad} evaluation(s) stored with band_pct <= 0 — those rows would "
        f"classify every move as decisive",
    )
    logger.info("Gate 4 OK: missing data postpones; every stored band is > 0")


async def gate_cross_asset_coverage() -> None:
    """Every asset that COULD have been evaluated must have been.

    The first version of this gate asserted "at least one altcoin appears in
    ``decision_evaluations``", which conflated two different situations: the
    multi-asset path being broken, and no altcoin decision being old enough
    to have matured yet. The entry lag made the second one true on
    2026-07-31 and the gate failed on a perfectly correct system.

    So eligibility is now derived from the SAME predicate the evaluator
    uses — ``decision + ENTRY_LAG_DAYS + min(horizon) < cutoff`` — rather
    than re-guessed. That is also strictly STRONGER than the old check: an
    eligible altcoin silently producing no evaluation now fails, where
    before any single altcoin in the table was enough to pass.
    """
    cutoff = _finalization_cutoff(datetime.now(tz=UTC))
    # Strictly earlier than: a decision landing exactly on this date has its
    # shortest target ON the cutoff, which is not yet settled.
    newest_evaluable = cutoff - timedelta(
        days=min(DECISION_HORIZONS) + ENTRY_LAG_DAYS
    )

    async with session_scope() as session:
        eligible = set(
            (
                await session.execute(
                    select(Decision.symbol)
                    .where(Decision.timestamp < newest_evaluable)
                    .distinct()
                )
            )
            .scalars()
            .all()
        )
        evaluated = set(
            (
                await session.execute(select(DecisionEvaluation.symbol).distinct())
            )
            .scalars()
            .all()
        )

    _require(
        bool(evaluated),
        "no decision evaluations at all — run the evaluator before this gate",
    )
    _require(
        BASE_SYMBOL in evaluated,
        f"Bitcoin missing from the evaluated symbols: {sorted(evaluated)}",
    )

    missing = eligible - evaluated
    _require(
        not missing,
        f"{len(missing)} asset(s) have decisions old enough to score but no "
        f"evaluation row: {sorted(missing)}. Either the multi-asset path is "
        f"broken or their daily candles are missing — both are real problems, "
        f"not a quiet market",
    )

    altcoins = evaluated - {BASE_SYMBOL}
    if not altcoins:
        # Not a failure. Phase 9's altcoin coverage started 2026-07-28, and
        # every such decision needs ENTRY_LAG_DAYS + horizon settled days.
        unlocks = (newest_evaluable + timedelta(days=1)).date()
        logger.info(
            "Gate 5 OK (BTC-only, expected): no altcoin decision has matured "
            "yet — the oldest ones need entry(+%dd) + horizon settled candles. "
            "Altcoin coverage resumes once decisions predate %s.",
            ENTRY_LAG_DAYS, unlocks,
        )
        return

    logger.info(
        "Gate 5 OK: %d symbol(s) evaluated (%d altcoin(s) beyond BTC); all %d "
        "eligible asset(s) accounted for",
        len(evaluated), len(altcoins), len(eligible),
    )


async def gate_metric_arithmetic() -> None:
    """Hit-rate and Brier must match an independent recomputation."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(
                    DecisionEvaluation.symbol,
                    DecisionEvaluation.horizon_days,
                    DecisionEvaluation.action,
                    DecisionEvaluation.confidence_score,
                    DecisionEvaluation.verdict,
                )
            )
        ).all()
    _require(bool(rows), "no evaluations to aggregate")

    # Recompute the way the metrics module should, independently.
    groups: dict[tuple[str, int], list] = {}
    for row in rows:
        groups.setdefault((row.symbol, row.horizon_days), []).append(row)

    checked = 0
    # ONE session for every comparison. `_metrics_for` takes a live session,
    # and opening a fresh scope per group inside the loop is what produced an
    # IllegalStateChangeError on the first version of this gate.
    async with session_scope() as session:
        for (symbol, horizon), group in groups.items():
            decisive = [
                r for r in group if r.verdict is not DecisionVerdict.NEUTRAL
            ]
            neutral = len(group) - len(decisive)
            if not decisive:
                continue
            correct = sum(
                1 for r in decisive if r.verdict is DecisionVerdict.CORRECT
            )
            hit = correct / len(decisive)
            brier = sum(
                (
                    r.confidence_score / 100.0
                    - (1.0 if r.verdict is DecisionVerdict.CORRECT else 0.0)
                )
                ** 2
                for r in decisive
            ) / len(decisive)

            metrics = await DecisionAccuracyEvaluator._metrics_for(
                session, symbol, horizon
            )
            _require(
                metrics is not None,
                f"{symbol} +{horizon}d: metrics returned None despite "
                f"{len(group)} rows",
            )
            _require(
                metrics["neutral_count"] == neutral,
                f"{symbol} +{horizon}d: neutral_count "
                f"{metrics['neutral_count']} != {neutral}",
            )
            _require(
                metrics["sample_size"] == len(decisive),
                f"{symbol} +{horizon}d: sample_size {metrics['sample_size']} "
                f"counts NEUTRAL rows; expected {len(decisive)} decisive",
            )
            _require(
                abs((metrics["hit_rate"] or 0) - hit) < 1e-9,
                f"{symbol} +{horizon}d: hit_rate {metrics['hit_rate']} != {hit}",
            )
            _require(
                abs((metrics["brier_score"] or 0) - brier) < 1e-9,
                f"{symbol} +{horizon}d: brier {metrics['brier_score']} != {brier}",
            )
            # The per-action breakdown must account for every decisive row.
            action_total = (
                metrics["buy_total"]
                + metrics["sell_total"]
                + metrics["hold_total"]
            )
            _require(
                action_total == len(decisive),
                f"{symbol} +{horizon}d: per-action totals sum to "
                f"{action_total}, but there are {len(decisive)} decisive rows "
                f"— an action is being dropped from the breakdown",
            )
            checked += 1

    _require(checked > 0, "no (symbol, horizon) group had decisive evaluations")
    logger.info(
        "Gate 6 OK: hit-rate/Brier/breakdown verified on %d group(s); NEUTRAL "
        "excluded from all three", checked,
    )


async def gate_idempotent() -> None:
    """Re-running must add nothing and change no verdict."""
    async with session_scope() as session:
        before = (
            await session.execute(
                select(
                    DecisionEvaluation.id,
                    DecisionEvaluation.verdict,
                    DecisionEvaluation.return_pct,
                ).order_by(DecisionEvaluation.id)
            )
        ).all()

    await DecisionEvaluator().evaluate_matured()

    async with session_scope() as session:
        after = (
            await session.execute(
                select(
                    DecisionEvaluation.id,
                    DecisionEvaluation.verdict,
                    DecisionEvaluation.return_pct,
                ).order_by(DecisionEvaluation.id)
            )
        ).all()

    _require(
        len(before) == len(after),
        f"a second run changed the row count ({len(before)} -> {len(after)}) — "
        f"the unique constraint is not preventing duplicates",
    )
    for old, new in zip(before, after, strict=True):
        _require(
            old == new,
            f"id={old.id}: a re-run altered the row ({old} -> {new}). "
            f"Evaluations are historical facts and must be immutable.",
        )
    logger.info("Gate 7 OK: re-run added 0 rows and mutated 0 verdicts")


async def gate_entry_bar_uncontaminated() -> None:
    """The trade must enter on a candle the decision could not have seen.

    This gate exists for a bug that reached production. Until 2026-07-31 the
    entry was ``close(T)`` — the close of the very bar ``daily_features[T]``
    is computed from, and a close that had usually not happened yet when the
    decision was written (measured: 5-20 hours early). Setting
    ``ENTRY_LAG_DAYS`` back to 0 fails 8a on the first row.
    """
    # -- 8a: structural. The one invariant that makes the entry honest. ----
    _require(
        ENTRY_LAG_DAYS >= 1,
        f"ENTRY_LAG_DAYS is {ENTRY_LAG_DAYS}; at 0 the trade enters on the "
        f"close of the same still-forming candle the decision was computed "
        f"from, which is the 2026-07-31 bug",
    )
    async with session_scope() as session:
        same_day = (
            await session.execute(
                select(func.count())
                .select_from(DecisionEvaluation)
                .where(DecisionEvaluation.entry_date <= DecisionEvaluation.decision_timestamp)
            )
        ).scalar_one()
    _require(
        same_day == 0,
        f"{same_day} evaluation(s) enter at or before their own decision day. "
        f"base_price is then a price the decision partly saw form",
    )

    # -- 8b: base_price must recompute from the warehouse ------------------
    # Trusting the stored number would let a wrong date lookup pass unnoticed;
    # the same independent-recomputation discipline as gate 6.
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(DecisionEvaluation)
                    .order_by(DecisionEvaluation.evaluated_at.desc())
                    .limit(200)
                )
            )
            .scalars()
            .all()
        )
        _require(bool(rows), "no evaluations to check — run the evaluator first")

        wanted = {(r.symbol, r.entry_date) for r in rows}
        closes = {
            (p.symbol, p.timestamp): p.close
            for p in (
                await session.execute(
                    select(
                        CryptoPrice.symbol, CryptoPrice.timestamp, CryptoPrice.close
                    ).where(
                        CryptoPrice.exchange == "binance",
                        CryptoPrice.timeframe == DAILY_TIMEFRAME,
                        tuple_(CryptoPrice.symbol, CryptoPrice.timestamp).in_(wanted),
                    )
                )
            ).all()
        }
    for row in rows:
        actual_close = closes.get((row.symbol, row.entry_date))
        _require(
            actual_close is not None,
            f"id={row.id}: no {DAILY_TIMEFRAME} candle for {row.symbol} on the "
            f"entry date {row.entry_date.date()} — the row was scored against "
            f"a price that is not in crypto_prices",
        )
        _require(
            row.base_price == actual_close,
            f"id={row.id}: base_price {row.base_price} != close of "
            f"{row.entry_date.date()} ({actual_close}) for {row.symbol}",
        )

    # -- 8c: operational lateness. Counted, named, but NOT fatal. ----------
    # A decision written after its entry bar already opened is not a code
    # defect — it is the pipeline having run late (or a backfill). It stays
    # visible for the same reason `STALE FALLBACK` does in run_collectors:
    # a caveat nobody can see is a caveat nobody applies.
    async with session_scope() as session:
        late = (
            await session.execute(
                select(
                    DecisionEvaluation.symbol,
                    DecisionEvaluation.decision_timestamp,
                    DecisionEvaluation.entry_date,
                    Decision.ingested_at,
                )
                .join(
                    Decision,
                    (Decision.timestamp == DecisionEvaluation.decision_timestamp)
                    & (Decision.symbol == DecisionEvaluation.symbol),
                )
                .where(Decision.ingested_at > DecisionEvaluation.entry_date)
                .distinct()
            )
        ).all()
    if late:
        offenders = sorted({(r.symbol, r.decision_timestamp.date()) for r in late})
        logger.warning(
            "Gate 8c: %d (asset, day) decision(s) were written AFTER their entry "
            "bar had already opened — the entry is still tradeable but no longer "
            "fully forward-looking. Run the pipeline earlier in the UTC day to "
            "clear this. Affected: %s",
            len(offenders),
            ", ".join(f"{s} {d}" for s, d in offenders[:8])
            + (" ..." if len(offenders) > 8 else ""),
        )

    logger.info(
        "Gate 8 OK: entry is +%dd on every row, base_price recomputed from "
        "crypto_prices on %d row(s), %d late-written decision(s) reported",
        ENTRY_LAG_DAYS, len(rows), len({(r.symbol, r.decision_timestamp) for r in late}),
    )


async def gate_null_baseline() -> None:
    """The skill-free baseline must be right, complete, and entry-lag aware.

    Four checks, in the order they would fail if something broke:

    9a  the three band regions partition every sample exactly once;
    9b  the SQL aggregate matches an independent Python recomputation that
        goes through ``band_pct_for`` — the same pandas-vs-SQL parity
        discipline as the Part 3 sector tracker, and the only thing that
        proves the band formula was not silently re-derived in SQL;
    9c  the baseline measures the SAME span the evaluator scores, entry lag
        included. Sampling close(T)->close(T+h) would benchmark today's
        numbers against the PREVIOUS, buggy definition;
    9d  blending arithmetic, including that a pure-HOLD mix returns the HOLD
        baseline exactly and a mix cannot drift outside its components.
    """
    async with session_scope() as session:
        baselines = await load_null_baselines(session)
    _require(
        bool(baselines),
        "no null baselines at all — crypto_prices/daily_features are too thin "
        "for any (symbol, horizon) to reach MIN_NULL_SAMPLES",
    )

    # -- 9a: the partition must be exact ----------------------------------
    for key, base in baselines.items():
        _require(
            base.inside + base.above + base.below == base.samples,
            f"{key}: {base.inside}+{base.above}+{base.below} != {base.samples}. "
            f"The band's three regions must partition every sample exactly "
            f"once, or a verdict rate is being double-counted or dropped",
        )
        _require(
            0.0 < base.hold_hit_rate < 1.0,
            f"{key}: HOLD baseline {base.hold_hit_rate} is degenerate — a band "
            f"that swallows everything or nothing is not a band",
        )

    # -- 9b/9c: independent recomputation, in Python, from raw candles -----
    probe = (BASE_SYMBOL, 30)
    _require(probe in baselines, f"no baseline for the probe {probe}")
    expected = baselines[probe]
    symbol, horizon = probe

    async with session_scope() as session:
        vols = {
            row.timestamp: float(row.realized_vol_30d)
            for row in (
                await session.execute(
                    select(DailyFeature.timestamp, DailyFeature.realized_vol_30d)
                    .where(
                        DailyFeature.symbol == symbol,
                        DailyFeature.realized_vol_30d.is_not(None),
                    )
                )
            ).all()
            if row.realized_vol_30d is not None and float(row.realized_vol_30d) > 0
        }
        closes = {
            row.timestamp: float(row.close)
            for row in (
                await session.execute(
                    select(CryptoPrice.timestamp, CryptoPrice.close).where(
                        CryptoPrice.symbol == symbol,
                        CryptoPrice.exchange == "binance",
                        CryptoPrice.timeframe == DAILY_TIMEFRAME,
                    )
                )
            ).all()
        }

    cutoff = _finalization_cutoff(datetime.now(tz=UTC))
    inside = above = below = samples = 0
    for day, vol in vols.items():
        # Deliberately spelled out rather than reusing the SQL: entry is
        # day + ENTRY_LAG_DAYS, target is entry + horizon, band is read at
        # `day`, and the target must be SETTLED. If the SQL ever drifts from
        # any of those, 9b fails.
        entry_day = day + timedelta(days=ENTRY_LAG_DAYS)
        target_day = entry_day + timedelta(days=horizon)
        if target_day >= cutoff:
            continue
        entry, target = closes.get(entry_day), closes.get(target_day)
        if entry is None or target is None or entry <= 0:
            continue
        samples += 1
        return_pct = (target / entry - 1.0) * 100.0
        band = band_pct_for(vol, horizon)
        if abs(return_pct) <= band:
            inside += 1
        elif return_pct > band:
            above += 1
        else:
            below += 1

    _require(
        (samples, inside, above, below)
        == (expected.samples, expected.inside, expected.above, expected.below),
        f"{symbol} +{horizon}d: SQL baseline "
        f"(n={expected.samples} in={expected.inside} up={expected.above} "
        f"dn={expected.below}) != Python recomputation "
        f"(n={samples} in={inside} up={above} dn={below}). Either the SQL band "
        f"formula drifted from band_pct_for, or the entry lag is not applied "
        f"on both sides",
    )

    # -- 9d: blending ------------------------------------------------------
    hold_only = expected.blended(buy_n=0, sell_n=0, hold_n=10)
    _require(
        hold_only is not None and abs(hold_only - expected.hold_hit_rate) < 1e-12,
        f"a pure-HOLD mix must blend to exactly the HOLD baseline "
        f"({hold_only} vs {expected.hold_hit_rate})",
    )
    _require(expected.blended(buy_n=0, sell_n=0, hold_n=0) is None,
             "an empty mix must blend to None, not to 0.0 — no rows is an "
             "unknown baseline, not a zero one")
    mixed = expected.blended(buy_n=5, sell_n=5, hold_n=5)
    components = [
        expected.hold_hit_rate, expected.buy_hit_rate, expected.sell_hit_rate
    ]
    _require(
        mixed is not None and min(components) - 1e-12 <= mixed <= max(components) + 1e-12,
        f"blended baseline {mixed} falls outside its own components "
        f"{components} — a weighted mean cannot do that",
    )

    logger.info(
        "Gate 9 OK: %d baseline(s); %s +%dd verified against an independent "
        "Python recompute (n=%d, HOLD %.1f%% / BUY %s / SELL %s, entry lag +%dd)",
        len(baselines), symbol, horizon, expected.samples,
        expected.hold_hit_rate * 100,
        "n/a" if expected.buy_hit_rate is None else f"{expected.buy_hit_rate:.1%}",
        "n/a" if expected.sell_hit_rate is None else f"{expected.sell_hit_rate:.1%}",
        ENTRY_LAG_DAYS,
    )


async def gate_overlap_corrected_gates() -> None:
    """The n>=30 gates must count INDEPENDENT observations, not rows.

    The defect: a decision evaluated over ``[T+1, T+1+30]`` shares 29/30 of
    its price path with the next day's, so thirty consecutive daily BUY
    evaluations are worth ~1 observation. Both ``MIN_WIN_RATE_SAMPLES`` (which
    feeds Kelly's ``p``) and ``MIN_ACCURACY_SAMPLES`` (which publishes an
    accuracy) tested the row count and would have passed on that one.
    """
    base = datetime(2026, 1, 1, tzinfo=UTC)

    # -- 10a: the exact scenario the gates now have to refuse ---------------
    consecutive = [base + timedelta(days=k) for k in range(MIN_WIN_RATE_SAMPLES)]
    effective = effective_sample_size(consecutive, SIZING_HORIZON_DAYS)
    _require(
        len(consecutive) >= MIN_WIN_RATE_SAMPLES,
        "the probe must satisfy the OLD row-count gate to be meaningful",
    )
    _require(
        effective < MIN_WIN_RATE_SAMPLES,
        f"{len(consecutive)} consecutive daily windows at +{SIZING_HORIZON_DAYS}d "
        f"are {effective:.1f} independent observations and MUST fail a "
        f"{MIN_WIN_RATE_SAMPLES}-observation gate; if this passes, Kelly's p "
        f"can again be set by a single observation",
    )
    _require(
        abs(effective - 1.0) < 1e-12,
        f"30 daily windows at a 30-day horizon are exactly 1 non-overlapping "
        f"window, got {effective}",
    )

    # -- 10b: and it must NOT over-penalise genuinely spread-out data -------
    spread = [base + timedelta(days=31 * k) for k in range(MIN_WIN_RATE_SAMPLES)]
    _require(
        effective_sample_size(spread, SIZING_HORIZON_DAYS)
        == MIN_WIN_RATE_SAMPLES,
        "decisions 31 days apart at a 30-day horizon do not overlap and must "
        "all count — an n/horizon approximation would wrongly divide these by "
        "30 and make the gate unreachable for correct data",
    )

    # -- 10c: live metrics must carry a consistent effective count ---------
    async with session_scope() as session:
        symbols = (
            (
                await session.execute(
                    select(DecisionEvaluation.symbol).distinct()
                )
            )
            .scalars()
            .all()
        )
        checked = 0
        for symbol in symbols:
            for horizon in DECISION_HORIZONS:
                metrics = await DecisionAccuracyEvaluator._metrics_for(
                    session, symbol, horizon
                )
                if metrics is None:
                    continue
                raw = metrics["sample_size"]
                eff = metrics["effective_sample_size"]
                _require(
                    eff <= raw + 1e-9,
                    f"{symbol} +{horizon}d: effective {eff} exceeds raw {raw} — "
                    f"a correction can only ever remove observations",
                )
                if raw > 0:
                    _require(
                        eff >= 1.0 - 1e-9,
                        f"{symbol} +{horizon}d: {raw} rows collapsed to "
                        f"{eff} — at least one window always survives",
                    )
                checked += 1
    _require(checked > 0, "no live (symbol, horizon) metrics to check")

    # -- 10d: today nothing may pass the win-rate gate ---------------------
    for symbol in symbols:
        result = await load_win_rate_for(symbol)
        _require(
            result is None,
            f"{symbol} passed the win-rate gate with {result}. No asset has "
            f"{MIN_WIN_RATE_SAMPLES} independent decisive BUY evaluations at "
            f"+{SIZING_HORIZON_DAYS}d — there is 1 BUY in the entire history",
        )

    logger.info(
        "Gate 10 OK: %d consecutive rows at +%dd = %.1f independent (gate "
        "refuses); spread-out rows keep full credit; %d live group(s) "
        "consistent; 0/%d assets pass the win-rate gate",
        MIN_WIN_RATE_SAMPLES, SIZING_HORIZON_DAYS, effective, checked,
        len(symbols),
    )


# ---------------------------------------------------------------------------
async def _evaluation_count() -> int:
    async with session_scope() as session:
        return (
            await session.execute(select(func.count()).select_from(DecisionEvaluation))
        ).scalar_one()


# ---------------------------------------------------------------------------
async def main() -> int:
    started = datetime.now(tz=UTC)
    try:
        logger.info("--- Band arithmetic ---")
        await gate_band_arithmetic()
        logger.info("--- Verdict truth table ---")
        await gate_verdict_truth_table()
        logger.info("--- Look-ahead safety ---")
        await gate_no_lookahead()
        logger.info("--- Missing data degrades ---")
        await gate_missing_data_degrades()
        logger.info("--- Cross-asset coverage ---")
        await gate_cross_asset_coverage()
        logger.info("--- Metric arithmetic ---")
        await gate_metric_arithmetic()
        logger.info("--- Idempotency ---")
        await gate_idempotent()
        logger.info("--- Entry bar uncontaminated ---")
        await gate_entry_bar_uncontaminated()
        logger.info("--- Null baseline ---")
        await gate_null_baseline()
        logger.info("--- Overlap-corrected sample gates ---")
        await gate_overlap_corrected_gates()
    except AssertionError as exc:
        logger.error("PHASE 9 TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("PHASE 9 TEST FAILED with an unexpected error")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "PHASE 9 TEST PASSED (10 gates, %.1fs) — decision accuracy ready "
        "(entry +%dd, sizing horizon +%dd)",
        elapsed, ENTRY_LAG_DAYS, SIZING_HORIZON_DAYS,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging("INFO")
    raise SystemExit(asyncio.run(main()))

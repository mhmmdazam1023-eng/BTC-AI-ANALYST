"""Phase 9 test: baseline strategy comparator.

No model calls. Synthetic rows for the strategy logic, the live warehouse for
the cross-checks.

Gates
-----
1. **Strategy logic** — each strategy returns the action it promises,
   ``sma_crossover`` HOLDs while either average is still warming up, and
   ``coin_flip`` is seeded (same input, same answer, forever).
2. **Scored by the SAME machinery** — a synthetic row with a known return
   and band produces exactly the verdict ``classify_verdict`` would give, so
   the comparator cannot drift into scoring itself differently.
3. **CROSS-CHECK vs the null model** — ``buy_and_hold``'s hit rate must equal
   ``NullBaseline.buy_hit_rate`` and ``always_hold``'s must equal
   ``hold_hit_rate``, per symbol, EXACTLY. Two independent implementations of
   the same quantity: one aggregates verdicts, the other counts band regions.
   Divergence means one of them is wrong and the scoreboard is unsafe.
4. **Per-action accounting** — the breakdown sums to the totals, and the
   action mix sums to 1. This is what keeps the HOLD-bias visible.
5. **Grid restriction** — asking for a specific set of days returns only
   those days, which is what makes the head-to-head honest.
6. **Snapshot grid parity** — ``edge_vs_baseline`` subtracts two hit rates,
   and they must come from the SAME population. Locks out the C5 bug where
   the baseline was scored over all history while the system was scored on
   its own decisions: it reported +13.0pp at +7d where the truth is -6.2pp.
7. **Daily snapshot (Stage A6)** — composes, persists, upserts in place, and
   keeps an uncomputable IC NULL rather than 0.0.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_baseline_comparison.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import (  # noqa: E402
    BASE_SYMBOL,
    DecisionVerdict,
    MeasurementSnapshot,
)
from src.evaluation.baseline_strategies import (  # noqa: E402
    SMA_FAST_BARS,
    SMA_SLOW_BARS,
    STRATEGIES,
    MarketRow,
    compare_baselines,
    load_market_rows,
    score_strategy,
)
from src.evaluation.decision_evaluator import (  # noqa: E402
    band_pct_for,
    classify_verdict,
)
from src.evaluation.measurement_snapshot import save_snapshots  # noqa: E402
from src.evaluation.null_model import load_null_baselines  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_baseline_comparison")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _row(
    *,
    symbol: str = "TEST/USDT",
    day: datetime | None = None,
    fast: float | None = 110.0,
    slow: float | None = 100.0,
    entry: float = 100.0,
    target: float = 100.0,
    vol: float = 0.40,
) -> MarketRow:
    return MarketRow(
        symbol=symbol,
        day=day or datetime(2026, 1, 1, tzinfo=UTC),
        close=entry,
        sma_fast=fast,
        sma_slow=slow,
        realized_vol_30d=vol,
        entry_close=entry,
        target_close=target,
    )


# ---------------------------------------------------------------------------
async def gate_strategy_logic() -> None:
    """Each strategy must do what its name claims, and warm up honestly."""
    _require(
        STRATEGIES["buy_and_hold"](_row()) == "BUY",
        "buy_and_hold must always BUY",
    )
    _require(
        STRATEGIES["always_hold"](_row()) == "HOLD",
        "always_hold must always HOLD",
    )

    sma = STRATEGIES["sma_crossover"]
    _require(sma(_row(fast=110.0, slow=100.0)) == "BUY", "fast above slow -> BUY")
    _require(sma(_row(fast=90.0, slow=100.0)) == "SELL", "fast below slow -> SELL")
    _require(sma(_row(fast=100.0, slow=100.0)) == "HOLD", "exact tie -> HOLD")

    # Warm-up honesty: a missing average must not be guessed around. This is
    # the same rule the intraday layer needed (WARMUP_BARS writes NULL rather
    # than a half-formed indicator).
    _require(
        sma(_row(fast=None, slow=100.0)) == "HOLD"
        and sma(_row(fast=110.0, slow=None)) == "HOLD",
        f"with fewer than {SMA_SLOW_BARS} bars the slow average does not "
        f"exist, and the strategy must HOLD rather than take a direction from "
        f"a partial window",
    )

    # Seeded: the same (symbol, day) must give the same call forever, or two
    # runs of the report cannot be compared with each other.
    flip = STRATEGIES["coin_flip"]
    first = [flip(_row(symbol=f"S{i}/USDT")) for i in range(50)]
    second = [flip(_row(symbol=f"S{i}/USDT")) for i in range(50)]
    _require(first == second, "coin_flip must be deterministic for a given key")
    _require(
        len(set(first)) == 3,
        f"coin_flip must actually use all three actions, saw {set(first)}",
    )
    logger.info(
        "Gate 1 OK: 4 strategies behave; SMA holds during warm-up (%d/%d bars); "
        "coin_flip reproducible", SMA_FAST_BARS, SMA_SLOW_BARS,
    )


async def gate_same_scoring_machinery() -> None:
    """The comparator must score exactly like a real decision does."""
    vol, horizon = 0.40, 30
    band = band_pct_for(vol, horizon)
    # A move of 1.5x the band is decisive; 0.5x is inside it.
    big = 100.0 * (1 + 1.5 * band / 100.0)
    small = 100.0 * (1 + 0.5 * band / 100.0)

    decisive = score_strategy(
        "buy_and_hold", [_row(entry=100.0, target=big, vol=vol)], horizon
    )
    _require(
        decisive.correct == 1 and decisive.neutral == 0,
        f"a BUY on a +1.5-band move must be CORRECT, got {decisive}",
    )
    _require(
        classify_verdict("BUY", (big / 100.0 - 1) * 100, band)
        is DecisionVerdict.CORRECT,
        "and classify_verdict must agree — if these ever differ the "
        "comparator has grown its own scoring rule",
    )

    inside = score_strategy(
        "buy_and_hold", [_row(entry=100.0, target=small, vol=vol)], horizon
    )
    _require(
        inside.neutral == 1 and inside.decisive == 0,
        f"a BUY inside the band must be NEUTRAL and excluded, got {inside}",
    )
    held = score_strategy(
        "always_hold", [_row(entry=100.0, target=small, vol=vol)], horizon
    )
    _require(
        held.correct == 1 and held.neutral == 0,
        f"HOLD inside the band must be CORRECT and is never NEUTRAL, got {held}",
    )
    logger.info("Gate 2 OK: verdicts identical to classify_verdict, band-scaled")


async def gate_cross_check_null_model() -> None:
    """Two independent paths to the same number must agree EXACTLY.

    ``buy_and_hold`` aggregates verdicts row by row; ``NullBaseline`` counts
    band regions in SQL. They are the same quantity. If they disagree, the
    null model that every hit rate is now judged against is wrong — which
    would quietly invalidate the whole scoreboard rather than fail loudly.
    """
    horizon = 30
    async with session_scope() as session:
        baselines = await load_null_baselines(session, (horizon,))
        rows = await load_market_rows(session, horizon)
    _require(bool(rows) and bool(baselines), "no data to cross-check")

    by_symbol: dict[str, list[MarketRow]] = {}
    for row in rows:
        by_symbol.setdefault(row.symbol, []).append(row)

    checked = 0
    for symbol, symbol_rows in by_symbol.items():
        baseline = baselines.get((symbol, horizon))
        if baseline is None:  # below MIN_NULL_SAMPLES — nothing published
            continue
        buy = score_strategy("buy_and_hold", symbol_rows, horizon)
        hold = score_strategy("always_hold", symbol_rows, horizon)

        _require(
            buy.total == baseline.samples,
            f"{symbol}: comparator saw {buy.total} rows, null model "
            f"{baseline.samples} — the two are reading different data, so any "
            f"agreement below would be luck",
        )
        _require(
            buy.hit_rate is not None
            and abs(buy.hit_rate - (baseline.buy_hit_rate or -1)) < 1e-12,
            f"{symbol}: buy_and_hold {buy.hit_rate} != null buy rate "
            f"{baseline.buy_hit_rate}",
        )
        _require(
            hold.hit_rate is not None
            and abs(hold.hit_rate - baseline.hold_hit_rate) < 1e-12,
            f"{symbol}: always_hold {hold.hit_rate} != null hold rate "
            f"{baseline.hold_hit_rate}",
        )
        checked += 1

    _require(checked >= 5, f"only {checked} symbol(s) cross-checked; expected 5+")
    logger.info(
        "Gate 3 OK: buy_and_hold == null BUY rate and always_hold == null HOLD "
        "rate, exactly, on %d symbol(s)", checked,
    )


async def gate_per_action_accounting() -> None:
    """The per-action breakdown must account for every row."""
    async with session_scope() as session:
        results = await compare_baselines(session, 7, symbols=[BASE_SYMBOL])
    _require(bool(results), "no baseline results to check")

    for name, result in results.items():
        totals = [sum(counts) for counts in result.by_action.values()]
        _require(
            sum(totals) == result.total,
            f"{name}: per-action rows sum to {sum(totals)} but total is "
            f"{result.total} — an action is being dropped",
        )
        correct = sum(counts[0] for counts in result.by_action.values())
        wrong = sum(counts[1] for counts in result.by_action.values())
        _require(
            correct == result.correct and wrong == result.wrong,
            f"{name}: per-action verdicts ({correct}/{wrong}) disagree with the "
            f"aggregate ({result.correct}/{result.wrong})",
        )
        mix = result.action_mix
        _require(
            abs(sum(mix.values()) - 1.0) < 1e-9,
            f"{name}: action mix sums to {sum(mix.values())}, not 1.0",
        )

    # The bias this whole decomposition exists to expose: always_hold beats
    # buy_and_hold on aggregate purely by never taking a direction.
    hold_rate = results["always_hold"].hit_rate
    buy_rate = results["buy_and_hold"].hit_rate
    _require(
        hold_rate is not None and buy_rate is not None and hold_rate > buy_rate,
        f"always_hold ({hold_rate}) is expected to out-score buy_and_hold "
        f"({buy_rate}) on aggregate — that gap IS the HOLD bias, and if it "
        f"vanished the band would no longer be a 1-sigma band",
    )
    logger.info(
        "Gate 4 OK: per-action accounting exact; HOLD bias visible "
        "(always_hold %.1f%% vs buy_and_hold %.1f%%)",
        hold_rate * 100, buy_rate * 100,
    )


async def gate_grid_restriction() -> None:
    """Restricting to a set of days must return only those days."""
    horizon = 7
    async with session_scope() as session:
        everything = await load_market_rows(session, horizon, symbols=[BASE_SYMBOL])
        _require(len(everything) > 10, "not enough history for this gate")

        wanted = sorted({row.day for row in everything})[:3]
        restricted = await load_market_rows(
            session, horizon, symbols=[BASE_SYMBOL], days=wanted
        )
        empty = await load_market_rows(
            session, horizon, symbols=[BASE_SYMBOL], days=[]
        )

    _require(
        {row.day for row in restricted} == set(wanted),
        f"grid restriction returned {sorted({r.day for r in restricted})}, "
        f"expected {wanted}",
    )
    _require(
        len(restricted) < len(everything),
        "restricting to 3 days must return fewer rows than the full history",
    )
    _require(
        empty == [],
        "an empty day list must return nothing, not silently fall back to the "
        "full history — that would make a head-to-head compare different grids",
    )
    logger.info(
        "Gate 5 OK: %d row(s) -> %d on a 3-day grid; empty grid returns nothing",
        len(everything), len(restricted),
    )


async def gate_snapshot_grid_parity() -> None:
    """`edge_vs_baseline` must subtract two rates from the SAME population.

    The bug this locks out (found 2026-08-09, C5): ``build_snapshot`` scored
    the baselines over every scoreable asset-day in history while
    ``system_hit_rate`` came from the system's own matured decisions, then
    subtracted one from the other. It reported +6.4pp at +1d where the
    honest same-grid answer was +1.4pp, and — worse — **+13.0pp at +7d where
    the same-grid answer is −6.2pp**. A sign error in the headline number,
    inside the row built specifically to be trusted.

    The check recomputes the baseline straight from ``decision_evaluations``
    (always_hold is correct exactly when ``|return| <= band``) and demands
    the snapshot match it.
    """
    from sqlalchemy import text

    from src.evaluation.measurement_snapshot import build_snapshot

    checked = 0
    for horizon in (1, 7, 30):
        payload = await build_snapshot(horizon)
        if payload is None or payload.get("baseline_best_hit_rate") is None:
            continue
        async with session_scope() as session:
            row = (
                await session.execute(
                    text(
                        """
                        SELECT count(*) AS n,
                               count(*) FILTER (
                                   WHERE abs(return_pct) <= band_pct
                               ) AS hold_ok
                        FROM decision_evaluations
                        WHERE horizon_days = :h
                        """
                    ),
                    {"h": horizon},
                )
            ).one()
        if row.n == 0:
            continue
        expected = row.hold_ok / row.n
        actual = payload["baseline_best_hit_rate"]
        # always_hold is the usual winner; if some other strategy beat it the
        # comparison below is not meaningful, so only assert when it is.
        if payload["baseline_best_name"] != "always_hold":
            logger.info(
                "  +%dd: best baseline is %s, not always_hold — parity check "
                "skipped for this horizon", horizon, payload["baseline_best_name"],
            )
            continue
        _require(
            abs(actual - expected) < 1e-9,
            f"+{horizon}d always_hold must score {expected:.4f} on the "
            f"system's own {row.n}-row grid, snapshot says {actual:.4f}. A "
            f"gap means the baseline is being measured on a different "
            f"population from the system it is subtracted from.",
        )
        checked += 1

    _require(
        checked > 0,
        "no horizon could be checked — the gate proved nothing. It needs at "
        "least one horizon with matured evaluations AND a scoreable baseline.",
    )
    logger.info(
        "Gate 6 OK: edge_vs_baseline verified against a same-grid recompute "
        "on %d horizon(s)", checked,
    )


async def gate_daily_snapshot() -> None:
    """The Stage A6 snapshot must compose, persist and re-run cleanly.

    This gate exists because the snapshot is the ONLY consumer of three
    otherwise report-only layers. If it silently stops writing, the trend
    just stops growing and nothing fails — the quietest possible breakage.
    """
    before = await _snapshot_count()
    written = await save_snapshots()
    _require(
        written > 0,
        "the snapshot wrote nothing; with matured evaluations and a scoreable "
        "baseline there should be at least one row",
    )

    # Re-running the same day must refresh in place. The daily job can
    # legitimately run twice, and the second run has strictly more matured
    # evaluations behind it — it must win, not duplicate or be rejected.
    again = await save_snapshots()
    after = await _snapshot_count()
    _require(
        again == written and after == before + written or after == before,
        f"re-running changed the row count ({before} -> {after} after writing "
        f"{written} twice) — the (timestamp, horizon_days) upsert is not "
        f"holding",
    )

    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(MeasurementSnapshot).order_by(
                        MeasurementSnapshot.horizon_days
                    )
                )
            )
            .scalars()
            .all()
        )
    _require(bool(rows), "no snapshot rows to inspect")

    for row in rows:
        # Composition check: the stored edge must equal the difference of the
        # two stored components. A snapshot that recomputed either of them
        # independently would drift from the modules that own them.
        if row.system_hit_rate is not None and row.null_hit_rate is not None:
            _require(
                row.edge_vs_null is not None
                and abs(
                    row.edge_vs_null - (row.system_hit_rate - row.null_hit_rate)
                )
                < 1e-12,
                f"+{row.horizon_days}d: edge_vs_null {row.edge_vs_null} != "
                f"{row.system_hit_rate} - {row.null_hit_rate}",
            )
        if (
            row.system_hit_rate is not None
            and row.baseline_best_hit_rate is not None
        ):
            _require(
                row.edge_vs_baseline is not None
                and abs(
                    row.edge_vs_baseline
                    - (row.system_hit_rate - row.baseline_best_hit_rate)
                )
                < 1e-12,
                f"+{row.horizon_days}d: edge_vs_baseline is not the stored "
                f"difference",
            )
        # An IC that cannot be computed must be NULL, never 0.0 — absent is
        # not zero, the same contract as the Part 4 features.
        if row.ic_panel_mean is None:
            _require(
                row.ic_n_days is None,
                f"+{row.horizon_days}d: ic_n_days is set while ic_panel_mean "
                f"is NULL — a sample size for a measurement that does not exist",
            )
        _require(
            row.system_effective_n is None
            or row.system_effective_n <= row.system_decisive + 1e-9,
            f"+{row.horizon_days}d: effective n {row.system_effective_n} "
            f"exceeds {row.system_decisive} decisive rows",
        )
    logger.info(
        "Gate 7 OK: snapshot wrote %d row(s), upsert idempotent, stored edges "
        "equal their components, absent IC stays NULL", written,
    )


async def _snapshot_count() -> int:
    async with session_scope() as session:
        return (
            await session.execute(
                select(func.count()).select_from(MeasurementSnapshot)
            )
        ).scalar_one()


# ---------------------------------------------------------------------------
async def main() -> int:
    started = datetime.now(tz=UTC)
    try:
        logger.info("--- Strategy logic ---")
        await gate_strategy_logic()
        logger.info("--- Same scoring machinery ---")
        await gate_same_scoring_machinery()
        logger.info("--- Cross-check vs null model ---")
        await gate_cross_check_null_model()
        logger.info("--- Per-action accounting ---")
        await gate_per_action_accounting()
        logger.info("--- Grid restriction ---")
        await gate_grid_restriction()
        logger.info("--- Snapshot grid parity ---")
        await gate_snapshot_grid_parity()
        logger.info("--- Daily measurement snapshot ---")
        await gate_daily_snapshot()
    except AssertionError as exc:
        logger.error("BASELINE COMPARISON TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("BASELINE COMPARISON TEST FAILED unexpectedly")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "BASELINE COMPARISON TEST PASSED (7 gates, %.1fs) — comparator agrees "
        "with the null model exactly, and edge_vs_baseline is same-grid",
        elapsed,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging("INFO")
    raise SystemExit(asyncio.run(main()))

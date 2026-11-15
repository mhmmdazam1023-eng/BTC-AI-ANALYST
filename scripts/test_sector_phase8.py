"""Phase 8 Part 3 test: sector rotation tracker.

Gates
-----
1. **Registry**   — every sector has enough members to be a sector, all
   symbols are registered against BTC, and BTC itself is not in any sector
   bucket.
2. **SQL parity** — the pandas aggregation is compared against an
   INDEPENDENT SQL ``AVG``/``COUNT`` over the same day. Two implementations
   must agree, or the number is not trustworthy.
3. **NaN safety** — a coin with no ``rs_30d`` (fresh listing) must not
   corrupt the sector's 30-day average, and must be excluded from
   ``coin_count_30d`` while still counting toward ``coin_count``.
4. **Status**     — the 2x2 quadrants each fire on their own corner, every
   status is reachable, and ACCUMULATING/PULLBACK are not swapped (the exact
   error the first Part 3 pass shipped with).
5. **Ranking**    — strictly ordered by ``avg_rs_7d``, 1-based, no gaps or
   duplicates; unusable averages sort last.
6. **Persistence**— ``sector_performance`` round-trips and re-running the
   same day updates in place rather than duplicating.
7. **Agent**      — the Master Agent accepts ``sector_context`` and the
   text reaches the prompt; omitting it leaves the prompt unchanged.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_sector_phase8.py
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select, text  # noqa: E402

from src.agents import MasterAgent  # noqa: E402
from src.agents.schemas import MasterDecision, Signal, SubAgentResponse  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import (  # noqa: E402
    BASE_SYMBOL,
    CoinRegistry,
    SectorFlowStatus,
    SectorPerformance,
)
from src.features import SectorTracker  # noqa: E402
from src.features.sector_tracker import MIN_COINS_FOR_CONFIDENCE  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_sector_phase8")

#: Sectors that must exist after Part 3 seeding.
EXPECTED_SECTORS = ("L1", "L2", "DeFi", "AI", "Meme", "RWA", "Infrastructure")

#: Sectors explicitly allowed to run below MIN_COINS_FOR_CONFIDENCE.
#:
#: EMPTY since 2026-08-09, and it must stay that way unless someone accepts
#: a thin sector again on purpose. Infrastructure was the sole entry — it
#: held only LINK, accepted 2026-07-28 on the understanding that the tracker
#: flagged it THIN everywhere. PYTH and TIA closed that gap, and leaving the
#: exemption behind would let Infrastructure silently collapse back to one
#: member without the gate objecting. A dead allowlist entry is a gate that
#: has quietly stopped guarding the thing it names.
#:
#: The change was not cosmetic: with LINK alone the sector read PULLBACK at
#: +4.67% over 30 days; with three members it reads OUT_OF_FLOW at -10.06%.
#: One token was telling the opposite story from its own sector.
THIN_SECTORS_ALLOWED: frozenset[str] = frozenset()

#: Averages are NUMERIC(18,6) round-tripped through float64.
TOLERANCE = 1e-6


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
async def gate_registry() -> None:
    """Every sector must have enough members to average meaningfully."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(CoinRegistry)
                .where(CoinRegistry.is_active.is_(True))
                .order_by(CoinRegistry.sector, CoinRegistry.symbol)
            )
        ).scalars().all()

    by_sector: dict[str, list[str]] = {}
    for row in rows:
        if row.symbol == BASE_SYMBOL:
            continue
        by_sector.setdefault(row.sector.value, []).append(row.symbol)

    missing = [s for s in EXPECTED_SECTORS if s not in by_sector]
    _require(
        not missing,
        f"sector(s) missing from the registry: {missing} — run "
        f"scripts/seed_sectors_phase8.py",
    )
    thin = {
        s: m
        for s, m in by_sector.items()
        if len(m) < MIN_COINS_FOR_CONFIDENCE and s not in THIN_SECTORS_ALLOWED
    }
    _require(
        not thin,
        f"sector(s) with <{MIN_COINS_FOR_CONFIDENCE} coins: {thin} — a sector "
        "average over one coin is that coin's noise. Add members, or add the "
        "sector to THIN_SECTORS_ALLOWED if that is a deliberate choice.",
    )
    for sector in EXPECTED_SECTORS:
        members = by_sector[sector]
        note = (
            "  [THIN — accepted]"
            if len(members) < MIN_COINS_FOR_CONFIDENCE
            else ""
        )
        logger.info(
            "Gate 1 OK: %-14s %d coins: %s%s",
            sector, len(members), ", ".join(members), note,
        )

    # LINK is an oracle network: Infrastructure, not DeFi (user, 2026-07-28).
    link = next((r for r in rows if r.symbol == "LINK/USDT"), None)
    _require(link is not None, "LINK/USDT missing from the registry")
    _require(
        link.sector.value == "Infrastructure",
        f"LINK/USDT should be Infrastructure, found {link.sector.value}",
    )
    logger.info("Gate 1 OK: LINK/USDT classified as Infrastructure")

    # BTC is the denominator, so it must never sit inside a sector bucket.
    base = next((r for r in rows if r.symbol == BASE_SYMBOL), None)
    _require(base is not None, f"{BASE_SYMBOL} must stay in the registry")
    _require(
        base.base_asset is None,
        f"{BASE_SYMBOL} must have base_asset NULL — it IS the numéraire",
    )
    for row in rows:
        if row.symbol == BASE_SYMBOL:
            continue
        _require(
            row.base_asset == BASE_SYMBOL,
            f"{row.symbol} should be benchmarked against {BASE_SYMBOL}, "
            f"got {row.base_asset}",
        )
    logger.info("Gate 1 OK: all %d altcoins benchmarked against BTC", len(rows) - 1)


async def gate_sql_parity(tracker: SectorTracker) -> datetime:
    """pandas aggregation must equal an independent SQL AVG/COUNT."""
    frame = await tracker.load_sector_data()
    as_of = frame.attrs["as_of"]
    snapshot = tracker.compute(frame)

    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT r.sector::text          AS sector,
                           AVG(f.rs_7d)            AS avg_7d,
                           AVG(f.rs_30d)           AS avg_30d,
                           COUNT(f.rs_7d)          AS n_7d,
                           COUNT(f.rs_30d)         AS n_30d
                    FROM daily_features f
                    JOIN coins_registry r ON r.symbol = f.symbol
                    WHERE f.timestamp = :day
                      AND r.is_active
                      AND f.symbol <> :base
                    GROUP BY r.sector
                    """
                ),
                {"day": as_of, "base": BASE_SYMBOL},
            )
        ).all()
    sql = {row[0]: row for row in rows}

    _require(bool(sql), "the SQL cross-check returned no rows")
    for reading in snapshot.rankings:
        _require(
            reading.sector in sql,
            f"{reading.sector} present in pandas but not in SQL",
        )
        _, avg_7d, avg_30d, n_7d, n_30d = sql[reading.sector]

        _require(
            reading.coin_count == n_7d,
            f"{reading.sector}: coin_count {reading.coin_count} != SQL COUNT "
            f"{n_7d}",
        )
        _require(
            reading.coin_count_30d == n_30d,
            f"{reading.sector}: coin_count_30d {reading.coin_count_30d} != "
            f"SQL COUNT {n_30d}",
        )
        for label, ours, theirs in (
            ("avg_rs_7d", reading.avg_rs_7d, avg_7d),
            ("avg_rs_30d", reading.avg_rs_30d, avg_30d),
        ):
            if theirs is None:
                _require(
                    ours is None,
                    f"{reading.sector}: SQL {label} is NULL but pandas gave {ours}",
                )
                continue
            _require(
                ours is not None
                and math.isclose(ours, float(theirs), rel_tol=0, abs_tol=TOLERANCE),
                f"{reading.sector}: {label} pandas={ours} vs SQL={float(theirs)}",
            )
        logger.info(
            "Gate 2 OK: %-14s pandas == SQL (7d %+.4f n=%d | 30d %s n=%d)",
            reading.sector, reading.avg_rs_7d, reading.coin_count,
            "n/a" if reading.avg_rs_30d is None else f"{reading.avg_rs_30d:+.4f}",
            reading.coin_count_30d,
        )

    # BTC must never have leaked into a sector bucket.
    async with session_scope() as session:
        leaked = (
            await session.execute(
                text(
                    "SELECT count(*) FROM coins_registry r "
                    "JOIN daily_features f ON f.symbol = r.symbol "
                    "WHERE f.timestamp = :day AND f.symbol = :base"
                ),
                {"day": as_of, "base": BASE_SYMBOL},
            )
        ).scalar_one()
    total_counted = sum(r.coin_count for r in snapshot.rankings)
    _require(
        total_counted == len(frame[frame["rs_7d"].notna()]),
        "sector coin counts do not add up to the loaded rows",
    )
    _require(
        BASE_SYMBOL not in set(frame["symbol"]),
        f"{BASE_SYMBOL} leaked into the sector frame (rs_7d is 0 by "
        f"construction and would drag every average toward zero); "
        f"{leaked} registry row(s) exist for it",
    )
    logger.info("Gate 2 OK: BTC excluded from the sector aggregation")
    return as_of


def gate_nan_safety(tracker: SectorTracker) -> None:
    """A fresh listing without rs_30d must not corrupt the 30-day average."""
    frame = pd.DataFrame(
        {
            "sector": ["AI", "AI", "AI", "Meme", "Meme"],
            "symbol": [
                "OLD1/USDT", "OLD2/USDT", "NEW/USDT", "M1/USDT", "M2/USDT",
            ],
            "rs_7d": [10.0, 20.0, 30.0, -5.0, -15.0],
            # NEW/USDT listed days ago: it has a 7d number but no 30d one.
            "rs_30d": [100.0, 200.0, float("nan"), 40.0, float("nan")],
        }
    )
    frame.attrs["as_of"] = datetime(2026, 1, 1, tzinfo=UTC)
    snapshot = tracker.compute(frame)
    readings = {reading.sector: reading for reading in snapshot.rankings}

    ai = readings["AI"]
    _require(
        math.isclose(ai.avg_rs_7d, 20.0, abs_tol=1e-9),
        f"AI avg_rs_7d should be (10+20+30)/3 = 20.0, got {ai.avg_rs_7d}",
    )
    # The NaN must be SKIPPED, not treated as zero. Zero would give 100.0.
    _require(
        math.isclose(ai.avg_rs_30d, 150.0, abs_tol=1e-9),
        f"AI avg_rs_30d should be (100+200)/2 = 150.0 (NaN skipped, NOT "
        f"counted as 0 which would give 100.0), got {ai.avg_rs_30d}",
    )
    _require(
        ai.coin_count == 3 and ai.coin_count_30d == 2,
        f"AI counts should be 3/2, got {ai.coin_count}/{ai.coin_count_30d}",
    )
    logger.info(
        "Gate 3 OK: NaN rs_30d skipped — AI 7d=%.1f (n=3), 30d=%.1f (n=2), "
        "not dragged to 100.0 by a zero-fill",
        ai.avg_rs_7d, ai.avg_rs_30d,
    )

    meme = readings["Meme"]
    _require(
        math.isclose(meme.avg_rs_30d, 40.0, abs_tol=1e-9),
        f"Meme avg_rs_30d should be 40.0 from its single valid coin, got "
        f"{meme.avg_rs_30d}",
    )
    _require(
        meme.coin_count_30d == 1,
        "a 30-day average resting on ONE coin must say so via coin_count_30d",
    )
    logger.info(
        "Gate 3 OK: a 30d average from 1 of 2 coins is flagged "
        "(coin_count=%d, coin_count_30d=%d)",
        meme.coin_count, meme.coin_count_30d,
    )

    # An all-NaN 30d column must yield None, never a crash or a zero.
    blank = pd.DataFrame(
        {
            "sector": ["RWA", "RWA"],
            "symbol": ["A/USDT", "B/USDT"],
            "rs_7d": [1.0, 3.0],
            "rs_30d": [float("nan"), float("nan")],
        }
    )
    blank.attrs["as_of"] = datetime(2026, 1, 1, tzinfo=UTC)
    reading = tracker.compute(blank).rankings[0]
    _require(
        reading.avg_rs_30d is None and reading.coin_count_30d == 0,
        f"an all-NaN 30d column must give None/0, got {reading.avg_rs_30d}/"
        f"{reading.coin_count_30d}",
    )
    _require(
        math.isclose(reading.avg_rs_7d, 2.0, abs_tol=1e-9),
        "the 7-day average must survive a fully missing 30-day column",
    )
    logger.info("Gate 3 OK: an entirely missing rs_30d yields None, not 0.0")


def gate_status(tracker: SectorTracker) -> None:
    """The 2x2 quadrants must each fire on exactly their own corner.

    Directions are named explicitly so the semantics cannot drift silently:
    ACCUMULATING is the BOTTOMING signal (weak month, bidding this week),
    PULLBACK is its mirror (strong month, correcting this week).
    """
    cases = (
        (5.0, 10.0, SectorFlowStatus.IN_FLOW, "strong week, strong month"),
        (5.0, -10.0, SectorFlowStatus.ACCUMULATING, "weak month, turning up"),
        (-5.0, 10.0, SectorFlowStatus.PULLBACK, "strong month, correcting"),
        (-5.0, -10.0, SectorFlowStatus.OUT_OF_FLOW, "weak week, weak month"),
        (0.0, 5.0, SectorFlowStatus.NEUTRAL, "7d exactly flat"),
        (5.0, 0.0, SectorFlowStatus.NEUTRAL, "30d exactly flat"),
        (None, 5.0, SectorFlowStatus.NEUTRAL, "no 7d data"),
        (-5.0, None, SectorFlowStatus.NEUTRAL, "no 30d data"),
        (None, None, SectorFlowStatus.NEUTRAL, "no data at all"),
    )
    for avg_7d, avg_30d, expected, label in cases:
        actual = tracker.classify(avg_7d, avg_30d)
        _require(
            actual is expected,
            f"classify(7d={avg_7d}, 30d={avg_30d}) [{label}] gave {actual}, "
            f"expected {expected}",
        )
    logger.info("Gate 4 OK: all %d quadrant cases classify correctly", len(cases))

    # Direction guard: the two diagonal states must NOT be swapped. This is
    # the exact error the first Part 3 implementation shipped with.
    _require(
        tracker.classify(2.0, -8.0) is SectorFlowStatus.ACCUMULATING,
        "weak month + bidding week must be ACCUMULATING (bottoming), not PULLBACK",
    )
    _require(
        tracker.classify(-2.0, 8.0) is SectorFlowStatus.PULLBACK,
        "strong month + correcting week must be PULLBACK, not ACCUMULATING",
    )
    logger.info(
        "Gate 4 OK: ACCUMULATING/PULLBACK are on the correct diagonals "
        "(not swapped)"
    )

    # Every quadrant must be reachable — no state can be shadowed by another.
    reached = {
        tracker.classify(a, b)
        for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0))
    }
    _require(
        reached == set(SectorFlowStatus),
        f"unreachable status(es): {set(SectorFlowStatus) - reached}",
    )
    logger.info("Gate 4 OK: all %d statuses reachable", len(SectorFlowStatus))


def gate_ranking(tracker: SectorTracker) -> None:
    """Ranking must be a strict 1..N ordering by avg_rs_7d."""
    frame = pd.DataFrame(
        {
            "sector": ["A", "B", "C", "D"],
            "symbol": ["a/USDT", "b/USDT", "c/USDT", "d/USDT"],
            "rs_7d": [-3.0, 9.0, 1.0, float("nan")],
            "rs_30d": [1.0, 1.0, 1.0, 1.0],
        }
    )
    frame.attrs["as_of"] = datetime(2026, 1, 1, tzinfo=UTC)
    rankings = tracker.compute(frame).rankings

    # D has no rs_7d at all, so it drops out entirely (coin_count == 0).
    order = [r.sector for r in rankings]
    _require(order == ["B", "C", "A"], f"expected B, C, A by strength; got {order}")
    _require(
        [r.ranking for r in rankings] == [1, 2, 3],
        f"rankings must be 1..N with no gaps, got {[r.ranking for r in rankings]}",
    )
    logger.info("Gate 5 OK: ranked %s, sectors with no rs_7d dropped", order)

    snapshot = tracker.compute(frame)
    _require(
        [r.sector for r in snapshot.top_sectors][0] == "B",
        "top_sectors must start with the strongest sector",
    )
    _require(
        [r.sector for r in snapshot.bottom_sectors][0] == "A",
        "bottom_sectors must start with the weakest sector",
    )
    logger.info(
        "Gate 5 OK: top=%s bottom=%s",
        [r.sector for r in snapshot.top_sectors],
        [r.sector for r in snapshot.bottom_sectors],
    )


async def gate_persistence(tracker: SectorTracker, as_of: datetime) -> None:
    """sector_performance must round-trip and stay idempotent."""
    snapshot = await tracker.run()

    async with session_scope() as session:
        rows = (
            await session.execute(
                select(SectorPerformance)
                .where(SectorPerformance.timestamp == snapshot.as_of)
                .order_by(SectorPerformance.ranking)
            )
        ).scalars().all()
    _require(
        len(rows) == len(snapshot.rankings),
        f"persisted {len(rows)} rows for {len(snapshot.rankings)} sectors",
    )
    stored = {row.sector.value: row for row in rows}
    for reading in snapshot.rankings:
        row = stored[reading.sector]
        _require(
            row.ranking == reading.ranking,
            f"{reading.sector}: stored ranking {row.ranking} != {reading.ranking}",
        )
        _require(
            SectorFlowStatus(row.status) is reading.status,
            f"{reading.sector}: stored status {row.status} != {reading.status}",
        )
        _require(
            row.coin_count == reading.coin_count
            and row.coin_count_30d == reading.coin_count_30d,
            f"{reading.sector}: stored counts differ from computed",
        )
        if reading.avg_rs_7d is not None:
            _require(
                math.isclose(
                    float(row.avg_rs_7d), reading.avg_rs_7d, abs_tol=TOLERANCE
                ),
                f"{reading.sector}: stored avg_rs_7d differs",
            )
    logger.info("Gate 6 OK: %d sector rows round-tripped intact", len(rows))

    await tracker.run()
    async with session_scope() as session:
        count = (
            await session.execute(
                select(func.count())
                .select_from(SectorPerformance)
                .where(SectorPerformance.timestamp == snapshot.as_of)
            )
        ).scalar_one()
    _require(
        count == len(snapshot.rankings),
        f"re-running duplicated rows: {count} for {len(snapshot.rankings)} sectors",
    )
    logger.info("Gate 6 OK: re-running the same day updates in place (%d rows)", count)


async def gate_master_agent(tracker: SectorTracker) -> None:
    """The Master Agent must accept sector_context and put it in the prompt."""
    snapshot = await tracker.run()
    context = snapshot.to_prompt_context()
    _require(len(context) > 40, f"prompt context suspiciously short: {context!r}")
    _require(
        "Sector Rotation" in context and "vs BTC" in context,
        f"prompt context missing its headline structure: {context!r}",
    )
    logger.info("Gate 7 OK: prompt context = %s", context[:120] + "...")

    panel = [
        SubAgentResponse(
            agent_name=name,
            signal=Signal.NEUTRAL,
            score=0,
            reasoning="Synthetic pillar for the sector-context wiring test.",
        )
        for name in ("technical", "macro", "onchain", "sentiment", "liquidity_risk")
    ]

    captured: dict = {}

    async def fake_ask(self, prompt, response_model):
        captured["prompt"] = prompt
        return MasterDecision(
            action="HOLD",
            confidence_score=50,
            executive_summary=(
                "Ringkasan stub untuk uji wiring konteks sektor; tidak "
                "memanggil model bahasa apa pun."
            ),
            risk_warning="Peringatan risiko stub untuk uji wiring.",
        )

    original_ask = MasterAgent.ask
    MasterAgent.ask = fake_ask
    try:
        agent = MasterAgent()
        await agent.decide("2026-01-01", panel, sector_context=context)
        _require(
            context in captured["prompt"],
            "sector_context did not reach the Master Agent's prompt",
        )
        _require(
            "SECTOR ROTATION" in captured["prompt"],
            "the sector block is missing its prompt heading",
        )
        logger.info("Gate 7 OK: sector context reaches the Master Agent prompt")

        # Backward compatibility: no sector_context -> no sector block.
        captured.clear()
        await agent.decide("2026-01-01", panel)
        _require(
            "SECTOR ROTATION" not in captured["prompt"],
            "omitting sector_context must leave the prompt unchanged",
        )
        logger.info(
            "Gate 7 OK: omitting sector_context leaves the prompt free of "
            "any sector block (backward compatible)"
        )
    finally:
        MasterAgent.ask = original_ask


# ---------------------------------------------------------------------------
async def main() -> int:
    started = datetime.now(tz=UTC)
    tracker = SectorTracker()
    try:
        logger.info("--- Registry composition ---")
        await gate_registry()
        logger.info("--- pandas vs SQL parity ---")
        as_of = await gate_sql_parity(tracker)
        logger.info("--- NaN safety ---")
        gate_nan_safety(tracker)
        logger.info("--- Status classification ---")
        gate_status(tracker)
        logger.info("--- Ranking ---")
        gate_ranking(tracker)
        logger.info("--- Persistence ---")
        await gate_persistence(tracker, as_of)
        logger.info("--- Master Agent integration ---")
        await gate_master_agent(tracker)
    except AssertionError as exc:
        logger.error("PHASE 8 PART 3 TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("PHASE 8 PART 3 TEST FAILED with an unexpected error")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "PHASE 8 PART 3 TEST PASSED (7 gates, %.1fs) — sector rotation tracker "
        "ready", elapsed,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main()))

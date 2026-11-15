"""Test: derivatives freshness — stale-change guard + the live path.

No exchange calls except gate 4, which is opt-in via ``--live``.

Gates
-----
1. **No fabricated change** — ``funding_rate_change_1d`` is NULL on every day
   whose ``funding_rate`` was forward-filled rather than observed, and present
   on every day where both that day and the previous one were observed. This
   is the defect that mattered most: with a 35-day fill cap, ``diff()`` over
   the filled series produced ``0.0`` for eight consecutive days while the
   newest real reading was 2026-07-31 — "funding is flat" asserted about days
   nobody looked at.
2. **Levels still fill** — the guard must NOT have nulled ``funding_rate``
   itself. Carrying the last known LEVEL forward is correct and deliberate;
   only the derived CHANGE is a fabrication.
3. **Separation** — the live table and the archive stay distinct: the live
   rows carry a non-Binance ``source``, and nothing from the live path has
   leaked into ``derivatives_metrics``.
4. **Live read** (``--live``, one network call) — a reading is stored, its
   ``observed_at`` is our clock and recent, and ``settles_at`` is a different
   value (proving the two clocks are not being confused).

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_live_funding.py
    python scripts/test_live_funding.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, select, text  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import LiveDerivative  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_live_funding")

PROBE_SOURCE = "__probe__"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
async def gate_no_fabricated_change() -> None:
    """A 1-day change may exist only where both days were observed."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT f.timestamp::date AS d,
                           f.funding_rate_change_1d AS chg,
                           EXISTS (SELECT 1 FROM derivatives_metrics m
                                   WHERE m.metric = 'funding_rate'
                                     AND m.timestamp::date = f.timestamp::date
                           ) AS obs_today,
                           EXISTS (SELECT 1 FROM derivatives_metrics m
                                   WHERE m.metric = 'funding_rate'
                                     AND m.timestamp::date
                                         = f.timestamp::date - 1
                           ) AS obs_prev
                    FROM daily_features f
                    WHERE f.symbol = 'BTC/USDT'
                      AND f.funding_rate IS NOT NULL
                    """
                )
            )
        ).all()
    _require(bool(rows), "no BTC feature rows with a funding rate to check")

    fabricated = [
        r for r in rows if r.chg is not None and not (r.obs_today and r.obs_prev)
    ]
    # The message is built eagerly, before _require sees the condition, so it
    # must be safe on the passing path too.
    _require(
        not fabricated,
        f"{len(fabricated)} day(s) report a funding change without having "
        f"observed funding on both days — e.g. "
        f"{fabricated[0].d if fabricated else '-'}. A change between two "
        f"forward-filled copies of one reading is not a measurement; it must "
        f"be NULL.",
    )

    # And the converse: the guard must not have thrown away real changes.
    # The FIRST observed day of all has no predecessor, so allow one.
    missing = [r for r in rows if r.chg is None and r.obs_today and r.obs_prev]
    _require(
        len(missing) <= 1,
        f"{len(missing)} day(s) had funding observed on BOTH days but report "
        f"no change — the guard is over-nulling real measurements "
        f"(e.g. {missing[0].d if missing else '-'})",
    )
    logger.info(
        "Gate 1 OK: %d row(s) checked, 0 fabricated changes, %d legitimately "
        "absent", len(rows), sum(1 for r in rows if r.chg is None),
    )


async def gate_levels_still_fill() -> None:
    """Nulling the CHANGE must not have nulled the LEVEL."""
    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT count(*) AS n,
                           count(funding_rate) AS have_level,
                           count(funding_rate_change_1d) AS have_change
                    FROM daily_features
                    WHERE symbol = 'BTC/USDT'
                      AND timestamp >= now() - INTERVAL '8 days'
                    """
                )
            )
        ).one()
    _require(
        row.have_level == row.n,
        f"only {row.have_level}/{row.n} recent days carry a funding LEVEL — "
        f"forward-filling the last known level is correct and must survive",
    )
    logger.info(
        "Gate 2 OK: last %d day(s) all carry a funding level; %d carry a "
        "change (the rest correctly NULL)", row.n, row.have_change,
    )


async def gate_paths_stay_separate() -> None:
    """The live source must never appear in the archive table."""
    async with session_scope() as session:
        live_sources = set(
            (
                await session.execute(select(LiveDerivative.source).distinct())
            ).scalars().all()
        )
        archive_exchanges = set(
            (
                await session.execute(
                    text(
                        "SELECT DISTINCT exchange FROM derivatives_metrics"
                    )
                )
            ).scalars().all()
        )
    overlap = live_sources & archive_exchanges
    _require(
        not overlap,
        f"source(s) {overlap} appear in BOTH live_derivatives and "
        f"derivatives_metrics. The whole point of two tables is that a "
        f"live cross-exchange reading can never be mistaken for the "
        f"archive's settled Binance series.",
    )
    logger.info(
        "Gate 3 OK: live source(s) %s disjoint from archive exchange(s) %s",
        sorted(live_sources) or ["(none yet)"], sorted(archive_exchanges),
    )


async def gate_live_read() -> None:
    """One real call: the reading is stored and stamped with OUR clock."""
    from src.collectors import LiveFundingFetcher

    before = datetime.now(tz=UTC)
    fetcher = LiveFundingFetcher()
    try:
        rows = await fetcher.fetch_data()
    finally:
        await fetcher.close()

    _require(len(rows) == 1, f"expected one reading, got {len(rows)}")
    row = rows[0]
    after = datetime.now(tz=UTC)

    _require(
        before <= row.observed_at <= after,
        f"observed_at {row.observed_at} must be OUR read time, between "
        f"{before} and {after} — using the exchange's clock here would make "
        f"the age meaningless",
    )
    _require(row.value is not None, "no funding value")
    _require(row.source != "binance", f"source must not claim binance: {row.source}")
    if row.settles_at is not None:
        _require(
            row.settles_at != row.observed_at,
            "settles_at and observed_at are different clocks and must not be "
            "the same value — settlement time does not answer 'how stale'",
        )
    logger.info(
        "Gate 4 OK: %s funding=%s observed_at=%s settles_at=%s",
        row.source, row.value,
        row.observed_at.isoformat(timespec="seconds"),
        row.settles_at.isoformat() if row.settles_at else None,
    )


# ---------------------------------------------------------------------------
async def _cleanup() -> None:
    async with session_scope() as session:
        await session.execute(
            delete(LiveDerivative).where(LiveDerivative.source == PROBE_SOURCE)
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="Also run gate 4, which makes one real exchange call.",
    )
    args = parser.parse_args()

    started = datetime.now(tz=UTC)
    try:
        logger.info("--- No fabricated funding change ---")
        await gate_no_fabricated_change()
        logger.info("--- Levels still forward-fill ---")
        await gate_levels_still_fill()
        logger.info("--- Live and archive stay separate ---")
        await gate_paths_stay_separate()
        if args.live:
            logger.info("--- Live read (1 network call) ---")
            await gate_live_read()
        else:
            logger.info("--- Live read SKIPPED (pass --live to include it) ---")
    except AssertionError as exc:
        logger.error("LIVE FUNDING TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("LIVE FUNDING TEST FAILED unexpectedly")
        return 1
    finally:
        try:
            await _cleanup()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cleanup issue (ignored): %s", exc)
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "LIVE FUNDING TEST PASSED (%d gates, %.1fs) — no fabricated changes, "
        "paths separate", 4 if args.live else 3, elapsed,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main()))

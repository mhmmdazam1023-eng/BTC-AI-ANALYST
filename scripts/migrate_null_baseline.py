"""Migration: skill-free baseline columns on decision_accuracy_metrics.

Why
---
``band_pct`` is exactly one sigma, so a HOLD that knows nothing is CORRECT
about 70% of the time and a BUY/SELL is not even testable about 70% of the
time. The first Phase 9 scoreboard read 76.5% (+1d) and 100% (+7d) and looked
like evidence; the skill-free baselines for those same bands are 73.4% and
70.7%. The number was never wrong — it was unreadable on its own.

Storing the baseline beside ``hit_rate`` is what makes it unreadable-alone
impossible, rather than asking whoever reads the table to remember. Four
columns, all nullable:

* ``null_hit_rate``      — expected hit_rate under zero skill, blended over
                           this row's own BUY/SELL/HOLD mix
* ``edge_vs_null``       — ``hit_rate - null_hit_rate``; the only one that
                           answers "is there anything here"
* ``null_decisive_rate`` — fraction of days the band did not swallow
* ``null_samples``       — asset-days behind the estimate (provenance)

They are NULLABLE on purpose. A missing baseline is an unknown baseline, not
zero edge — the same NULL-vs-0 discipline as the Part 4 feature contract.

No backfill is needed or possible: ``decision_accuracy_metrics`` rows are
snapshots, and recomputing an old row's baseline from today's history would
silently rewrite what the scoreboard said at the time. New rows carry the
columns; old ones keep NULL and stay honest.

Usage::

    python scripts/migrate_null_baseline.py --dry-run
    python scripts/migrate_null_baseline.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.migrate_null_baseline")

TABLE = "decision_accuracy_metrics"
COLUMNS: tuple[tuple[str, str], ...] = (
    ("null_hit_rate", "DOUBLE PRECISION"),
    ("edge_vs_null", "DOUBLE PRECISION"),
    ("null_decisive_rate", "DOUBLE PRECISION"),
    ("null_samples", "INTEGER"),
)


async def migrate(*, dry_run: bool) -> int:
    async with session_scope() as session:
        existing = {
            row[0]
            for row in (
                await session.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = :t"
                    ),
                    {"t": TABLE},
                )
            ).all()
        }
        rows = (
            await session.execute(text(f"SELECT count(*) FROM {TABLE}"))
        ).scalar_one()

        missing = [(name, sql) for name, sql in COLUMNS if name not in existing]
        logger.info(
            "State: %s has %d row(s); %d of %d baseline column(s) missing",
            TABLE, rows, len(missing), len(COLUMNS),
        )

        if not missing:
            logger.info("All baseline columns already present — nothing to do")
            return 0

        if dry_run:
            logger.info("--- DRY RUN — no changes written ---")
            for name, sql in missing:
                logger.info("WOULD add %s.%s %s (nullable)", TABLE, name, sql)
            return 0

        for name, sql in missing:
            # ADD COLUMN IF NOT EXISTS keeps this safe to re-run even if the
            # information_schema read above raced another migration.
            await session.execute(
                text(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS {name} {sql}")
            )
            logger.info("Added %s.%s (%s, nullable)", TABLE, name, sql)

    logger.info(
        "Migration complete. New metric rows will carry the baseline; existing "
        "rows keep NULL rather than being back-dated with today's history."
    )
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="report the plan, write nothing"
    )
    args = parser.parse_args()
    try:
        return await migrate(dry_run=args.dry_run)
    except Exception:
        logger.exception("Migration FAILED — the transaction was rolled back")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)


if __name__ == "__main__":
    setup_logging("INFO")
    raise SystemExit(asyncio.run(main()))

"""Migration: effective_sample_size on decision_accuracy_metrics.

Why
---
Every ``n >= 30`` gate in this system counted ROWS. Rows are not
observations: a decision evaluated over ``[T+1, T+1+30]`` shares 29/30 of its
price path with the one made a day later, and 21 assets on the same day carry
``btc_corr_30d`` of 0.5-0.9. ``MIN_ACCURACY_SAMPLES = 30`` therefore read as
"enough history to publish an accuracy" while admitting as little as one
independent observation.

The gate was never wrong in intent — it was measuring the wrong quantity. It
now gates on ``effective_sample_size``, and this column stores that number
beside the raw one so a published row shows both.

Nullable, and no backfill: existing rows were gated on the raw count, and
recomputing their effective size today would state a number that row was
never actually held to. There are no rows yet in practice, but the rule
matters more than the current count.

Usage::

    python scripts/migrate_effective_samples.py --dry-run
    python scripts/migrate_effective_samples.py
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

logger = logging.getLogger("scripts.migrate_effective_samples")

TABLE = "decision_accuracy_metrics"
COLUMN = "effective_sample_size"
COLUMN_TYPE = "DOUBLE PRECISION"


async def migrate(*, dry_run: bool) -> int:
    async with session_scope() as session:
        present = (
            await session.execute(
                text(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = :c"
                ),
                {"t": TABLE, "c": COLUMN},
            )
        ).scalar_one()
        rows = (
            await session.execute(text(f"SELECT count(*) FROM {TABLE}"))
        ).scalar_one()

        logger.info(
            "State: %s has %d row(s); column %s %s",
            TABLE, rows, COLUMN, "present" if present else "MISSING",
        )
        if present:
            logger.info("Nothing to do")
            return 0
        if dry_run:
            logger.info("--- DRY RUN — no changes written ---")
            logger.info(
                "WOULD add %s.%s %s (nullable, no backfill)",
                TABLE, COLUMN, COLUMN_TYPE,
            )
            return 0

        await session.execute(
            text(
                f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS "
                f"{COLUMN} {COLUMN_TYPE}"
            )
        )
        logger.info("Added %s.%s (%s, nullable)", TABLE, COLUMN, COLUMN_TYPE)

    logger.info(
        "Migration complete. The publication gate now tests independent "
        "observations; new rows carry both counts."
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

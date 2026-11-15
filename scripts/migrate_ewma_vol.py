"""Migration: ewma_vol_30d on daily_features (2026-08-02).

Why
---
``realized_vol_30d`` weights a return from 30 days ago exactly like
yesterday's, so it registers a volatility regime change only once the change
has half-filled its window. Measured over 34.8k asset-days across all 21
assets, at a nominal 90% band:

    horizon        +1d    +7d   +30d
    realized_30  89.2%  87.5%  84.3%
    EWMA 0.97    91.0%  89.8%  87.0%

Better at every horizon, and essentially calibrated out to a week.

``realized_vol_30d`` is deliberately KEPT and unchanged. It defines the Phase
9 verdict band, and redefining that would invalidate all stored
``decision_evaluations``, the null model computed from them, and the
continuity of ``measurement_snapshots``. The new column is consumed by the
scenario engine and Kelly's variance only.

The column is nullable and NOT backfilled here — ``FeatureEngine`` computes
it on the next run over whatever history it loads. Run::

    python scripts/migrate_ewma_vol.py
    python scripts/run_collectors.py --mode full   # recomputes features

Usage::

    python scripts/migrate_ewma_vol.py --dry-run
    python scripts/migrate_ewma_vol.py
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
from src.features.engine import EWMA_LAMBDA  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.migrate_ewma_vol")

TABLE = "daily_features"
COLUMN = "ewma_vol_30d"


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
                "WOULD add %s.%s NUMERIC(38,12) nullable (lambda=%.2f), "
                "no backfill", TABLE, COLUMN, EWMA_LAMBDA,
            )
            return 0

        await session.execute(
            text(
                f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS "
                f"{COLUMN} NUMERIC(38,12)"
            )
        )
        logger.info("Added %s.%s (lambda=%.2f)", TABLE, COLUMN, EWMA_LAMBDA)

    logger.info(
        "Migration complete. Recompute features to populate it: "
        "python scripts/run_collectors.py --mode full"
    )
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
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

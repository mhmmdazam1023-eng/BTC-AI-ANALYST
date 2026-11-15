"""Migration: entry lag for decision evaluations (2026-07-31, user "Opsi A").

Why this migration exists
-------------------------
Until now ``DecisionEvaluator`` measured the return from ``close(T)``, where
T is the decision day. Measured against the live table, decisions were being
written 5 to 20 hours BEFORE that close settled — the 2026-07-31 run wrote at
13:07-14:14 UTC for a candle that closed 2026-08-01 00:00 UTC. Worse, day T's
candle is the same (still-forming) bar ``daily_features[T]`` is built from, so
the entry price was contaminated by the very bar the decision read.

The evaluator now enters at ``close(T + ENTRY_LAG_DAYS)``. Every existing
``decision_evaluations`` row was computed under the old rule and is therefore
not comparable with anything written from here on, so this script retires
them rather than leaving two incompatible definitions in one table.

What it does (all idempotent, safe to re-run)
---------------------------------------------
1. ``ALTER TABLE decision_evaluations ADD COLUMN IF NOT EXISTS entry_date``
   (nullable at first — an existing table cannot gain a NOT NULL column while
   it still holds rows).
2. Snapshots the old rows into ``decision_evaluations_pre_entry_lag``, the
   same rollback-point pattern used by ``daily_features_pre_phase8``. The
   snapshot is created ONCE and never overwritten: a second run must not be
   able to clobber the original evidence with an already-purged table.
3. Purges ``decision_evaluations`` and ``decision_accuracy_metrics`` — the
   metrics are aggregates of the rows being retired, so leaving them would
   publish accuracy computed under a definition that no longer exists.
4. ``SET NOT NULL`` on ``entry_date``, which an empty table satisfies
   trivially. Doing it here rather than "later" means the constraint is in
   force BEFORE the first new row is written, not after.

Recompute afterwards with ``python scripts/run_daily.py`` (Stage A5) or::

    python -c "import asyncio; from src.evaluation.decision_evaluator import \
DecisionEvaluator; asyncio.run(DecisionEvaluator().evaluate_matured())"

Usage::

    python scripts/migrate_entry_lag.py --dry-run
    python scripts/migrate_entry_lag.py
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
from src.evaluation.decision_evaluator import ENTRY_LAG_DAYS  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.migrate_entry_lag")

SNAPSHOT_TABLE = "decision_evaluations_pre_entry_lag"


async def _scalar(session, sql: str) -> int:
    return (await session.execute(text(sql))).scalar_one()


async def migrate(*, dry_run: bool) -> int:
    async with session_scope() as session:
        # ---- inspect current state -------------------------------------
        has_column = await _scalar(
            session,
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name='decision_evaluations' AND column_name='entry_date'",
        )
        snapshot_exists = await _scalar(
            session,
            "SELECT count(*) FROM information_schema.tables "
            f"WHERE table_name='{SNAPSHOT_TABLE}'",
        )
        evaluations = await _scalar(
            session, "SELECT count(*) FROM decision_evaluations"
        )
        metrics = await _scalar(
            session, "SELECT count(*) FROM decision_accuracy_metrics"
        )

        logger.info(
            "State: entry_date column=%s | snapshot table=%s | "
            "%d evaluation row(s) | %d accuracy metric row(s)",
            "present" if has_column else "MISSING",
            "present" if snapshot_exists else "absent",
            evaluations, metrics,
        )

        if dry_run:
            logger.info("--- DRY RUN — no changes written ---")
            if not has_column:
                logger.info("WOULD add column decision_evaluations.entry_date")
            if evaluations and not snapshot_exists:
                logger.info(
                    "WOULD snapshot %d row(s) into %s", evaluations, SNAPSHOT_TABLE
                )
            if evaluations or metrics:
                logger.info(
                    "WOULD purge %d evaluation(s) and %d metric row(s) — all "
                    "computed with entry=close(T), which the evaluator no "
                    "longer produces",
                    evaluations, metrics,
                )
            logger.info("WOULD set entry_date NOT NULL")
            return 0

        # ---- 1) add the column ------------------------------------------
        if not has_column:
            await session.execute(
                text(
                    "ALTER TABLE decision_evaluations "
                    "ADD COLUMN IF NOT EXISTS entry_date TIMESTAMPTZ"
                )
            )
            logger.info("Added decision_evaluations.entry_date (nullable for now)")
        else:
            logger.info("Column entry_date already present — skipping")

        # ---- 2) snapshot, once and only once ----------------------------
        if evaluations and not snapshot_exists:
            await session.execute(
                text(
                    f"CREATE TABLE {SNAPSHOT_TABLE} AS "
                    "SELECT * FROM decision_evaluations"
                )
            )
            kept = await _scalar(session, f"SELECT count(*) FROM {SNAPSHOT_TABLE}")
            logger.info(
                "Snapshotted %d row(s) to %s (rollback point — drop it once you "
                "are satisfied with the recomputed verdicts)", kept, SNAPSHOT_TABLE,
            )
        elif snapshot_exists:
            logger.info(
                "%s already exists — NOT overwriting it; the original "
                "pre-migration evidence stays put", SNAPSHOT_TABLE,
            )
        else:
            logger.info("No evaluation rows to snapshot")

        # ---- 3) purge the incomparable rows -----------------------------
        if evaluations or metrics:
            await session.execute(text("DELETE FROM decision_accuracy_metrics"))
            await session.execute(text("DELETE FROM decision_evaluations"))
            logger.info(
                "Purged %d evaluation(s) + %d accuracy metric row(s). They were "
                "computed with entry=close(T); mixing them with entry=close(T+%d) "
                "rows would average two different measurements together",
                evaluations, metrics, ENTRY_LAG_DAYS,
            )
        else:
            logger.info("Nothing to purge")

        # ---- 4) tighten the constraint ----------------------------------
        remaining = await _scalar(
            session, "SELECT count(*) FROM decision_evaluations WHERE entry_date IS NULL"
        )
        if remaining:
            logger.error(
                "%d row(s) still have a NULL entry_date — refusing to SET NOT "
                "NULL. This should be unreachable after the purge above",
                remaining,
            )
            return 1
        await session.execute(
            text(
                "ALTER TABLE decision_evaluations "
                "ALTER COLUMN entry_date SET NOT NULL"
            )
        )
        logger.info("entry_date is now NOT NULL")

    logger.info(
        "Migration complete. Recompute with run_daily.py (Stage A5); every new "
        "row will enter at close(T+%d).", ENTRY_LAG_DAYS,
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

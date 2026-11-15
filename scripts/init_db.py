"""Initialise the TimescaleDB data warehouse.

Creates every ORM table defined in ``src.db.models`` and converts each one
into a TimescaleDB hypertable. The script is **idempotent** — safe to re-run.

Usage (from the project root, with the TimescaleDB container running)::

    python scripts/init_db.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Allow `python scripts/init_db.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, get_engine  # noqa: E402
from src.db.models import HYPERTABLES, Base  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.init_db")


async def init_database() -> None:
    """Create the extension, all tables, and the hypertables."""
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            logger.info("Ensuring TimescaleDB extension is installed...")
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))

            logger.info("Creating ORM tables (if they do not exist)...")
            await conn.run_sync(Base.metadata.create_all)

            for table_name, time_column in HYPERTABLES:
                logger.info(
                    "Converting '%s' into a hypertable (time column: '%s')...",
                    table_name,
                    time_column,
                )
                await conn.execute(
                    text(
                        "SELECT create_hypertable("
                        ":table_name, :time_column, "
                        "if_not_exists => TRUE, migrate_data => TRUE);"
                    ),
                    {"table_name": table_name, "time_column": time_column},
                )

        logger.info(
            "Database initialised successfully: %d tables / hypertables ready.",
            len(HYPERTABLES),
        )
    finally:
        await dispose_engine()


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "Initialising warehouse at %s:%s/%s ...",
        settings.postgres_host,
        settings.postgres_port,
        settings.postgres_db,
    )
    try:
        asyncio.run(init_database())
    except ConnectionRefusedError:
        logger.error(
            "Could not connect to PostgreSQL at %s:%s. "
            "Is the TimescaleDB container running? Try: docker compose up -d",
            settings.postgres_host,
            settings.postgres_port,
        )
        raise SystemExit(1)
    except Exception:
        logger.exception("Database initialisation failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

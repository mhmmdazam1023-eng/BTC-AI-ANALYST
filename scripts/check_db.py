"""Database health check.

Verifies, using the application credentials from ``.env``:
1. Connectivity + TimescaleDB extension version.
2. All expected hypertables exist.
3. A full write -> read -> delete round-trip through the async ORM.

Usage (from the project root)::

    python scripts/check_db.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, text  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.health import assert_tables_accessible  # noqa: E402
from src.db.models import HYPERTABLES, CryptoPrice  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.check_db")

_SENTINEL_EXCHANGE = "_healthcheck"


async def check() -> None:
    try:
        await _run_checks()
    finally:
        await dispose_engine()


async def _run_checks() -> None:
    expected = {name for name, _ in HYPERTABLES}

    async with session_scope() as session:
        # 1) Extension version
        version = (
            await session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
            )
        ).scalar_one_or_none()
        if version is None:
            raise RuntimeError("TimescaleDB extension is NOT installed in this database")
        logger.info("TimescaleDB extension version: %s", version)

        # 2) Hypertables present
        rows = (
            await session.execute(
                text("SELECT hypertable_name FROM timescaledb_information.hypertables")
            )
        ).scalars().all()
        found = set(rows)
        missing = expected - found
        if missing:
            raise RuntimeError(f"Missing hypertables: {sorted(missing)}")
        logger.info("Hypertables OK (%d): %s", len(found & expected), sorted(found & expected))

        # 2b) Table privileges — the failure mode that surfaces as a wall of
        # `permission denied for table ...` tracebacks halfway through a
        # daily run. Checked up front so the diagnosis is one actionable line.
        await assert_tables_accessible(get_settings().postgres_db)

    # 3) ORM round-trip on crypto_prices
    now = datetime.now(tz=UTC).replace(microsecond=0)
    probe = CryptoPrice(
        timestamp=now,
        exchange=_SENTINEL_EXCHANGE,
        symbol="BTC/USDT",
        timeframe="1d",
        open=Decimal("1"),
        high=Decimal("2"),
        low=Decimal("0.5"),
        close=Decimal("1.5"),
        volume=Decimal("42"),
    )
    async with session_scope() as session:
        session.add(probe)
    async with session_scope() as session:
        fetched = (
            await session.execute(
                select(CryptoPrice).where(CryptoPrice.exchange == _SENTINEL_EXCHANGE)
            )
        ).scalar_one()
        assert fetched.close == Decimal("1.5"), "round-trip value mismatch"
        await session.delete(fetched)
    logger.info("ORM write -> read -> delete round-trip OK (crypto_prices)")

    logger.info("DATABASE HEALTH CHECK PASSED")


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    try:
        asyncio.run(check())
    except Exception:
        logger.exception("Database health check FAILED")
        raise SystemExit(1)
    finally:
        # engine already disposed inside check() sessions' lifecycle at exit
        pass


if __name__ == "__main__":
    main()

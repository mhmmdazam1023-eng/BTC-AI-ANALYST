"""Phase 8 - Part 1 migration: single-asset warehouse -> multi-asset.

Fully **idempotent** — safe to run repeatedly, and safe to run on a fresh
database (where ``init_db.py`` already created everything in the new shape).

What it does
------------
1. Creates the Phase 8 dimension tables ``coins_registry`` and
   ``tokenomics_events`` (plus their native enum types).
2. Adds any enum label that exists in Python but not yet in PostgreSQL —
   so extending :class:`~src.db.models.CoinSector` only requires re-running
   this script.
3. Migrates ``daily_features`` from PK ``(timestamp)`` to PK
   ``(timestamp, symbol)``:
   * adds ``symbol`` (backfilled to ``BTC/USDT`` — every pre-Phase-8 row IS
     Bitcoin, so no history is reinterpreted),
   * adds ``asset_close`` / ``asset_volume`` (backfilled from
     ``btc_close`` / ``btc_volume``, which are identical on a BTC row),
   * adds ``price_btc_ratio`` / ``rs_7d`` / ``rs_30d`` (left NULL; the
     FeatureEngine fills them on its next run),
   * swaps the primary key and adds ``ix_daily_features_symbol_ts``.
4. Ensures the multi-symbol index on ``crypto_prices`` exists.
5. Seeds ``coins_registry`` and the (verifiable) Bitcoin halving events with
   ``ON CONFLICT DO NOTHING`` — re-running NEVER clobbers manual curation
   such as flipping ``is_active`` off.

BITCOIN SAFETY: this migration only ADDS columns and re-keys the table. No
existing value is overwritten, no row is deleted. Run
``scripts/test_features_phase3.py`` afterwards to confirm the BTC pipeline
still passes its original quality gates.

Usage (from the project root)::

    python scripts/migrate_phase8.py
    python scripts/migrate_phase8.py --dry-run   # report only, change nothing
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncConnection  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, get_engine, session_scope  # noqa: E402
from src.db.models import (  # noqa: E402
    BASE_SYMBOL,
    Base,
    CoinRegistry,
    CoinSector,
    MarketCapTier,
    MarketRegime,
    SectorFlowStatus,
    TokenomicsEvent,
    TokenomicsEventType,
    VolRegime,
)
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.migrate_phase8")


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------
#: Initial Coin DNA. ``base_asset`` is the benchmark for relative strength;
#: NULL for Bitcoin because it IS the numéraire of this system.
COIN_SEED: tuple[dict, ...] = (
    {
        "symbol": "BTC/USDT",
        "name": "Bitcoin",
        "sector": CoinSector.L1,
        "chain": "Bitcoin",
        "base_asset": None,
        "is_active": True,
    },
    {
        "symbol": "ETH/USDT",
        "name": "Ethereum",
        "sector": CoinSector.L1,
        "chain": "Ethereum",
        "base_asset": BASE_SYMBOL,
        "is_active": True,
    },
    {
        "symbol": "SOL/USDT",
        "name": "Solana",
        "sector": CoinSector.L1,
        "chain": "Solana",
        "base_asset": BASE_SYMBOL,
        "is_active": True,
    },
    {
        "symbol": "BNB/USDT",
        "name": "BNB",
        "sector": CoinSector.L1,
        "chain": "BNB Chain",
        "base_asset": BASE_SYMBOL,
        "is_active": True,
    },
    {
        "symbol": "TAO/USDT",
        "name": "Bittensor",
        "sector": CoinSector.AI,
        "chain": "Bittensor",
        "base_asset": BASE_SYMBOL,
        "is_active": True,
    },
)

#: Only *verifiable protocol facts* are seeded. Token unlock schedules for
#: SOL / TAO / BNB are deliberately NOT seeded: inventing dates and
#: percentages would poison the very supply-shock signal this table exists
#: to carry. Populate them from each project's official vesting docs.
#:
#: ``supply_impact_pct`` for a halving = change to the EMISSION RATE
#: (-50%), not to the circulating stock — see TokenomicsEvent's docstring.
TOKENOMICS_SEED: tuple[dict, ...] = (
    {
        "symbol": "BTC/USDT",
        "event_date": datetime(2020, 5, 11, tzinfo=UTC),
        "event_type": TokenomicsEventType.HALVING,
        "supply_impact_pct": Decimal("-50"),
        "description": "Third Bitcoin halving at block 630,000: block subsidy 12.5 -> 6.25 BTC.",
    },
    {
        "symbol": "BTC/USDT",
        "event_date": datetime(2024, 4, 20, tzinfo=UTC),
        "event_type": TokenomicsEventType.HALVING,
        "supply_impact_pct": Decimal("-50"),
        "description": "Fourth Bitcoin halving at block 840,000: block subsidy 6.25 -> 3.125 BTC.",
    },
)

#: Columns added to daily_features by Phase 8: (name, DDL type).
#: ``price_btc_ratio`` is scale 30 on purpose — see the model docstring: a
#: ratio needs far more decimals than a price for low-priced assets
#: (PEPE/BTC is 4.4e-11).
_NEW_FEATURE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("asset_close", "NUMERIC(38, 12)"),
    ("asset_volume", "NUMERIC(38, 12)"),
    ("price_btc_ratio", "NUMERIC(40, 30)"),
    ("rs_7d", "NUMERIC(38, 12)"),
    ("rs_30d", "NUMERIC(38, 12)"),
    # --- Part 4: Coin DNA. All nullable; see the model docstrings. ---
    ("days_to_next_unlock", "INTEGER"),
    ("supply_shock_score_30d", "NUMERIC(18, 6)"),
    ("inflation_rate_7d", "NUMERIC(18, 6)"),
    ("supply_utilization", "NUMERIC(18, 8)"),
    ("vol_percentile_30d", "NUMERIC(9, 4)"),
    ("vol_regime", "vol_regime"),
    ("liquidity_score", "NUMERIC(24, 10)"),
    ("market_cap_tier", "market_cap_tier"),
    ("btc_beta_30d", "NUMERIC(18, 8)"),
    ("btc_corr_30d", "NUMERIC(18, 8)"),
    ("momentum_score", "NUMERIC(18, 6)"),
    ("trend_strength", "NUMERIC(18, 6)"),
)

#: Columns added to OTHER tables by Phase 8: (table, column, DDL type).
_NEW_TABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("coins_registry", "coingecko_id", "VARCHAR(64)"),
)

#: Columns whose type must be widened on databases migrated before a
#: precision fix: (column, target type, target scale). Idempotent — only
#: altered when the live scale differs. Raised 12 -> 18 in Part 1, then
#: 18 -> 30 in Part 3 once real meme-coin ratios (1e-11) arrived.
_RETYPE_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("price_btc_ratio", "NUMERIC(40, 30)", 30),
)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------
async def _column_exists(conn: AsyncConnection, table: str, column: str) -> bool:
    return bool(
        (
            await conn.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = :c"
                ),
                {"t": table, "c": column},
            )
        ).scalar_one_or_none()
    )


async def _primary_key_def(conn: AsyncConnection, table: str) -> str | None:
    return (
        await conn.execute(
            text(
                "SELECT pg_get_constraintdef(con.oid) FROM pg_constraint con "
                "JOIN pg_class rel ON rel.oid = con.conrelid "
                "WHERE rel.relname = :t AND con.contype = 'p'"
            ),
            {"t": table},
        )
    ).scalar_one_or_none()


async def _enum_labels(conn: AsyncConnection, type_name: str) -> set[str]:
    rows = (
        await conn.execute(
            text(
                "SELECT e.enumlabel FROM pg_enum e "
                "JOIN pg_type t ON t.oid = e.enumtypid WHERE t.typname = :n"
            ),
            {"n": type_name},
        )
    ).scalars().all()
    return set(rows)


# ---------------------------------------------------------------------------
# Migration steps
# ---------------------------------------------------------------------------
async def create_new_tables(dry_run: bool) -> None:
    """Create coins_registry / tokenomics_events (+ enum types) if missing."""
    engine = get_engine()
    async with engine.begin() as conn:
        targets = [CoinRegistry.__table__, TokenomicsEvent.__table__]
        missing = [
            table.name
            for table in targets
            if not await _column_exists(conn, table.name, "symbol")
        ]
        if not missing:
            logger.info("Step 1 SKIP: coins_registry & tokenomics_events already exist")
            return
        logger.info("Step 1: creating tables %s", missing)
        if dry_run:
            return
        # checkfirst=True: never touches the eight existing hypertables.
        await conn.run_sync(Base.metadata.create_all, tables=targets, checkfirst=True)
        logger.info("Step 1 OK: %s created", missing)


#: Every native enum in the schema: (PostgreSQL type name, Python enum).
_ENUM_TYPES: tuple[tuple[str, type], ...] = (
    ("coin_sector", CoinSector),
    ("tokenomics_event_type", TokenomicsEventType),
    ("market_regime", MarketRegime),
    ("sector_flow_status", SectorFlowStatus),
    ("vol_regime", VolRegime),
    ("market_cap_tier", MarketCapTier),
)


async def ensure_enum_types(dry_run: bool) -> None:
    """Create any enum TYPE that does not exist yet.

    ``Base.metadata.create_all`` only creates an enum type as a side effect
    of creating a table that uses it. Phase 8 Part 4 adds enum-typed COLUMNS
    to ``daily_features``, which already exists — so create_all skips the
    table and never creates ``vol_regime``/``market_cap_tier``, and the
    subsequent ``ALTER TABLE ... ADD COLUMN vol_regime`` fails on an unknown
    type. This step closes that gap.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        for type_name, enum_cls in _ENUM_TYPES:
            exists = bool(
                (
                    await conn.execute(
                        text("SELECT 1 FROM pg_type WHERE typname = :n"),
                        {"n": type_name},
                    )
                ).scalar_one_or_none()
            )
            if exists:
                logger.debug("Step 1b SKIP: enum type '%s' exists", type_name)
                continue
            labels = [member.value for member in enum_cls]
            for label in labels:
                if "'" in label:
                    raise ValueError(f"refusing unsafe enum label: {label!r}")
            logger.info("Step 1b: creating enum type '%s' %s", type_name, labels)
            if dry_run:
                continue
            rendered = ", ".join(f"'{label}'" for label in labels)
            # DO/EXCEPTION is the idiomatic idempotent CREATE TYPE: PostgreSQL
            # has no CREATE TYPE IF NOT EXISTS.
            await conn.execute(
                text(
                    "DO $$ BEGIN "
                    f"CREATE TYPE {type_name} AS ENUM ({rendered}); "
                    "EXCEPTION WHEN duplicate_object THEN NULL; END $$;"
                )
            )
            logger.info("Step 1b OK: enum type '%s' created", type_name)


async def add_table_columns(dry_run: bool) -> None:
    """Add Phase 8 columns to tables other than ``daily_features``."""
    engine = get_engine()
    async with engine.begin() as conn:
        for table, column, ddl_type in _NEW_TABLE_COLUMNS:
            if await _column_exists(conn, table, column):
                logger.info("Step 6 SKIP: %s.%s already present", table, column)
                continue
            logger.info("Step 6: adding %s.%s %s", table, column, ddl_type)
            if not dry_run:
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")
                )


async def sync_enum_labels(dry_run: bool) -> None:
    """Add Python enum members that PostgreSQL does not know about yet.

    Covers EVERY native enum in the schema, not just the Part 1 pair: a
    label added to a Python enum without the matching ``ALTER TYPE`` fails
    only at INSERT time, in production, on the first day the new value
    occurs. ``sector_flow_status`` gained ``PULLBACK`` in Part 3 exactly
    this way.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        for type_name, enum_cls in _ENUM_TYPES:
            existing = await _enum_labels(conn, type_name)
            if not existing:
                logger.warning(
                    "Step 2: enum type '%s' not found — expected if step 1 was a "
                    "dry run; otherwise check table creation",
                    type_name,
                )
                continue
            wanted = {member.value for member in enum_cls}
            missing = sorted(wanted - existing)
            if not missing:
                logger.info(
                    "Step 2 SKIP: enum '%s' already has all %d labels",
                    type_name, len(wanted),
                )
                continue
            logger.info("Step 2: adding labels to '%s': %s", type_name, missing)
            if dry_run:
                continue
            for label in missing:
                # Safe to interpolate: `label` comes from the Python enum, and
                # ALTER TYPE ... ADD VALUE cannot take a bind parameter.
                if not label.replace("_", "").replace(" ", "").isalnum():
                    raise ValueError(f"refusing unsafe enum label: {label!r}")
                await conn.execute(
                    text(f"ALTER TYPE {type_name} ADD VALUE IF NOT EXISTS '{label}'")
                )
            logger.info("Step 2 OK: '%s' extended with %s", type_name, missing)


async def migrate_daily_features(dry_run: bool) -> None:
    """Re-key ``daily_features`` to (timestamp, symbol) and add new columns."""
    engine = get_engine()
    async with engine.begin() as conn:
        # --- 3a) new columns -------------------------------------------------
        if not await _column_exists(conn, "daily_features", "symbol"):
            logger.info("Step 3a: adding 'symbol' to daily_features (default %s)", BASE_SYMBOL)
            if not dry_run:
                # DEFAULT + NOT NULL in one statement: PostgreSQL 11+ does this
                # without rewriting the table, and it backfills every existing
                # (Bitcoin-only) row in the same pass.
                await conn.execute(
                    text(
                        "ALTER TABLE daily_features ADD COLUMN symbol VARCHAR(32) "
                        f"NOT NULL DEFAULT '{BASE_SYMBOL}'"
                    )
                )
        else:
            logger.info("Step 3a SKIP: daily_features.symbol already present")

        for column, ddl_type in _NEW_FEATURE_COLUMNS:
            if await _column_exists(conn, "daily_features", column):
                logger.debug("Step 3b SKIP: daily_features.%s already present", column)
                continue
            logger.info("Step 3b: adding daily_features.%s %s", column, ddl_type)
            if not dry_run:
                await conn.execute(
                    text(f"ALTER TABLE daily_features ADD COLUMN {column} {ddl_type}")
                )

        # --- 3b2) widen columns that were created at too small a scale ------
        for column, ddl_type, wanted_scale in _RETYPE_COLUMNS:
            actual_scale = (
                await conn.execute(
                    text(
                        "SELECT numeric_scale FROM information_schema.columns "
                        "WHERE table_name = 'daily_features' AND column_name = :c"
                    ),
                    {"c": column},
                )
            ).scalar_one_or_none()
            if actual_scale is None or actual_scale == wanted_scale:
                logger.debug("Step 3b2 SKIP: %s already scale %s", column, actual_scale)
                continue
            logger.info(
                "Step 3b2: widening daily_features.%s from scale %s to %s",
                column, actual_scale, wanted_scale,
            )
            if not dry_run:
                await conn.execute(
                    text(
                        f"ALTER TABLE daily_features ALTER COLUMN {column} "
                        f"TYPE {ddl_type}"
                    )
                )

        # --- 3c) backfill own-asset price context on legacy BTC rows ---------
        # Pre-Phase-8 rows are all Bitcoin, where asset_* == btc_* by
        # definition. Only NULLs are touched, so this can never overwrite a
        # value the FeatureEngine has already computed.
        if not dry_run:
            result = await conn.execute(
                text(
                    "UPDATE daily_features SET asset_close = btc_close, "
                    "asset_volume = btc_volume "
                    "WHERE symbol = :s AND asset_close IS NULL AND btc_close IS NOT NULL"
                ),
                {"s": BASE_SYMBOL},
            )
            logger.info("Step 3c: backfilled asset_close/volume on %d BTC rows", result.rowcount)

        # --- 3d) primary key swap -------------------------------------------
        pk_def = await _primary_key_def(conn, "daily_features")
        if pk_def and "symbol" in pk_def:
            logger.info("Step 3d SKIP: PK already composite -> %s", pk_def)
        else:
            logger.info("Step 3d: swapping PK %s -> (timestamp, symbol)", pk_def)
            if not dry_run:
                await conn.execute(
                    text("ALTER TABLE daily_features DROP CONSTRAINT daily_features_pkey")
                )
                await conn.execute(
                    text(
                        "ALTER TABLE daily_features ADD CONSTRAINT daily_features_pkey "
                        'PRIMARY KEY ("timestamp", symbol)'
                    )
                )
                logger.info("Step 3d OK: composite PK in place")

        # --- 3e) supporting index -------------------------------------------
        if not dry_run:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_daily_features_symbol_ts "
                    'ON daily_features (symbol, "timestamp")'
                )
            )
            logger.info("Step 3e OK: ix_daily_features_symbol_ts ensured")


async def migrate_decisions(dry_run: bool) -> None:
    """Re-key ``decisions`` to (timestamp, symbol) — Phase 8 Part 6.

    Before this, one row per DAY. A multi-asset run would upsert all 21
    coins onto the same row and keep whichever finished last, silently
    reducing a full portfolio scan to a single decision. Existing rows are
    all Bitcoin, so the backfill is exact.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        if not await _column_exists(conn, "decisions", "symbol"):
            logger.info("Step 7a: adding 'symbol' to decisions (default %s)", BASE_SYMBOL)
            if not dry_run:
                await conn.execute(
                    text(
                        "ALTER TABLE decisions ADD COLUMN symbol VARCHAR(32) "
                        f"NOT NULL DEFAULT '{BASE_SYMBOL}'"
                    )
                )
        else:
            logger.info("Step 7a SKIP: decisions.symbol already present")

        pk_def = await _primary_key_def(conn, "decisions")
        if pk_def and "symbol" in pk_def:
            logger.info("Step 7b SKIP: decisions PK already composite -> %s", pk_def)
        else:
            logger.info("Step 7b: swapping decisions PK %s -> (timestamp, symbol)", pk_def)
            if not dry_run:
                await conn.execute(
                    text("ALTER TABLE decisions DROP CONSTRAINT decisions_pkey")
                )
                await conn.execute(
                    text(
                        "ALTER TABLE decisions ADD CONSTRAINT decisions_pkey "
                        'PRIMARY KEY ("timestamp", symbol)'
                    )
                )
                logger.info("Step 7b OK: composite PK in place")

        if not dry_run:
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_decisions_symbol_ts "
                    'ON decisions (symbol, "timestamp")'
                )
            )
            logger.info("Step 7c OK: ix_decisions_symbol_ts ensured")


async def ensure_price_index(dry_run: bool) -> None:
    """Guarantee the multi-symbol lookup index on ``crypto_prices``."""
    if dry_run:
        logger.info("Step 4: would ensure ix_crypto_prices_symbol_tf_ts")
        return
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_crypto_prices_symbol_tf_ts "
                'ON crypto_prices (symbol, timeframe, "timestamp")'
            )
        )
    logger.info("Step 4 OK: ix_crypto_prices_symbol_tf_ts ensured")


async def seed_registry(dry_run: bool) -> None:
    """Insert the initial Coin DNA + halving events (never overwrites)."""
    if dry_run:
        logger.info(
            "Step 5: would seed %d coins and %d tokenomics events",
            len(COIN_SEED), len(TOKENOMICS_SEED),
        )
        return

    async with session_scope() as session:
        coins = pg_insert(CoinRegistry.__table__).values(list(COIN_SEED))
        # DO NOTHING, not DO UPDATE: a re-run must never resurrect a symbol
        # the operator deliberately deactivated.
        await session.execute(coins.on_conflict_do_nothing(index_elements=["symbol"]))

        events = pg_insert(TokenomicsEvent.__table__).values(list(TOKENOMICS_SEED))
        await session.execute(
            events.on_conflict_do_nothing(constraint="uq_tokenomics_event")
        )

    async with session_scope() as session:
        registry = (
            await session.execute(
                text(
                    "SELECT symbol, name, sector, chain, base_asset, is_active "
                    "FROM coins_registry ORDER BY symbol"
                )
            )
        ).all()
        events = (
            await session.execute(
                text("SELECT count(*) FROM tokenomics_events")
            )
        ).scalar_one()

    logger.info("Step 5 OK: coins_registry now holds %d symbols", len(registry))
    for row in registry:
        logger.info(
            "    %-10s %-10s sector=%-14s chain=%-10s base=%-9s active=%s",
            row[0], row[1], row[2], row[3], row[4] or "-", row[5],
        )
    logger.info("Step 5 OK: tokenomics_events holds %d rows", events)


# ---------------------------------------------------------------------------
async def migrate(dry_run: bool) -> None:
    try:
        await create_new_tables(dry_run)
        # Enum TYPES before any ALTER that references them; LABELS after,
        # so a type created above is immediately complete.
        await ensure_enum_types(dry_run)
        await sync_enum_labels(dry_run)
        await migrate_daily_features(dry_run)
        await migrate_decisions(dry_run)
        await add_table_columns(dry_run)
        await ensure_price_index(dry_run)
        await seed_registry(dry_run)
    finally:
        await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 Part 1 schema migration.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without executing any DDL/DML.",
    )
    arguments = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "PHASE 8 PART 1 MIGRATION %s(target %s:%s/%s)",
        "[DRY RUN] " if arguments.dry_run else "",
        settings.postgres_host, settings.postgres_port, settings.postgres_db,
    )
    try:
        asyncio.run(migrate(arguments.dry_run))
    except Exception:
        logger.exception("Migration FAILED — the database is unchanged past the last OK step")
        raise SystemExit(1)
    logger.info("=" * 68)
    logger.info(
        "MIGRATION %s— daily_features is now keyed by (timestamp, symbol)",
        "DRY RUN COMPLETE " if arguments.dry_run else "COMPLETE ",
    )
    logger.info("=" * 68)


if __name__ == "__main__":
    main()

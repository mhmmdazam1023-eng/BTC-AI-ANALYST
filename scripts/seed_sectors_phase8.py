"""Expand ``coins_registry`` so every sector has a real sample (Part 3).

Phase 8 Part 1 seeded 5 coins, 4 of them L1. A sector average over one
coin is not a sector — it is that coin's noise wearing a sector's name.
This script raises every sector to 3 members so rotation is measurable.

Idempotent and NON-DESTRUCTIVE: rows are inserted with
``ON CONFLICT DO NOTHING``, so re-running never resurrects a symbol the
operator deliberately deactivated, and never rewrites a sector the
operator re-classified by hand.

Symbol verification (2026-07-28, against Binance spot)
------------------------------------------------------
Two of the symbols in the original brief no longer exist and would have
seeded permanently-empty sectors:

* ``MATIC/USDT`` -> **delisted**, migrated to ``POL/USDT`` (Polygon 2.0).
* ``RNDR/USDT``  -> **delisted**, rebranded to ``RENDER/USDT``.

``--verify`` re-runs that check against the live exchange before writing,
so this script can be trusted again after future migrations.

Usage (from the project root)::

    python scripts/seed_sectors_phase8.py --verify
    python scripts/seed_sectors_phase8.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

from src.collectors import BinanceMultiFetcher  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import BASE_SYMBOL, CoinRegistry, CoinSector  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.seed_sectors_phase8")

#: (symbol, name, sector, chain). ``base_asset`` is BTC/USDT for all of
#: them — BTC is the numéraire every relative-strength feature divides by.
#:
#: Sector calls are judgement, not fact; they are one UPDATE away from
#: being changed. The debatable ones are flagged inline.
SECTOR_SEED: tuple[tuple[str, str, CoinSector, str], ...] = (
    # --- L1 -------------------------------------------------------------
    ("ETH/USDT", "Ethereum", CoinSector.L1, "Ethereum"),
    ("SOL/USDT", "Solana", CoinSector.L1, "Solana"),
    ("BNB/USDT", "BNB", CoinSector.L1, "BNB Chain"),
    ("AVAX/USDT", "Avalanche", CoinSector.L1, "Avalanche"),
    ("ADA/USDT", "Cardano", CoinSector.L1, "Cardano"),
    # --- L2 -------------------------------------------------------------
    ("ARB/USDT", "Arbitrum", CoinSector.L2, "Arbitrum"),
    ("OP/USDT", "Optimism", CoinSector.L2, "Optimism"),
    # POL replaces the delisted MATIC (Polygon 2.0 token migration).
    ("POL/USDT", "Polygon", CoinSector.L2, "Polygon"),
    # --- DeFi -----------------------------------------------------------
    ("UNI/USDT", "Uniswap", CoinSector.DEFI, "Ethereum"),
    ("AAVE/USDT", "Aave", CoinSector.DEFI, "Ethereum"),
    # LINK is an oracle network, not an exchange or lending protocol.
    # Classified as Infrastructure (user decision, 2026-07-28) — it was
    # briefly seeded under DeFi in the first Part 3 pass.
    ("LINK/USDT", "Chainlink", CoinSector.INFRASTRUCTURE, "Ethereum"),
    # Added 2026-08-09 to retire the [THIN] flag on Infrastructure. With one
    # member the sector average WAS Chainlink — the tracker was reporting a
    # single token's noise under a sector's name, which is exactly what the
    # THIN warning exists to admit. Both verified tradable on Binance spot
    # and both CoinGecko ids verified to resolve to the right ticker on the
    # same day (the MATIC->POL and RNDR->RENDER delistings are why that check
    # is not optional).
    #
    # PYTH is an oracle, the same function as LINK, so it needs no argument.
    ("PYTH/USDT", "Pyth Network", CoinSector.INFRASTRUCTURE, "Solana"),
    # TIA is the DEBATABLE one, flagged rather than hidden — same treatment as
    # PENDLE in RWA. Celestia runs its own consensus, so a case exists for L1.
    # It is placed here because the tracker measures where capital rotates
    # BETWEEN NARRATIVES, and Celestia's narrative is modular data
    # availability — plumbing other chains build on, not a smart-contract
    # platform competing with ETH/SOL. Move it to L1 if that reading changes;
    # it is one line, and `--reclassify` exists for exactly this.
    ("TIA/USDT", "Celestia", CoinSector.INFRASTRUCTURE, "Celestia"),
    # --- AI -------------------------------------------------------------
    ("TAO/USDT", "Bittensor", CoinSector.AI, "Bittensor"),
    ("FET/USDT", "Artificial Superintelligence Alliance", CoinSector.AI, "Ethereum"),
    # RENDER replaces the delisted RNDR ticker (same project, rebranded).
    ("RENDER/USDT", "Render", CoinSector.AI, "Solana"),
    # --- Meme -----------------------------------------------------------
    ("DOGE/USDT", "Dogecoin", CoinSector.MEME, "Dogecoin"),
    ("PEPE/USDT", "Pepe", CoinSector.MEME, "Ethereum"),
    ("WIF/USDT", "dogwifhat", CoinSector.MEME, "Solana"),
    # --- RWA ------------------------------------------------------------
    ("ONDO/USDT", "Ondo Finance", CoinSector.RWA, "Ethereum"),
    ("POLYX/USDT", "Polymesh", CoinSector.RWA, "Polymesh"),
    # PENDLE is yield tokenisation — read as RWA-adjacent by most rotation
    # screens, but a defensible DeFi call too. Included so RWA has three
    # members rather than two; reclassify if you disagree.
    ("PENDLE/USDT", "Pendle", CoinSector.RWA, "Ethereum"),
    # NOTE: PAXG (tokenised gold) is deliberately EXCLUDED despite being a
    # textbook RWA. Its BTC-relative strength tracks gold-vs-Bitcoin, not
    # capital rotating between crypto narratives, so it would inject a
    # commodity signal into a narrative metric.
)


#: CoinGecko ids per symbol (Phase 8 Part 4). VERIFIED against the live API
#: on 2026-07-28 — all 21 resolved and every returned ticker matched, so
#: there are no id collisions. Never derive these from the ticker at
#: runtime: CoinGecko hosts many coins per symbol, and a wrong id would
#: file another asset's supply under our symbol as a plausible-looking
#: number. Filling a NULL is not clobbering curation, so these are applied
#: automatically; an id an operator has already set is left alone.
COINGECKO_IDS: dict[str, str] = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "SOL/USDT": "solana",
    "BNB/USDT": "binancecoin",
    "AVAX/USDT": "avalanche-2",
    "ADA/USDT": "cardano",
    "ARB/USDT": "arbitrum",
    "OP/USDT": "optimism",
    "POL/USDT": "polygon-ecosystem-token",
    "UNI/USDT": "uniswap",
    "AAVE/USDT": "aave",
    "LINK/USDT": "chainlink",
    # Verified live 2026-08-09: both ids resolve and their tickers match.
    "PYTH/USDT": "pyth-network",
    "TIA/USDT": "celestia",
    "TAO/USDT": "bittensor",
    "FET/USDT": "fetch-ai",
    "RENDER/USDT": "render-token",
    "DOGE/USDT": "dogecoin",
    "PEPE/USDT": "pepe",
    "WIF/USDT": "dogwifcoin",
    "ONDO/USDT": "ondo-finance",
    "POLYX/USDT": "polymesh",
    "PENDLE/USDT": "pendle",
}


async def backfill_coingecko_ids(dry_run: bool) -> int:
    """Fill ``coingecko_id`` where it is NULL. Never overwrites a set value."""
    filled = 0
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(CoinRegistry.symbol, CoinRegistry.coingecko_id)
            )
        ).all()
        for symbol, current in rows:
            wanted = COINGECKO_IDS.get(symbol)
            if wanted is None or current:
                continue
            logger.info("CoinGecko id: %-13s -> %s", symbol, wanted)
            if not dry_run:
                await session.execute(
                    CoinRegistry.__table__.update()
                    .where(CoinRegistry.__table__.c.symbol == symbol)
                    .values(coingecko_id=wanted)
                )
            filled += 1

    async with session_scope() as session:
        unmapped = (
            await session.execute(
                select(CoinRegistry.symbol).where(
                    CoinRegistry.is_active.is_(True),
                    CoinRegistry.coingecko_id.is_(None),
                )
            )
        ).scalars().all()
    if unmapped and not dry_run:
        logger.warning(
            "%d active symbol(s) still have no coingecko_id: %s — their "
            "supply features will stay NULL", len(unmapped), ", ".join(unmapped),
        )
    logger.info("CoinGecko ids filled: %d", filled)
    return filled


#: Deliberate re-classifications for registries seeded BEFORE a taxonomy
#: decision. Plain inserts use ON CONFLICT DO NOTHING and therefore cannot
#: change an existing row — which is the right default, since it protects
#: manual curation. Applying these is an explicit, opt-in act (--reclassify).
SECTOR_CORRECTIONS: dict[str, CoinSector] = {
    # Seeded under DeFi in the first Part 3 pass; an oracle network belongs
    # in Infrastructure (user decision, 2026-07-28).
    "LINK/USDT": CoinSector.INFRASTRUCTURE,
}


async def apply_corrections(dry_run: bool) -> int:
    """Apply :data:`SECTOR_CORRECTIONS` to existing rows. Opt-in only."""
    changed = 0
    async with session_scope() as session:
        for symbol, target in SECTOR_CORRECTIONS.items():
            current = (
                await session.execute(
                    select(CoinRegistry.sector).where(CoinRegistry.symbol == symbol)
                )
            ).scalar_one_or_none()
            if current is None:
                logger.warning("Reclassify SKIP: %s is not in the registry", symbol)
                continue
            if current is target:
                logger.info(
                    "Reclassify SKIP: %s already in %s", symbol, target.value
                )
                continue
            logger.info(
                "Reclassify: %s %s -> %s", symbol, current.value, target.value
            )
            if not dry_run:
                await session.execute(
                    CoinRegistry.__table__.update()
                    .where(CoinRegistry.__table__.c.symbol == symbol)
                    .values(sector=target)
                )
            changed += 1
    if changed and not dry_run:
        logger.warning(
            "%d symbol(s) re-classified — sector_performance history written "
            "before now used the OLD grouping. Re-run the tracker to refresh "
            "today's row; older rows stay as they were computed.",
            changed,
        )
    return changed


async def verify_symbols(symbols: list[str]) -> list[str]:
    """Return the symbols that are NOT active Binance spot markets."""
    fetcher = BinanceMultiFetcher(limit=1)
    try:
        markets = await fetcher.exchange.load_markets()
        active = {
            symbol
            for symbol, meta in markets.items()
            if meta.get("spot") and meta.get("active")
        }
    finally:
        await fetcher.close()
    missing = [symbol for symbol in symbols if symbol not in active]
    logger.info(
        "Verified %d symbol(s) against %d active Binance spot markets: %s",
        len(symbols), len(active),
        "all present" if not missing else f"MISSING {missing}",
    )
    return missing


async def seed(dry_run: bool, verify: bool, reclassify: bool) -> int:
    symbols = [row[0] for row in SECTOR_SEED]

    if verify:
        missing = await verify_symbols(symbols)
        if missing:
            logger.error(
                "Refusing to seed: %s not tradable on Binance spot. A "
                "delisted symbol would create a permanently empty sector "
                "average. Update SECTOR_SEED with the current ticker.",
                missing,
            )
            return 1

    async with session_scope() as session:
        existing = set(
            (await session.execute(select(CoinRegistry.symbol))).scalars().all()
        )
    new = [row for row in SECTOR_SEED if row[0] not in existing]
    logger.info(
        "Registry holds %d symbol(s); %d of the %d seed entries are new",
        len(existing), len(new), len(SECTOR_SEED),
    )

    if dry_run:
        for symbol, name, sector, chain in new:
            logger.info("  would add %-13s %-38s %-14s %s", symbol, name, sector.value, chain)
        return 0

    if new:
        payload = [
            {
                "symbol": symbol,
                "name": name,
                "sector": sector,
                "chain": chain,
                "base_asset": BASE_SYMBOL,
                "is_active": True,
            }
            for symbol, name, sector, chain in new
        ]
        async with session_scope() as session:
            statement = pg_insert(CoinRegistry.__table__).values(payload)
            await session.execute(
                statement.on_conflict_do_nothing(index_elements=["symbol"])
            )
        logger.info("Inserted %d new registry entries", len(payload))
    else:
        logger.info("Nothing to insert — registry already covers the seed")

    # Filling a NULL id is not clobbering curation, so this runs by default.
    await backfill_coingecko_ids(dry_run)

    if reclassify:
        await apply_corrections(dry_run)
    elif SECTOR_CORRECTIONS:
        logger.info(
            "Pending re-classification(s) available: %s — pass --reclassify "
            "to apply (inserts alone cannot change an existing row)",
            ", ".join(SECTOR_CORRECTIONS),
        )

    # --- report the resulting sector composition ---
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

    logger.info("--- active registry composition (%d symbols + base) ---", len(rows) - 1)
    thin = []
    for sector, members in sorted(by_sector.items()):
        logger.info("  %-14s %d: %s", sector, len(members), ", ".join(members))
        if len(members) < 2:
            thin.append(sector)
    if thin:
        logger.warning(
            "Sector(s) with fewer than 2 members: %s — their averages are a "
            "single coin's noise, not a sector signal", thin,
        )
    return 0


async def main(dry_run: bool, verify: bool, reclassify: bool) -> int:
    try:
        return await seed(dry_run, verify, reclassify)
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed sector coins into coins_registry.")
    parser.add_argument("--dry-run", action="store_true", help="Report only.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check every symbol against live Binance spot markets first.",
    )
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="Also apply SECTOR_CORRECTIONS to existing rows (opt-in: plain "
        "seeding never modifies a row that is already there).",
    )
    arguments = parser.parse_args()
    setup_logging(get_settings().log_level)
    raise SystemExit(
        asyncio.run(main(arguments.dry_run, arguments.verify, arguments.reclassify))
    )

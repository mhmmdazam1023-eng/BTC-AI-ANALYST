"""Load tokenomics events from a JSON or CSV file (Phase 8 Part 4).

Unlock and burn schedules are project-specific, get revised, and have no
reliable free API. They are therefore curated by hand and loaded here —
deliberately, rather than scraped from a source that would be wrong in ways
nobody notices. An empty table is a correct state; an invented one is not.

Input format
------------
JSON — a list of objects, or ``{"events": [...]}``::

    [
      {
        "symbol": "SOL/USDT",
        "event_date": "2025-03-01",
        "event_type": "unlock",
        "supply_impact_pct": 2.5,
        "description": "Cliff unlock for early investors"
      }
    ]

CSV — the same field names as the header row.

Validation (every row must pass, or NOTHING is written)
-------------------------------------------------------
* ``symbol`` must already exist in ``coins_registry`` (the FK would reject
  it anyway; failing here gives a readable message instead of a constraint
  error halfway through).
* ``event_type`` must be one of the ``tokenomics_event_type`` enum values.
* ``event_date`` must parse as a date.
* ``supply_impact_pct`` must be between -100 and +100. A single event
  cannot move more than the entire supply, so anything outside that range
  is a data-entry error — most often a fraction (0.05) written where a
  percentage (5.0) was meant, or the reverse.
* Duplicate (symbol, event_date, event_type) rows inside ONE file are
  rejected; the same key across runs is handled by ON CONFLICT DO NOTHING.

Sign convention — read this before writing a file
--------------------------------------------------
``supply_impact_pct`` is POSITIVE when supply enters circulation and
NEGATIVE when it leaves:

* ``unlock``         — positive, % of circulating supply released.
* ``burn``           — negative, % of circulating supply destroyed.
* ``halving``        — negative, change to the EMISSION RATE (-50), NOT to
  the supply stock. Different unit; the feature engine deliberately
  excludes halvings from ``supply_shock_score_30d`` for that reason.
* ``mainnet_launch`` — positive, or null when there is no direct effect.

Usage (from the project root)::

    python scripts/seed_tokenomics.py --file data/unlocks.json
    python scripts/seed_tokenomics.py --file data/unlocks.csv --dry-run
    python scripts/seed_tokenomics.py --show
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import (  # noqa: E402
    CoinRegistry,
    TokenomicsEvent,
    TokenomicsEventType,
)
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.seed_tokenomics")

#: A single event cannot move more than the whole supply.
MIN_IMPACT_PCT = Decimal("-100")
MAX_IMPACT_PCT = Decimal("100")

#: Impact magnitudes above this are accepted but flagged — a 40% unlock is
#: possible yet rare enough to be worth a second look before it silently
#: dominates every supply feature for that coin.
SUSPICIOUS_IMPACT_PCT = Decimal("40")

REQUIRED_FIELDS = ("symbol", "event_date", "event_type")


class TokenomicsSeedError(ValueError):
    """Raised when the input file is malformed or fails validation."""


# ---------------------------------------------------------------------------
def load_file(path: Path) -> list[dict]:
    """Read a JSON or CSV file into a list of raw dicts."""
    if not path.exists():
        raise TokenomicsSeedError(f"file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("events", payload.get("data"))
        if not isinstance(payload, list):
            raise TokenomicsSeedError(
                "JSON must be a list of events, or an object with an "
                "'events' list"
            )
        return payload
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise TokenomicsSeedError(f"unsupported file type '{suffix}' (use .json or .csv)")


def _parse_date(raw: object, index: int) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    text = str(raw).strip()
    if not text:
        raise TokenomicsSeedError(f"row {index}: event_date is empty")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TokenomicsSeedError(
            f"row {index}: event_date {text!r} is not ISO format (YYYY-MM-DD)"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    # Normalise to UTC midnight so the unique key is stable regardless of
    # whether the file carried a time component.
    return parsed.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def _parse_impact(raw: object, index: int) -> Decimal | None:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        value = Decimal(str(raw).strip())
    except (InvalidOperation, ValueError) as exc:
        raise TokenomicsSeedError(
            f"row {index}: supply_impact_pct {raw!r} is not a number"
        ) from exc
    if not value.is_finite():
        raise TokenomicsSeedError(f"row {index}: supply_impact_pct is not finite")
    if not (MIN_IMPACT_PCT <= value <= MAX_IMPACT_PCT):
        raise TokenomicsSeedError(
            f"row {index}: supply_impact_pct {value} is outside "
            f"[{MIN_IMPACT_PCT}, {MAX_IMPACT_PCT}]. A single event cannot move "
            f"more than the entire supply — check whether a fraction was "
            f"written where a percentage was meant (0.05 vs 5.0)."
        )
    return value


def validate(raw_rows: list[dict], known_symbols: set[str]) -> list[dict]:
    """Validate every row. Raises on the FIRST problem; writes nothing."""
    if not raw_rows:
        raise TokenomicsSeedError("the file contains no events")

    valid_types = {member.value for member in TokenomicsEventType}
    seen: set[tuple[str, datetime, str]] = set()
    cleaned: list[dict] = []

    for index, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict):
            raise TokenomicsSeedError(f"row {index}: expected an object, got {row!r}")
        missing = [field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            raise TokenomicsSeedError(f"row {index}: missing field(s) {missing}")

        symbol = str(row["symbol"]).strip()
        if symbol not in known_symbols:
            raise TokenomicsSeedError(
                f"row {index}: symbol {symbol!r} is not in coins_registry. "
                f"Register the coin first (scripts/seed_sectors_phase8.py) — "
                f"the foreign key would reject it anyway."
            )

        event_type = str(row["event_type"]).strip()
        if event_type not in valid_types:
            raise TokenomicsSeedError(
                f"row {index}: event_type {event_type!r} is not one of "
                f"{sorted(valid_types)}"
            )

        event_date = _parse_date(row["event_date"], index)
        impact = _parse_impact(row.get("supply_impact_pct"), index)

        key = (symbol, event_date, event_type)
        if key in seen:
            raise TokenomicsSeedError(
                f"row {index}: duplicate (symbol, event_date, event_type) "
                f"{key} inside the same file"
            )
        seen.add(key)

        if impact is not None and abs(impact) > SUSPICIOUS_IMPACT_PCT:
            logger.warning(
                "row %d: %s %s on %s has a very large impact (%s%%) — accepted, "
                "but worth double-checking",
                index, symbol, event_type, event_date.date(), impact,
            )
        if event_type == TokenomicsEventType.UNLOCK.value and impact is not None and impact < 0:
            logger.warning(
                "row %d: %s unlock on %s has a NEGATIVE impact (%s%%) — unlocks "
                "add supply and are normally positive; check the sign",
                index, symbol, event_date.date(), impact,
            )
        if event_type == TokenomicsEventType.BURN.value and impact is not None and impact > 0:
            logger.warning(
                "row %d: %s burn on %s has a POSITIVE impact (%s%%) — burns "
                "remove supply and are normally negative; check the sign",
                index, symbol, event_date.date(), impact,
            )

        cleaned.append(
            {
                "symbol": symbol,
                "event_date": event_date,
                "event_type": event_type,
                "supply_impact_pct": impact,
                "description": (str(row.get("description") or "").strip() or None),
            }
        )

    logger.info("Validated %d event(s) — all rows passed", len(cleaned))
    return cleaned


# ---------------------------------------------------------------------------
async def known_registry_symbols() -> set[str]:
    async with session_scope() as session:
        return set(
            (await session.execute(select(CoinRegistry.symbol))).scalars().all()
        )


async def insert_events(events: list[dict]) -> int:
    """Idempotent insert. Existing keys are left untouched."""
    async with session_scope() as session:
        statement = pg_insert(TokenomicsEvent.__table__).values(events)
        result = await session.execute(
            statement.on_conflict_do_nothing(constraint="uq_tokenomics_event")
        )
    return result.rowcount if result.rowcount is not None else 0


async def show_events() -> None:
    """Print what is currently stored, grouped by symbol."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(TokenomicsEvent).order_by(
                    TokenomicsEvent.symbol, TokenomicsEvent.event_date
                )
            )
        ).scalars().all()
    if not rows:
        logger.info(
            "tokenomics_events is EMPTY. That is a valid state — supply "
            "features degrade to NULL/0 rather than failing."
        )
        return
    logger.info("--- tokenomics_events (%d row(s)) ---", len(rows))
    current = None
    for row in rows:
        if row.symbol != current:
            current = row.symbol
            logger.info("  %s", current)
        impact = "n/a" if row.supply_impact_pct is None else f"{row.supply_impact_pct:+}%"
        logger.info(
            "    %s  %-14s %8s  %s",
            row.event_date.date(), row.event_type, impact,
            (row.description or "")[:60],
        )


async def run(path: Path | None, dry_run: bool, show: bool) -> int:
    try:
        if show or path is None:
            await show_events()
            if path is None and not show:
                logger.error("Nothing to do: pass --file or --show")
                return 1
            if path is None:
                return 0

        raw_rows = load_file(path)
        logger.info("Read %d raw row(s) from %s", len(raw_rows), path)
        events = validate(raw_rows, await known_registry_symbols())

        if dry_run:
            logger.info("--- DRY RUN, nothing written ---")
            for event in events:
                impact = (
                    "n/a" if event["supply_impact_pct"] is None
                    else f"{event['supply_impact_pct']:+}%"
                )
                logger.info(
                    "  %-13s %s  %-14s %8s",
                    event["symbol"], event["event_date"].date(),
                    event["event_type"], impact,
                )
            return 0

        inserted = await insert_events(events)
        logger.info(
            "Inserted %d new event(s); %d already present (left untouched)",
            inserted, len(events) - inserted,
        )
        await show_events()
        return 0
    except TokenomicsSeedError as exc:
        logger.error("VALIDATION FAILED — nothing was written: %s", exc)
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed tokenomics events from a file.")
    parser.add_argument("--file", type=Path, default=None, help="JSON or CSV input.")
    parser.add_argument("--dry-run", action="store_true", help="Validate only.")
    parser.add_argument(
        "--show", action="store_true", help="Print the stored events and exit."
    )
    arguments = parser.parse_args()
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(run(arguments.file, arguments.dry_run, arguments.show)))

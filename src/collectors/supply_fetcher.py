"""Circulating / total supply collector (CoinGecko, Phase 8 Part 4).

Feeds the ``supply_metrics`` hypertable, which in turn drives the supply
side of the Coin DNA features (``inflation_rate_7d``,
``supply_utilization``, ``liquidity_score``, ``market_cap_tier``).

One request, not N
------------------
CoinGecko's ``/coins/markets`` accepts a comma-separated ``ids`` list and
returns every coin in a single response. The whole 21-coin registry
therefore costs **one** HTTP call per run, which keeps this collector far
below the free tier's limit no matter how large the registry grows. Ids are
chunked at :data:`_MAX_IDS_PER_REQUEST` purely to keep the URL sane.

Ids come from ``coins_registry.coingecko_id``
---------------------------------------------
Never guessed from the ticker. CoinGecko has multiple coins per symbol
(several "RENDER"s, dozens of "PEPE"s), so deriving the id would eventually
fetch a different asset's supply and file it under a registry symbol —
a wrong number that looks completely plausible. A coin with a NULL id is
skipped and reported, which is loud and harmless.

Snapshot, not history
---------------------
The endpoint returns supply AS OF NOW. There is no free historical supply
series, so this table only grows forward from first collection. Features
derived from it are NULL for older rows by design; interpolating backwards
would fabricate the very quantity ``inflation_rate_7d`` measures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar

from sqlalchemy import select

from src.collectors.base import BaseDataFetcher, FetcherError
from src.config import Settings
from src.db.engine import session_scope
from src.db.models import CoinRegistry, SupplyMetric

#: CoinGecko public API root.
_COINGECKO_API = "https://api.coingecko.com/api/v3"

#: Ids per request. One call covers the whole registry today; chunking only
#: guards against an unreasonably long URL as the registry grows.
_MAX_IDS_PER_REQUEST = 100


class SupplyFetcher(BaseDataFetcher):
    """Collects circulating/total supply for every registered coin."""

    source_name: ClassVar[str] = "coingecko"

    #: CoinGecko's free tier is stricter than most; be patient rather than
    #: hammering it. ``_request_json`` already honours ``Retry-After``.
    max_attempts: ClassVar[int] = 4
    backoff_min_seconds: ClassVar[float] = 2.0
    backoff_max_seconds: ClassVar[float] = 60.0

    def __init__(
        self,
        *,
        symbols: list[str] | None = None,
        as_of: datetime | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._symbols_override = symbols
        #: Observation day; defaults to today's UTC midnight so the row
        #: lines up with the daily feature spine.
        self.as_of = as_of or datetime.now(tz=UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        #: Registry symbols with no CoinGecko id (reported, never guessed).
        self.unmapped_symbols: list[str] = []

    # ------------------------------------------------------------------
    async def load_id_map(self) -> dict[str, str]:
        """``{coingecko_id: symbol}`` for active, mapped registry coins."""
        async with session_scope() as session:
            query = select(CoinRegistry.symbol, CoinRegistry.coingecko_id).where(
                CoinRegistry.is_active.is_(True)
            )
            if self._symbols_override is not None:
                query = query.where(CoinRegistry.symbol.in_(self._symbols_override))
            rows = (await session.execute(query)).all()

        mapping: dict[str, str] = {}
        self.unmapped_symbols = []
        for symbol, coingecko_id in rows:
            if not coingecko_id:
                self.unmapped_symbols.append(symbol)
                continue
            mapping[coingecko_id] = symbol

        if self.unmapped_symbols:
            self.log.warning(
                "[coingecko] %d symbol(s) have no coingecko_id and are skipped: "
                "%s — set the id in coins_registry (never guessed from the "
                "ticker: CoinGecko has many coins per symbol)",
                len(self.unmapped_symbols), ", ".join(sorted(self.unmapped_symbols)),
            )
        self.log.info("[coingecko] %d mapped symbol(s) to fetch", len(mapping))
        return mapping

    # ------------------------------------------------------------------
    @staticmethod
    def _to_decimal(value: Any) -> Decimal | None:
        """CoinGecko float -> Decimal; None/garbage/negative -> None."""
        if value is None:
            return None
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None
        if not number.is_finite() or number < 0:
            return None
        return number

    async def fetch_data(self) -> list[SupplyMetric]:
        """Fetch supply for every mapped coin. One request per chunk."""
        id_map = await self.load_id_map()
        if not id_map:
            self.log.warning(
                "[coingecko] no coin has a coingecko_id — nothing to fetch"
            )
            return []

        ids = sorted(id_map)
        rows: list[SupplyMetric] = []
        skipped = 0
        for start in range(0, len(ids), _MAX_IDS_PER_REQUEST):
            chunk = ids[start : start + _MAX_IDS_PER_REQUEST]
            self.log.info(
                "[coingecko] requesting supply for %d coin(s) in one call", len(chunk)
            )
            payload = await self._request_json(
                f"{_COINGECKO_API}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "ids": ",".join(chunk),
                    "per_page": len(chunk),
                    "page": 1,
                    "sparkline": "false",
                },
            )
            if not isinstance(payload, list):
                raise FetcherError(
                    f"[coingecko] expected a JSON array, got "
                    f"{type(payload).__name__}"
                )

            returned = set()
            for entry in payload:
                if not isinstance(entry, dict) or "id" not in entry:
                    skipped += 1
                    self.log.warning("[coingecko] skipping malformed entry: %r", entry)
                    continue
                coingecko_id = entry["id"]
                symbol = id_map.get(coingecko_id)
                if symbol is None:
                    # Not requested — ignore rather than filing it under a
                    # symbol we did not ask about.
                    continue
                returned.add(coingecko_id)
                circulating = self._to_decimal(entry.get("circulating_supply"))
                total = self._to_decimal(entry.get("total_supply"))
                if circulating is None and total is None:
                    skipped += 1
                    self.log.warning(
                        "[coingecko] %s (%s) returned no usable supply — skipped",
                        symbol, coingecko_id,
                    )
                    continue
                rows.append(
                    SupplyMetric(
                        timestamp=self.as_of,
                        symbol=symbol,
                        circulating_supply=circulating,
                        total_supply=total,
                    )
                )

            for missing_id in set(chunk) - returned:
                skipped += 1
                self.log.warning(
                    "[coingecko] id '%s' (%s) was not in the response — check "
                    "that the id is still valid",
                    missing_id, id_map.get(missing_id, "?"),
                )

        if skipped:
            self.log.warning("[coingecko] %d coin(s) produced no supply row", skipped)
        if not rows:
            self.log.warning("[coingecko] no usable supply rows — returning none")
            return []
        self.log.info(
            "[coingecko] fetched supply for %d coin(s) as of %s",
            len(rows), self.as_of.date(),
        )
        return rows

    # ------------------------------------------------------------------
    async def save_to_db(self, rows) -> int:
        """Idempotent upsert into ``supply_metrics``.

        Values are refreshed on conflict: re-running on the same day should
        record the latest snapshot, not the first one seen.
        """
        return await self._upsert(
            SupplyMetric,
            rows,
            update_columns=("circulating_supply", "total_supply"),
        )

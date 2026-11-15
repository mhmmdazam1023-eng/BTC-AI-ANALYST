"""Blockchain.com on-chain metrics collector (free, no API key).

Fetches daily historical series from the public charts API
(``https://api.blockchain.info/charts/{chart}``) and maps them to the
``on_chain_metrics`` hypertable (``source='BLOCKCHAIN_INFO'``, ``asset='BTC'``).

Default charts (Phase 2 Part 2 scope):

* ``hash-rate``       — estimated network hash rate, in TH/s
* ``n-transactions``  — confirmed transactions per day
* ``total-bitcoins``  — circulating BTC supply

Granularity note: some charts (``total-bitcoins``) are **per-block** rather
than daily; every chart is therefore normalised to one observation per UTC
day (last point of the day wins) before persistence.

Rate-limit posture (Blockchain.com is strict):
* All HTTP goes through :meth:`BaseDataFetcher._request_json`, which retries
  HTTP 429 honouring the ``Retry-After`` header with exponential backoff.
* This fetcher additionally overrides the retry budget (more attempts,
  longer max backoff) and inserts a **polite pause between chart requests**
  so three back-to-back calls never look like a burst.
* A chart that keeps failing is skipped (logged as ERROR) without killing
  the other charts.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar

import httpx

from src.collectors.base import BaseDataFetcher, FetcherError, RetryableHTTPError
from src.config import Settings
from src.db.models import OnChainMetric

#: Chart slug -> human description (for logs only).
DEFAULT_CHARTS: dict[str, str] = {
    "hash-rate": "Estimated network hash rate (TH/s)",
    "n-transactions": "Confirmed transactions per day",
    "total-bitcoins": "Circulating BTC supply",
}


class BlockchainFetcher(BaseDataFetcher):
    """Collects daily on-chain series from the Blockchain.com charts API."""

    source_name: ClassVar[str] = "blockchain_info"

    BASE_URL: ClassVar[str] = "https://api.blockchain.info/charts"

    # --- stricter retry posture for a strictly rate-limited API ---
    max_attempts: ClassVar[int] = 6
    backoff_min_seconds: ClassVar[float] = 2.0
    backoff_max_seconds: ClassVar[float] = 120.0
    #: Polite pause between consecutive chart requests (seconds).
    inter_request_delay_seconds: ClassVar[float] = 2.0

    #: Everything _request_json can raise after its retry budget is spent.
    _FETCH_FAILURES: ClassVar[tuple[type[BaseException], ...]] = (
        FetcherError,
        RetryableHTTPError,
        httpx.HTTPError,
    )

    def __init__(
        self,
        *,
        charts: dict[str, str] | None = None,
        timespan: str = "all",
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.charts = dict(charts or DEFAULT_CHARTS)
        self.timespan = timespan

    # ------------------------------------------------------------------
    async def fetch_data(self) -> list[OnChainMetric]:
        """Fetch every configured chart; failing charts are skipped, not fatal."""
        rows: list[OnChainMetric] = []
        failed_charts: list[str] = []

        for index, (chart, description) in enumerate(self.charts.items()):
            if index > 0:
                self.log.debug(
                    "[blockchain_info] pausing %.1fs before next chart (rate-limit courtesy)",
                    self.inter_request_delay_seconds,
                )
                await asyncio.sleep(self.inter_request_delay_seconds)

            self.log.info(
                "[blockchain_info] fetching chart '%s' (%s, timespan=%s)",
                chart, description, self.timespan,
            )
            try:
                payload = await self._request_json(
                    f"{self.BASE_URL}/{chart}",
                    params={
                        "timespan": self.timespan,
                        "format": "json",
                        "sampled": "false",  # full daily resolution
                    },
                )
            except self._FETCH_FAILURES as exc:
                failed_charts.append(chart)
                self.log.error(
                    "[blockchain_info] chart '%s' failed after %d attempts — "
                    "skipping this chart. Detail: %s",
                    chart, self.max_attempts, exc,
                )
                continue

            rows.extend(self._map_chart(chart, payload))

        if failed_charts:
            self.log.warning(
                "[blockchain_info] %d/%d charts failed: %s",
                len(failed_charts), len(self.charts), failed_charts,
            )
        rows = self._deduplicate(rows)
        if not rows:
            self.log.warning("[blockchain_info] no usable observations across all charts")
        else:
            self.log.info(
                "[blockchain_info] fetched %d observations across %d chart(s)",
                len(rows), len(self.charts) - len(failed_charts),
            )
        return rows

    # ------------------------------------------------------------------
    def _map_chart(self, chart: str, payload: Any) -> list[OnChainMetric]:
        """Validate and map one charts-API payload into ORM rows."""
        if not isinstance(payload, dict) or not isinstance(payload.get("values"), list):
            self.log.error(
                "[blockchain_info] chart '%s': unexpected payload shape: %s",
                chart, str(payload)[:200],
            )
            return []

        values = payload["values"]
        if not values:
            self.log.warning("[blockchain_info] chart '%s' returned 0 points", chart)
            return []

        unit = payload.get("unit", "?")
        # Some charts (e.g. total-bitcoins) are per-block, not daily. We
        # normalise everything to ONE observation per UTC day, keeping the
        # last point of the day (points arrive in chronological order).
        daily: dict[datetime, OnChainMetric] = {}
        skipped = 0
        negative = 0
        for point in values:
            try:
                if not isinstance(point, dict):
                    raise ValueError("point is not an object")
                ts_raw, value_raw = point["x"], point["y"]
                if ts_raw is None or value_raw is None:
                    raise ValueError("null x/y in point")
                value = Decimal(str(value_raw))
                if value < 0:
                    negative += 1
                    raise ValueError(f"negative value {value}")
                day = datetime.fromtimestamp(int(ts_raw), tz=UTC).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                daily[day] = OnChainMetric(
                    timestamp=day,
                    source="BLOCKCHAIN_INFO",
                    asset="BTC",
                    metric=chart,
                    value=value,
                )
            except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError, OSError) as exc:
                skipped += 1
                if skipped <= 5:  # avoid log flooding on huge charts
                    self.log.warning(
                        "[blockchain_info] chart '%s': skipping malformed point %r: %s",
                        chart, point, exc,
                    )

        if skipped:
            self.log.warning(
                "[blockchain_info] chart '%s': skipped %d malformed point(s) "
                "(%d negative)", chart, skipped, negative,
            )
        rows = [daily[day] for day in sorted(daily)]
        if rows:
            if len(values) > len(rows):
                self.log.info(
                    "[blockchain_info] chart '%s': downsampled %d raw points "
                    "-> %d daily observations", chart, len(values), len(rows),
                )
            self.log.info(
                "[blockchain_info] chart '%s': %d daily observations (unit=%s, %s -> %s)",
                chart, len(rows), unit,
                rows[0].timestamp.date(), rows[-1].timestamp.date(),
            )
        return rows

    # ------------------------------------------------------------------
    def _deduplicate(self, rows: list[OnChainMetric]) -> list[OnChainMetric]:
        """Drop duplicate primary keys (a single INSERT may not hit one twice)."""
        unique: dict[tuple, OnChainMetric] = {}
        for row in rows:
            unique[(row.timestamp, row.source, row.asset, row.metric)] = row
        dropped = len(rows) - len(unique)
        if dropped:
            self.log.warning(
                "[blockchain_info] dropped %d duplicate observation(s)", dropped
            )
        return list(unique.values())

    # ------------------------------------------------------------------
    async def save_to_db(self, rows) -> int:
        """Idempotent upsert into ``on_chain_metrics`` (values refreshed)."""
        return await self._upsert(OnChainMetric, rows, update_columns=("value",))

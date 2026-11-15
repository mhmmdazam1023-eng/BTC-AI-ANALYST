"""Deribit derivatives/liquidity collector (Public API v2, no key needed).

Chosen over Binance Futures because ``fapi.binance.com`` is ISP-blocked on
this machine (see CLAUDE.md). Deribit is the reference venue for BTC options
and its DVOL index; funding/OI dynamics are key liquidity-regime signals in
the research corpus (Zhivkov et al. 2026; Krause 2026).

Metrics collected into ``derivatives_metrics`` (exchange ``deribit``):

* ``dvol``          — daily close of the BTC DVOL implied-volatility index
                      (symbol ``BTC-DVOL``; history since the index launch,
                      March 2021), via ``public/get_volatility_index_data``
                      with continuation-token pagination.
* ``funding_8h``    — trailing 8h funding rate of ``BTC-PERPETUAL``,
                      downsampled hourly -> daily (last reading of the day),
                      via ``public/get_funding_rate_history`` fetched in
                      30-day windows.
* ``open_interest`` — current OI snapshot of ``BTC-PERPETUAL`` (USD), via
                      ``public/get_book_summary_by_currency``. Deribit does
                      not expose OI history publicly, so history accrues
                      one snapshot per collection day.
* ``open_interest_futures_total`` — snapshot sum of OI across all BTC
                      futures instruments (symbol ``BTC-FUTURES``).

Each metric group is fetched independently: one failing group is logged and
skipped without killing the others. All JSON-RPC envelopes are validated
(``error`` object -> :class:`FetcherError`).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar

import httpx

from src.collectors.base import BaseDataFetcher, FetcherError, RetryableHTTPError
from src.config import Settings
from src.db.models import DerivativesMetric

_MS_PER_DAY = 86_400_000


class DeribitFetcher(BaseDataFetcher):
    """Collects DVOL, funding-rate history, and OI snapshots from Deribit."""

    source_name: ClassVar[str] = "deribit"

    BASE_URL: ClassVar[str] = "https://www.deribit.com/api/v2"

    #: Polite pause between consecutive API calls (public rate limits).
    inter_request_delay_seconds: ClassVar[float] = 0.25
    #: Safety caps for pagination loops.
    max_dvol_pages: ClassVar[int] = 30
    max_funding_windows: ClassVar[int] = 130
    #: Funding history window per request (Deribit caps roughly at a month).
    funding_window_days: ClassVar[int] = 30

    _FETCH_FAILURES: ClassVar[tuple[type[BaseException], ...]] = (
        FetcherError,
        RetryableHTTPError,
        httpx.HTTPError,
    )

    def __init__(
        self,
        *,
        currency: str = "BTC",
        history_start: str = "2021-01-01",
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.currency = currency.upper()
        self.history_start_ms = int(
            datetime.fromisoformat(history_start).replace(tzinfo=UTC).timestamp() * 1000
        )

    # ------------------------------------------------------------------
    # JSON-RPC plumbing
    # ------------------------------------------------------------------
    async def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        """Call a public JSON-RPC method and return the validated ``result``."""
        payload = await self._request_json(f"{self.BASE_URL}/public/{method}", params=params)
        if not isinstance(payload, dict):
            raise FetcherError(
                f"[deribit] {method}: unexpected payload type {type(payload).__name__}"
            )
        error = payload.get("error")
        if error:
            raise FetcherError(
                f"[deribit] {method}: API error "
                f"{error.get('code', '?')}: {error.get('message', error)}"
            )
        if "result" not in payload:
            raise FetcherError(
                f"[deribit] {method}: missing 'result' in response: {str(payload)[:200]}"
            )
        return payload["result"]

    @staticmethod
    def _day(ts_ms: int) -> datetime:
        """Millisecond timestamp -> UTC midnight of that day."""
        return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    # ------------------------------------------------------------------
    # Metric group 1: DVOL history (paginated)
    # ------------------------------------------------------------------
    async def _fetch_dvol(self) -> list[DerivativesMetric]:
        symbol = f"{self.currency}-DVOL"
        end_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        daily: dict[datetime, DerivativesMetric] = {}
        skipped = 0

        for page in range(1, self.max_dvol_pages + 1):
            self.log.debug("[deribit] DVOL page %d (end=%d)", page, end_ms)
            result = await self._rpc(
                "get_volatility_index_data",
                {
                    "currency": self.currency,
                    "start_timestamp": self.history_start_ms,
                    "end_timestamp": end_ms,
                    "resolution": "86400",  # daily candles
                },
            )
            data = result.get("data") if isinstance(result, dict) else None
            if not isinstance(data, list) or not data:
                break

            for point in data:
                try:
                    # [timestamp_ms, open, high, low, close]
                    if not isinstance(point, (list, tuple)) or len(point) < 5:
                        raise ValueError("malformed DVOL point")
                    ts_ms, close = point[0], point[4]
                    if ts_ms is None or close is None:
                        raise ValueError("null DVOL fields")
                    day = self._day(int(ts_ms))
                    daily.setdefault(
                        day,
                        DerivativesMetric(
                            timestamp=day,
                            exchange="deribit",
                            symbol=symbol,
                            metric="dvol",
                            value=Decimal(str(close)),
                        ),
                    )
                except (TypeError, ValueError, InvalidOperation, OverflowError) as exc:
                    skipped += 1
                    if skipped <= 5:
                        self.log.warning(
                            "[deribit] skipping malformed DVOL point %r: %s", point, exc
                        )

            continuation = result.get("continuation")
            if not continuation or int(continuation) >= end_ms:
                break
            end_ms = int(continuation)
            await asyncio.sleep(self.inter_request_delay_seconds)

        rows = [daily[day] for day in sorted(daily)]
        if skipped:
            self.log.warning("[deribit] DVOL: skipped %d malformed point(s)", skipped)
        if rows:
            self.log.info(
                "[deribit] DVOL: %d daily closes (%s -> %s, latest=%s)",
                len(rows), rows[0].timestamp.date(), rows[-1].timestamp.date(),
                rows[-1].value,
            )
        else:
            self.log.warning("[deribit] DVOL: no data returned")
        return rows

    # ------------------------------------------------------------------
    # Metric group 2: funding-rate history (windowed)
    # ------------------------------------------------------------------
    async def _fetch_funding(self) -> list[DerivativesMetric]:
        instrument = f"{self.currency}-PERPETUAL"
        window_ms = self.funding_window_days * _MS_PER_DAY
        cursor_end = int(datetime.now(tz=UTC).timestamp() * 1000)
        points: list[dict[str, Any]] = []
        windows = 0

        while cursor_end > self.history_start_ms and windows < self.max_funding_windows:
            cursor_start = max(self.history_start_ms, cursor_end - window_ms)
            self.log.debug(
                "[deribit] funding window %d: %s -> %s",
                windows + 1, self._day(cursor_start).date(), self._day(cursor_end).date(),
            )
            result = await self._rpc(
                "get_funding_rate_history",
                {
                    "instrument_name": instrument,
                    "start_timestamp": cursor_start,
                    "end_timestamp": cursor_end,
                },
            )
            if isinstance(result, list):
                points.extend(item for item in result if isinstance(item, dict))
            cursor_end = cursor_start
            windows += 1
            await asyncio.sleep(self.inter_request_delay_seconds)

        # Downsample hourly readings -> last reading per UTC day.
        latest_per_day: dict[datetime, dict[str, Any]] = {}
        skipped = 0
        for item in points:
            try:
                ts_ms = int(item["timestamp"])
                if item.get("interest_8h") is None:
                    raise ValueError("null interest_8h")
                day = self._day(ts_ms)
                current = latest_per_day.get(day)
                if current is None or ts_ms > int(current["timestamp"]):
                    latest_per_day[day] = item
            except (KeyError, TypeError, ValueError) as exc:
                skipped += 1
                if skipped <= 5:
                    self.log.warning(
                        "[deribit] skipping malformed funding item %r: %s", item, exc
                    )

        rows: list[DerivativesMetric] = []
        for day in sorted(latest_per_day):
            try:
                rows.append(
                    DerivativesMetric(
                        timestamp=day,
                        exchange="deribit",
                        symbol=instrument,
                        metric="funding_8h",
                        value=Decimal(str(latest_per_day[day]["interest_8h"])),
                    )
                )
            except (InvalidOperation, TypeError, ValueError) as exc:
                skipped += 1
                self.log.warning("[deribit] bad funding value on %s: %s", day, exc)

        if skipped:
            self.log.warning("[deribit] funding: skipped %d malformed item(s)", skipped)
        if rows:
            self.log.info(
                "[deribit] funding_8h: %d hourly points -> %d daily readings (%s -> %s)",
                len(points), len(rows),
                rows[0].timestamp.date(), rows[-1].timestamp.date(),
            )
        else:
            self.log.warning("[deribit] funding: no data returned")
        return rows

    # ------------------------------------------------------------------
    # Metric group 3: open-interest snapshot (no public history)
    # ------------------------------------------------------------------
    async def _fetch_open_interest(self) -> list[DerivativesMetric]:
        result = await self._rpc(
            "get_book_summary_by_currency",
            {"currency": self.currency, "kind": "future"},
        )
        if not isinstance(result, list) or not result:
            self.log.warning("[deribit] book summary returned no instruments")
            return []

        today = self._day(int(datetime.now(tz=UTC).timestamp() * 1000))
        perpetual = f"{self.currency}-PERPETUAL"
        rows: list[DerivativesMetric] = []
        total = Decimal(0)
        counted = 0

        for summary in result:
            if not isinstance(summary, dict):
                continue
            oi_raw = summary.get("open_interest")
            name = summary.get("instrument_name", "")
            if oi_raw is None:
                continue
            try:
                oi = Decimal(str(oi_raw))
            except (InvalidOperation, TypeError, ValueError) as exc:
                self.log.warning("[deribit] bad open_interest for %s: %s", name, exc)
                continue
            total += oi
            counted += 1
            if name == perpetual:
                rows.append(
                    DerivativesMetric(
                        timestamp=today,
                        exchange="deribit",
                        symbol=perpetual,
                        metric="open_interest",
                        value=oi,
                    )
                )

        if counted:
            rows.append(
                DerivativesMetric(
                    timestamp=today,
                    exchange="deribit",
                    symbol=f"{self.currency}-FUTURES",
                    metric="open_interest_futures_total",
                    value=total,
                )
            )
            self.log.info(
                "[deribit] OI snapshot @ %s: perpetual in %d instrument(s), total=%s USD",
                today.date(), counted, total,
            )
        else:
            self.log.warning("[deribit] no usable open_interest values in book summary")
        return rows

    # ------------------------------------------------------------------
    async def fetch_data(self) -> list[DerivativesMetric]:
        """Fetch all metric groups; a failing group is skipped, not fatal."""
        groups = (
            ("dvol", self._fetch_dvol),
            ("funding_8h", self._fetch_funding),
            ("open_interest", self._fetch_open_interest),
        )
        rows: list[DerivativesMetric] = []
        failed: list[str] = []
        for index, (name, fetch) in enumerate(groups):
            if index > 0:
                await asyncio.sleep(self.inter_request_delay_seconds)
            try:
                rows.extend(await fetch())
            except self._FETCH_FAILURES as exc:
                failed.append(name)
                self.log.error(
                    "[deribit] metric group '%s' failed — skipping it. Detail: %s",
                    name, exc,
                )

        if failed:
            self.log.warning(
                "[deribit] %d/%d metric groups failed: %s", len(failed), len(groups), failed
            )
        if not rows:
            self.log.warning("[deribit] no usable observations across all metric groups")
        else:
            self.log.info("[deribit] fetched %d observations in total", len(rows))
        return rows

    # ------------------------------------------------------------------
    async def save_to_db(self, rows) -> int:
        """Idempotent upsert into ``derivatives_metrics`` (values refreshed)."""
        return await self._upsert(DerivativesMetric, rows, update_columns=("value",))

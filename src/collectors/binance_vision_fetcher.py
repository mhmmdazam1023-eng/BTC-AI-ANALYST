"""Binance Vision derivatives collector (official public data archive).

WHY THIS SOURCE: every offshore derivatives venue API is ISP-blocked on this
machine — ``fapi.binance.com``, ``deribit.com``, ``bybit.com`` and ``okx.com``
all fail with the same DNS-hijack TLS signature (verified 2026-07-08, see
CLAUDE.md). ``data.binance.vision`` — Binance's official static archive of
historical market data — IS reachable, and provides USDⓈ-M futures funding
rates and open-interest metrics as CSV-in-ZIP files.

Metrics collected into ``derivatives_metrics`` (exchange ``binance``,
symbol ``BTCUSDT`` USDⓈ-M perpetual):

* ``funding_rate``        — realised funding per 8h interval, downsampled to
                            the last event of each UTC day. Source: monthly
                            ZIPs ``futures/um/monthly/fundingRate/...``
                            (the still-running month is not yet published —
                            its absence is expected and handled).
* ``open_interest``       — OI in BTC, last 5-minute reading of each day.
* ``open_interest_value`` — OI notional in USDT, same downsampling.
                            Source: daily ZIPs ``futures/um/daily/metrics/...``
                            over a rolling backfill window (default 365 days,
                            per user decision 2026-07-08; history accrues
                            further with every scheduled run).

Liquidity-pillar grounding: funding-rate dynamics and OI are the key
liquidity-regime signals in the research corpus (Zhivkov et al. 2026).
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import ClassVar

import httpx
import pandas as pd

from src.collectors.base import BaseDataFetcher, FetcherError, RetryableHTTPError
from src.config import Settings
from src.db.models import DerivativesMetric


class BinanceVisionFetcher(BaseDataFetcher):
    """Collects funding-rate & OI history from the Binance Vision archive."""

    source_name: ClassVar[str] = "binance_vision"

    BASE_URL: ClassVar[str] = "https://data.binance.vision/data/futures/um"

    #: Polite pause between consecutive archive downloads.
    inter_request_delay_seconds: ClassVar[float] = 0.15

    _FETCH_FAILURES: ClassVar[tuple[type[BaseException], ...]] = (
        FetcherError,
        RetryableHTTPError,
        httpx.HTTPError,
    )

    def __init__(
        self,
        *,
        symbol: str = "BTCUSDT",
        funding_start: str = "2021-01-01",
        oi_backfill_days: int = 365,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        if oi_backfill_days < 1:
            raise ValueError(f"oi_backfill_days must be >= 1, got {oi_backfill_days}")
        self.symbol = symbol.upper()
        self.funding_start = datetime.fromisoformat(funding_start).replace(tzinfo=UTC)
        self.oi_backfill_days = oi_backfill_days

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    async def _download_csv(self, url: str) -> pd.DataFrame | None:
        """Download a ZIP archive and return its single CSV as a DataFrame.

        Returns ``None`` for not-yet-published archives (HTTP 404).
        """
        content = await self._request_bytes(url, allow_404=True)
        if content is None:
            return None
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = archive.namelist()
                if not names:
                    raise ValueError("empty ZIP archive")
                with archive.open(names[0]) as handle:
                    return pd.read_csv(handle)
        except (zipfile.BadZipFile, ValueError, pd.errors.ParserError) as exc:
            raise FetcherError(
                f"[binance_vision] corrupt archive at {url}: {exc}"
            ) from exc

    @staticmethod
    def _floor_day(moment: datetime) -> datetime:
        return moment.replace(hour=0, minute=0, second=0, microsecond=0)

    def _make_row(self, day: datetime, metric: str, value: Decimal) -> DerivativesMetric:
        return DerivativesMetric(
            timestamp=day,
            exchange="binance",
            symbol=self.symbol,
            metric=metric,
            value=value,
        )

    # ------------------------------------------------------------------
    # Metric group 1: funding-rate history (monthly archives)
    # ------------------------------------------------------------------
    def _funding_months(self) -> list[str]:
        """All YYYY-MM strings from ``funding_start`` through the current month."""
        months: list[str] = []
        cursor = self.funding_start.replace(day=1)
        now = datetime.now(tz=UTC)
        while cursor <= now:
            months.append(f"{cursor.year:04d}-{cursor.month:02d}")
            cursor = (cursor + timedelta(days=32)).replace(day=1)
        return months

    async def _fetch_funding(self) -> list[DerivativesMetric]:
        latest_per_day: dict[datetime, tuple[int, Decimal]] = {}
        months = self._funding_months()
        missing = 0
        skipped = 0

        for index, month in enumerate(months):
            if index > 0:
                await asyncio.sleep(self.inter_request_delay_seconds)
            url = (
                f"{self.BASE_URL}/monthly/fundingRate/{self.symbol}/"
                f"{self.symbol}-fundingRate-{month}.zip"
            )
            frame = await self._download_csv(url)
            if frame is None:
                missing += 1
                self.log.debug(
                    "[binance_vision] funding archive %s not published (expected "
                    "for the running month)", month,
                )
                continue
            if "calc_time" not in frame.columns or "last_funding_rate" not in frame.columns:
                self.log.warning(
                    "[binance_vision] funding %s: unexpected columns %s — skipping",
                    month, list(frame.columns),
                )
                continue

            for record in frame.itertuples(index=False):
                try:
                    ts_ms = int(record.calc_time)
                    rate = Decimal(str(record.last_funding_rate))
                    day = self._floor_day(datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC))
                    current = latest_per_day.get(day)
                    if current is None or ts_ms > current[0]:
                        latest_per_day[day] = (ts_ms, rate)
                except (TypeError, ValueError, InvalidOperation, OverflowError, OSError) as exc:
                    skipped += 1
                    if skipped <= 5:
                        self.log.warning(
                            "[binance_vision] funding %s: skipping bad record %r: %s",
                            month, record, exc,
                        )

        rows = [
            self._make_row(day, "funding_rate", latest_per_day[day][1])
            for day in sorted(latest_per_day)
        ]
        if skipped:
            self.log.warning("[binance_vision] funding: skipped %d bad record(s)", skipped)
        if rows:
            self.log.info(
                "[binance_vision] funding_rate: %d daily readings from %d monthly "
                "archives (%d unpublished), %s -> %s",
                len(rows), len(months) - missing, missing,
                rows[0].timestamp.date(), rows[-1].timestamp.date(),
            )
        else:
            self.log.warning("[binance_vision] funding: no data returned")
        return rows

    # ------------------------------------------------------------------
    # Metric group 2: open-interest history (daily metric archives)
    # ------------------------------------------------------------------
    async def _fetch_open_interest(self) -> list[DerivativesMetric]:
        today = self._floor_day(datetime.now(tz=UTC))
        latest_per_day: dict[datetime, tuple[datetime, Decimal, Decimal]] = {}
        missing = 0
        skipped = 0

        for offset in range(self.oi_backfill_days, 0, -1):
            day = today - timedelta(days=offset)
            if offset != self.oi_backfill_days:
                await asyncio.sleep(self.inter_request_delay_seconds)
            url = (
                f"{self.BASE_URL}/daily/metrics/{self.symbol}/"
                f"{self.symbol}-metrics-{day.date().isoformat()}.zip"
            )
            frame = await self._download_csv(url)
            if frame is None:
                missing += 1
                continue
            required = {"create_time", "sum_open_interest", "sum_open_interest_value"}
            if not required.issubset(frame.columns):
                self.log.warning(
                    "[binance_vision] metrics %s: unexpected columns %s — skipping",
                    day.date(), list(frame.columns),
                )
                continue

            for record in frame.itertuples(index=False):
                try:
                    moment = datetime.strptime(
                        str(record.create_time), "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=UTC)
                    oi = Decimal(str(record.sum_open_interest))
                    oi_value = Decimal(str(record.sum_open_interest_value))
                    if oi < 0 or oi_value < 0:
                        raise ValueError("negative open interest")
                    bucket = self._floor_day(moment)
                    current = latest_per_day.get(bucket)
                    if current is None or moment > current[0]:
                        latest_per_day[bucket] = (moment, oi, oi_value)
                except (TypeError, ValueError, InvalidOperation, OverflowError) as exc:
                    skipped += 1
                    if skipped <= 5:
                        self.log.warning(
                            "[binance_vision] metrics %s: skipping bad record: %s",
                            day.date(), exc,
                        )

        rows: list[DerivativesMetric] = []
        for bucket in sorted(latest_per_day):
            _, oi, oi_value = latest_per_day[bucket]
            rows.append(self._make_row(bucket, "open_interest", oi))
            rows.append(self._make_row(bucket, "open_interest_value", oi_value))

        if skipped:
            self.log.warning("[binance_vision] metrics: skipped %d bad record(s)", skipped)
        if rows:
            self.log.info(
                "[binance_vision] open_interest: %d daily observations "
                "(%d daily archives read, %d unpublished/missing)",
                len(rows) // 2, self.oi_backfill_days - missing, missing,
            )
        else:
            self.log.warning("[binance_vision] open_interest: no data returned")
        return rows

    # ------------------------------------------------------------------
    async def fetch_data(self) -> list[DerivativesMetric]:
        """Fetch all metric groups; a failing group is skipped, not fatal."""
        groups = (
            ("funding_rate", self._fetch_funding),
            ("open_interest", self._fetch_open_interest),
        )
        rows: list[DerivativesMetric] = []
        failed: list[str] = []
        for name, fetch in groups:
            try:
                rows.extend(await fetch())
            except self._FETCH_FAILURES as exc:
                failed.append(name)
                self.log.error(
                    "[binance_vision] metric group '%s' failed — skipping it. "
                    "Detail: %s", name, exc,
                )

        if failed:
            self.log.warning(
                "[binance_vision] %d/%d metric groups failed: %s",
                len(failed), len(groups), failed,
            )
        if not rows:
            self.log.warning("[binance_vision] no usable observations at all")
        else:
            self.log.info("[binance_vision] fetched %d observations in total", len(rows))
        return rows

    # ------------------------------------------------------------------
    async def save_to_db(self, rows) -> int:
        """Idempotent upsert into ``derivatives_metrics`` (values refreshed)."""
        return await self._upsert(DerivativesMetric, rows, update_columns=("value",))

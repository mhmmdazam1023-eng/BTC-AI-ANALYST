"""FRED macro indicator collector (``fredapi`` — mandated tech stack).

Fetches macro series and maps them to the ``macro_indicators`` hypertable.
Default series (per Phase 2 Part 1 scope):

* ``DTWEXBGS`` — Nominal Broad U.S. Dollar Index (the DXY-style dollar gauge)
* ``DGS10``   — 10-Year Treasury Constant Maturity Yield

Resilience contract (user requirement):
* **Empty API key** -> log a warning and return ``[]`` — NEVER crash.
* **Invalid API key** -> FRED replies with an ``api_key`` error message;
  the fetcher logs a warning, aborts remaining series, returns ``[]``.
* Network failures are retried with tenacity exponential backoff; a series
  that keeps failing is skipped (logged) without killing the whole batch.
* ``fredapi`` is synchronous (urllib-based), so calls run in a worker thread
  via ``asyncio.to_thread`` to keep the event loop responsive.
"""

from __future__ import annotations

import asyncio
from datetime import UTC
from decimal import Decimal, InvalidOperation
from typing import ClassVar
from urllib.error import URLError
from xml.etree.ElementTree import ParseError

import pandas as pd
from fredapi import Fred

from src.collectors.base import BaseDataFetcher
from src.config import Settings
from src.db.models import MacroIndicator

#: Phase 2 Part 1 series (id -> human description, for logs only).
DEFAULT_SERIES: dict[str, str] = {
    "DTWEXBGS": "Nominal Broad U.S. Dollar Index (DXY proxy)",
    "DGS10": "10-Year Treasury Constant Maturity Yield (%)",
}

#: Default history window; matches the macro literature in the research
#: corpus (Lin et al. 2025 sample starts January 2018).
DEFAULT_OBSERVATION_START = "2018-01-01"


class _InvalidFredApiKeyError(RuntimeError):
    """Internal sentinel: the configured FRED API key was rejected."""


class FredFetcher(BaseDataFetcher):
    """Collects macro series from FRED into ``macro_indicators``."""

    source_name: ClassVar[str] = "fred"

    #: Retryable failures. ``fredapi`` converts HTTP errors into
    #: ``ValueError`` internally, so what reaches us as URLError/timeouts is
    #: genuine network trouble; ParseError covers 5xx HTML bodies.
    RETRYABLE_ERRORS: ClassVar[tuple[type[BaseException], ...]] = (
        URLError,
        TimeoutError,
        ConnectionError,
        ParseError,
    )

    def __init__(
        self,
        *,
        series: dict[str, str] | None = None,
        observation_start: str | None = DEFAULT_OBSERVATION_START,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.series = dict(series or DEFAULT_SERIES)
        self.observation_start = observation_start
        self._fred: Fred | None = None

    # ------------------------------------------------------------------
    def _get_client(self) -> Fred | None:
        """Return the fredapi client, or ``None`` when no key is configured."""
        api_key = self.settings.fred_api_key.get_secret_value().strip()
        if not api_key:
            self.log.warning(
                "[fred] FRED_API_KEY is empty — skipping FRED collection. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
                "and set it in .env."
            )
            return None
        if self._fred is None:
            self._fred = Fred(api_key=api_key)
            self.log.debug("[fred] fredapi client created")
        return self._fred

    # ------------------------------------------------------------------
    async def fetch_data(self) -> list[MacroIndicator]:
        """Fetch all configured series; returns ``[]`` gracefully on key issues."""
        fred = self._get_client()
        if fred is None:
            return []

        rows: list[MacroIndicator] = []
        for series_id, description in self.series.items():
            self.log.info(
                "[fred] fetching series '%s' (%s, start=%s)",
                series_id, description, self.observation_start or "full history",
            )
            try:
                series = await self._fetch_series(fred, series_id)
            except _InvalidFredApiKeyError:
                # Already logged; no point trying the remaining series.
                return []
            if series is None:
                continue

            rows.extend(self._map_series(series_id, series))

        if not rows:
            self.log.warning("[fred] no usable observations across all series")
        else:
            self.log.info(
                "[fred] fetched %d observations across %d series",
                len(rows), len(self.series),
            )
        return rows

    # ------------------------------------------------------------------
    async def _fetch_series(self, fred: Fred, series_id: str) -> pd.Series | None:
        """Fetch one series with retry; graceful ``None`` on fatal errors."""
        kwargs: dict[str, str] = {}
        if self.observation_start:
            kwargs["observation_start"] = self.observation_start

        try:
            async for attempt in self._retryer(self.RETRYABLE_ERRORS):
                with attempt:
                    attempt_no = attempt.retry_state.attempt_number
                    self.log.debug(
                        "[fred] get_series('%s') attempt %d/%d",
                        series_id, attempt_no, self.max_attempts,
                    )
                    # fredapi is blocking -> run in a worker thread.
                    return await asyncio.to_thread(
                        fred.get_series, series_id, **kwargs
                    )
        except ValueError as exc:
            # fredapi wraps FRED API errors (bad key, unknown series, 400s)
            # into ValueError with the server message.
            message = str(exc)
            if "api_key" in message.lower():
                self.log.warning(
                    "[fred] FRED rejected the API key — check FRED_API_KEY in "
                    ".env. Skipping FRED collection. Server said: %s", message,
                )
                raise _InvalidFredApiKeyError(message) from exc
            self.log.error(
                "[fred] series '%s' failed (not retryable): %s", series_id, message
            )
            return None
        except self.RETRYABLE_ERRORS as exc:
            self.log.error(
                "[fred] series '%s': network failure after %d attempts: %s",
                series_id, self.max_attempts, exc,
            )
            return None

        return None  # defensive; retry loop always returns or raises

    # ------------------------------------------------------------------
    def _map_series(self, series_id: str, series: pd.Series) -> list[MacroIndicator]:
        """Validate and map a pandas Series into ``MacroIndicator`` rows."""
        if series.empty:
            self.log.warning("[fred] series '%s' returned 0 observations", series_id)
            return []

        total = len(series)
        series = series.dropna()  # FRED uses NaN for holidays/gaps
        dropped = total - len(series)
        if dropped:
            self.log.debug(
                "[fred] series '%s': dropped %d NaN observation(s)", series_id, dropped
            )

        rows: list[MacroIndicator] = []
        skipped = 0
        for index, value in series.items():
            try:
                timestamp = pd.Timestamp(index).to_pydatetime().replace(tzinfo=UTC)
                rows.append(
                    MacroIndicator(
                        timestamp=timestamp,
                        source="FRED",
                        indicator=series_id,
                        value=Decimal(str(value)),
                    )
                )
            except (TypeError, ValueError, InvalidOperation) as exc:
                skipped += 1
                self.log.warning(
                    "[fred] skipping malformed observation %s=%r for '%s': %s",
                    index, value, series_id, exc,
                )
        if skipped:
            self.log.warning(
                "[fred] series '%s': skipped %d malformed observation(s)",
                series_id, skipped,
            )

        if rows:
            self.log.info(
                "[fred] series '%s': %d observations (%s -> %s)",
                series_id, len(rows),
                rows[0].timestamp.date(), rows[-1].timestamp.date(),
            )
        return rows

    # ------------------------------------------------------------------
    async def save_to_db(self, rows) -> int:
        """Idempotent upsert into ``macro_indicators``.

        ``value`` is refreshed on conflict because FRED revises historical
        observations (e.g. index recalculations).
        """
        return await self._upsert(MacroIndicator, rows, update_columns=("value",))

"""Market-index collector: Stooq primary, FRED fallback.

Fetches daily closes for the S&P 500 and Nasdaq 100 into the **wide-format**
``market_indices`` table (one row per trading day). Kept separate from
``macro_indicators`` so equity benchmarks never mix with FRED macro series.

ENVIRONMENT REALITY (verified 2026-07-08, see CLAUDE.md):
* ``pandas-datareader`` 0.11.1 — the only version compatible with this
  machine's Python 3.14 — **removed its Stooq reader entirely**
  (``DataReader(..., 'stooq')`` raises ``NotImplementedError``).
* ``stooq.com`` itself now sits behind a JavaScript proof-of-work anti-bot
  wall, so its CSV endpoint returns an HTML challenge page to any
  non-browser client. Bypassing an anti-bot wall is not something this
  project does.

Strategy, in order:
1. **Stooq direct CSV** via httpx (self-healing: works automatically if the
   wall is ever lifted or the machine moves to an unrestricted network).
   Non-CSV responses are detected and treated as "unavailable".
2. **FRED fallback** (series ``SP500`` and ``NASDAQ100``, ~10 years of
   history) via the already-proven ``fredapi`` + key from ``.env``. The
   fallback fills the exact same ``market_indices`` columns; the ``source``
   used is logged loudly so data lineage stays clear.

Research grounding: BTC's Nasdaq beta is the strongest daily-return driver
in the corpus (Krause 2026) — these series feed the Macro/Risk agents.
"""

from __future__ import annotations

import asyncio
import io
from datetime import UTC
from decimal import Decimal, InvalidOperation
from typing import ClassVar
from urllib.error import URLError
from xml.etree.ElementTree import ParseError

import httpx
import pandas as pd

from src.collectors.base import BaseDataFetcher, FetcherError, RetryableHTTPError
from src.config import Settings
from src.db.models import MarketIndex

#: Stooq symbol -> target column on ``market_indices``.
STOOQ_SYMBOLS: dict[str, str] = {
    "^spx": "sp500_close",
    "^ndx": "nasdaq_close",
}

#: FRED fallback series -> target column (FRED serves ~10y of daily closes).
FRED_FALLBACK_SERIES: dict[str, str] = {
    "SP500": "sp500_close",
    "NASDAQ100": "nasdaq_close",
}

DEFAULT_START = "2018-01-01"


class StooqFetcher(BaseDataFetcher):
    """Collects daily index closes into ``market_indices`` (Stooq -> FRED)."""

    source_name: ClassVar[str] = "stooq"

    STOOQ_CSV_URL: ClassVar[str] = "https://stooq.com/q/d/l/"

    # Fail fast on the fragile primary source so the FRED fallback kicks in
    # quickly (Stooq's anti-bot wall often manifests as connect timeouts).
    max_attempts: ClassVar[int] = 2
    http_timeout_seconds: ClassVar[float] = 10.0

    _FETCH_FAILURES: ClassVar[tuple[type[BaseException], ...]] = (
        FetcherError,
        RetryableHTTPError,
        httpx.HTTPError,
    )
    _FRED_RETRYABLE: ClassVar[tuple[type[BaseException], ...]] = (
        URLError,
        TimeoutError,
        ConnectionError,
        ParseError,
    )

    def __init__(
        self,
        *,
        start: str = DEFAULT_START,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.start = start

    # ------------------------------------------------------------------
    async def fetch_data(self) -> list[MarketIndex]:
        """Try Stooq for every symbol; fall back to FRED when Stooq is walled."""
        merged: dict[object, dict[str, Decimal]] = {}
        stooq_ok = True

        for symbol, column in STOOQ_SYMBOLS.items():
            self.log.info("[stooq] trying Stooq CSV for '%s' -> %s", symbol, column)
            observations = await self._fetch_stooq_symbol(symbol)
            if observations is None:
                stooq_ok = False
                break  # walled/unavailable — no point trying the next symbol
            for day, value in observations:
                merged.setdefault(day, {})[column] = value

        if not stooq_ok:
            self.log.warning(
                "[stooq] Stooq unavailable (anti-bot wall or network) — "
                "falling back to FRED series %s", sorted(FRED_FALLBACK_SERIES),
            )
            merged = await self._fetch_fred_fallback()

        if not merged:
            self.log.warning("[stooq] no usable observations from Stooq nor FRED")
            return []

        rows = [
            MarketIndex(timestamp=day, **columns)
            for day, columns in sorted(merged.items())
        ]
        self.log.info(
            "[stooq] fetched %d trading days via %s (%s -> %s)",
            len(rows), "Stooq" if stooq_ok else "FRED fallback",
            rows[0].timestamp.date(), rows[-1].timestamp.date(),
        )
        return rows

    # ------------------------------------------------------------------
    # Primary source: Stooq direct CSV
    # ------------------------------------------------------------------
    async def _fetch_stooq_symbol(self, symbol: str) -> list[tuple] | None:
        """Fetch one symbol from Stooq; ``None`` when the source is unusable."""
        start_compact = self.start.replace("-", "")
        try:
            content = await self._request_bytes(
                self.STOOQ_CSV_URL,
                params={"s": symbol, "i": "d", "d1": start_compact},
                headers={"Accept": "text/csv"},
            )
        except self._FETCH_FAILURES as exc:
            self.log.warning("[stooq] request for '%s' failed: %s", symbol, exc)
            return None
        if not content:
            return None

        head = content[:256].lstrip().lower()
        if head.startswith(b"<") or b"<html" in head:
            self.log.warning(
                "[stooq] '%s': received an HTML page instead of CSV — Stooq's "
                "anti-bot wall is active", symbol,
            )
            return None

        try:
            frame = pd.read_csv(io.BytesIO(content))
        except (pd.errors.ParserError, ValueError) as exc:
            self.log.warning("[stooq] '%s': unparseable CSV: %s", symbol, exc)
            return None
        if frame.empty or "Close" not in frame.columns or "Date" not in frame.columns:
            self.log.warning(
                "[stooq] '%s': unexpected CSV shape (columns=%s)",
                symbol, list(frame.columns),
            )
            return None

        observations: list[tuple] = []
        skipped = 0
        for record in frame.itertuples(index=False):
            try:
                value = Decimal(str(record.Close))
                if value <= 0:
                    raise ValueError(f"non-positive close {value}")
                day = (
                    pd.Timestamp(record.Date)
                    .to_pydatetime()
                    .replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)
                )
                observations.append((day, value))
            except (TypeError, ValueError, InvalidOperation) as exc:
                skipped += 1
                if skipped <= 5:
                    self.log.warning(
                        "[stooq] '%s': skipping observation %r: %s", symbol, record, exc
                    )
        if skipped:
            self.log.warning("[stooq] '%s': skipped %d observation(s)", symbol, skipped)
        self.log.info("[stooq] '%s': %d observations via Stooq", symbol, len(observations))
        return observations

    # ------------------------------------------------------------------
    # Fallback source: FRED (SP500 / NASDAQ100)
    # ------------------------------------------------------------------
    async def _fetch_fred_fallback(self) -> dict[object, dict[str, Decimal]]:
        api_key = self.settings.fred_api_key.get_secret_value().strip()
        if not api_key:
            self.log.warning(
                "[stooq] FRED fallback needs FRED_API_KEY (empty in .env) — "
                "market indices unavailable this run."
            )
            return {}

        from fredapi import Fred  # deferred: only needed on the fallback path

        fred = Fred(api_key=api_key)
        merged: dict[object, dict[str, Decimal]] = {}

        for series_id, column in FRED_FALLBACK_SERIES.items():
            series = await self._fetch_fred_series(fred, series_id)
            if series is None:
                continue
            total = len(series)
            series = series.dropna()
            self.log.info(
                "[stooq] FRED '%s': %d observations (%d NaN dropped)",
                series_id, len(series), total - len(series),
            )
            for index, value in series.items():
                try:
                    close = Decimal(str(value))
                    if close <= 0:
                        raise ValueError(f"non-positive close {close}")
                    day = (
                        pd.Timestamp(index)
                        .to_pydatetime()
                        .replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)
                    )
                    merged.setdefault(day, {})[column] = close
                except (TypeError, ValueError, InvalidOperation) as exc:
                    self.log.warning(
                        "[stooq] FRED '%s': skipping %s=%r: %s",
                        series_id, index, value, exc,
                    )
        return merged

    async def _fetch_fred_series(self, fred, series_id: str):
        """One FRED series with retry; graceful ``None`` on failure."""
        try:
            async for attempt in self._retryer(self._FRED_RETRYABLE):
                with attempt:
                    self.log.debug(
                        "[stooq] FRED get_series('%s') attempt %d/%d",
                        series_id,
                        attempt.retry_state.attempt_number,
                        self.max_attempts,
                    )
                    return await asyncio.to_thread(
                        fred.get_series, series_id, observation_start=self.start
                    )
        except ValueError as exc:
            self.log.error("[stooq] FRED series '%s' failed: %s", series_id, exc)
            return None
        except self._FRED_RETRYABLE as exc:
            self.log.error(
                "[stooq] FRED series '%s': network failure after %d attempts: %s",
                series_id, self.max_attempts, exc,
            )
            return None
        return None

    # ------------------------------------------------------------------
    async def save_to_db(self, rows) -> int:
        """Idempotent upsert into ``market_indices`` (closes refreshed)."""
        return await self._upsert(
            MarketIndex, rows, update_columns=("sp500_close", "nasdaq_close")
        )

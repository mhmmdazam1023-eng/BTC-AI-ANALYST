"""Base class for all data collectors.

Contract (per project rules in CLAUDE.md):

* Reusable async HTTP session (:class:`httpx.AsyncClient`) shared by all
  HTTP-based fetchers. SDK-based fetchers (ccxt, fredapi — mandated by the
  tech stack) manage their own transport but reuse the same retry policy.
* Retry with **tenacity exponential backoff**, specifically for rate limits
  (HTTP 429 — honouring ``Retry-After`` when the server sends it), timeouts,
  and server errors (HTTP 5xx). Client errors (4xx) fail fast.
* Abstract :meth:`fetch_data` / :meth:`save_to_db` that every fetcher must
  implement, plus a shared idempotent **upsert** helper (safe re-runs: same
  natural key never duplicates, values are refreshed).
* Specific logging on every attempt and every failure path — never a bare
  ``pass`` or ``print``.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Sequence
from typing import Any, ClassVar

import httpx
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import Base

logger = logging.getLogger(__name__)

#: Maximum number of rows per INSERT statement. asyncpg caps bind parameters
#: at 32767; 2000 rows x <=10 columns stays comfortably below that limit.
_UPSERT_CHUNK_SIZE = 2000

#: Transport-level httpx errors that are always worth retrying.
RETRYABLE_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.TransportError,
)


class FetcherError(RuntimeError):
    """Non-retryable fetcher failure (bad request, unexpected schema, ...)."""


class RetryableHTTPError(RuntimeError):
    """Transient HTTP failure (429 / 5xx) that should be retried.

    Carries the server-provided ``Retry-After`` delay (seconds) when present
    so the backoff strategy can honour it instead of hammering the API.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class BaseDataFetcher(abc.ABC):
    """Abstract base for every external-API data collector."""

    #: Short identifier used in logs and (usually) in the ``source`` column.
    source_name: ClassVar[str] = "base"

    # --- retry policy defaults (subclasses may override) ---
    max_attempts: ClassVar[int] = 5
    backoff_min_seconds: ClassVar[float] = 1.0
    backoff_max_seconds: ClassVar[float] = 30.0
    #: Hard cap when honouring a server-sent ``Retry-After`` header.
    retry_after_cap_seconds: ClassVar[float] = 120.0
    #: Default timeout for the shared httpx client.
    http_timeout_seconds: ClassVar[float] = 30.0

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._http: httpx.AsyncClient | None = None
        self.log = logging.getLogger(f"src.collectors.{self.source_name}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abc.abstractmethod
    async def fetch_data(self) -> list[Base]:
        """Fetch from the external API and return validated ORM rows.

        Must return ``[]`` (with a logged warning) when the API yields no
        usable data, and raise :class:`FetcherError` on unrecoverable
        failures after the retry budget is exhausted.
        """

    @abc.abstractmethod
    async def save_to_db(self, rows: Sequence[Base]) -> int:
        """Persist rows into the warehouse; return the number of rows written.

        Implementations should delegate to :meth:`_upsert` so that re-running
        a collection window is idempotent.
        """

    # ------------------------------------------------------------------
    # Orchestration convenience
    # ------------------------------------------------------------------
    async def run(self) -> int:
        """Fetch then save; returns the number of rows persisted."""
        self.log.info("[%s] collection run started", self.source_name)
        rows = await self.fetch_data()
        if not rows:
            self.log.warning("[%s] no rows fetched — nothing to persist", self.source_name)
            return 0
        saved = await self.save_to_db(rows)
        self.log.info("[%s] collection run finished: %d rows persisted", self.source_name, saved)
        return saved

    # ------------------------------------------------------------------
    # Shared HTTP session (httpx.AsyncClient, reusable)
    # ------------------------------------------------------------------
    @property
    def http(self) -> httpx.AsyncClient:
        """Lazily-created, reusable async HTTP client."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.http_timeout_seconds),
                headers={"User-Agent": "bitcoin-ai-analyst/0.1"},
                follow_redirects=True,
            )
            self.log.debug("[%s] httpx.AsyncClient created", self.source_name)
        return self._http

    # ------------------------------------------------------------------
    # Retry machinery (tenacity)
    # ------------------------------------------------------------------
    def _wait_strategy(self):
        """Exponential backoff that also honours server ``Retry-After`` hints."""
        exponential = wait_exponential(
            multiplier=1,
            min=self.backoff_min_seconds,
            max=self.backoff_max_seconds,
        )

        def _wait(retry_state: RetryCallState) -> float:
            delay = exponential(retry_state)
            outcome = retry_state.outcome
            exc = outcome.exception() if outcome is not None and outcome.failed else None
            retry_after = getattr(exc, "retry_after", None)
            if retry_after:
                delay = max(delay, min(float(retry_after), self.retry_after_cap_seconds))
            return delay

        return _wait

    def _retryer(
        self, retry_on: tuple[type[BaseException], ...]
    ) -> AsyncRetrying:
        """Build the tenacity retry loop used by all fetch operations."""
        return AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=self._wait_strategy(),
            retry=retry_if_exception_type(retry_on),
            before_sleep=before_sleep_log(self.log, logging.WARNING),
            reraise=True,
        )

    async def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """GET ``url`` and return parsed JSON, with full retry + validation.

        Retries: transport errors/timeouts, HTTP 429 (honouring Retry-After),
        HTTP 5xx. Fails fast on other 4xx and on non-JSON bodies.
        """
        retry_on = RETRYABLE_TRANSPORT_ERRORS + (RetryableHTTPError,)
        async for attempt in self._retryer(retry_on):
            with attempt:
                attempt_no = attempt.retry_state.attempt_number
                self.log.debug(
                    "[%s] GET %s (attempt %d/%d)",
                    self.source_name, url, attempt_no, self.max_attempts,
                )
                try:
                    response = await self.http.get(url, params=params, headers=headers)
                except RETRYABLE_TRANSPORT_ERRORS as exc:
                    self.log.warning(
                        "[%s] transport error on GET %s (attempt %d/%d): %s",
                        self.source_name, url, attempt_no, self.max_attempts, exc,
                    )
                    raise

                # ---- HTTP status validation ----
                status = response.status_code
                if status == 429 or status >= 500:
                    header = response.headers.get("Retry-After", "")
                    retry_after = float(header) if header.replace(".", "", 1).isdigit() else None
                    self.log.warning(
                        "[%s] HTTP %d on GET %s (attempt %d/%d, Retry-After=%s)",
                        self.source_name, status, url, attempt_no, self.max_attempts, header or "n/a",
                    )
                    raise RetryableHTTPError(
                        f"[{self.source_name}] HTTP {status} for {url}",
                        status_code=status,
                        retry_after=retry_after,
                    )
                if status >= 400:
                    raise FetcherError(
                        f"[{self.source_name}] HTTP {status} for {url}: "
                        f"{response.text[:300]}"
                    )

                # ---- Body validation ----
                try:
                    return response.json()
                except ValueError as exc:
                    raise FetcherError(
                        f"[{self.source_name}] non-JSON body from {url}: "
                        f"{response.text[:300]}"
                    ) from exc

        # Unreachable: AsyncRetrying(reraise=True) returns or raises above.
        raise FetcherError(f"[{self.source_name}] retry loop exhausted for {url}")

    async def _request_bytes(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_404: bool = False,
    ) -> bytes | None:
        """GET ``url`` and return the raw body, with the same retry policy.

        Used for binary archives (e.g. Binance Vision ZIP files). With
        ``allow_404=True`` a missing resource returns ``None`` instead of
        raising — expected for archive files not yet published.
        """
        retry_on = RETRYABLE_TRANSPORT_ERRORS + (RetryableHTTPError,)
        async for attempt in self._retryer(retry_on):
            with attempt:
                attempt_no = attempt.retry_state.attempt_number
                self.log.debug(
                    "[%s] GET(bytes) %s (attempt %d/%d)",
                    self.source_name, url, attempt_no, self.max_attempts,
                )
                try:
                    response = await self.http.get(url, params=params, headers=headers)
                except RETRYABLE_TRANSPORT_ERRORS as exc:
                    self.log.warning(
                        "[%s] transport error on GET %s (attempt %d/%d): %s",
                        self.source_name, url, attempt_no, self.max_attempts, exc,
                    )
                    raise

                status = response.status_code
                if status == 404 and allow_404:
                    self.log.debug("[%s] 404 (allowed) for %s", self.source_name, url)
                    return None
                if status == 429 or status >= 500:
                    header = response.headers.get("Retry-After", "")
                    retry_after = float(header) if header.replace(".", "", 1).isdigit() else None
                    self.log.warning(
                        "[%s] HTTP %d on GET %s (attempt %d/%d, Retry-After=%s)",
                        self.source_name, status, url, attempt_no, self.max_attempts, header or "n/a",
                    )
                    raise RetryableHTTPError(
                        f"[{self.source_name}] HTTP {status} for {url}",
                        status_code=status,
                        retry_after=retry_after,
                    )
                if status >= 400:
                    raise FetcherError(
                        f"[{self.source_name}] HTTP {status} for {url}: "
                        f"{response.text[:300]}"
                    )
                return response.content

        raise FetcherError(f"[{self.source_name}] retry loop exhausted for {url}")

    # ------------------------------------------------------------------
    # Idempotent persistence helper
    # ------------------------------------------------------------------
    async def _upsert(
        self,
        model: type[Base],
        rows: Sequence[Base],
        *,
        update_columns: Sequence[str],
    ) -> int:
        """Idempotent bulk upsert keyed on the model's composite primary key.

        Existing rows get their ``update_columns`` (and ``ingested_at``)
        refreshed — correct for still-forming candles and for FRED's
        historical revisions.
        """
        if not rows:
            self.log.warning("[%s] _upsert called with 0 rows — skipping", self.source_name)
            return 0

        table = model.__table__
        pk_columns = [c.name for c in table.primary_key.columns]
        data_columns = [c.name for c in table.columns if c.name != "ingested_at"]
        payload = [
            {name: getattr(row, name) for name in data_columns} for row in rows
        ]

        written = 0
        async with session_scope() as session:
            for start in range(0, len(payload), _UPSERT_CHUNK_SIZE):
                chunk = payload[start : start + _UPSERT_CHUNK_SIZE]
                stmt = pg_insert(table).values(chunk)
                set_ = {col: stmt.excluded[col] for col in update_columns}
                set_["ingested_at"] = func.now()
                stmt = stmt.on_conflict_do_update(
                    index_elements=pk_columns, set_=set_
                )
                await session.execute(stmt)
                written += len(chunk)
                self.log.debug(
                    "[%s] upserted chunk %d-%d into '%s'",
                    self.source_name, start, start + len(chunk), table.name,
                )

        self.log.info(
            "[%s] upserted %d rows into '%s' (conflict target: %s)",
            self.source_name, written, table.name, pk_columns,
        )
        return written

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def close(self) -> None:
        """Release network resources. Subclasses must call ``super().close()``."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
            self.log.debug("[%s] httpx.AsyncClient closed", self.source_name)
        self._http = None

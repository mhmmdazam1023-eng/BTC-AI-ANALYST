"""Binance OHLCV collector (``ccxt.async_support`` — mandated tech stack).

Two classes, one transport:

* :class:`BinanceFetcher` — one configured spot pair (default ``BTC/USDT``).
  Unchanged since Phase 2; Phase 1-7 keep using it as-is.
* :class:`BinanceMultiFetcher` — Phase 8. Reads the symbol list from
  ``coins_registry`` and ingests them **hierarchically** (BTC, then ETH,
  then altcoins), reusing the parent's ccxt client, throttler, retry policy
  and validation.

Both map candles to the ``crypto_prices`` hypertable.

Environment workarounds (verified on this machine, 2026-07-08):
* ``api.binance.com`` is ISP-blocked in Indonesia (DNS hijack — TLS cert
  mismatch). Public market data is therefore fetched from
  ``data-api.binance.vision``, Binance's official public-data host
  (configurable via ``BINANCE_PUBLIC_API_URL`` in ``.env``). Markets are
  restricted to **spot** so ccxt never touches the blocked futures hosts
  (``fapi``/``dapi``).
* ccxt's default aiodns/pycares resolver cannot read this machine's DNS
  configuration (``DNSError: Could not contact DNS servers``); the exchange
  is therefore given an aiohttp session using the **system resolver**
  (``ThreadedResolver``) with certifi's CA bundle — the same trust path
  httpx uses.

Resilience:
* ccxt's built-in throttler is enabled (``enableRateLimit``) so requests are
  spaced out client-side — first line of defence against bans.
* tenacity retries transient ``ccxt.NetworkError`` (covers request timeouts,
  HTTP 5xx via ``ExchangeNotAvailable``, and 429 via ``RateLimitExceeded``)
  with exponential backoff. ``ccxt.ExchangeError`` (bad symbol, geo-block)
  fails fast with a clear message.
* Malformed candles are skipped (and logged) instead of poisoning the batch.
"""

from __future__ import annotations

import ssl
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import ClassVar

import aiohttp
import certifi
import ccxt.async_support as ccxt_async
from sqlalchemy import func, select

from src.collectors.base import BaseDataFetcher, FetcherError
from src.config import Settings
from src.db.engine import session_scope
from src.db.models import (
    BASE_SYMBOL,
    HIERARCHY_HEAD,
    INTRADAY_TIMEFRAME,
    CoinRegistry,
    CryptoPrice,
)
from src.timeframes import bar_label, floor_to_bar

#: Binance caps klines requests at 1000 candles per call.
_BINANCE_MAX_LIMIT = 1000

#: Safety cap on ``since`` pagination: 1000 daily candles/page, so 12 pages
#: is ~33 years of daily history — far beyond Bitcoin's existence, while
#: still guaranteeing the loop terminates.
_MAX_BACKFILL_PAGES = 12


class BinanceFetcher(BaseDataFetcher):
    """Collects OHLCV candles from Binance spot via ccxt."""

    source_name: ClassVar[str] = "binance"

    #: ccxt exception classes worth retrying. NetworkError covers
    #: RequestTimeout, ExchangeNotAvailable (5xx), DDoSProtection and
    #: RateLimitExceeded (429).
    RETRYABLE_CCXT_ERRORS: ClassVar[tuple[type[BaseException], ...]] = (
        ccxt_async.NetworkError,
    )

    def __init__(
        self,
        *,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        limit: int = 365,
        since: str | None = None,
        drop_unclosed: bool = False,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        if not 1 <= limit <= _BINANCE_MAX_LIMIT:
            raise ValueError(
                f"limit must be between 1 and {_BINANCE_MAX_LIMIT}, got {limit}"
            )
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        #: Phase 8 Part 8 — drop the still-forming bar before saving.
        #:
        #: Default False keeps Phase 1-7 behaviour EXACTLY: the daily pipeline
        #: depends on today's forming candle being stored (``daily_features``
        #: refreshes it each run, and the pre-Phase-8 snapshot comparison
        #: confirmed that is by design). Only the intraday layer wants the
        #: opposite guarantee.
        #:
        #: Restricted to the intraday timeframe because the cutoff is computed
        #: with 4-hour bar arithmetic; silently applying it to '1d' would floor
        #: to a 4h boundary and drop candles that are in fact complete.
        if drop_unclosed and timeframe != INTRADAY_TIMEFRAME:
            raise ValueError(
                f"drop_unclosed is only supported for timeframe="
                f"'{INTRADAY_TIMEFRAME}' (4-hour bar arithmetic), got "
                f"'{timeframe}'"
            )
        self.drop_unclosed = drop_unclosed
        #: ISO date (``YYYY-MM-DD``). When set, candles are PAGINATED from
        #: this date to now instead of taking the last ``limit`` candles.
        #: Binance caps a single klines call at 1000 candles, so deep
        #: history (Phase 8 regime backtests need 2021-2022) is only
        #: reachable by walking forward with ``since``. Leaving it None
        #: preserves the exact Phase 2 behaviour.
        self.since = since
        self._exchange: ccxt_async.binance | None = None

    # ------------------------------------------------------------------
    @property
    def exchange(self) -> ccxt_async.binance:
        """Lazily-created ccxt async exchange with client-side rate limiting.

        Must be first accessed from within a running event loop (aiohttp
        sessions are loop-bound) — which is always the case here, since it is
        only used inside :meth:`fetch_data`.
        """
        if self._exchange is None:
            exchange = ccxt_async.binance(
                {
                    "enableRateLimit": True,  # ccxt throttler — avoid bans
                    # Spot-only: prevents load_markets() from calling the
                    # ISP-blocked futures hosts (fapi/dapi.binance.com).
                    "options": {"defaultType": "spot", "fetchMarkets": ["spot"]},
                }
            )
            # System DNS resolver + certifi CAs: ccxt's default aiodns
            # resolver fails on this machine ("Could not contact DNS
            # servers"). ccxt reuses this session and closes it in .close().
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            exchange.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    resolver=aiohttp.ThreadedResolver(), ssl=ssl_context
                )
            )
            public_api_url = self.settings.binance_public_api_url.strip()
            if public_api_url:
                exchange.urls["api"]["public"] = public_api_url
                self.log.info(
                    "[binance] using public market-data host: %s", public_api_url
                )
            self._exchange = exchange
            self.log.debug("[binance] ccxt async client created (spot-only)")
        return self._exchange

    # ------------------------------------------------------------------
    async def _fetch_page(
        self, symbol: str, *, since_ms: int | None, limit: int
    ) -> list[list[float]]:
        """One retry-wrapped ``fetch_ohlcv`` call. Raises FetcherError."""
        raw: list[list[float]] = []
        try:
            async for attempt in self._retryer(self.RETRYABLE_CCXT_ERRORS):
                with attempt:
                    attempt_no = attempt.retry_state.attempt_number
                    self.log.debug(
                        "[binance] fetch_ohlcv %s since=%s attempt %d/%d",
                        symbol, since_ms, attempt_no, self.max_attempts,
                    )
                    raw = await self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe=self.timeframe,
                        since=since_ms,
                        limit=limit,
                    )
        except ccxt_async.ExchangeError as exc:
            # Non-retryable: unknown symbol, geo restriction, etc.
            raise FetcherError(
                f"[binance] exchange error (not retryable) for "
                f"{symbol} {self.timeframe}: {exc}"
            ) from exc
        except ccxt_async.NetworkError as exc:
            raise FetcherError(
                f"[binance] network failure after {self.max_attempts} attempts "
                f"for {symbol} {self.timeframe}: {exc}"
            ) from exc
        return raw

    async def _fetch_paginated(self, symbol: str) -> list[list[float]]:
        """Walk forward from ``self.since`` in 1000-candle pages.

        Termination is guarded three ways so a misbehaving endpoint can
        never spin forever: a short page ends the walk, the cursor must
        strictly advance, and :data:`_MAX_BACKFILL_PAGES` caps the total.
        """
        start = datetime.fromisoformat(self.since).replace(tzinfo=UTC)
        cursor = int(start.timestamp() * 1000)
        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        self.log.info(
            "[binance] paginating %s %s from %s",
            symbol, self.timeframe, start.date(),
        )

        collected: list[list[float]] = []
        for page in range(1, _MAX_BACKFILL_PAGES + 1):
            batch = await self._fetch_page(
                symbol, since_ms=cursor, limit=_BINANCE_MAX_LIMIT
            )
            if not batch:
                break
            collected.extend(batch)
            last_ts = int(batch[-1][0])
            self.log.debug(
                "[binance] %s page %d: %d candles up to %s",
                symbol, page, len(batch),
                datetime.fromtimestamp(last_ts / 1000, tz=UTC).date(),
            )
            if len(batch) < _BINANCE_MAX_LIMIT or last_ts >= now_ms:
                break
            if last_ts + 1 <= cursor:  # cursor must strictly advance
                self.log.warning(
                    "[binance] %s pagination cursor stalled at %s — stopping",
                    symbol, cursor,
                )
                break
            cursor = last_ts + 1
        else:
            self.log.warning(
                "[binance] %s hit the %d-page backfill cap — history may be "
                "incomplete; re-run with a later --since",
                symbol, _MAX_BACKFILL_PAGES,
            )
        return collected

    # ------------------------------------------------------------------
    async def fetch_symbol(self, symbol: str) -> list[CryptoPrice]:
        """Fetch + validate one symbol's candles (the reusable unit of work).

        Split out of :meth:`fetch_data` so the multi-asset collector can call
        it per symbol while sharing this class's ccxt client, throttler,
        retry policy and validation rules.
        """
        self.log.info(
            "[binance] fetching %s %s OHLCV (limit=%d)",
            symbol, self.timeframe, self.limit,
        )

        if self.since is None:
            raw = await self._fetch_page(symbol, since_ms=None, limit=self.limit)
        else:
            raw = await self._fetch_paginated(symbol)

        # ---- Response validation & mapping ----
        if not raw:
            self.log.warning(
                "[binance] empty OHLCV response for %s %s — returning no rows",
                symbol, self.timeframe,
            )
            return []
        if not isinstance(raw, list):
            raise FetcherError(
                f"[binance] unexpected OHLCV payload type: {type(raw).__name__}"
            )

        rows: list[CryptoPrice] = []
        skipped = 0
        for candle in raw:
            try:
                if not isinstance(candle, (list, tuple)) or len(candle) < 6:
                    raise ValueError("malformed candle structure")
                ts_ms, o, h, low, c, v = candle[:6]
                if any(field is None for field in (ts_ms, o, h, low, c, v)):
                    raise ValueError("candle contains null fields")
                rows.append(
                    CryptoPrice(
                        timestamp=datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC),
                        exchange="binance",
                        symbol=symbol,
                        timeframe=self.timeframe,
                        open=Decimal(str(o)),
                        high=Decimal(str(h)),
                        low=Decimal(str(low)),
                        close=Decimal(str(c)),
                        volume=Decimal(str(v)),
                    )
                )
            except (TypeError, ValueError, InvalidOperation, OverflowError) as exc:
                skipped += 1
                self.log.warning("[binance] skipping malformed candle %r: %s", candle, exc)

        if skipped:
            self.log.warning(
                "[binance] %s: skipped %d malformed candle(s)", symbol, skipped
            )
        if not rows:
            self.log.warning(
                "[binance] %s: all candles were malformed — returning no rows", symbol
            )
            return []

        if self.drop_unclosed:
            rows = self._keep_closed_bars(rows, symbol)
            if not rows:
                return []

        self.log.info(
            "[binance] fetched %d valid candles for %s %s (range %s -> %s)",
            len(rows), symbol, self.timeframe,
            rows[0].timestamp.date(), rows[-1].timestamp.date(),
        )
        return rows

    # ------------------------------------------------------------------
    def _keep_closed_bars(
        self, rows: list[CryptoPrice], symbol: str
    ) -> list[CryptoPrice]:
        """Drop bars that are still forming (Phase 8 Part 8, defence layer 1).

        ccxt returns the in-progress bar as the last element. Its high/low are
        "so far", not final, so storing it would let a later reader treat a
        partial range as a complete one — the bug class already fixed twice in
        this project. Filtering here means ``crypto_prices`` at the intraday
        timeframe is clean for EVERY future reader, not just the ones that
        remember to filter.

        Consequence, stated plainly: intraday data lags real time by up to
        4 hours. That is the price of "closed only".
        """
        cutoff = floor_to_bar(datetime.now(tz=UTC))
        closed = [row for row in rows if row.timestamp < cutoff]
        dropped = len(rows) - len(closed)

        if dropped:
            self.log.info(
                "[binance] %s %s: dropped %d still-forming bar(s) at/after %s "
                "— only CLOSED bars are stored",
                symbol, self.timeframe, dropped, bar_label(cutoff),
            )
        if not closed:
            self.log.warning(
                "[binance] %s %s: every returned bar is still forming "
                "(cutoff %s) — returning no rows",
                symbol, self.timeframe, bar_label(cutoff),
            )
        return closed

    async def fetch_data(self) -> list[CryptoPrice]:
        """Fetch OHLCV candles for this fetcher's single configured symbol."""
        return await self.fetch_symbol(self.symbol)

    # ------------------------------------------------------------------
    async def save_to_db(self, rows) -> int:
        """Idempotent upsert into ``crypto_prices``.

        OHLCV values are refreshed on conflict because the most recent daily
        candle is still forming when fetched intraday.
        """
        return await self._upsert(
            CryptoPrice,
            rows,
            update_columns=("open", "high", "low", "close", "volume"),
        )

    # ------------------------------------------------------------------
    async def close(self) -> None:
        """Close the ccxt aiohttp session, then the shared httpx client."""
        if self._exchange is not None:
            try:
                await self._exchange.close()
                self.log.debug("[binance] ccxt client closed")
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                self.log.warning("[binance] error while closing ccxt client: %s", exc)
            finally:
                self._exchange = None
        await super().close()


class BinanceMultiFetcher(BinanceFetcher):
    """Registry-driven multi-asset collector (Phase 8).

    Reads the symbol list from ``coins_registry`` (``is_active = TRUE``)
    instead of hard-coding ``BTC/USDT``, and writes every asset into the same
    ``crypto_prices`` hypertable.

    Hierarchical ingestion (mandatory rule)
    ---------------------------------------
    Symbols are fetched **sequentially** in a fixed order: ``BTC/USDT``
    first, then ``ETH/USDT``, then the remaining altcoins alphabetically.
    This matters because BTC is the denominator of every altcoin feature
    (``price_btc_ratio``, ``rs_7d``, ``rs_30d``) — the majors must be the
    ones that get the rate-limit budget and the freshest candles, never an
    altcoin. Sequential is deliberate: running them concurrently would make
    the ordering meaningless and multiply the request burst.

    ``BTC/USDT`` is force-included even if it is missing or deactivated in
    the registry. Bitcoin is this system's numéraire and Phases 1-7 depend
    on it; deactivating it would break the existing product, so the fetcher
    refuses to honour that and says so loudly. ETH carries no such
    guarantee — it is fetched only when registered and active.

    Partial-failure policy: strict, with a bounded fallback
    -------------------------------------------------------
    A symbol is **load-bearing** when other symbols' features are divided by
    it — i.e. it is :data:`BASE_SYMBOL`, or some active coin declares it as
    its ``base_asset``. Today that set is exactly ``{BTC/USDT}``. It is
    derived from the registry rather than hard-coded so that if Part 2
    benchmarks an L2 against ETH, ETH becomes load-bearing automatically.

    When a load-bearing symbol's live fetch fails, the run does NOT silently
    degrade. It escalates in three steps:

    1. Look for that symbol's most recent stored candle in ``crypto_prices``.
    2. If it is at most :attr:`base_fallback_max_age_days` old, log a WARNING,
       record it in :attr:`fallback_symbols`, and continue — the altcoins'
       relative features will be computed against that recent-but-stored
       price, which is a known, bounded compromise.
    3. If there is no stored candle, or it is staler than that budget, raise
       :class:`FetcherError` (a ``RuntimeError``) and abort the collection.
       Computing relative strength against a price of unknown age is worse
       than collecting nothing: the numbers would look perfectly valid.

    Non-load-bearing symbols (ETH today, and every altcoin) stay isolated:
    a failure is logged and recorded in :attr:`failed_symbols`, and the run
    continues. Nothing divides by them, so losing one costs only that asset.

    Because ``BASE_SYMBOL`` is fetched first, the abort path normally fires
    before any altcoin request is made.
    """

    #: When True (default, per user decision 2026-07-28), a load-bearing
    #: symbol that fails BOTH the live fetch AND the stored-candle fallback
    #: aborts the whole collection instead of producing altcoin features
    #: against an unknown-age denominator.
    abort_if_head_fails: ClassVar[bool] = True

    #: How stale a stored candle may be and still serve as the denominator
    #: fallback. 1 = today's or yesterday's candle is acceptable.
    base_fallback_max_age_days: ClassVar[int] = 1

    def __init__(
        self,
        *,
        symbols: list[str] | None = None,
        timeframe: str = "1d",
        limit: int = 365,
        since: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(
            symbol=BASE_SYMBOL,
            timeframe=timeframe,
            limit=limit,
            since=since,
            settings=settings,
        )
        #: Explicit override; when None the list comes from coins_registry.
        self._symbols_override = symbols
        #: Rows fetched per symbol, populated during :meth:`fetch_data`.
        self.symbol_stats: dict[str, int] = {}
        #: Symbols whose fetch failed this run (surfaced in the summary).
        self.failed_symbols: list[str] = []
        #: Load-bearing symbols served from the stored-candle fallback
        #: instead of a live fetch (surfaced in the summary).
        self.fallback_symbols: list[str] = []

    # ------------------------------------------------------------------
    async def load_active_symbols(
        self, *, denominators: set[str] | None = None
    ) -> list[str]:
        """Active symbols from ``coins_registry``, in hierarchical order."""
        if self._symbols_override is not None:
            active = list(self._symbols_override)
            self.log.info("[binance] using %d explicit symbol(s)", len(active))
        else:
            async with session_scope() as session:
                active = list(
                    (
                        await session.execute(
                            select(CoinRegistry.symbol)
                            .where(CoinRegistry.is_active.is_(True))
                            .order_by(CoinRegistry.symbol)
                        )
                    ).scalars().all()
                )
            self.log.info(
                "[binance] coins_registry returned %d active symbol(s)", len(active)
            )

        if BASE_SYMBOL not in active:
            self.log.warning(
                "[binance] %s is not in the active symbol list — force-including "
                "it: it is the numéraire every altcoin feature divides by, and "
                "Phases 1-7 depend on it",
                BASE_SYMBOL,
            )
            active.append(BASE_SYMBOL)

        ordered = self._order_hierarchically(active, denominators)
        self.log.info("[binance] ingestion order: %s", " -> ".join(ordered))
        return ordered

    @staticmethod
    def _order_hierarchically(
        symbols: list[str], denominators: set[str] | None = None
    ) -> list[str]:
        """BTC, then ETH, then any other denominator, then the rest (A-Z).

        Denominators are pulled forward so that if one of them is going to
        abort the run, it does so before altcoin requests are spent.
        """
        unique = set(symbols)
        head = [symbol for symbol in HIERARCHY_HEAD if symbol in unique]
        extra = sorted((denominators or set()) & unique - set(head))
        tail = sorted(unique - set(head) - set(extra))
        return head + extra + tail

    # ------------------------------------------------------------------
    async def load_required_denominators(self) -> set[str]:
        """Symbols other symbols' features are divided by ("load-bearing").

        Always contains :data:`BASE_SYMBOL`. Also contains every distinct
        ``base_asset`` declared by an active coin, so the strict guard
        follows the registry instead of a hard-coded list.
        """
        if self._symbols_override is not None:
            return {BASE_SYMBOL}
        async with session_scope() as session:
            declared = (
                await session.execute(
                    select(CoinRegistry.base_asset)
                    .where(
                        CoinRegistry.is_active.is_(True),
                        CoinRegistry.base_asset.is_not(None),
                    )
                    .distinct()
                )
            ).scalars().all()
        return {BASE_SYMBOL, *declared}

    async def stored_candle_age_days(self, symbol: str) -> tuple[datetime | None, int]:
        """Newest stored candle for ``symbol`` and its age in whole days.

        Age is measured from today's UTC midnight, matching how daily candle
        timestamps are stored: today's candle -> 0, yesterday's -> 1.
        Returns ``(None, -1)`` when the symbol has no stored candle at all.
        """
        async with session_scope() as session:
            latest = (
                await session.execute(
                    select(func.max(CryptoPrice.timestamp)).where(
                        CryptoPrice.exchange == "binance",
                        CryptoPrice.symbol == symbol,
                        CryptoPrice.timeframe == self.timeframe,
                    )
                )
            ).scalar_one_or_none()
        if latest is None:
            return None, -1
        today_midnight = datetime.now(tz=UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return latest, (today_midnight - latest).days

    async def _handle_denominator_failure(
        self, symbol: str, exc: BaseException
    ) -> None:
        """Fallback-or-abort for a load-bearing symbol. May raise."""
        stored_at, age_days = await self.stored_candle_age_days(symbol)

        if stored_at is not None and 0 <= age_days <= self.base_fallback_max_age_days:
            self.fallback_symbols.append(symbol)
            self.log.warning(
                "[binance] %s live fetch FAILED (%s: %s) — falling back to the "
                "stored candle from %s (%d day(s) old, budget %d). Altcoin "
                "relative features will be computed against that price.",
                symbol, type(exc).__name__, exc, stored_at.date(), age_days,
                self.base_fallback_max_age_days,
            )
            return

        reason = (
            f"no stored candle at all for {symbol}"
            if stored_at is None
            else f"stored candle from {stored_at.date()} is {age_days} day(s) "
            f"old, past the {self.base_fallback_max_age_days}-day budget"
        )
        if not self.abort_if_head_fails:
            self.log.error(
                "[binance] %s unavailable live AND via fallback (%s) — "
                "continuing anyway because abort_if_head_fails is disabled; "
                "downstream relative features will be unreliable",
                symbol, reason,
            )
            return

        self.log.error(
            "[binance] ABORTING collection: %s unavailable live (%s: %s) AND "
            "via DB fallback (%s). Refusing to compute relative features "
            "against a denominator of unknown age.",
            symbol, type(exc).__name__, exc, reason,
        )
        raise FetcherError(
            f"[binance] {symbol} data unavailable (live + DB fallback): {reason}"
        ) from exc

    # ------------------------------------------------------------------
    async def fetch_data(self) -> list[CryptoPrice]:
        """Fetch every active symbol in hierarchical order.

        Altcoin failures are isolated; a load-bearing symbol's failure goes
        through :meth:`_handle_denominator_failure`, which either falls back
        to a recent stored candle or aborts the run.
        """
        denominators = await self.load_required_denominators()
        symbols = await self.load_active_symbols(denominators=denominators)
        self.symbol_stats = {}
        self.failed_symbols = []
        self.fallback_symbols = []
        self.log.info(
            "[binance] load-bearing (strict) symbols: %s", ", ".join(sorted(denominators))
        )

        rows: list[CryptoPrice] = []
        for symbol in symbols:
            try:
                fetched = await self.fetch_symbol(symbol)
            except Exception as exc:  # noqa: BLE001 - per-symbol isolation
                self.failed_symbols.append(symbol)
                self.symbol_stats[symbol] = 0
                if symbol in denominators:
                    # Raises unless a fresh-enough stored candle exists.
                    await self._handle_denominator_failure(symbol, exc)
                else:
                    self.log.warning(
                        "[binance] %s FAILED (%s: %s) — nothing is divided by "
                        "it, continuing with the remaining symbols",
                        symbol, type(exc).__name__, exc,
                    )
                continue
            self.symbol_stats[symbol] = len(fetched)
            rows.extend(fetched)

        self.log.info(
            "[binance] multi-asset fetch complete: %d rows across %d/%d symbols",
            len(rows), len(symbols) - len(self.failed_symbols), len(symbols),
        )
        for symbol, count in self.symbol_stats.items():
            self.log.info("[binance]     %-10s %5d candles", symbol, count)
        if self.fallback_symbols:
            self.log.warning(
                "[binance] served from stored-candle fallback: %s",
                ", ".join(self.fallback_symbols),
            )
        if self.failed_symbols:
            self.log.error(
                "[binance] symbols with no data this run: %s",
                ", ".join(self.failed_symbols),
            )
        return rows

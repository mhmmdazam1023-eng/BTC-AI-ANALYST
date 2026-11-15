"""Near-real-time funding rate for the EXECUTION path (2026-08-09).

Why this exists, in one paragraph
---------------------------------
``derivatives_metrics.funding_rate`` comes from Binance Vision's **monthly**
archives, and there is no daily equivalent — verified 2026-08-08:
``futures/um/daily/fundingRate/`` returns 404 while ``monthly/fundingRate/``
returns 200. The running month has no archive until it ends, so the series can
sit ~31 days stale in early August (measured that day: 9 days). No scheduling
change fixes a source that has not published yet.

That is acceptable for training — funding is 1 of 50 features and its
consistency matters more than its freshness — and unacceptable for anything
near execution. So this writes to a SEPARATE table
(:class:`~src.db.models.LiveDerivative`) which is never merged into the
archive. See that class's docstring for the full reasoning.

Why Gate.io
-----------
Binance's own futures API is ISP-blocked here, as are Deribit, Bybit and OKX.
Of the exchanges probed via ccxt on 2026-08-09 — bitget, gate, kraken, mexc,
bingx, htx, kucoinfutures — **gate was the only one both reachable and
implementing** ``fetch_funding_rate``. Kraken's ccxt binding does not support
it at all; the rest failed to connect.

**Gate.io funding is NOT Binance funding.** The two track each other closely
because arbitrage forces them to, but they are set independently and differ.
``source`` is stored on every row for exactly this reason, and no reader may
present one as the other.

What it does NOT collect
------------------------
Open interest: ``gate.fetchOpenInterest()`` is not implemented in ccxt. That
gap is deliberate rather than papered over — and it matters much less, because
OI comes from Binance Vision's **daily** archives and is at most ~1 day stale,
which is the freshest that source can be.
"""

from __future__ import annotations

import ssl
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar

import aiohttp
import ccxt.async_support as ccxt_async
import certifi

from src.collectors.base import BaseDataFetcher, FetcherError
from src.db.engine import session_scope
from src.db.models import LiveDerivative

#: ccxt id of the one exchange that answered. Changing this is a data-source
#: change, not a config tweak: it alters what the stored numbers mean, so the
#: `source` column must move with it.
EXCHANGE_ID = "gate"

#: ccxt unified symbol for the USDT-margined perpetual.
DEFAULT_CONTRACT = "BTC/USDT:USDT"

#: Above this, a reading is too old to act on without re-checking. Not
#: enforced here — this module only records — but every reader should apply
#: it, and putting the number in one place stops three consumers inventing
#: three different thresholds.
LIVE_STALE_AFTER_MINUTES = 60


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


class LiveFundingFetcher(BaseDataFetcher):
    """One funding-rate reading, stamped with when we actually read it."""

    source_name: ClassVar[str] = "live_funding"

    RETRYABLE_CCXT_ERRORS: ClassVar[tuple[type[Exception], ...]] = (
        ccxt_async.NetworkError,
    )

    def __init__(self, *, symbol: str = "BTC/USDT", contract: str | None = None) -> None:
        super().__init__()
        #: Warehouse symbol, so this joins to everything else.
        self.symbol = symbol
        #: Exchange-side contract id, which is a different string.
        self.contract = contract or DEFAULT_CONTRACT
        self._exchange: ccxt_async.Exchange | None = None

    # ------------------------------------------------------------------
    @property
    def exchange(self) -> ccxt_async.Exchange:
        """Lazily-created ccxt client using the SYSTEM DNS resolver.

        ccxt's default aiodns/pycares resolver cannot read this machine's DNS
        configuration — the same failure the Binance fetcher works around, and
        the same fix (``ThreadedResolver`` + certifi CAs). Reusing that pattern
        rather than inventing a second one.
        """
        if self._exchange is None:
            exchange = getattr(ccxt_async, EXCHANGE_ID)(
                {"enableRateLimit": True, "timeout": 20_000}
            )
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            exchange.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    resolver=aiohttp.ThreadedResolver(), ssl=ssl_context
                )
            )
            self._exchange = exchange
            self.log.debug("[%s] ccxt client created (%s)", self.source_name, EXCHANGE_ID)
        return self._exchange

    async def close(self) -> None:
        if self._exchange is not None:
            try:
                await self._exchange.close()
            finally:
                self._exchange = None
        await super().close()

    # ------------------------------------------------------------------
    async def fetch_data(self) -> list[LiveDerivative]:
        """Read the current funding rate. Raises :class:`FetcherError`."""
        try:
            async for attempt in self._retryer(self.RETRYABLE_CCXT_ERRORS):
                with attempt:
                    payload = await self.exchange.fetch_funding_rate(self.contract)
        except ccxt_async.NotSupported as exc:
            raise FetcherError(
                f"[{self.source_name}] {EXCHANGE_ID} does not implement "
                f"fetch_funding_rate: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - normalised to FetcherError
            raise FetcherError(
                f"[{self.source_name}] could not read funding from "
                f"{EXCHANGE_ID}: {type(exc).__name__}: {exc}"
            ) from exc

        rate = _decimal(payload.get("fundingRate"))
        if rate is None:
            raise FetcherError(
                f"[{self.source_name}] {EXCHANGE_ID} returned no fundingRate "
                f"for {self.contract}"
            )

        # OUR clock, not the exchange's. `fundingTimestamp` is the next
        # settlement — a different question from "how old is this reading".
        observed_at = datetime.now(tz=UTC)

        settles_at = None
        raw_settle = payload.get("fundingTimestamp")
        if raw_settle:
            try:
                settles_at = datetime.fromtimestamp(int(raw_settle) / 1000, tz=UTC)
            except (TypeError, ValueError, OSError):
                settles_at = None

        row = LiveDerivative(
            observed_at=observed_at,
            symbol=self.symbol,
            source=EXCHANGE_ID,
            metric="funding_rate",
            value=rate,
            settles_at=settles_at,
            funding_interval=payload.get("interval"),
            mark_price=_decimal(payload.get("markPrice")),
            index_price=_decimal(payload.get("indexPrice")),
        )
        self.log.info(
            "[%s] %s funding_rate=%s (interval %s, settles %s) read at %s",
            self.source_name, EXCHANGE_ID, rate, payload.get("interval"),
            settles_at.isoformat() if settles_at else "unknown",
            observed_at.isoformat(timespec="seconds"),
        )
        return [row]

    # ------------------------------------------------------------------
    async def save_to_db(self, rows) -> int:
        """Append the readings. No upsert: each is a distinct observation.

        Two reads a minute apart are two facts, not a correction of one — the
        opposite of the archive tables, where a re-published day supersedes
        what was stored before.
        """
        if not rows:
            return 0
        async with session_scope() as session:
            session.add_all(list(rows))
        self.log.info(
            "[%s] stored %d live reading(s) in 'live_derivatives'",
            self.source_name, len(rows),
        )
        return len(rows)

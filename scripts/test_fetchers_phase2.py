"""Phase 2 fetcher test.

Covers:
* Part 1 — Binance (``crypto_prices``), FRED (``macro_indicators``)
* Part 2 — Blockchain.com (``on_chain_metrics``), Alternative.me
  (``sentiment_data``)
* Part 3 — Deribit (``derivatives_metrics``), Stooq (``market_indices``)

For each fetcher this script:
1. runs ``fetch_data()``,
2. validates the returned rows against the database model schema,
3. attempts ``save_to_db()`` — if the database is unreachable, or the FRED
   API key is missing/invalid, it degrades gracefully (logged, no crash),
4. **hands back every row the save INSERTED**, and verifies at the end that
   the newest ``crypto_prices`` date still covers the same number of symbols
   it did at the start (2026-08-09).

Step 4 is not tidiness. Before it existed, ``test_binance`` left a BTC-only
candle for the current date in ``crypto_prices``; the next run of
``test_features_phase3`` turned that into a single BTC ``daily_features`` row,
and three other suites — sector, tokenomics, multi-asset — then failed because
"the latest date" contained no altcoins. The first full sweep looked green and
the second did not. See the borrow-and-return section below for what is
returned and what deliberately is not.

Exit code: 0 when nothing FAILED (SKIPPED is acceptable), 1 otherwise.

Usage (from the project root)::

    python scripts/test_fetchers_phase2.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, func, inspect, select, tuple_  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402

from src.collectors import (  # noqa: E402
    AlternativeMeFetcher,
    BinanceFetcher,
    BinanceVisionFetcher,
    BlockchainFetcher,
    FetcherError,
    FredFetcher,
    StooqFetcher,
)
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import (  # noqa: E402
    CryptoPrice,
    DerivativesMetric,
    MacroIndicator,
    MarketIndex,
    OnChainMetric,
    SentimentData,
)
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_fetchers_phase2")

#: Errors that mean "database not reachable" rather than "code is broken".
_DB_UNAVAILABLE_ERRORS = (SQLAlchemyError, OSError, ConnectionError)


# ---------------------------------------------------------------------------
# Schema validators
# ---------------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_crypto_rows(rows: list) -> None:
    """Validate BinanceFetcher output against the ``crypto_prices`` schema."""
    _require(len(rows) > 0, "no crypto rows returned")
    for row in rows:
        _require(isinstance(row, CryptoPrice), f"unexpected row type: {type(row)}")
        _require(isinstance(row.timestamp, datetime), "timestamp is not datetime")
        _require(row.timestamp.tzinfo is not None, "timestamp must be tz-aware (UTC)")
        _require(row.exchange == "binance", f"unexpected exchange: {row.exchange}")
        _require(row.symbol == "BTC/USDT", f"unexpected symbol: {row.symbol}")
        _require(row.timeframe == "1d", f"unexpected timeframe: {row.timeframe}")
        for field in ("open", "high", "low", "close", "volume"):
            value = getattr(row, field)
            _require(isinstance(value, Decimal), f"{field} is not Decimal: {type(value)}")
        _require(row.high >= row.low, f"high < low at {row.timestamp}")
        _require(row.volume >= 0, f"negative volume at {row.timestamp}")
    timestamps = [row.timestamp for row in rows]
    _require(len(set(timestamps)) == len(timestamps), "duplicate candle timestamps")
    logger.info(
        "Binance schema validation OK: %d candles (%s -> %s)",
        len(rows), rows[0].timestamp.date(), rows[-1].timestamp.date(),
    )


def validate_macro_rows(rows: list, expected_series: set[str]) -> None:
    """Validate FredFetcher output against the ``macro_indicators`` schema."""
    _require(len(rows) > 0, "no macro rows returned")
    seen: set[str] = set()
    for row in rows:
        _require(isinstance(row, MacroIndicator), f"unexpected row type: {type(row)}")
        _require(isinstance(row.timestamp, datetime), "timestamp is not datetime")
        _require(row.timestamp.tzinfo is not None, "timestamp must be tz-aware (UTC)")
        _require(row.source == "FRED", f"unexpected source: {row.source}")
        _require(row.indicator in expected_series, f"unexpected series: {row.indicator}")
        _require(isinstance(row.value, Decimal), f"value is not Decimal: {type(row.value)}")
        seen.add(row.indicator)
    logger.info(
        "FRED schema validation OK: %d observations across series %s",
        len(rows), sorted(seen),
    )


def validate_on_chain_rows(rows: list, expected_metrics: set[str]) -> None:
    """Validate BlockchainFetcher output against the ``on_chain_metrics`` schema."""
    _require(len(rows) > 0, "no on-chain rows returned")
    seen: set[str] = set()
    keys: set[tuple] = set()
    for row in rows:
        _require(isinstance(row, OnChainMetric), f"unexpected row type: {type(row)}")
        _require(isinstance(row.timestamp, datetime), "timestamp is not datetime")
        _require(row.timestamp.tzinfo is not None, "timestamp must be tz-aware (UTC)")
        _require(row.source == "BLOCKCHAIN_INFO", f"unexpected source: {row.source}")
        _require(row.asset == "BTC", f"unexpected asset: {row.asset}")
        _require(row.metric in expected_metrics, f"unexpected metric: {row.metric}")
        _require(isinstance(row.value, Decimal), f"value is not Decimal: {type(row.value)}")
        # >= 0: hash-rate legitimately rounds to 0.0 in early-2009 history.
        _require(row.value >= 0, f"negative value {row.value} for {row.metric}")
        keys.add((row.timestamp, row.metric))
        seen.add(row.metric)
    _require(len(keys) == len(rows), "duplicate (timestamp, metric) keys")
    logger.info(
        "Blockchain.com schema validation OK: %d observations across metrics %s",
        len(rows), sorted(seen),
    )


def validate_sentiment_rows(rows: list) -> None:
    """Validate AlternativeMeFetcher output against the ``sentiment_data`` schema."""
    _require(len(rows) > 0, "no sentiment rows returned")
    timestamps: set[datetime] = set()
    for row in rows:
        _require(isinstance(row, SentimentData), f"unexpected row type: {type(row)}")
        _require(isinstance(row.timestamp, datetime), "timestamp is not datetime")
        _require(row.timestamp.tzinfo is not None, "timestamp must be tz-aware (UTC)")
        _require(row.source == "ALTERNATIVE_ME", f"unexpected source: {row.source}")
        _require(row.metric == "fear_greed_index", f"unexpected metric: {row.metric}")
        _require(isinstance(row.value, Decimal), f"value is not Decimal: {type(row.value)}")
        _require(0 <= row.value <= 100, f"score {row.value} outside 0..100")
        _require(bool(row.value_text), "empty value_text classification")
        timestamps.add(row.timestamp)
    _require(len(timestamps) == len(rows), "duplicate sentiment timestamps")
    logger.info(
        "Alternative.me schema validation OK: %d daily readings (latest: %s '%s')",
        len(rows), rows[-1].value, rows[-1].value_text,
    )


def validate_derivatives_rows(
    rows: list, expected_exchange: str, expected_metrics: set[str]
) -> None:
    """Validate derivatives fetcher output against ``derivatives_metrics``."""
    _require(len(rows) > 0, "no derivatives rows returned")
    seen: set[str] = set()
    keys: set[tuple] = set()
    for row in rows:
        _require(isinstance(row, DerivativesMetric), f"unexpected row type: {type(row)}")
        _require(isinstance(row.timestamp, datetime), "timestamp is not datetime")
        _require(row.timestamp.tzinfo is not None, "timestamp must be tz-aware (UTC)")
        _require(
            row.exchange == expected_exchange, f"unexpected exchange: {row.exchange}"
        )
        _require(row.symbol.startswith("BTC"), f"unexpected symbol: {row.symbol}")
        _require(row.metric in expected_metrics, f"unexpected metric: {row.metric}")
        _require(isinstance(row.value, Decimal), f"value is not Decimal: {type(row.value)}")
        if row.metric == "dvol":
            _require(0 < row.value < 500, f"implausible DVOL {row.value}")
        if row.metric.startswith("open_interest"):
            _require(row.value >= 0, f"negative OI {row.value}")
        # funding rates may legitimately be negative — no sign check.
        keys.add((row.timestamp, row.symbol, row.metric))
        seen.add(row.metric)
    _require(len(keys) == len(rows), "duplicate (timestamp, symbol, metric) keys")
    logger.info(
        "Derivatives (%s) schema validation OK: %d observations across metrics %s",
        expected_exchange, len(rows), sorted(seen),
    )


def validate_market_index_rows(rows: list) -> None:
    """Validate StooqFetcher output against the ``market_indices`` schema."""
    _require(len(rows) > 0, "no market index rows returned")
    timestamps: set[datetime] = set()
    with_spx = with_ndx = 0
    for row in rows:
        _require(isinstance(row, MarketIndex), f"unexpected row type: {type(row)}")
        _require(isinstance(row.timestamp, datetime), "timestamp is not datetime")
        _require(row.timestamp.tzinfo is not None, "timestamp must be tz-aware (UTC)")
        _require(
            row.sp500_close is not None or row.nasdaq_close is not None,
            f"row {row.timestamp} has neither index value",
        )
        for field in ("sp500_close", "nasdaq_close"):
            value = getattr(row, field)
            if value is not None:
                _require(isinstance(value, Decimal), f"{field} is not Decimal")
                _require(value > 0, f"non-positive {field}: {value}")
        with_spx += row.sp500_close is not None
        with_ndx += row.nasdaq_close is not None
        timestamps.add(row.timestamp)
    _require(len(timestamps) == len(rows), "duplicate trading-day timestamps")
    logger.info(
        "Stooq schema validation OK: %d trading days (spx=%d, ndx=%d filled)",
        len(rows), with_spx, with_ndx,
    )


# ---------------------------------------------------------------------------
# Borrow-and-return (2026-08-09)
# ---------------------------------------------------------------------------
# This test writes to PRODUCTION tables, and for a long time it never gave
# back what it took. The damage was indirect, delayed, and order-dependent,
# which is why it went unnoticed for so long:
#
#   1. `test_binance` fetches BTC/USDT ONLY and saves it, so `crypto_prices`
#      gains a candle for the current date that no other symbol has.
#   2. On the NEXT run of `test_features_phase3`, FeatureEngine sees that date
#      as the newest and writes exactly ONE `daily_features` row — BTC.
#   3. Everything that reads "the latest date" then finds zero altcoins:
#      `test_sector_phase8` (SectorTrackerError), `test_tokenomics_phase8`
#      (no altcoin has both beta and correlation) and
#      `test_multi_asset_phase8` (no decision row persisted) all fail.
#
# So the first full sweep looked green and the second did not. "All suites
# pass" was never stable with respect to run order.
#
# The rule this restores is the one the project already learned from gate 9d
# of `test_regime_phase8` and the `decisions` cleanup in
# `test_multi_asset_phase8`: **a test that writes to a production table must
# RETURN what it borrowed** — not refuse to run, and not delete indiscriminately.
#
# What is returned, and what deliberately is NOT
# ----------------------------------------------
# Rows the test INSERTED (a primary key absent beforehand) are removed: those
# are the ones that invent an incomplete "latest date".
#
# Rows the test merely REFRESHED (key already present, values re-fetched) are
# LEFT ALONE. Restoring them would roll live data back to an older copy of
# itself — strictly worse, and indistinguishable from what `run_collectors`
# does every day. Nothing is lost either way: the real collector re-fetches
# the same candles for all 21 symbols on the next run.
#: Keys per round trip. A single ``IN ((a,b,c), (a,b,c), ...)`` over every key
#: at once does NOT survive these table sizes: the first version of this
#: helper sent 19,254 x 4 columns for `on_chain_metrics` and PostgreSQL
#: answered ``StatementTooComplexError: stack depth limit exceeded``, while
#: `macro_indicators` and `derivatives_metrics` got ``OutOfMemoryError``. It is
#: a planner resource limit, not a fixed parameter cap, so it degrades by
#: table size and by server load rather than failing at a predictable count —
#: which is exactly why it must be chunked instead of "kept under the limit".
_KEY_CHUNK = 500


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _primary_key_of(model, row) -> tuple:
    return tuple(getattr(row, column.name) for column in inspect(model).primary_key)


async def _existing_keys(model, keys: list[tuple]) -> set[tuple]:
    """Which of ``keys`` are already in the table, before we write anything.

    Raises on failure. The caller MUST NOT treat that as "database
    unavailable": an unusable snapshot means the test cannot promise to give
    back what it takes, which is a different situation entirely.
    """
    if not keys:
        return set()
    pk_columns = [
        getattr(model, column.name) for column in inspect(model).primary_key
    ]
    found: set[tuple] = set()
    async with session_scope() as session:
        for chunk in _chunks(keys, _KEY_CHUNK):
            rows = (
                await session.execute(
                    select(*pk_columns).where(tuple_(*pk_columns).in_(chunk))
                )
            ).all()
            found.update(tuple(row) for row in rows)
    return found


async def _return_borrowed(model, keys: list[tuple]) -> int:
    """Delete exactly the rows this test inserted. Never raises."""
    if not keys:
        return 0
    pk_columns = [
        getattr(model, column.name) for column in inspect(model).primary_key
    ]
    removed = 0
    try:
        async with session_scope() as session:
            for chunk in _chunks(keys, _KEY_CHUNK):
                result = await session.execute(
                    delete(model).where(tuple_(*pk_columns).in_(chunk))
                )
                removed += result.rowcount or 0
        return removed
    except Exception as exc:  # noqa: BLE001 - cleanup must not mask results
        logger.error(
            "[cleanup] could not remove %d borrowed row(s) from '%s' (%s: %s). "
            "%d were removed before the failure. Run "
            "`python scripts/run_daily.py` to refill the incomplete date.",
            len(keys), model.__tablename__, type(exc).__name__, exc, removed,
        )
        return removed


# ---------------------------------------------------------------------------
# Persistence helper (graceful when DB is down)
# ---------------------------------------------------------------------------
async def try_save(fetcher, rows: list, model, borrowed: list) -> str:
    """Attempt save_to_db(), then hand back every row it inserted.

    ``borrowed`` accumulates ``(model, keys)`` so the summary can report how
    much was returned — a silent cleanup is how a broken cleanup hides.
    """
    # The snapshot gets its OWN error scope. Folding it into the block below
    # made a bug in this file surface as "[fred] database unavailable" —
    # blaming a healthy component for a defect here, and hiding it behind a
    # status the suite treats as an acceptable pass. Never let your own
    # failure wear someone else's name.
    keys = [_primary_key_of(model, row) for row in rows]
    try:
        # Snapshot BEFORE the write: afterwards every key exists and the two
        # cases are indistinguishable.
        pre_existing = await _existing_keys(model, keys)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[%s] could not snapshot '%s' before writing (%s: %s) — REFUSING "
            "to save, because a write that cannot be identified afterwards "
            "cannot be given back. Fetch + validation already passed.",
            fetcher.source_name, model.__tablename__, type(exc).__name__, exc,
        )
        return "FETCH OK, SAVE REFUSED (no rollback possible)"

    try:
        written = await fetcher.save_to_db(rows)

        inserted = [key for key in keys if key not in pre_existing]
        removed = await _return_borrowed(model, inserted)
        if inserted:
            borrowed.append((model.__tablename__, removed))

        async with session_scope() as session:
            total = (
                await session.execute(select(func.count()).select_from(model))
            ).scalar_one()
        logger.info(
            "[%s] persisted %d rows (%d were new and have been returned); "
            "'%s' now holds %d total rows",
            fetcher.source_name, written, removed, model.__tablename__, total,
        )
        return f"SAVED ({written} rows, {removed} returned, table total {total})"
    except _DB_UNAVAILABLE_ERRORS as exc:
        logger.warning(
            "[%s] database unavailable — skipping persistence "
            "(fetch + validation already OK). Detail: %s",
            fetcher.source_name, exc,
        )
        return "FETCH OK, DB SKIPPED"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
async def test_binance(results: dict[str, str], borrowed: list) -> None:
    fetcher = BinanceFetcher(symbol="BTC/USDT", timeframe="1d", limit=365)
    try:
        rows = await fetcher.fetch_data()
        if not rows:
            results["binance"] = "FAILED (empty response)"
            return
        validate_crypto_rows(rows)
        status = await try_save(fetcher, rows, CryptoPrice, borrowed)
        results["binance"] = status
    except FetcherError as exc:
        logger.error("Binance fetcher failed: %s", exc)
        results["binance"] = f"FAILED ({exc})"
    except AssertionError as exc:
        logger.error("Binance schema validation failed: %s", exc)
        results["binance"] = f"FAILED (validation: {exc})"
    finally:
        await fetcher.close()


async def test_fred(results: dict[str, str], borrowed: list) -> None:
    settings = get_settings()
    fetcher = FredFetcher()
    expected_series = set(fetcher.series)
    try:
        rows = await fetcher.fetch_data()
        if not rows:
            if not settings.fred_api_key.get_secret_value().strip():
                # Graceful-by-design path: key not configured yet.
                results["fred"] = "SKIPPED (FRED_API_KEY empty — handled gracefully)"
            else:
                results["fred"] = "FAILED (key set but no data returned)"
            return
        validate_macro_rows(rows, expected_series)
        status = await try_save(fetcher, rows, MacroIndicator, borrowed)
        results["fred"] = status
    except FetcherError as exc:
        logger.error("FRED fetcher failed: %s", exc)
        results["fred"] = f"FAILED ({exc})"
    except AssertionError as exc:
        logger.error("FRED schema validation failed: %s", exc)
        results["fred"] = f"FAILED (validation: {exc})"
    finally:
        await fetcher.close()


async def test_blockchain(results: dict[str, str], borrowed: list) -> None:
    fetcher = BlockchainFetcher()
    expected_metrics = set(fetcher.charts)
    try:
        rows = await fetcher.fetch_data()
        if not rows:
            results["blockchain"] = "FAILED (no data from any chart)"
            return
        validate_on_chain_rows(rows, expected_metrics)
        status = await try_save(fetcher, rows, OnChainMetric, borrowed)
        results["blockchain"] = status
    except FetcherError as exc:
        logger.error("Blockchain.com fetcher failed: %s", exc)
        results["blockchain"] = f"FAILED ({exc})"
    except AssertionError as exc:
        logger.error("Blockchain.com schema validation failed: %s", exc)
        results["blockchain"] = f"FAILED (validation: {exc})"
    finally:
        await fetcher.close()


async def test_alternative_me(results: dict[str, str], borrowed: list) -> None:
    fetcher = AlternativeMeFetcher()
    try:
        rows = await fetcher.fetch_data()
        if not rows:
            results["alt_me"] = "FAILED (empty history)"
            return
        validate_sentiment_rows(rows)
        status = await try_save(fetcher, rows, SentimentData, borrowed)
        results["alt_me"] = status
    except FetcherError as exc:
        logger.error("Alternative.me fetcher failed: %s", exc)
        results["alt_me"] = f"FAILED ({exc})"
    except AssertionError as exc:
        logger.error("Alternative.me schema validation failed: %s", exc)
        results["alt_me"] = f"FAILED (validation: {exc})"
    finally:
        await fetcher.close()


async def test_binance_vision(results: dict[str, str], borrowed: list) -> None:
    fetcher = BinanceVisionFetcher()
    expected_metrics = {"funding_rate", "open_interest", "open_interest_value"}
    try:
        rows = await fetcher.fetch_data()
        if not rows:
            results["binance_vision"] = "FAILED (no data from any metric group)"
            return
        validate_derivatives_rows(rows, "binance", expected_metrics)
        status = await try_save(fetcher, rows, DerivativesMetric, borrowed)
        results["binance_vision"] = status
    except FetcherError as exc:
        logger.error("Binance Vision fetcher failed: %s", exc)
        results["binance_vision"] = f"FAILED ({exc})"
    except AssertionError as exc:
        logger.error("Binance Vision schema validation failed: %s", exc)
        results["binance_vision"] = f"FAILED (validation: {exc})"
    finally:
        await fetcher.close()


async def test_stooq(results: dict[str, str], borrowed: list) -> None:
    fetcher = StooqFetcher()
    try:
        rows = await fetcher.fetch_data()
        if not rows:
            # Fragile source: by contract it degrades to [] instead of crashing.
            results["stooq"] = "SKIPPED (Stooq unavailable — handled gracefully)"
            return
        validate_market_index_rows(rows)
        status = await try_save(fetcher, rows, MarketIndex, borrowed)
        results["stooq"] = status
    except FetcherError as exc:
        logger.error("Stooq fetcher failed: %s", exc)
        results["stooq"] = f"FAILED ({exc})"
    except AssertionError as exc:
        logger.error("Stooq schema validation failed: %s", exc)
        results["stooq"] = f"FAILED (validation: {exc})"
    finally:
        await fetcher.close()


async def _newest_daily_coverage() -> tuple[datetime | None, int]:
    """``(newest 1d date in crypto_prices, how many symbols have it)``.

    THE invariant this whole section exists to protect, stated directly.
    The original bug was not "rows were left behind" in the abstract — it was
    that the newest date ended up with ONE symbol instead of 21, and every
    reader of "the latest date" then found no altcoins. Checking the symptom
    catches the regression no matter which mechanism reintroduces it.
    """
    async with session_scope() as session:
        row = (
            await session.execute(
                select(
                    func.date(CryptoPrice.timestamp),
                    func.count(func.distinct(CryptoPrice.symbol)),
                )
                .where(CryptoPrice.timeframe == "1d")
                .group_by(func.date(CryptoPrice.timestamp))
                .order_by(func.date(CryptoPrice.timestamp).desc())
                .limit(1)
            )
        ).first()
    return (row[0], row[1]) if row else (None, 0)


# ---------------------------------------------------------------------------
async def main() -> int:
    results: dict[str, str] = {}
    #: (table, rows returned) per fetcher that inserted anything.
    borrowed: list[tuple[str, int]] = []
    coverage_before = await _newest_daily_coverage()
    try:
        await test_binance(results, borrowed)
        await test_fred(results, borrowed)
        await test_blockchain(results, borrowed)
        await test_alternative_me(results, borrowed)
        await test_binance_vision(results, borrowed)
        # DeribitFetcher is NOT exercised here: deribit.com (like bybit.com
        # and okx.com) is ISP-blocked on this machine (TLS hostname-mismatch
        # DNS hijack). The fetcher stays in the codebase for unrestricted
        # networks; Binance Vision covers the derivatives pillar instead.
        results["deribit"] = "SKIPPED (deribit.com ISP-blocked — see CLAUDE.md)"
        await test_stooq(results, borrowed)
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001 - shutdown must not mask results
            logger.warning("Engine disposal issue (ignored): %s", exc)

    logger.info("=" * 64)
    logger.info("PHASE 2 FETCHER TEST SUMMARY (Part 1-3)")
    for name, status in results.items():
        logger.info("  %-8s : %s", name, status)
    # Reported, never silent: a cleanup nobody can see is a cleanup nobody
    # notices breaking, and this test spent months leaving rows behind.
    if borrowed:
        logger.info("-" * 64)
        logger.info(
            "BORROW-AND-RETURN: %d row(s) inserted by this test were removed",
            sum(count for _, count in borrowed),
        )
        for table, count in borrowed:
            logger.info("  %-22s %d row(s) returned", table, count)
    else:
        logger.info("-" * 64)
        logger.info(
            "BORROW-AND-RETURN: nothing new was inserted — every row this "
            "test wrote already existed, so there was nothing to give back."
        )
    coverage_after = await _newest_daily_coverage()
    coverage_ok = coverage_before == coverage_after
    if coverage_ok:
        logger.info(
            "COVERAGE UNCHANGED: newest 1d date %s still has %d symbol(s)",
            coverage_after[0], coverage_after[1],
        )
    else:
        logger.error(
            "COVERAGE CHANGED: newest 1d date was %s with %d symbol(s), now "
            "%s with %d. This test has left `crypto_prices` with a date the "
            "rest of the pipeline cannot use — the exact breakage the "
            "borrow-and-return section exists to prevent. Run "
            "`python scripts/run_daily.py` to repair, then fix the cleanup.",
            coverage_before[0], coverage_before[1],
            coverage_after[0], coverage_after[1],
        )
    logger.info("=" * 64)

    if not coverage_ok:
        return 1
    failed = [name for name, status in results.items() if status.startswith("FAILED")]
    if failed:
        logger.error("Test FAILED for: %s", ", ".join(failed))
        return 1
    logger.info("All fetcher tests passed (SKIPPED counts as pass by design).")
    return 0


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main()))

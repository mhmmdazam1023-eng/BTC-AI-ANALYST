"""Phase 8 Part 1 test: multi-asset foundation + Coin DNA + relative strength.

Gates
-----
1.  **Registry**   — ``coins_registry`` holds the seeded Coin DNA, the native
    enum rejects an unknown sector, and the FK rejects an unknown symbol.
2.  **Ordering**   — hierarchical ingestion really puts BTC then ETH first.
2b. **Guard**      — a load-bearing symbol's live-fetch outage falls back to
    a fresh stored candle, and ABORTS when no fresh candle exists.
3.  **Prices**     — ``crypto_prices`` holds every active symbol.
4.  **Keying**     — ``daily_features`` is keyed by (timestamp, symbol) and
    contains no duplicate (day, symbol) pair.
5.  **Correctness**— ``price_btc_ratio`` is recomputed INDEPENDENTLY from
    ``crypto_prices`` and must match (excluding the still-forming candle);
    ``rs_7d``/``rs_30d`` are re-derived from the ratio series and must match.
6.  **Identity**   — for BTC, ratio is exactly 1 and rs exactly 0.
7.  **No gaps**    — the recent 30-day window has no NULL in the new
    columns for ANY symbol (the "safe forward-fill" requirement).
8.  **BTC intact** — Bitcoin still has its full history and its own row is
    self-consistent (``asset_close == btc_close``).

Gates 1 and 4's negative checks run inside a transaction that is always
rolled back, so the test never leaves anything behind.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_altcoin_phase8.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select, text  # noqa: E402
from sqlalchemy.exc import DBAPIError, IntegrityError  # noqa: E402

from src.collectors import BinanceMultiFetcher, FetcherError  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import (  # noqa: E402
    BASE_SYMBOL,
    HIERARCHY_HEAD,
    CoinRegistry,
    CryptoPrice,
    DailyFeature,
)
from src.forecasting.evaluator import FINALIZATION_BUFFER_DAYS  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_altcoin_phase8")

#: Symbols the Phase 8 seed must provide.
EXPECTED_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "TAO/USDT")

#: Window that must be gap-free for every symbol.
RECENT_COMPLETE_DAYS = 30

#: Relative tolerance when re-deriving a feature from raw prices. NUMERIC
#: round-tripping through float64 costs ~1e-15; 1e-9 is generous but still
#: orders of magnitude tighter than any real computation error.
TOLERANCE = 1e-9

#: Reconciliation (gate 5) compares a STORED derived feature against the
#: LIVE source prices, so it must ignore days whose candle is still forming.
#: Today's candle legitimately moves every time a collector runs, which makes
#: `daily_features` lag `crypto_prices` until the next FeatureEngine pass —
#: a scheduling artefact, not a computation error. Reusing the Phase 6
#: evaluator's constant keeps ONE definition of "finalized" in the codebase
#: (see src/forecasting/evaluator.py, fix of 2026-07-25).
_RECONCILE_CUTOFF_DAYS = FINALIZATION_BUFFER_DAYS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@contextmanager
def _quiet_rollback_logs():
    """Silence ``session_scope``'s rollback traceback for EXPECTED failures.

    The negative checks below deliberately violate constraints. Without this,
    a fully passing run prints several alarming stack traces, which trains
    the reader to ignore real ones.
    """
    engine_logger = logging.getLogger("src.db.engine")
    previous = engine_logger.level
    engine_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        engine_logger.setLevel(previous)


async def _rejects(sql: str, params: dict | None = None) -> bool:
    """True when the database REFUSES this statement (constraint held)."""
    with _quiet_rollback_logs():
        try:
            async with session_scope() as session:
                await session.execute(text(sql), params or {})
        except (DBAPIError, IntegrityError):
            return True
    return False


# ---------------------------------------------------------------------------
async def gate_registry() -> None:
    """Coin DNA is seeded, and its enum + FK constraints actually bite."""
    async with session_scope() as session:
        rows = (
            await session.execute(select(CoinRegistry).order_by(CoinRegistry.symbol))
        ).scalars().all()
        found = {row.symbol: row for row in rows}
        for symbol in EXPECTED_SYMBOLS:
            _require(symbol in found, f"coins_registry is missing {symbol}")
            _require(found[symbol].is_active, f"{symbol} is not active")
            _require(bool(found[symbol].chain), f"{symbol} has no chain")

        _require(
            found[BASE_SYMBOL].base_asset is None,
            "BTC/USDT must have base_asset NULL — it IS the numéraire",
        )
        for symbol in ("SOL/USDT", "TAO/USDT", "ETH/USDT", "BNB/USDT"):
            _require(
                found[symbol].base_asset == BASE_SYMBOL,
                f"{symbol} should be benchmarked against {BASE_SYMBOL}",
            )
        _require(
            found["TAO/USDT"].sector.value == "AI",
            f"TAO sector should be 'AI', got {found['TAO/USDT'].sector}",
        )
        logger.info(
            "Gate 1a OK: %d coins registered -> %s",
            len(found), ", ".join(f"{s}({found[s].sector.value})" for s in found),
        )

    # --- negative check: the native enum must reject an unknown sector ---
    _require(
        await _rejects(
            "INSERT INTO coins_registry (symbol, name, sector, chain, is_active) "
            "VALUES ('ZZZ/USDT', 'Bogus', 'NotASector', 'Nowhere', true)"
        ),
        "coin_sector enum accepted an invalid value — integrity is fake",
    )
    logger.info("Gate 1b OK: coin_sector enum rejects unknown sectors")

    # --- negative check: the FK must reject an unregistered symbol ---
    _require(
        await _rejects(
            "INSERT INTO tokenomics_events "
            "(symbol, event_date, event_type, supply_impact_pct) "
            "VALUES ('NOPE/USDT', now(), 'unlock', 1.0)"
        ),
        "tokenomics_events FK accepted an unregistered symbol",
    )
    logger.info("Gate 1c OK: tokenomics_events FK rejects unregistered symbols")

    async with session_scope() as session:
        events = (
            await session.execute(text("SELECT count(*) FROM tokenomics_events"))
        ).scalar_one()
    logger.info("Gate 1d OK: tokenomics_events reachable (%d seeded rows)", events)


async def gate_hierarchy() -> None:
    """BTC and ETH must lead the ingestion order, whatever the registry says."""
    ordered = BinanceMultiFetcher._order_hierarchically(
        ["TAO/USDT", "SOL/USDT", "ETH/USDT", "BNB/USDT", "BTC/USDT"]
    )
    _require(
        ordered[: len(HIERARCHY_HEAD)] == list(HIERARCHY_HEAD),
        f"hierarchy violated: {ordered}",
    )
    _require(len(ordered) == 5, f"symbols lost while ordering: {ordered}")

    # Even a registry that omits BTC entirely must still ingest it first.
    fetcher = BinanceMultiFetcher(symbols=["SOL/USDT", "TAO/USDT"])
    forced = await fetcher.load_active_symbols()
    _require(
        forced[0] == BASE_SYMBOL,
        f"{BASE_SYMBOL} must be force-included and lead; got {forced}",
    )
    logger.info("Gate 2 OK: ingestion order %s", " -> ".join(ordered))


class _BaseSymbolOutage(BinanceMultiFetcher):
    """Simulates the live Binance fetch failing for the numéraire only.

    Altcoins return no candles so the probe never touches the network or
    writes anything; only the failure-handling path is exercised.
    """

    async def fetch_symbol(self, symbol: str) -> list:
        if symbol == BASE_SYMBOL:
            raise FetcherError("simulated Binance outage for the numéraire")
        return []


async def gate_denominator_guard() -> None:
    """Load-bearing failure must fall back when fresh, abort when stale."""
    fetcher = BinanceMultiFetcher()
    denominators = await fetcher.load_required_denominators()
    _require(
        BASE_SYMBOL in denominators,
        f"{BASE_SYMBOL} must always be treated as load-bearing; got {denominators}",
    )
    logger.info("Gate 8a OK: load-bearing set = %s", ", ".join(sorted(denominators)))

    stored_at, age_days = await fetcher.stored_candle_age_days(BASE_SYMBOL)
    _require(stored_at is not None, f"no stored candle at all for {BASE_SYMBOL}")
    _require(
        age_days <= BinanceMultiFetcher.base_fallback_max_age_days,
        f"{BASE_SYMBOL} stored candle is {age_days} day(s) old — the warehouse "
        f"itself is stale, run the collectors",
    )
    logger.info(
        "Gate 8b OK: newest stored %s candle is %s (%d day(s) old, budget %d)",
        BASE_SYMBOL, stored_at.date(), age_days,
        BinanceMultiFetcher.base_fallback_max_age_days,
    )

    # --- fresh DB: the outage must be survived via the stored candle ---
    with _quiet_rollback_logs():
        survivor = _BaseSymbolOutage()
        try:
            await survivor.fetch_data()
        except FetcherError as exc:  # pragma: no cover - failure path
            raise AssertionError(
                f"fallback should have rescued a fresh warehouse, but aborted: {exc}"
            ) from exc
        finally:
            await survivor.close()
    _require(
        BASE_SYMBOL in survivor.fallback_symbols,
        f"{BASE_SYMBOL} outage was not recorded in fallback_symbols: "
        f"{survivor.fallback_symbols}",
    )
    logger.info("Gate 8c OK: fresh stored candle rescues a live-fetch outage")

    # --- stale DB: a budget of -1 makes every stored candle too old ---
    class _NoFallbackBudget(_BaseSymbolOutage):
        base_fallback_max_age_days = -1

    aborted = False
    with _quiet_rollback_logs():
        strict = _NoFallbackBudget()
        try:
            await strict.fetch_data()
        except FetcherError:
            aborted = True
        finally:
            await strict.close()
    _require(
        aborted,
        "with no usable fallback the collection MUST abort, not silently "
        "compute relative features against an unknown-age denominator",
    )
    _require(
        BinanceMultiFetcher.abort_if_head_fails is True,
        "abort_if_head_fails must default to True (strict policy)",
    )
    logger.info("Gate 8d OK: stale/absent fallback aborts the collection")


async def gate_prices() -> None:
    """Every registered active symbol has candles in crypto_prices."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(
                    CryptoPrice.symbol,
                    func.count().label("n"),
                    func.min(CryptoPrice.timestamp),
                    func.max(CryptoPrice.timestamp),
                )
                .where(
                    CryptoPrice.exchange == "binance",
                    CryptoPrice.timeframe == "1d",
                )
                .group_by(CryptoPrice.symbol)
            )
        ).all()
    stats = {row[0]: row for row in rows}
    for symbol in EXPECTED_SYMBOLS:
        _require(symbol in stats, f"crypto_prices has no candles for {symbol}")
        _require(
            stats[symbol][1] >= 200,
            f"{symbol}: only {stats[symbol][1]} candles, expected >=200",
        )
        logger.info(
            "Gate 3 OK: %-9s %5d candles (%s -> %s)",
            symbol, stats[symbol][1], stats[symbol][2].date(), stats[symbol][3].date(),
        )


async def gate_keying() -> None:
    """daily_features is keyed by (timestamp, symbol) with no duplicates."""
    async with session_scope() as session:
        pk_def = (
            await session.execute(
                text(
                    "SELECT pg_get_constraintdef(con.oid) FROM pg_constraint con "
                    "JOIN pg_class rel ON rel.oid = con.conrelid "
                    "WHERE rel.relname='daily_features' AND con.contype='p'"
                )
            )
        ).scalar_one()
        _require(
            "symbol" in pk_def and "timestamp" in pk_def,
            f"daily_features PK is not composite: {pk_def}",
        )

        duplicates = (
            await session.execute(
                text(
                    "SELECT count(*) FROM (SELECT timestamp, symbol FROM daily_features "
                    "GROUP BY 1,2 HAVING count(*) > 1) d"
                )
            )
        ).scalar_one()
        _require(duplicates == 0, f"{duplicates} duplicate (day, symbol) pairs")

    # --- negative check: a duplicate (day, symbol) must be refused ---
    async with session_scope() as session:
        day = (
            await session.execute(
                select(func.max(DailyFeature.timestamp)).where(
                    DailyFeature.symbol == BASE_SYMBOL
                )
            )
        ).scalar_one()
    _require(
        await _rejects(
            "INSERT INTO daily_features (timestamp, symbol) VALUES (:t, :s)",
            {"t": day, "s": BASE_SYMBOL},
        ),
        "composite PK did not prevent a duplicate (day, symbol)",
    )
    logger.info("Gate 4 OK: PK %s enforced, no duplicates", pk_def)


async def gate_relative_strength() -> None:
    """Re-derive ratio and RS from raw prices; the stored values must match.

    Gate 5a crosses table boundaries (``daily_features`` vs ``crypto_prices``)
    so it skips the last :data:`_RECONCILE_CUTOFF_DAYS` day(s): a still-forming
    candle moves on every collector run, and `daily_features` only catches up
    on the next FeatureEngine pass. Gate 5b compares ``daily_features`` against
    itself, so it needs no such cutoff — every row in a block is written by one
    engine run and is therefore always mutually consistent.
    """
    logger.info(
        "Gate 5: reconciliation skips the last %d day(s) — still-forming candles",
        _RECONCILE_CUTOFF_DAYS,
    )
    async with session_scope() as session:
        # --- price_btc_ratio vs an independent join on crypto_prices ---
        worst = (
            await session.execute(
                text(
                    """
                    SELECT f.symbol,
                           max(abs(f.price_btc_ratio - (p.close / b.close))
                               / nullif(abs(p.close / b.close), 0)) AS rel_err,
                           count(*) AS n
                    FROM daily_features f
                    JOIN crypto_prices p ON p.timestamp = f.timestamp
                         AND p.symbol = f.symbol AND p.exchange = 'binance'
                         AND p.timeframe = '1d'
                    JOIN crypto_prices b ON b.timestamp = f.timestamp
                         AND b.symbol = :base AND b.exchange = 'binance'
                         AND b.timeframe = '1d'
                    WHERE f.price_btc_ratio IS NOT NULL
                      AND f.timestamp < date_trunc('day', now() at time zone 'utc')
                          - make_interval(days => :cutoff)
                    GROUP BY f.symbol ORDER BY f.symbol
                    """
                ),
                {"base": BASE_SYMBOL, "cutoff": _RECONCILE_CUTOFF_DAYS},
            )
        ).all()
        _require(bool(worst), "no rows available to verify price_btc_ratio")
        for symbol, rel_err, count in worst:
            _require(
                float(rel_err) < TOLERANCE,
                f"{symbol}: price_btc_ratio off by {float(rel_err):.2e} "
                f"(> {TOLERANCE:.0e}) across {count} rows",
            )
            logger.info(
                "Gate 5a OK: %-9s price_btc_ratio matches raw prices "
                "(max rel. error %.2e over %d rows)",
                symbol, float(rel_err), count,
            )

        # --- rs_7d / rs_30d vs the ratio series shifted by the window ---
        for window in (7, 30):
            column = f"rs_{window}d"
            rows = (
                await session.execute(
                    text(
                        f"""
                        SELECT cur.symbol,
                               max(abs(cur.{column}
                                   - ((cur.price_btc_ratio / prev.price_btc_ratio) - 1)
                                     * 100)) AS abs_err,
                               count(*) AS n
                        FROM daily_features cur
                        JOIN daily_features prev
                          ON prev.symbol = cur.symbol
                         AND prev.timestamp = cur.timestamp
                             - make_interval(days => :w)
                        WHERE cur.{column} IS NOT NULL
                          AND prev.price_btc_ratio IS NOT NULL
                          AND prev.price_btc_ratio <> 0
                        GROUP BY cur.symbol ORDER BY cur.symbol
                        """
                    ),
                    {"w": window},
                )
            ).all()
            _require(bool(rows), f"no rows available to verify {column}")
            for symbol, abs_err, count in rows:
                # abs_err is in percentage points; 1e-6 pp is far below any
                # meaningful signal and covers NUMERIC/float round-tripping.
                _require(
                    float(abs_err) < 1e-6,
                    f"{symbol}: {column} off by {float(abs_err):.2e} pp "
                    f"across {count} rows",
                )
            logger.info(
                "Gate 5b OK: %s re-derives from the ratio series for all %d symbols",
                column, len(rows),
            )


async def gate_btc_identity() -> None:
    """Bitcoin's own row must be self-consistent: ratio 1, rs 0."""
    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT count(*) AS total,
                           count(*) FILTER (WHERE price_btc_ratio IS DISTINCT FROM 1),
                           count(*) FILTER (WHERE rs_7d IS NOT NULL AND rs_7d <> 0),
                           count(*) FILTER (WHERE rs_30d IS NOT NULL AND rs_30d <> 0),
                           count(*) FILTER (WHERE asset_close IS DISTINCT FROM btc_close)
                    FROM daily_features WHERE symbol = :base
                    """
                ),
                {"base": BASE_SYMBOL},
            )
        ).one()
    total, bad_ratio, bad_rs7, bad_rs30, bad_close = row
    _require(total >= 300, f"BTC has only {total} feature rows — history lost")
    _require(bad_ratio == 0, f"{bad_ratio} BTC rows where price_btc_ratio != 1")
    _require(bad_rs7 == 0, f"{bad_rs7} BTC rows where rs_7d != 0")
    _require(bad_rs30 == 0, f"{bad_rs30} BTC rows where rs_30d != 0")
    _require(bad_close == 0, f"{bad_close} BTC rows where asset_close != btc_close")
    logger.info(
        "Gate 6 OK: %d BTC rows, ratio==1 and rs==0 throughout, "
        "asset_close==btc_close throughout", total,
    )


async def gate_no_gaps() -> None:
    """The recent window must be NULL-free for every symbol (safe ffill)."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT symbol,
                           count(*) FILTER (WHERE asset_close IS NULL),
                           count(*) FILTER (WHERE btc_close IS NULL),
                           count(*) FILTER (WHERE price_btc_ratio IS NULL),
                           count(*) FILTER (WHERE rs_7d IS NULL),
                           count(*) FILTER (WHERE rs_30d IS NULL),
                           count(*)
                    FROM daily_features
                    WHERE timestamp > (SELECT max(timestamp) FROM daily_features)
                                      - make_interval(days => :d)
                    GROUP BY symbol ORDER BY symbol
                    """
                ),
                {"d": RECENT_COMPLETE_DAYS},
            )
        ).all()
    _require(
        len(rows) >= len(EXPECTED_SYMBOLS),
        f"only {len(rows)} symbols present in the recent window",
    )
    for symbol, n_close, n_btc, n_ratio, n_rs7, n_rs30, total in rows:
        _require(
            total >= RECENT_COMPLETE_DAYS - 1,
            f"{symbol}: only {total} rows in the recent window",
        )
        for label, count in (
            ("asset_close", n_close), ("btc_close", n_btc),
            ("price_btc_ratio", n_ratio), ("rs_7d", n_rs7), ("rs_30d", n_rs30),
        ):
            _require(
                count == 0,
                f"{symbol}: {count} NULL '{label}' in the last "
                f"{RECENT_COMPLETE_DAYS} days — the BTC join left gaps",
            )
        logger.info("Gate 7 OK: %-9s %d recent rows, zero NULLs", symbol, total)


# ---------------------------------------------------------------------------
async def main() -> int:
    started = datetime.now(tz=UTC)
    gates = (
        ("Registry & Coin DNA", gate_registry),
        ("Hierarchical ingestion", gate_hierarchy),
        ("Denominator fallback/abort guard", gate_denominator_guard),
        ("Multi-asset prices", gate_prices),
        ("Composite keying", gate_keying),
        ("Relative strength math", gate_relative_strength),
        ("Bitcoin identity", gate_btc_identity),
        ("No gaps after safe ffill", gate_no_gaps),
    )
    try:
        for name, gate in gates:
            logger.info("--- %s ---", name)
            await gate()
    except AssertionError as exc:
        logger.error("PHASE 8 TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("PHASE 8 TEST FAILED with an unexpected error")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 68)
    logger.info(
        "PHASE 8 PART 1 TEST PASSED (%d gates, %.1fs) — multi-asset foundation ready",
        len(gates), elapsed,
    )
    logger.info("=" * 68)
    return 0


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main()))

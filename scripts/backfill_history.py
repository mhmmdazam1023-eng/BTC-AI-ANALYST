"""Deep historical backfill of ``crypto_prices`` (Phase 8 Part 2 helper).

The daily collectors take the last N candles, which Binance caps at 1000
(~2.7 years). Regime backtesting needs the 2021 bull and 2022 bear markets,
so this script paginates ``fetch_ohlcv`` backwards via ``since``.

Opt-in and idempotent: every candle is upserted on its natural key, so
re-running only refreshes values. It is NOT part of ``run_daily.py`` —
deep history is a one-off, not a daily cost.

Effect on other phases
----------------------
* **Phase 6 forecasting** is unaffected in substance: ``ForecastEngine``
  already caps its return pool at ``LOOKBACK_DAYS`` (3 years) via
  ``log_returns_full[-LOOKBACK_DAYS:]``. Before this backfill it ran BELOW
  that target and said so in the logs; afterwards it simply reaches its
  intended window.
* **Feature Engineering** will extend ``daily_features`` back to wherever
  the source tables reach. Columns whose sources start later (e.g.
  ``open_interest``, only backfilled 365 days) stay NULL in old rows —
  which is honest, and the recent-window quality gate is unaffected.

Usage (from the project root)::

    python scripts/backfill_history.py --since 2020-01-01
    python scripts/backfill_history.py --since 2020-01-01 --symbols BTC/USDT
    python scripts/backfill_history.py --since 2020-01-01 --skip-features
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from src.collectors import BinanceMultiFetcher  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import CryptoPrice  # noqa: E402
from src.features import FeatureEngine  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.backfill_history")


async def report_coverage(label: str) -> None:
    """Log first/last candle per symbol so the effect is visible."""
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
                .order_by(CryptoPrice.symbol)
            )
        ).all()
    logger.info("--- crypto_prices coverage %s ---", label)
    for symbol, count, first, last in rows:
        logger.info(
            "    %-10s %5d candles  %s -> %s",
            symbol, count, first.date(), last.date(),
        )


async def backfill(since: str, symbols: list[str] | None, skip_features: bool) -> int:
    started = time.monotonic()
    try:
        datetime.fromisoformat(since)
    except ValueError as exc:
        raise SystemExit(f"--since must be ISO YYYY-MM-DD, got {since!r}") from exc

    await report_coverage("BEFORE")

    fetcher = BinanceMultiFetcher(since=since, symbols=symbols)
    try:
        rows = await fetcher.fetch_data()
        if not rows:
            logger.error("No candles returned — nothing to write.")
            return 1
        saved = await fetcher.save_to_db(rows)
        logger.info(
            "Backfill upserted %d candles across %d symbol(s) in %.1fs",
            saved, len(fetcher.symbol_stats), time.monotonic() - started,
        )
    finally:
        await fetcher.close()

    await report_coverage("AFTER")

    if skip_features:
        logger.warning(
            "--skip-features set: daily_features NOT recomputed. Run "
            "scripts/run_collectors.py before using the new history."
        )
        return 0

    logger.info("Recomputing daily_features over the extended history...")
    written = await FeatureEngine().run()
    logger.info("FeatureEngine wrote %d rows", written)
    return 0


async def main(since: str, symbols: list[str] | None, skip_features: bool) -> int:
    try:
        return await backfill(since, symbols, skip_features)
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep historical candle backfill.")
    parser.add_argument(
        "--since",
        default="2020-01-01",
        help="ISO start date (YYYY-MM-DD). Default: 2020-01-01.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Restrict to these symbols; default = all active in coins_registry.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Do not recompute daily_features afterwards.",
    )
    arguments = parser.parse_args()
    setup_logging(get_settings().log_level)
    raise SystemExit(
        asyncio.run(main(arguments.since, arguments.symbols, arguments.skip_features))
    )

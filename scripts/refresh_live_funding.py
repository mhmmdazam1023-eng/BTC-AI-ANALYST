"""Read a fresh funding rate now, and say how it compares to the archive.

For the execution path (the separate "Trading Assistant" bridge), which
reported ``derivatives_metrics.funding_rate`` sitting 5+ days stale. It is
stale for a structural reason: Binance Vision publishes funding in MONTHLY
archives and has no daily equivalent, so the running month simply does not
exist yet. Verified 2026-08-08 — ``daily/fundingRate/`` 404,
``monthly/fundingRate/`` 200.

This does not try to make the archive faster. It runs a SEPARATE path with a
different source and a different table, and prints both side by side with
their ages, so a caller can see exactly which number it is acting on.

Usage::

    python scripts/refresh_live_funding.py            # read + store + report
    python scripts/refresh_live_funding.py --read-only  # report what is stored
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from src.collectors import FetcherError, LiveFundingFetcher  # noqa: E402
from src.collectors.live_funding_fetcher import (  # noqa: E402
    LIVE_STALE_AFTER_MINUTES,
)
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import DerivativesMetric, LiveDerivative  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.refresh_live_funding")


def _age(then: datetime | None, now: datetime) -> str:
    if then is None:
        return "n/a"
    minutes = (now - then).total_seconds() / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} days"


async def report(symbol: str) -> int:
    now = datetime.now(tz=UTC)

    async with session_scope() as session:
        live = (
            await session.execute(
                select(LiveDerivative)
                .where(
                    LiveDerivative.symbol == symbol,
                    LiveDerivative.metric == "funding_rate",
                )
                .order_by(LiveDerivative.observed_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        archive_ts = (
            await session.execute(
                select(func.max(DerivativesMetric.timestamp)).where(
                    DerivativesMetric.metric == "funding_rate"
                )
            )
        ).scalar_one_or_none()
        archive_value = None
        if archive_ts is not None:
            archive_value = (
                await session.execute(
                    select(DerivativesMetric.value).where(
                        DerivativesMetric.metric == "funding_rate",
                        DerivativesMetric.timestamp == archive_ts,
                    )
                )
            ).scalar_one_or_none()

    print()
    print("=" * 74)
    print(f"  FUNDING RATE — {symbol}   (now {now:%Y-%m-%d %H:%M} UTC)")
    print("=" * 74)
    print(f"  {'path':<10}{'source':<10}{'value':>16}{'observed':>22}{'age':>12}")
    print("  " + "-" * 68)

    def _row(path: str, source: str, value, when: datetime | None) -> str:
        shown = "n/a" if value is None else f"{float(value):+.8f}"
        stamp = f"{when:%Y-%m-%d %H:%M}Z" if when else "n/a"
        return (
            f"  {path:<10}{source:<10}{shown:>16}{stamp:>22}"
            f"{_age(when, now):>12}"
        )

    if live is not None:
        print(_row("LIVE", live.source, live.value, live.observed_at))
    else:
        print(f"  {'LIVE':<10}{'-':<10}{'none stored':>16}{'n/a':>22}{'n/a':>12}")
    print(_row("ARCHIVE", "binance", archive_value, archive_ts))
    print("  " + "-" * 68)

    # The comparison is the point, and so is refusing to average them.
    if live is not None and archive_value is not None:
        gap = float(live.value) - float(archive_value)
        print(
            f"  live - archive = {gap:+.8f}  (per {live.funding_interval or '?'} "
            f"interval)"
        )
    print(
        "  These are DIFFERENT EXCHANGES and must not be averaged or\n"
        "  substituted for each other. The archive is the training series;\n"
        "  the live reading is the only one an execution decision may use."
    )

    if live is None:
        print("\n  VERDICT: no live reading — do not act on the archive value.")
        return 1
    minutes = (now - live.observed_at).total_seconds() / 60.0
    if minutes > LIVE_STALE_AFTER_MINUTES:
        print(
            f"\n  VERDICT: live reading is {_age(live.observed_at, now)} old "
            f"(> {LIVE_STALE_AFTER_MINUTES} min) — REFRESH before acting."
        )
        return 1
    print(f"\n  VERDICT: live reading is fresh ({_age(live.observed_at, now)} old).")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Report what is already stored without calling the exchange.",
    )
    args = parser.parse_args()

    try:
        if not args.read_only:
            fetcher = LiveFundingFetcher(symbol=args.symbol)
            try:
                rows = await fetcher.fetch_data()
                await fetcher.save_to_db(rows)
            except FetcherError as exc:
                # A failed read must not erase the last good one: the report
                # still runs and shows how old that reading now is, which is
                # exactly what the caller needs in order to decide.
                logger.error("Live funding unavailable: %s", exc)
            finally:
                await fetcher.close()
        return await report(args.symbol)
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main()))

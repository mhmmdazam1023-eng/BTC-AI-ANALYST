"""Data-collection pipeline orchestrator (Phase 2 - Part 4).

Runs every active fetcher and prints an execution summary. Designed for both
manual runs and Windows Task Scheduler (see README.md).

Concurrency model
-----------------
* Binance Spot, FRED, Alternative.me, Binance Vision, and Stooq/FRED run
  **concurrently** (``asyncio.gather``) — different hosts, generous limits.
* Blockchain.com runs **sequentially afterwards, alone** — its rate limits
  are strict, so it never competes with other traffic from this machine.

Error isolation (hard requirement)
----------------------------------
Every fetcher is wrapped in its own try/except inside :func:`run_one`; a
failing/timing-out fetcher is reported as FAILED in the summary while all
others keep running. The pipeline is idempotent (every fetcher upserts on
its natural key), so re-runs never duplicate data.

Modes
-----
* ``--mode incremental`` (default) — small lookback windows for daily
  scheduled runs (fast, gentle on the APIs).
* ``--mode full``        — full historical backfill (first run / recovery).

Usage::

    python scripts/run_collectors.py                # incremental
    python scripts/run_collectors.py --mode full    # full backfill
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collectors import (  # noqa: E402
    AlternativeMeFetcher,
    BinanceMultiFetcher,
    BinanceVisionFetcher,
    BlockchainFetcher,
    FredFetcher,
    StooqFetcher,
    LiveFundingFetcher,
    SupplyFetcher,
)
from src.collectors.base import BaseDataFetcher  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine  # noqa: E402
from src.features import FeatureEngine  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.run_collectors")


@dataclass
class CollectorResult:
    """Outcome of one fetcher run, for the final summary."""

    name: str
    status: str = "FAILED"  # SUCCESS | SKIPPED | FAILED
    rows: int = 0
    duration_seconds: float = 0.0
    detail: str = ""

    def line(self) -> str:
        text = f"{self.name:<28s}: {self.status}"
        if self.status == "SUCCESS":
            text += f" ({self.rows} rows upserted)"
            if self.detail:  # e.g. a multi-asset run that lost one symbol
                text += f" — {self.detail}"
        elif self.detail:
            text += f" ({self.detail})"
        return f"{text} [{self.duration_seconds:.1f}s]"


# ---------------------------------------------------------------------------
# Fetcher construction per mode
# ---------------------------------------------------------------------------
def build_fetchers(mode: str) -> tuple[list[tuple[str, BaseDataFetcher]], tuple[str, BaseDataFetcher]]:
    """Return (concurrent_group, sequential_blockchain) for the given mode."""
    today = datetime.now(tz=UTC)

    if mode == "full":
        concurrent: list[tuple[str, BaseDataFetcher]] = [
            # limit=1000 (Binance's per-call maximum), raised from 365 in
            # Phase 8: BTC already carries ~1000 days of history, and an
            # altcoin with only 365 could not be compared against it on
            # equal footing. Pure upsert — this only adds rows.
            ("Binance Spot (multi-asset)", BinanceMultiFetcher(limit=1000)),
            ("FRED Macro", FredFetcher()),
            ("Alternative.me Sentiment", AlternativeMeFetcher(limit=0)),
            ("Binance Vision Derivatives", BinanceVisionFetcher()),
            ("Market Indices (Stooq/FRED)", StooqFetcher()),
            ("Supply Metrics (CoinGecko)", SupplyFetcher()),
        # Live funding (2026-08-09). NOT a replacement for the Binance
        # Vision archive above — a different exchange writing to a different
        # table, because the archive's funding is monthly-published and can
        # legitimately be ~31 days stale. One reading per daily run only
        # starts the history; the execution path should call
        # `scripts/refresh_live_funding.py` whenever it actually needs a
        # current number.
        ("Live Funding (gate.io)", LiveFundingFetcher()),
        ]
        blockchain = ("Blockchain.com On-Chain", BlockchainFetcher(timespan="all"))
        return concurrent, blockchain

    # --- incremental: small overlapping windows; upsert dedupes overlap ---
    start_90d = (today - timedelta(days=90)).date().isoformat()
    funding_start = (today - timedelta(days=65)).replace(day=1).date().isoformat()
    concurrent = [
        ("Binance Spot (multi-asset)", BinanceMultiFetcher(limit=45)),
        ("FRED Macro", FredFetcher(observation_start=start_90d)),
        ("Alternative.me Sentiment", AlternativeMeFetcher(limit=45)),
        (
            "Binance Vision Derivatives",
            BinanceVisionFetcher(funding_start=funding_start, oi_backfill_days=7),
        ),
        ("Market Indices (Stooq/FRED)", StooqFetcher(start=start_90d)),
        # CoinGecko returns a snapshot, so mode makes no difference here.
        ("Supply Metrics (CoinGecko)", SupplyFetcher()),
        # See the note in the full-mode branch: separate table, separate
        # exchange, deliberately not merged into the archive.
        ("Live Funding (gate.io)", LiveFundingFetcher()),
    ]
    blockchain = ("Blockchain.com On-Chain", BlockchainFetcher(timespan="1year"))
    return concurrent, blockchain


# ---------------------------------------------------------------------------
# Isolated execution of one fetcher
# ---------------------------------------------------------------------------
async def run_one(name: str, fetcher: BaseDataFetcher) -> CollectorResult:
    """Run one fetcher fully isolated: NEVER raises, always returns a result."""
    result = CollectorResult(name=name)
    started = time.monotonic()
    logger.info(">>> %s: starting", name)
    try:
        rows = await fetcher.fetch_data()
        if not rows:
            result.status = "SKIPPED"
            result.detail = "no data returned — see warnings above (key/wall/API)"
            logger.warning("<<< %s: SKIPPED (no data)", name)
        else:
            result.rows = await fetcher.save_to_db(rows)
            result.status = "SUCCESS"
            # A multi-symbol fetcher can succeed overall while losing one
            # asset, or while serving a denominator from the stored-candle
            # fallback. Both must be visible, not hidden behind a green line.
            notes: list[str] = []
            fallback = getattr(fetcher, "fallback_symbols", None)
            if fallback:
                notes.append(f"STALE FALLBACK for: {', '.join(fallback)}")
            partial = getattr(fetcher, "failed_symbols", None)
            if partial:
                notes.append(f"NO DATA for: {', '.join(partial)}")
            if notes:
                result.detail = "; ".join(notes)
                logger.error("<<< %s: partial success — %s", name, result.detail)
            logger.info("<<< %s: SUCCESS, %d rows upserted", name, result.rows)
    except Exception as exc:  # noqa: BLE001 - error isolation is mandatory here
        result.status = "FAILED"
        result.detail = f"{type(exc).__name__}: {exc}"
        logger.exception("<<< %s: FAILED — pipeline continues with the rest", name)
    finally:
        result.duration_seconds = time.monotonic() - started
        try:
            await fetcher.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must not mask results
            logger.warning("%s: error while closing fetcher (ignored): %s", name, exc)
    return result


# ---------------------------------------------------------------------------
async def run_pipeline(mode: str) -> tuple[int, list[CollectorResult]]:
    """Run all stages; returns (exit_code, per-stage results).

    Importable by scripts/run_daily.py — does NOT dispose the engine.
    """
    started = time.monotonic()
    logger.info("Pipeline starting (mode=%s)", mode)
    concurrent_group, (blockchain_name, blockchain_fetcher) = build_fetchers(mode)

    # Stage 1: everything except Blockchain.com, concurrently.
    results: list[CollectorResult] = list(
        await asyncio.gather(
            *(run_one(name, fetcher) for name, fetcher in concurrent_group)
        )
    )

    # Stage 2: Blockchain.com alone (strict rate limits — no competing traffic).
    logger.info("Concurrent stage done — running %s sequentially", blockchain_name)
    results.append(await run_one(blockchain_name, blockchain_fetcher))

    # Stage 3: recompute daily_features so the agents always see fresh data
    # (approved by user 2026-07-10). Isolated like every other stage.
    feature_result = CollectorResult(name="Feature Engineering")
    feature_started = time.monotonic()
    try:
        feature_result.rows = await FeatureEngine().run()
        feature_result.status = "SUCCESS"
    except Exception as exc:  # noqa: BLE001 - same isolation contract
        feature_result.status = "FAILED"
        feature_result.detail = f"{type(exc).__name__}: {exc}"
        logger.exception("Feature Engineering FAILED")
    finally:
        feature_result.duration_seconds = time.monotonic() - feature_started
    results.append(feature_result)

    total_seconds = time.monotonic() - started
    total_rows = sum(r.rows for r in results)
    failed = [r.name for r in results if r.status == "FAILED"]

    logger.info("=== PIPELINE EXECUTION SUMMARY ===")
    for result in results:
        logger.info(result.line())
    logger.info("-" * 34)
    logger.info(
        "Total: %d rows upserted across %d/%d fetchers in %.1fs (mode=%s)",
        total_rows,
        sum(1 for r in results if r.status == "SUCCESS"),
        len(results),
        total_seconds,
        mode,
    )
    logger.info("==================================")

    if failed:
        logger.error("Pipeline finished WITH FAILURES: %s", ", ".join(failed))
        return 1, results
    logger.info("Pipeline finished cleanly (SKIPPED entries are non-fatal).")
    return 0, results


async def main(mode: str) -> int:
    try:
        exit_code, _ = await run_pipeline(mode)
        return exit_code
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data-collection pipeline.")
    parser.add_argument(
        "--mode",
        choices=("incremental", "full"),
        default="incremental",
        help="incremental: small lookback for daily runs (default); "
        "full: complete historical backfill.",
    )
    arguments = parser.parse_args()
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main(arguments.mode)))

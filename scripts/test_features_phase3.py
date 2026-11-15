"""Phase 3 test: FeatureEngine end-to-end + data-quality gates.

Steps:
1. Run :class:`FeatureEngine` (load 6 source tables -> compute -> upsert
   into ``daily_features``).
2. Validate mathematical sanity: RSI in [0, 100], Bollinger band ordering,
   ATR/realized-vol positivity, OI/volume ratio positivity.
3. Validate the missing-value contract: after capped forward-fill, the most
   recent 30 rows must be COMPLETE (no NULL in any feature column) — this is
   the window the Phase 4 agents will actually consume.
4. Read back from the database and print the latest feature vector.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_features_phase3.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import (  # noqa: E402
    BASE_SYMBOL,
    DAILY_FEATURE_COLUMNS,
    DailyFeature,
)
from src.features import FeatureEngine  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_features_phase3")

#: Recent window that must be gap-free for the Phase 4 agents.
RECENT_COMPLETE_DAYS = 30

#: Columns allowed to be NULL even in the recent window.
#:
#: Phase 8 Part 4 introduced the first legitimately-absent features. These
#: are NOT pipeline faults — they are honest gaps, and filling them would
#: mean inventing data:
#: * ``days_to_next_unlock`` — NULL when no unlock is on file for the coin.
#:   Bitcoin's next halving is not seeded, so BTC is NULL here.
#: * supply-derived — ``supply_metrics`` only grows FORWARD from the day
#:   CoinGecko collection starts (no free historical supply series), and
#:   ``inflation_rate_7d`` additionally needs 7 days of it.
#: * ``market_cap_tier`` — derived from circulating supply, same gap.
RECENT_NULLABLE_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "days_to_next_unlock",
        "inflation_rate_7d",
        "supply_utilization",
        "liquidity_score",
        "market_cap_tier",
        # funding_rate_change_1d (2026-08-09). The LEVEL still forward-fills
        # and is still required above; the CHANGE is NULL on any day whose
        # funding was carried rather than observed.
        #
        # Binance Vision publishes funding in MONTHLY archives with no daily
        # equivalent, so the running month is legitimately absent and the
        # level is filled for up to 35 days. Running diff() over that filled
        # series produced a change of exactly 0.0 — measured 2026-08-08:
        # eight consecutive days of "funding is flat" about days nobody
        # observed, feeding the liquidity pillar. NULL is the honest answer.
        #
        # This entry is what the gate is FOR: a column may only join this
        # list with a reason, so "expected NULL" can never quietly become
        # "nobody noticed it went NULL".
        "funding_rate_change_1d",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def validate_database() -> None:
    """Read back BITCOIN's ``daily_features`` rows and enforce the gates.

    Phase 8 made ``daily_features`` multi-asset, so every query here filters
    on ``symbol``. This test's contract is unchanged: it certifies that the
    BITCOIN pipeline still meets the exact gates it met before altcoins
    existed. Altcoin-specific checks live in ``test_altcoin_phase8.py``.
    """
    async with session_scope() as session:
        total = (
            await session.execute(
                select(func.count())
                .select_from(DailyFeature)
                .where(DailyFeature.symbol == BASE_SYMBOL)
            )
        ).scalar_one()
        _require(total >= 300, f"expected >=300 BTC feature rows, found {total}")

        recent_rows = (
            (
                await session.execute(
                    select(DailyFeature)
                    .where(DailyFeature.symbol == BASE_SYMBOL)
                    .order_by(DailyFeature.timestamp.desc())
                    .limit(RECENT_COMPLETE_DAYS)
                )
            )
            .scalars()
            .all()
        )
        _require(
            len(recent_rows) == RECENT_COMPLETE_DAYS,
            f"expected {RECENT_COMPLETE_DAYS} recent rows, got {len(recent_rows)}",
        )

        # --- Gate 1: completeness of the recent window ---
        for row in recent_rows:
            for column in DAILY_FEATURE_COLUMNS:
                if column in RECENT_NULLABLE_EXCEPTIONS:
                    continue
                _require(
                    getattr(row, column) is not None,
                    f"NULL '{column}' on {row.timestamp.date()} — the recent "
                    f"{RECENT_COMPLETE_DAYS}-day window must be complete",
                )
        logger.info(
            "Gate 1 OK: last %d days complete across all %d feature columns",
            RECENT_COMPLETE_DAYS, len(DAILY_FEATURE_COLUMNS),
        )

        # --- Gate 2: mathematical sanity on the recent window ---
        for row in recent_rows:
            day = row.timestamp.date()
            _require(0 <= row.rsi_14 <= 100, f"RSI out of range on {day}: {row.rsi_14}")
            _require(
                row.bb_lower <= row.bb_middle <= row.bb_upper,
                f"Bollinger ordering broken on {day}",
            )
            _require(
                row.bb_lower <= row.btc_close * 2,  # loose sanity anchor
                f"implausible bb_lower vs close on {day}",
            )
            _require(row.atr_14 > 0, f"non-positive ATR on {day}")
            _require(row.realized_vol_7d > 0, f"non-positive 7d vol on {day}")
            _require(row.realized_vol_30d > 0, f"non-positive 30d vol on {day}")
            _require(0 <= row.fear_greed_value <= 100, f"F&G out of range on {day}")
            _require(row.oi_volume_ratio > 0, f"non-positive OI/volume on {day}")
            _require(row.hash_rate_ma_7d > 0, f"non-positive hash-rate MA on {day}")
            _require(row.tx_count_ma_7d > 0, f"non-positive tx-count MA on {day}")
        logger.info("Gate 2 OK: mathematical sanity holds on the recent window")

        # --- Report: overall coverage + latest vector ---
        latest = recent_rows[0]
        logger.info("Latest feature vector (%s):", latest.timestamp.date())
        for column in DAILY_FEATURE_COLUMNS:
            logger.info("  %-24s = %s", column, getattr(latest, column))


async def main() -> int:
    engine = FeatureEngine()
    try:
        written = await engine.run()
        _require(written >= 300, f"FeatureEngine wrote only {written} rows")
        await validate_database()
    except AssertionError as exc:
        logger.error("PHASE 3 TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("PHASE 3 TEST FAILED with an unexpected error")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    logger.info("=" * 64)
    logger.info("PHASE 3 FEATURE TEST PASSED — daily_features is agent-ready")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main()))

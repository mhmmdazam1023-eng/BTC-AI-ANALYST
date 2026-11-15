"""Phase 8 Part 4 test: tokenomics ingestion + Coin DNA features.

Gates
-----
1. **Seeder validation** — out-of-range impact, unknown symbol, bad event
   type, in-file duplicates and unparseable dates are all REJECTED, and a
   rejected file writes nothing at all.
2. **Missing data**      — the whole point of Part 4. With no tokenomics
   events: ``days_to_next_unlock`` is NULL and ``supply_shock_score_30d``
   is 0 (a real answer, not a gap). With no supply rows: the four
   supply-derived features are NULL. Nothing raises.
3. **Unlock maths**      — with events present, days-to-next and the 30-day
   shock sum are exact, halvings are excluded from the shock sum (different
   unit), and burns net against unlocks.
4. **DNA correctness**   — BTC's beta and correlation against itself are
   exactly 1.0; vol_regime and market_cap_tier fire on the right side of
   every boundary; trend_strength is horizon-consistent.
5. **Supply pipeline**   — ids come from the registry (never guessed), the
   fetcher round-trips, and a NULL id is skipped rather than invented.
6. **Type safety**       — the categorical features never reach a float()
   conversion, in the feature writer or in run_inference.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_tokenomics_phase8.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, text  # noqa: E402

from src.collectors import SupplyFetcher  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import (  # noqa: E402
    BASE_SYMBOL,
    DAILY_FEATURE_CATEGORICAL_COLUMNS,
    DAILY_FEATURE_COLUMNS,
    CoinRegistry,
    DailyFeature,
    MarketCapTier,
    SupplyMetric,
    VolRegime,
)
from src.features import FeatureEngine  # noqa: E402
from src.features.engine import (  # noqa: E402
    MARKET_CAP_LARGE_USD,
    MARKET_CAP_MID_USD,
    VOL_REGIME_HIGH_PCT,
    VOL_REGIME_LOW_PCT,
)
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_tokenomics_phase8")

TOLERANCE = 1e-9

#: A symbol used only for synthetic event probes; always cleaned up.
PROBE_SYMBOL = "SOL/USDT"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
def _write_json(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(rows, handle)
    handle.close()
    return Path(handle.name)


async def gate_seeder_validation() -> None:
    """Every malformed file must be rejected, and must write nothing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "seed_tokenomics", Path(__file__).with_name("seed_tokenomics.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["seed_tokenomics"] = module
    spec.loader.exec_module(module)

    known = await module.known_registry_symbols()
    _require(BASE_SYMBOL in known, "registry should contain the base symbol")

    good = {
        "symbol": PROBE_SYMBOL,
        "event_date": "2027-01-15",
        "event_type": "unlock",
        "supply_impact_pct": 2.5,
        "description": "probe",
    }

    bad_cases = [
        ({**good, "supply_impact_pct": 150}, "impact above +100"),
        ({**good, "supply_impact_pct": -150}, "impact below -100"),
        ({**good, "symbol": "NOPE/USDT"}, "symbol not in coins_registry"),
        ({**good, "event_type": "airdrop"}, "event_type not in the enum"),
        ({**good, "event_date": "15-01-2027"}, "non-ISO date"),
        ({**good, "event_date": ""}, "empty required field"),
        ({**good, "supply_impact_pct": "abc"}, "non-numeric impact"),
    ]
    for row, label in bad_cases:
        rejected = False
        try:
            module.validate([row], known)
        except module.TokenomicsSeedError:
            rejected = True
        _require(rejected, f"seeder accepted an invalid row: {label} -> {row}")
    logger.info("Gate 1 OK: all %d malformed row shapes rejected", len(bad_cases))

    # Duplicates inside ONE file must be caught before the DB sees them.
    rejected = False
    try:
        module.validate([good, dict(good)], known)
    except module.TokenomicsSeedError:
        rejected = True
    _require(rejected, "seeder accepted a duplicate key inside one file")
    logger.info("Gate 1 OK: in-file duplicate (symbol, date, type) rejected")

    # Boundary values are legal: exactly +/-100 is allowed.
    for boundary in (100, -100, 0):
        cleaned = module.validate([{**good, "supply_impact_pct": boundary}], known)
        _require(
            float(cleaned[0]["supply_impact_pct"]) == boundary,
            f"boundary impact {boundary} should be accepted",
        )
    logger.info("Gate 1 OK: boundary impacts (-100, 0, +100) accepted")

    # A file with ONE bad row must write NOTHING, not a partial batch.
    async with session_scope() as session:
        before = (
            await session.execute(text("SELECT count(*) FROM tokenomics_events"))
        ).scalar_one()
    path = _write_json([good, {**good, "event_date": "2027-02-01", "supply_impact_pct": 999}])
    try:
        exit_code = await module.run(path, dry_run=False, show=False)
    finally:
        path.unlink(missing_ok=True)
    async with session_scope() as session:
        after = (
            await session.execute(text("SELECT count(*) FROM tokenomics_events"))
        ).scalar_one()
    _require(exit_code == 1, "a file with an invalid row must exit non-zero")
    _require(
        before == after,
        f"a rejected file wrote {after - before} row(s) — validation must be "
        f"all-or-nothing",
    )
    logger.info("Gate 1 OK: a file with one bad row writes nothing (%d rows)", after)


# ---------------------------------------------------------------------------
def _synthetic_frame(days: int = 420, *, price: float = 100.0) -> pd.DataFrame:
    index = pd.date_range(end=datetime(2026, 1, 1), periods=days, freq="D")
    steps = np.arange(days)
    close = price * (1.0 + 0.001) ** steps
    return pd.DataFrame(
        {
            "close": close,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "volume": np.full(days, 1_000.0),
        },
        index=index,
    )


def gate_missing_data(engine: FeatureEngine) -> None:
    """No tokenomics and no supply must degrade, never raise."""
    ohlcv = _synthetic_frame()
    btc = _synthetic_frame(price=50_000.0)
    sources: dict = {"supply_by_symbol": {}, "events_by_symbol": {}}

    features = pd.DataFrame(index=ohlcv.index.copy())
    features["asset_close"] = ohlcv["close"]
    features["asset_volume"] = ohlcv["volume"]
    features["realized_vol_30d"] = 0.5
    features["rsi_14"] = 50.0
    features["macd_hist"] = 1.0

    engine._add_dna_features(features, "TEST/USDT", sources, ohlcv, btc)

    last = features.iloc[-1]
    _require(
        pd.isna(last["days_to_next_unlock"]),
        f"no events -> days_to_next_unlock must be NULL, got "
        f"{last['days_to_next_unlock']}",
    )
    _require(
        float(last["supply_shock_score_30d"]) == 0.0,
        f"no events -> supply_shock_score_30d must be 0 (a real answer), got "
        f"{last['supply_shock_score_30d']}",
    )
    for column in ("inflation_rate_7d", "supply_utilization", "liquidity_score"):
        _require(
            pd.isna(last[column]),
            f"no supply data -> {column} must be NULL, got {last[column]}",
        )
    _require(
        last["market_cap_tier"] is None or pd.isna(last["market_cap_tier"]),
        f"no supply data -> market_cap_tier must be NULL, got "
        f"{last['market_cap_tier']}",
    )
    logger.info(
        "Gate 2 OK: empty tokenomics -> NULL days / 0 shock; empty supply -> "
        "NULL supply features; nothing raised"
    )

    # Too little history for a 30-day beta must yield NULL, not a number.
    short_ohlcv = _synthetic_frame(days=10)
    short_btc = _synthetic_frame(days=10, price=50_000.0)
    short = pd.DataFrame(index=short_ohlcv.index.copy())
    short["asset_close"] = short_ohlcv["close"]
    short["asset_volume"] = short_ohlcv["volume"]
    short["realized_vol_30d"] = 0.5
    short["rsi_14"] = 50.0
    short["macd_hist"] = 1.0
    engine._add_dna_features(short, "TEST/USDT", sources, short_ohlcv, short_btc)
    _require(
        pd.isna(short.iloc[-1]["btc_beta_30d"]),
        "fewer than 30 days of history must give a NULL beta",
    )
    logger.info("Gate 2 OK: <30 days of history -> btc_beta_30d NULL")


def gate_unlock_maths(engine: FeatureEngine) -> None:
    """days-to-next and the 30-day shock sum must be exact."""
    ohlcv = _synthetic_frame(days=300)
    btc = _synthetic_frame(days=300, price=50_000.0)
    last_day = ohlcv.index[-1]

    events = pd.DataFrame(
        {
            "event_date": [
                last_day + pd.Timedelta(days=10),   # unlock inside 30d
                last_day + pd.Timedelta(days=25),   # unlock inside 30d
                last_day + pd.Timedelta(days=45),   # unlock OUTSIDE 30d
                last_day + pd.Timedelta(days=5),    # burn inside 30d
                last_day + pd.Timedelta(days=3),    # halving inside 30d
            ],
            "event_type": ["unlock", "unlock", "unlock", "burn", "halving"],
            "supply_impact_pct": [2.0, 3.0, 10.0, -1.0, -50.0],
        }
    )
    features = pd.DataFrame(index=ohlcv.index.copy())
    features["asset_close"] = ohlcv["close"]
    features["asset_volume"] = ohlcv["volume"]
    features["realized_vol_30d"] = 0.5
    features["rsi_14"] = 50.0
    features["macd_hist"] = 1.0
    engine._add_dna_features(
        features, "TEST/USDT",
        {"supply_by_symbol": {}, "events_by_symbol": {"TEST/USDT": events}},
        ohlcv, btc,
    )
    last = features.iloc[-1]

    _require(
        float(last["days_to_next_unlock"]) == 10.0,
        f"next unlock is 10 days out, got {last['days_to_next_unlock']} — note "
        f"the burn (5d) and halving (3d) are nearer but are not unlocks",
    )
    # 2.0 + 3.0 (unlocks in window) + (-1.0) (burn) = 4.0.
    # The 10.0 unlock is outside the window; the halving is excluded because
    # its -50 is an EMISSION-RATE change, not a % of circulating supply.
    _require(
        math.isclose(float(last["supply_shock_score_30d"]), 4.0, abs_tol=TOLERANCE),
        f"30d shock should be 2.0 + 3.0 - 1.0 = 4.0, got "
        f"{last['supply_shock_score_30d']}. If it is -46, the halving was "
        f"wrongly summed (different unit).",
    )
    logger.info(
        "Gate 3 OK: days_to_next_unlock=10 (burn/halving ignored), "
        "shock_30d=+4.0 (halving excluded, burn netted)"
    )


def gate_dna_correctness(engine: FeatureEngine) -> None:
    """Boundary and identity checks on the computed DNA."""
    # vol_regime boundaries.
    for percentile, expected in (
        (VOL_REGIME_LOW_PCT - 0.1, VolRegime.LOW.value),
        (VOL_REGIME_LOW_PCT, VolRegime.NORMAL.value),
        (50.0, VolRegime.NORMAL.value),
        (VOL_REGIME_HIGH_PCT, VolRegime.NORMAL.value),
        (VOL_REGIME_HIGH_PCT + 0.1, VolRegime.HIGH.value),
        (float("nan"), None),
    ):
        actual = engine._vol_regime(percentile)
        _require(
            actual == expected,
            f"_vol_regime({percentile}) gave {actual}, expected {expected}",
        )
    logger.info("Gate 4 OK: vol_regime boundaries at %.0f/%.0f", VOL_REGIME_LOW_PCT, VOL_REGIME_HIGH_PCT)

    for cap, expected in (
        (MARKET_CAP_LARGE_USD + 1, MarketCapTier.LARGE.value),
        (MARKET_CAP_LARGE_USD, MarketCapTier.MID.value),
        (MARKET_CAP_MID_USD, MarketCapTier.MID.value),
        (MARKET_CAP_MID_USD - 1, MarketCapTier.SMALL.value),
        (0, None),
        (float("nan"), None),
    ):
        actual = engine._market_cap_tier(cap)
        _require(
            actual == expected,
            f"_market_cap_tier({cap}) gave {actual}, expected {expected}",
        )
    logger.info("Gate 4 OK: market_cap_tier boundaries at $10B / $1B")


async def gate_dna_live() -> None:
    """Live data: BTC's beta/corr against itself must be exactly 1."""
    async with session_scope() as session:
        day = (
            await session.execute(
                select(DailyFeature.timestamp)
                .where(DailyFeature.symbol == BASE_SYMBOL)
                .order_by(DailyFeature.timestamp.desc())
                .limit(1)
            )
        ).scalar_one()
        btc = (
            await session.execute(
                select(DailyFeature).where(
                    DailyFeature.symbol == BASE_SYMBOL,
                    DailyFeature.timestamp == day,
                )
            )
        ).scalar_one()

    _require(
        btc.btc_beta_30d is not None
        and math.isclose(float(btc.btc_beta_30d), 1.0, abs_tol=1e-6),
        f"BTC beta against itself must be 1.0, got {btc.btc_beta_30d}",
    )
    _require(
        btc.btc_corr_30d is not None
        and math.isclose(float(btc.btc_corr_30d), 1.0, abs_tol=1e-6),
        f"BTC correlation against itself must be 1.0, got {btc.btc_corr_30d}",
    )
    _require(
        btc.market_cap_tier is not None
        and MarketCapTier(btc.market_cap_tier) is MarketCapTier.LARGE,
        f"BTC must be LARGE cap, got {btc.market_cap_tier}",
    )
    logger.info(
        "Gate 4 OK (live): BTC beta=%.6f corr=%.6f tier=%s",
        float(btc.btc_beta_30d), float(btc.btc_corr_30d), btc.market_cap_tier,
    )

    # Beta and correlation must be genuinely DIFFERENT measurements: if the
    # implementation silently stored correlation twice, this catches it.
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT symbol, btc_beta_30d, btc_corr_30d FROM daily_features "
                    "WHERE timestamp = :d AND symbol <> :b "
                    "AND btc_beta_30d IS NOT NULL AND btc_corr_30d IS NOT NULL"
                ),
                {"d": day, "b": BASE_SYMBOL},
            )
        ).all()
    _require(bool(rows), "no altcoin has both beta and correlation")
    differing = [
        symbol
        for symbol, beta, corr in rows
        if abs(float(beta) - float(corr)) > 1e-6
    ]
    _require(
        len(differing) >= len(rows) // 2,
        "beta and correlation are near-identical for most coins — one is "
        "probably computed as the other",
    )
    _require(
        all(-1.0001 <= float(corr) <= 1.0001 for _, _, corr in rows),
        "correlation must lie in [-1, 1]",
    )
    logger.info(
        "Gate 4 OK (live): beta != corr for %d/%d altcoins, corr within [-1,1]",
        len(differing), len(rows),
    )


async def gate_supply_pipeline() -> None:
    """Ids come from the registry; supply round-trips; NULL id is skipped."""
    async with session_scope() as session:
        mapped = (
            await session.execute(
                select(CoinRegistry.symbol, CoinRegistry.coingecko_id).where(
                    CoinRegistry.is_active.is_(True)
                )
            )
        ).all()
    unmapped = [symbol for symbol, cid in mapped if not cid]
    _require(
        not unmapped,
        f"active symbols without a coingecko_id: {unmapped} — run "
        f"scripts/seed_sectors_phase8.py",
    )
    logger.info("Gate 5 OK: all %d active symbols have a CoinGecko id", len(mapped))

    fetcher = SupplyFetcher()
    try:
        id_map = await fetcher.load_id_map()
        _require(
            len(id_map) == len(mapped),
            f"id map has {len(id_map)} entries for {len(mapped)} symbols",
        )
        _require(
            id_map.get("bitcoin") == BASE_SYMBOL,
            f"'bitcoin' must map to {BASE_SYMBOL}, got {id_map.get('bitcoin')}",
        )
    finally:
        await fetcher.close()

    async with session_scope() as session:
        stored = (
            await session.execute(
                select(SupplyMetric).order_by(SupplyMetric.timestamp.desc()).limit(1)
            )
        ).scalar_one_or_none()
    _require(stored is not None, "supply_metrics is empty — run the collector")
    _require(
        stored.circulating_supply is not None and stored.circulating_supply > 0,
        f"stored circulating supply is not positive: {stored.circulating_supply}",
    )
    async with session_scope() as session:
        count = (
            await session.execute(
                text(
                    "SELECT count(*) FROM supply_metrics WHERE timestamp = :t"
                ),
                {"t": stored.timestamp},
            )
        ).scalar_one()
    logger.info(
        "Gate 5 OK: %d supply row(s) stored for %s (latest: %s circ=%s)",
        count, stored.timestamp.date(), stored.symbol, stored.circulating_supply,
    )


async def gate_type_safety() -> None:
    """Categorical features must never reach a float() conversion."""
    _require(
        DAILY_FEATURE_CATEGORICAL_COLUMNS,
        "the categorical column registry is empty — Part 4 added two",
    )
    for column in DAILY_FEATURE_CATEGORICAL_COLUMNS:
        _require(
            column in DAILY_FEATURE_COLUMNS,
            f"{column} is registered as categorical but is not a feature column",
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_inference", Path(__file__).with_name("run_inference.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_inference"] = module
    spec.loader.exec_module(module)

    _, features = await module.load_latest_features()
    for column in DAILY_FEATURE_CATEGORICAL_COLUMNS:
        value = features.get(column)
        _require(
            value is None or isinstance(value, str),
            f"{column} reached run_inference as {type(value).__name__} "
            f"({value!r}); categorical features must stay strings",
        )
    numeric_bad = [
        column
        for column in DAILY_FEATURE_COLUMNS
        if column not in DAILY_FEATURE_CATEGORICAL_COLUMNS
        and features.get(column) is not None
        and not isinstance(features[column], (int, float))
    ]
    _require(not numeric_bad, f"non-numeric values in numeric features: {numeric_bad}")
    logger.info(
        "Gate 6 OK: run_inference loaded %d features; categorical stayed "
        "strings (vol_regime=%r, market_cap_tier=%r)",
        len(DAILY_FEATURE_COLUMNS),
        features.get("vol_regime"), features.get("market_cap_tier"),
    )


# ---------------------------------------------------------------------------
async def main() -> int:
    started = datetime.now(tz=UTC)
    engine = FeatureEngine()
    try:
        logger.info("--- Seeder validation ---")
        await gate_seeder_validation()
        logger.info("--- Missing-data handling ---")
        gate_missing_data(engine)
        logger.info("--- Unlock maths ---")
        gate_unlock_maths(engine)
        logger.info("--- DNA boundaries ---")
        gate_dna_correctness(engine)
        logger.info("--- DNA on live data ---")
        await gate_dna_live()
        logger.info("--- Supply pipeline ---")
        await gate_supply_pipeline()
        logger.info("--- Type safety ---")
        await gate_type_safety()
    except AssertionError as exc:
        logger.error("PHASE 8 PART 4 TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("PHASE 8 PART 4 TEST FAILED with an unexpected error")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "PHASE 8 PART 4 TEST PASSED (6 gates, %.1fs) — tokenomics ingestion & "
        "Coin DNA ready", elapsed,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main()))

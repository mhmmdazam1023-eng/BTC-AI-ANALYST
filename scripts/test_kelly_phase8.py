"""Phase 8 Part 7 test: scenario engine + Kelly position sizing.

Gates
-----
1. **Ideal case**    — win rate 60%, payoff 2:1 gives a positive, sane
   fraction after half-Kelly and the cap.
2. **Dangerous case**— HIGH volatility plus an unlock in 7 days forces the
   position to exactly zero.
3. **Missing data**  — no win-rate history must not crash, and must fall
   back to 0.50 rather than to whatever a six-sample average happened to be.
4. **Conservatism**  — full Kelly is never used as-is; a negative edge and a
   non-BUY action both give zero; the per-asset cap holds.
5. **Null safety**   — a NULL ``days_to_next_unlock`` must NOT be treated as
   an imminent unlock, and a NULL vol_regime must not apply the vol overlay.
6. **Scenarios**     — probabilities always sum to 1.0 after any tilt;
   regime and supply-shock tilts move mass in the right direction.
7. **Persistence**   — recommendations round-trip, are idempotent, and carry
   their provenance (win-rate source/sample size, scenario source).
8. **Band calibration** — the empirical sqrt-time correction widens with
   horizon, clamps outside the measured range, is SQUARED in variance space,
   and EWMA volatility is preferred over the equally-weighted estimate.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_kelly_phase8.py
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, text  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import BASE_SYMBOL, PositionRecommendation  # noqa: E402
from src.forecasting import (  # noqa: E402
    PositionPipeline,
    PositionSizer,
    ScenarioEngine,
    ScenarioSet,
    portfolio_exposure,
)
from src.forecasting.kelly_criterion import (  # noqa: E402
    DEFAULT_WIN_RATE,
    KELLY_FRACTION,
    MAX_POSITION_FRACTION,
    MIN_WIN_RATE_SAMPLES,
    UNLOCK_BLACKOUT_DAYS,
)
from src.forecasting.scenario_engine import (  # noqa: E402
    horizon_band_multiplier,
)
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_kelly_phase8")

TOLERANCE = 1e-9

#: Skill-free BUY rate for the fixtures. 0.50 is chosen for the SYNTHETIC
#: cases only — real assets are nowhere near it (BTC +30d is 0.609), which is
#: precisely why sizing takes it as an input instead of assuming it.
NULL_WIN_RATE = 0.50

#: Variance of a 30-day return at ~95% annualised vol: 0.955^2 * 30/365.
HORIZON_VARIANCE = 0.075


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _scenarios(
    *,
    bull: float = 0.20,
    bear: float = 0.10,
    symbol: str = "TEST/USDT",
    source: str = "monte_carlo",
) -> ScenarioSet:
    """A scenario set with a chosen payoff ratio (bull / bear)."""
    return ScenarioSet(
        symbol=symbol,
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        horizon_days=30,
        source=source,
        prob_bull=0.25,
        prob_base=0.50,
        prob_bear=0.25,
        expected_return_bull=bull,
        expected_return_base=0.01,
        expected_drawdown_bear=bear,
        base_price=100.0,
    )


# ---------------------------------------------------------------------------
def gate_ideal_case(sizer: PositionSizer) -> None:
    """Win rate 60%, payoff 2:1 -> a positive, sane fraction."""
    scenarios = _scenarios(bull=0.20, bear=0.10)  # payoff = 2.0
    _require(
        math.isclose(scenarios.payoff_ratio, 2.0, abs_tol=TOLERANCE),
        f"fixture payoff should be 2.0, got {scenarios.payoff_ratio}",
    )

    # f* = mu / sigma^2, where mu = (p - p_null)*(gain + loss).
    # (0.60 - 0.50) * (0.20 + 0.10) = 0.030;  0.030 / 0.075 = 0.40
    mu = sizer.excess_return(0.60, NULL_WIN_RATE, 0.20, 0.10)
    _require(
        math.isclose(mu, 0.030, abs_tol=1e-9),
        f"excess return should be 0.030, got {mu}",
    )
    full = sizer.continuous_kelly(mu, HORIZON_VARIANCE)
    _require(
        math.isclose(full, 0.40, abs_tol=1e-9),
        f"continuous Kelly for mu=0.03 sigma^2=0.075 should be 0.40, got {full}",
    )

    # THE FIX, asserted directly: a 10-point edge over the null is worth less
    # in a noisier asset. The old binary formula had this backwards — its
    # payoff ratio was exactly exp(z*sigma), so more volatility meant a
    # BIGGER position (ONDO 16.5% vs BNB 5.5% on volatility alone).
    calm = sizer.continuous_kelly(mu, HORIZON_VARIANCE)
    wild = sizer.continuous_kelly(mu, HORIZON_VARIANCE * 4)
    _require(
        wild < calm and math.isclose(wild, calm / 4, abs_tol=1e-9),
        f"quadrupling variance must quarter the position ({calm} -> {wild}); "
        f"if it rises, sizing is once again rewarding volatility",
    )

    # And no measured skill must mean NO position, exactly.
    _require(
        sizer.continuous_kelly(
            sizer.excess_return(NULL_WIN_RATE, NULL_WIN_RATE, 0.20, 0.10),
            HORIZON_VARIANCE,
        )
        == 0.0,
        "p == p_null must give exactly 0.0 — an unmeasured edge is not a "
        "small edge, and the old formula returned ~8% here from lognormal "
        "convexity alone",
    )

    sizing = sizer.size(
        "TEST/USDT", "BUY", scenarios, {},
        win_rate=0.60, win_rate_source="empirical", win_rate_samples=120,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
    )
    # 0.40 full -> 0.20 half -> under the 0.25 cap.
    _require(
        math.isclose(sizing.adjusted_fraction, 0.20, abs_tol=1e-9),
        f"half Kelly of 0.40 should be 0.20, got {sizing.adjusted_fraction}",
    )
    _require(
        0.10 <= sizing.adjusted_fraction <= 0.20,
        f"expected a 10-20% position, got {sizing.position_size_pct:.2f}%",
    )
    _require(
        sizing.adjusted_fraction < sizing.kelly_fraction,
        "the adjusted fraction must always be below raw full Kelly",
    )
    logger.info(
        "Gate 1 OK: p=0.60 vs null 0.50, sigma^2=%.3f -> f*=%.4f, half %.4f "
        "(%.1f%% position); 4x variance quarters it; p==null gives exactly 0",
        HORIZON_VARIANCE, sizing.kelly_fraction, sizing.adjusted_fraction,
        sizing.position_size_pct,
    )


def gate_band_calibration() -> None:
    """The sqrt-time correction must widen with horizon, and be squared in
    variance.

    Square-root-of-time alone is measurably too narrow on this data: a
    nominal 90% band covers 87.0% at +30d and 86.2% at +60d. The multiplier
    table fixes that empirically. This gate locks its SHAPE — a future edit
    that flattens it back to 1.0 everywhere silently restores the
    under-coverage, and nothing else would fail.
    """
    # Monotone non-decreasing: longer horizon never needs a NARROWER band.
    values = [horizon_band_multiplier(h) for h in (1, 3, 7, 14, 30, 60)]
    _require(
        all(a <= b for a, b in zip(values, values[1:])),
        f"the multiplier must never shrink with horizon, got {values}",
    )
    _require(
        values[0] < 1.0 < values[-1],
        f"sqrt-time is too WIDE at +1d and too NARROW at +60d; the table must "
        f"straddle 1.0, got {values[0]} .. {values[-1]}",
    )
    _require(
        horizon_band_multiplier(30) > 1.10,
        f"+30d needs a >10% widening (measured 1.160), got "
        f"{horizon_band_multiplier(30)}",
    )

    # Interpolation between anchors, clamping outside them. Extrapolating to
    # 180 days would be inventing a number the measurement does not support.
    mid = horizon_band_multiplier(45)
    _require(
        horizon_band_multiplier(30) < mid < horizon_band_multiplier(60),
        f"+45d must interpolate between its neighbours, got {mid}",
    )
    _require(
        horizon_band_multiplier(180) == horizon_band_multiplier(60),
        "beyond the measured range the multiplier must CLAMP, not extrapolate",
    )

    # And the variance must square it: sigma scales by k, variance by k^2.
    dna = {"ewma_vol_30d": 0.60}
    var = PositionPipeline._horizon_variance(dna, 30)
    expected = (0.60**2) * (30 / 365.0) * (horizon_band_multiplier(30) ** 2)
    _require(
        var is not None and math.isclose(var, expected, rel_tol=1e-12),
        f"horizon variance {var} != {expected} — the multiplier must be "
        f"SQUARED in variance space; applying it linearly under-widens by its "
        f"own square root and still looks plausible",
    )

    # EWMA preferred, realized as fallback — never crash when one is absent.
    _require(
        PositionPipeline._horizon_variance({"realized_vol_30d": 0.60}, 30) is not None,
        "must fall back to realized_vol_30d when EWMA is absent",
    )
    ewma_first = PositionPipeline._horizon_variance(
        {"ewma_vol_30d": 0.40, "realized_vol_30d": 0.90}, 30
    )
    _require(
        ewma_first is not None and ewma_first < expected,
        "EWMA must WIN over realized_vol_30d when both are present",
    )
    _require(
        PositionPipeline._horizon_variance({}, 30) is None,
        "no volatility at all must give None, not a guess",
    )
    logger.info(
        "Gate 8 OK: multiplier %.3f (+1d) -> %.3f (+30d) -> %.3f (+60d), "
        "clamped beyond; squared in variance; EWMA preferred",
        values[0], horizon_band_multiplier(30), values[-1],
    )


def gate_dangerous_case(sizer: PositionSizer) -> None:
    """HIGH volatility + unlock in 7 days -> exactly zero."""
    scenarios = _scenarios(bull=0.30, bear=0.10)  # payoff 3.0, strong edge
    dangerous = {
        "vol_regime": "HIGH",
        "days_to_next_unlock": 7,
        "supply_shock_score_30d": 8.0,
    }
    sizing = sizer.size(
        "RISKY/USDT", "BUY", scenarios, dangerous,
        win_rate=0.65, win_rate_source="empirical", win_rate_samples=200,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
    )
    _require(
        sizing.adjusted_fraction == 0.0,
        f"an unlock in {dangerous['days_to_next_unlock']} days must force zero, "
        f"got {sizing.adjusted_fraction}",
    )
    _require(
        "unlock_blackout" in sizing.risk_factors,
        f"the blackout reason must be recorded: {sizing.risk_factors}",
    )
    _require(
        sizing.kelly_fraction > 0,
        "raw Kelly should still be reported as positive — the zero comes from "
        "the overlay, and hiding that would erase why",
    )
    logger.info(
        "Gate 2 OK: unlock in 7d zeroes a %.4f raw Kelly (reason recorded)",
        sizing.kelly_fraction,
    )

    # HIGH vol alone must halve rather than zero.
    high_vol_only = sizer.size(
        "RISKY/USDT", "BUY", scenarios, {"vol_regime": "HIGH"},
        win_rate=0.65, win_rate_source="empirical", win_rate_samples=200,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
    )
    normal_vol = sizer.size(
        "RISKY/USDT", "BUY", scenarios, {"vol_regime": "NORMAL"},
        win_rate=0.65, win_rate_source="empirical", win_rate_samples=200,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
    )
    _require(
        high_vol_only.adjusted_fraction < normal_vol.adjusted_fraction,
        "HIGH volatility must reduce the position versus NORMAL",
    )
    _require(
        high_vol_only.adjusted_fraction > 0,
        "HIGH volatility alone should REDUCE, not zero",
    )
    logger.info(
        "Gate 2 OK: HIGH vol %.4f vs NORMAL %.4f (reduced, not zeroed)",
        high_vol_only.adjusted_fraction, normal_vol.adjusted_fraction,
    )


async def gate_missing_data(sizer: PositionSizer) -> None:
    """Thin or absent history must fall back, not crash or over-trust."""
    win_rate, source, samples = await sizer.load_win_rate()
    _require(
        0.0 <= win_rate <= 1.0, f"win rate out of range: {win_rate}"
    )
    if samples < MIN_WIN_RATE_SAMPLES:
        _require(
            math.isclose(win_rate, DEFAULT_WIN_RATE, abs_tol=TOLERANCE),
            f"with only {samples} sample(s) the DEFAULT {DEFAULT_WIN_RATE} must "
            f"be used, got {win_rate} — a thin empirical estimate must never "
            f"reach Kelly",
        )
        _require(
            source.startswith("default"),
            f"the source must say the default was used, got {source!r}",
        )
    logger.info(
        "Gate 3 OK: win rate %.4f from %d sample(s) via '%s' (threshold %d)",
        win_rate, samples, source, MIN_WIN_RATE_SAMPLES,
    )

    # A brand-new asset: no DNA at all must still size without crashing.
    sizing = sizer.size(
        "NEW/USDT", "BUY", _scenarios(), {},
        win_rate=DEFAULT_WIN_RATE,
        win_rate_source="default_insufficient_data",
        win_rate_samples=0,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
    )
    _require(
        sizing.adjusted_fraction >= 0,
        "a new asset must produce a valid fraction, not an error",
    )
    _require(
        sizing.win_rate_samples == 0
        and sizing.win_rate_source == "default_insufficient_data",
        "provenance must record that no history backed this number",
    )
    logger.info(
        "Gate 3 OK: an asset with no history sizes to %.4f with provenance "
        "'%s' (n=0)", sizing.adjusted_fraction, sizing.win_rate_source,
    )


def gate_conservatism(sizer: PositionSizer) -> None:
    """Full Kelly is never used raw; no edge and non-BUY both give zero."""
    _require(
        KELLY_FRACTION < 1.0,
        f"fractional Kelly must be < 1.0, found {KELLY_FRACTION}",
    )

    # Negative edge -> zero.
    no_edge = sizer.size(
        "TEST/USDT", "BUY", _scenarios(bull=0.05, bear=0.20), {},
        win_rate=0.40, win_rate_source="empirical", win_rate_samples=100,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
    )
    _require(
        no_edge.adjusted_fraction == 0.0,
        f"a negative edge must give zero, got {no_edge.adjusted_fraction}",
    )
    _require("no_edge" in no_edge.risk_factors, "the no-edge reason must be recorded")

    # Non-BUY actions -> zero, whatever the maths says.
    for action in ("HOLD", "SELL"):
        sizing = sizer.size(
            "TEST/USDT", action, _scenarios(bull=0.40, bear=0.05), {},
            win_rate=0.80, win_rate_source="empirical", win_rate_samples=500,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
        )
        _require(
            sizing.adjusted_fraction == 0.0,
            f"{action} must never be sized, got {sizing.adjusted_fraction}",
        )
        _require(
            "direction_gate" in sizing.risk_factors,
            f"{action} should record the direction gate",
        )
    logger.info("Gate 4 OK: negative edge and HOLD/SELL all give exactly zero")

    # The cap must bind on an extreme edge.
    extreme = sizer.size(
        "TEST/USDT", "BUY", _scenarios(bull=1.00, bear=0.05), {},
        win_rate=0.95, win_rate_source="empirical", win_rate_samples=500,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
    )
    _require(
        extreme.adjusted_fraction <= MAX_POSITION_FRACTION + TOLERANCE,
        f"the cap must bind: got {extreme.adjusted_fraction} > "
        f"{MAX_POSITION_FRACTION}",
    )
    _require(
        "position_cap" in extreme.risk_factors,
        "the cap should be recorded when it binds",
    )
    logger.info(
        "Gate 4 OK: an extreme edge (raw f*=%.3f) is capped at %.0f%%",
        extreme.kelly_fraction, MAX_POSITION_FRACTION * 100,
    )

    # Portfolio-level exposure must be surfaced, not silently exceeded.
    book = [
        sizer.size(
            f"A{i}/USDT", "BUY", _scenarios(bull=1.00, bear=0.05), {},
            win_rate=0.95, win_rate_source="empirical", win_rate_samples=500,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
        )
        for i in range(10)
    ]
    exposure = portfolio_exposure(book)
    _require(
        exposure["over_allocated"],
        f"10 assets at the cap should flag over-allocation, got {exposure}",
    )
    logger.info(
        "Gate 4 OK: 10 capped positions = %.0f%% total, flagged as "
        "over-allocated (per-asset caps do not bound the book)",
        exposure["total_pct"],
    )


def gate_null_safety(sizer: PositionSizer) -> None:
    """A NULL must never trigger a risk overlay."""
    scenarios = _scenarios(bull=0.20, bear=0.10)
    baseline = sizer.size(
        "TEST/USDT", "BUY", scenarios, {},
        win_rate=0.60, win_rate_source="empirical", win_rate_samples=100,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
    )
    null_dna = sizer.size(
        "TEST/USDT", "BUY", scenarios,
        {"vol_regime": None, "days_to_next_unlock": None,
         "supply_shock_score_30d": None},
        win_rate=0.60, win_rate_source="empirical", win_rate_samples=100,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
    )
    _require(
        math.isclose(
            null_dna.adjusted_fraction, baseline.adjusted_fraction, abs_tol=TOLERANCE
        ),
        "explicit NULLs must behave exactly like absent keys, got "
        f"{null_dna.adjusted_fraction} vs {baseline.adjusted_fraction}",
    )
    _require(
        null_dna.adjusted_fraction > 0,
        "a NULL unlock date must NOT be read as an imminent unlock — that "
        "would zero out every asset whose tokenomics nobody has entered",
    )
    _require(
        "unlock_blackout" not in null_dna.risk_factors,
        "a NULL unlock date must not trigger the blackout",
    )
    logger.info(
        "Gate 5 OK: NULL DNA leaves the position at %.4f (no overlay fired)",
        null_dna.adjusted_fraction,
    )

    # A far-off unlock must not trigger it either.
    far = sizer.size(
        "TEST/USDT", "BUY", scenarios,
        {"days_to_next_unlock": UNLOCK_BLACKOUT_DAYS + 1},
        win_rate=0.60, win_rate_source="empirical", win_rate_samples=100,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
    )
    _require(
        far.adjusted_fraction > 0,
        f"an unlock {UNLOCK_BLACKOUT_DAYS + 1} days out must not zero the "
        f"position",
    )
    boundary = sizer.size(
        "TEST/USDT", "BUY", scenarios,
        {"days_to_next_unlock": UNLOCK_BLACKOUT_DAYS - 1},
        win_rate=0.60, win_rate_source="empirical", win_rate_samples=100,
        null_win_rate=NULL_WIN_RATE,
        horizon_variance=HORIZON_VARIANCE,
    )
    _require(
        boundary.adjusted_fraction == 0.0,
        f"an unlock {UNLOCK_BLACKOUT_DAYS - 1} days out MUST zero the position",
    )
    logger.info(
        "Gate 5 OK: blackout boundary is exactly %d days (%d->0, %d->sized)",
        UNLOCK_BLACKOUT_DAYS, UNLOCK_BLACKOUT_DAYS - 1, UNLOCK_BLACKOUT_DAYS + 1,
    )


def gate_scenarios() -> None:
    """Probabilities must always sum to 1.0, and tilts must point right."""
    engine = ScenarioEngine()

    for regime in (None, "BULL_TRENDING", "BEAR_TRENDING", "CRASH_RECOVERY",
                   "SIDEWAYS_LOW_VOL"):
        for shock in (None, 0.0, 3.0, 12.0, 40.0):
            bull, base, bear, _ = engine.apply_tilts(regime, shock)
            total = bull + base + bear
            _require(
                math.isclose(total, 1.0, abs_tol=1e-9),
                f"probabilities for regime={regime} shock={shock} sum to "
                f"{total}, must be 1.0",
            )
            _require(
                min(bull, base, bear) > 0,
                f"no probability may be zero or negative: {bull},{base},{bear}",
            )
    logger.info("Gate 6 OK: probabilities sum to 1.0 across all regime/shock combos")

    neutral_bull, _, neutral_bear, _ = engine.apply_tilts(None, None)
    up_bull, _, up_bear, _ = engine.apply_tilts("BULL_TRENDING", None)
    down_bull, _, down_bear, _ = engine.apply_tilts("BEAR_TRENDING", None)
    _require(
        up_bull > neutral_bull and up_bear < neutral_bear,
        "BULL_TRENDING must move mass toward the bull case",
    )
    _require(
        down_bear > neutral_bear and down_bull < neutral_bull,
        "BEAR_TRENDING must move mass toward the bear case",
    )
    logger.info(
        "Gate 6 OK: regime tilts point the right way (bull %.2f/%.2f/%.2f)",
        down_bull, neutral_bull, up_bull,
    )

    _, _, shocked_bear, tilts = engine.apply_tilts(None, 20.0)
    _require(
        shocked_bear > neutral_bear,
        "a large supply shock must increase the bear probability",
    )
    _require("supply_shock" in tilts, "the supply-shock tilt must be recorded")
    _, _, small_bear, small_tilts = engine.apply_tilts(None, 2.0)
    _require(
        math.isclose(small_bear, neutral_bear, abs_tol=TOLERANCE),
        "a shock below the threshold must not tilt anything",
    )
    _require("supply_shock" not in small_tilts, "no tilt should be recorded")
    logger.info(
        "Gate 6 OK: supply shock tilts bear %.2f -> %.2f above the %.0f%% "
        "threshold only", neutral_bear, shocked_bear, 5.0,
    )

    # A bear band with no downside must refuse to produce a payoff ratio.
    no_downside = _scenarios(bull=0.20, bear=0.0)
    _require(
        no_downside.payoff_ratio is None,
        "zero downside must give a None payoff, never a free lunch",
    )
    logger.info("Gate 6 OK: zero downside yields no payoff ratio (not infinity)")


async def gate_persistence() -> None:
    """Recommendations must round-trip with provenance, idempotently."""
    pipeline = PositionPipeline()
    run = await pipeline.run()
    if not run.sizings:
        logger.warning(
            "Gate 7 SKIP: no decisions available to size (run the inference "
            "stage first) — persistence not exercised"
        )
        return

    async with session_scope() as session:
        rows = (
            await session.execute(
                select(PositionRecommendation).where(
                    PositionRecommendation.timestamp == run.as_of
                )
            )
        ).scalars().all()
    stored = {row.symbol: row for row in rows}
    _require(
        len(stored) == len(run.sizings),
        f"persisted {len(stored)} row(s) for {len(run.sizings)} sizing(s)",
    )

    for sizing in run.sizings:
        row = stored[sizing.symbol]
        _require(
            row.win_rate_source == sizing.win_rate_source
            and row.win_rate_samples == sizing.win_rate_samples,
            f"{sizing.symbol}: win-rate provenance not persisted — a thin "
            f"estimate must never lose its sample count",
        )
        _require(
            row.risk_factors is not None,
            f"{sizing.symbol}: risk_factors must be stored so a zero can "
            f"always be explained",
        )
        _require(
            math.isclose(
                float(row.adjusted_fraction), sizing.adjusted_fraction, abs_tol=1e-8
            ),
            f"{sizing.symbol}: stored fraction differs from computed",
        )
        _require(
            float(row.adjusted_fraction) <= MAX_POSITION_FRACTION + TOLERANCE,
            f"{sizing.symbol}: a stored position exceeds the cap",
        )
    logger.info(
        "Gate 7 OK: %d recommendation(s) round-tripped with provenance intact",
        len(stored),
    )

    await pipeline.run()
    async with session_scope() as session:
        count = (
            await session.execute(
                text(
                    "SELECT count(*) FROM position_recommendations "
                    "WHERE timestamp = :t"
                ),
                {"t": run.as_of},
            )
        ).scalar_one()
    _require(
        count == len(run.sizings),
        f"re-running duplicated rows: {count} for {len(run.sizings)} assets",
    )
    logger.info("Gate 7 OK: re-running updates in place (%d rows)", count)


# ---------------------------------------------------------------------------
async def main() -> int:
    started = datetime.now(tz=UTC)
    sizer = PositionSizer()
    try:
        logger.info("--- Ideal case (p=0.60, b=2.0) ---")
        gate_ideal_case(sizer)
        logger.info("--- Band calibration (sqrt-time correction) ---")
        gate_band_calibration()
        logger.info("--- Dangerous case (HIGH vol + unlock in 7d) ---")
        gate_dangerous_case(sizer)
        logger.info("--- Missing data ---")
        await gate_missing_data(sizer)
        logger.info("--- Conservatism guarantees ---")
        gate_conservatism(sizer)
        logger.info("--- NULL safety ---")
        gate_null_safety(sizer)
        logger.info("--- Scenario probabilities ---")
        gate_scenarios()
        logger.info("--- Persistence ---")
        await gate_persistence()
    except AssertionError as exc:
        logger.error("PHASE 8 PART 7 TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("PHASE 8 PART 7 TEST FAILED with an unexpected error")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "PHASE 8 PART 7 TEST PASSED (7 gates, %.1fs) — scenario engine & "
        "Kelly sizing ready", elapsed,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main()))

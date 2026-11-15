"""Phase 8 Part 2 test: regime detection + dynamic weighting.

Gates
-----
1. **Weight table**  — every regime's weights total exactly 1.0, cover all
   five pillars, and are non-negative. UNDETERMINED/unknown/low-confidence
   all fall back to equal weights.
2. **Backward compat** — fusing a panel with no weights gives EXACTLY the
   equal-weight average (the pre-Phase-8 behaviour), and renormalisation
   keeps the weights summing to 1.0 when pillars are missing.
3. **Historical truth** — the detector is run against real ``daily_features``
   at known dates: the 2021 bull legs must classify BULL_TRENDING and the
   2022 bear must classify BEAR_TRENDING. This is the gate that would have
   caught the original bug where the entire 2022 bear read as
   CRASH_RECOVERY.
4. **Crash recovery** — the CRASH_RECOVERY path must be REACHABLE on real
   data (June 2021, post-China-ban bounce), not merely suppressed.
5. **Performance**   — a full detection must complete in under 1 second.
6. **Persistence**   — ``market_regimes`` round-trips the verdict, evidence
   and scores; re-running the same day updates instead of duplicating.
7. **Synthetic**     — hand-built frames drive each regime deterministically,
   including UNDETERMINED, with no dependence on live market conditions.
8. **Transition**    — a regime seen for the FIRST time is damped by
   ``TRANSITION_CONFIDENCE_FACTOR`` and must clear a higher bar (65 raw
   instead of 45). Two moves are explicitly NOT transitions: a cold start
   (no prior regime) and leaving ``UNDETERMINED`` (emergence of clarity).
   Every outcome still totals exactly 1.0.
9. **Lookup**        — ``previous_regime()`` never reads back the row
   ``run()`` just wrote, and ignores classifications older than
   ``TRANSITION_LOOKBACK_DAYS``.
10. **Wiring**       — ``execute_inference()`` runs end-to-end with the model
    calls and the decision write stubbed out. Everything else here tests
    pure functions; this is the only gate that would catch a broken
    reference inside the orchestration itself.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_regime_phase8.py
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, text  # noqa: E402

from src.agents import (  # noqa: E402
    DEFAULT_WEIGHTS_PCT,
    PILLARS,
    REGIME_WEIGHTS,
    DynamicWeightingEngine,
    MasterAgent,
)
from src.agents.weighting_engine import (  # noqa: E402
    MIN_CONFIDENCE_TO_APPLY,
    TRANSITION_CONFIDENCE_FACTOR,
)
from src.agents.schemas import Signal, SubAgentResponse  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import MarketRegime, MarketRegimeRow  # noqa: E402
from src.features import RegimeDetector  # noqa: E402
from src.features.regime_detector import (  # noqa: E402
    MIN_HISTORY_DAYS,
    TRANSITION_LOOKBACK_DAYS,
)
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_regime_phase8")

#: Hard performance budget from the task specification.
MAX_DETECTION_SECONDS = 1.0

#: Known market dates and the regime a competent analyst would name.
#: Only unambiguous points are asserted; transitional chop is excluded on
#: purpose, because pinning a test to a judgement call makes it brittle.
HISTORICAL_EXPECTATIONS: tuple[tuple[str, str, str], ...] = (
    ("2021-01-15", "BULL_TRENDING", "Bull 2021, first leg (BTC 30k->40k)"),
    ("2021-04-14", "BULL_TRENDING", "Bull 2021, ATH ~64k"),
    ("2021-11-09", "BULL_TRENDING", "Bull 2021, ATH ~69k"),
    ("2022-01-24", "BEAR_TRENDING", "Bear 2022, breakdown below 35k"),
    ("2022-05-12", "BEAR_TRENDING", "Bear 2022, LUNA collapse"),
    ("2022-06-18", "BEAR_TRENDING", "Bear 2022, capitulation to 17k"),
    ("2022-09-20", "BEAR_TRENDING", "Bear 2022, grind lower"),
    ("2022-11-21", "BEAR_TRENDING", "Bear 2022, post-FTX"),
    ("2024-03-14", "BULL_TRENDING", "Bull 2024, ETF-driven ATH"),
)

#: A date where CRASH_RECOVERY must fire: the June 2021 bounce after the
#: China mining ban halved BTC from its April high.
CRASH_RECOVERY_DATE = "2021-06-14"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
def gate_weight_table() -> None:
    """Every weight vector must total exactly 1.0 over the five pillars."""
    engine = DynamicWeightingEngine()

    for regime in MarketRegime:
        weights = engine.weights_for(regime, confidence=100)
        total = sum(weights.values())
        _require(
            set(weights) == set(PILLARS),
            f"{regime.value}: pillars {sorted(weights)} != {sorted(PILLARS)}",
        )
        _require(
            math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-12),
            f"{regime.value}: weights total {total!r}, must be exactly 1.0",
        )
        _require(
            all(value >= 0 for value in weights.values()),
            f"{regime.value}: negative weight in {weights}",
        )
    logger.info(
        "Gate 1a OK: all %d regimes total exactly 1.0 across %d pillars",
        len(MarketRegime), len(PILLARS),
    )

    equal = engine.weights_for(MarketRegime.UNDETERMINED)
    _require(
        all(math.isclose(v, 0.2) for v in equal.values()),
        f"UNDETERMINED must be equal weights, got {equal}",
    )
    _require(
        engine.weights_for(None) == equal,
        "a missing regime must fall back to equal weights",
    )
    _require(
        engine.weights_for("NOT_A_REGIME") == equal,
        "an unknown regime label must fall back to equal weights",
    )
    low = engine.weights_for(MarketRegime.BULL_TRENDING, confidence=10)
    _require(
        low == equal,
        f"a low-confidence regime must not skew the weights, got {low}",
    )
    strong = engine.weights_for(MarketRegime.BULL_TRENDING, confidence=95)
    _require(strong != equal, "a strong regime read MUST skew the weights")
    _require(
        strong["technical"] > strong["onchain"],
        "BULL_TRENDING must favour technical over on-chain",
    )
    crash = engine.weights_for(MarketRegime.CRASH_RECOVERY, confidence=95)
    _require(
        crash["sentiment"] > crash["macro"]
        and crash["liquidity_risk"] > crash["macro"],
        "CRASH_RECOVERY must favour sentiment/liquidity over macro",
    )
    logger.info("Gate 1b OK: fallbacks and regime skews behave as specified")

    # The declared integer table is the source of truth for "sums to 100".
    for regime, table in (
        *REGIME_WEIGHTS.items(),
        (MarketRegime.UNDETERMINED, DEFAULT_WEIGHTS_PCT),
    ):
        _require(
            sum(table.values()) == 100,
            f"{regime}: integer table sums to {sum(table.values())}, not 100",
        )
    logger.info("Gate 1c OK: integer weight tables sum to exactly 100")


def gate_transition_damping() -> None:
    """A regime seen for the first time must clear a higher bar."""
    engine = DynamicWeightingEngine()
    equal = engine.weights_for(MarketRegime.UNDETERMINED)
    bull = MarketRegime.BULL_TRENDING
    bear = MarketRegime.BEAR_TRENDING

    # The damped bar: MIN_CONFIDENCE_TO_APPLY / TRANSITION_CONFIDENCE_FACTOR.
    bar = MIN_CONFIDENCE_TO_APPLY / TRANSITION_CONFIDENCE_FACTOR
    _require(
        math.isclose(bar, 45 / 0.7, rel_tol=1e-9),
        f"unexpected transition bar {bar}",
    )

    # (a) Persisted regime at moderate confidence -> regime weights.
    persisted = engine.evaluate(bull, bull, 50)
    _require(
        not persisted.used_default and not persisted.is_transition,
        f"a persisted regime at 50 must apply its weights: {persisted.summary()}",
    )

    # (b) SAME confidence but freshly transitioned -> damped below the bar.
    fresh = engine.evaluate(bull, bear, 50)
    _require(fresh.is_transition, "a bull-after-bear read must be a transition")
    _require(
        fresh.used_default and fresh.weights == equal,
        f"50 * 0.7 = 35 < 45 must fall back to equal weights: {fresh.summary()}",
    )
    logger.info(
        "Gate 8a OK: confidence 50 applies when persisted, falls back on "
        "transition (50 -> %.1f < %d)",
        fresh.adjusted_confidence, MIN_CONFIDENCE_TO_APPLY,
    )

    # (c) A convincing fresh regime still gets through.
    strong = engine.evaluate(bull, bear, 90)
    _require(
        strong.is_transition and not strong.used_default,
        f"90 * 0.7 = 63 >= 45 must still apply regime weights: {strong.summary()}",
    )
    _require(
        strong.weights != equal,
        "a strong fresh regime must skew the weights",
    )
    logger.info(
        "Gate 8b OK: a convincing fresh regime (90 -> %.0f) still applies",
        strong.adjusted_confidence,
    )

    # (d) Boundary: 65 survives a transition, 64 does not.
    _require(
        not engine.evaluate(bull, bear, 65).used_default,
        "confidence 65 (65*0.7=45.5) must survive a transition",
    )
    _require(
        engine.evaluate(bull, bear, 64).used_default,
        "confidence 64 (64*0.7=44.8) must NOT survive a transition",
    )
    logger.info("Gate 8c OK: transition boundary is exactly 65 (raw confidence)")

    # (e) EDGE CASE — no prior observation is NOT a transition. A cold start
    #     must not be penalised for a flip that never happened.
    cold = engine.evaluate(bull, None, 50)
    _require(
        not cold.is_transition,
        "previous_regime=None must NOT count as a transition (cold start)",
    )
    _require(
        not cold.used_default,
        f"a cold start at confidence 50 must apply regime weights: {cold.summary()}",
    )
    logger.info("Gate 8d1 OK: cold start (no prior regime) is not penalised")

    # (e2) EDGE CASE — leaving UNDETERMINED is the emergence of clarity,
    #      NOT a whipsaw: there was no prior conviction to flip against, so
    #      the normal bar applies (user decision, 2026-07-28).
    clarified = engine.evaluate(bull, MarketRegime.UNDETERMINED, 50)
    _require(
        not clarified.is_transition,
        "UNDETERMINED -> definite regime must NOT count as a transition",
    )
    _require(
        not clarified.used_default,
        f"clarity at confidence 50 must apply regime weights: "
        f"{clarified.summary()}",
    )
    _require(
        clarified.weights == engine.weights_for(bull, 50),
        "clarified weights must match the undamped regime weights",
    )
    # ...while a genuine strong->strong flip at the SAME confidence is damped.
    _require(
        engine.evaluate(bull, bear, 50).used_default,
        "a definite->definite flip at 50 must still be damped — the "
        "UNDETERMINED exemption must not leak into real transitions",
    )
    logger.info(
        "Gate 8d2 OK: UNDETERMINED -> %s at confidence 50 applies regime "
        "weights (clarity), while %s -> %s at 50 is still damped",
        bull.value, bear.value, bull.value,
    )

    # (f) The thin wrapper matches the rich evaluation.
    _require(
        engine.get_weights_with_transition(bull, bear, 50) == fresh.weights,
        "get_weights_with_transition must match evaluate().weights",
    )
    # (g) Transition INTO UNDETERMINED is still just equal weights.
    undetermined = engine.evaluate(MarketRegime.UNDETERMINED, bull, 90)
    _require(
        undetermined.weights == equal and undetermined.used_default,
        "UNDETERMINED must always be equal weights regardless of transition",
    )
    # (h) Every transition outcome still totals exactly 1.0.
    for decision in (persisted, fresh, strong, cold, clarified, undetermined):
        total = sum(decision.weights.values())
        _require(
            math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-12),
            f"transition path produced weights totalling {total}: {decision.summary()}",
        )
    logger.info("Gate 8e OK: every transition outcome still totals exactly 1.0")


async def gate_transition_lookup() -> None:
    """previous_regime() must ignore stale rows and never read its own write."""
    detector = RegimeDetector()
    snapshot = await detector.run()

    # run() must capture the PREVIOUS row, never the one it just wrote.
    _require(
        snapshot.previous_as_of is None
        or snapshot.previous_as_of < snapshot.as_of,
        f"previous_as_of {snapshot.previous_as_of} must precede "
        f"{snapshot.as_of} — run() read back its own write",
    )
    logger.info(
        "Gate 9a OK: run() captured previous=%s (as of %s), not its own write",
        snapshot.previous_regime.value if snapshot.previous_regime else "None",
        snapshot.previous_as_of.date() if snapshot.previous_as_of else "n/a",
    )

    # A prior row far in the past must NOT count as a fresh transition.
    stale_probe = snapshot.as_of + timedelta(days=TRANSITION_LOOKBACK_DAYS + 30)
    stale = await detector.previous_regime(stale_probe)
    _require(
        stale is None,
        f"a regime {TRANSITION_LOOKBACK_DAYS + 30} days old must not count as "
        f"a fresh transition, got {stale}",
    )
    logger.info(
        "Gate 9b OK: a prior classification older than %d days is not a "
        "fresh transition", TRANSITION_LOOKBACK_DAYS,
    )

    # And a strictly-earlier probe finds nothing before the first row.
    earliest = await detector.previous_regime(
        datetime(2000, 1, 1, tzinfo=UTC)
    )
    _require(earliest is None, "no regime can precede the year 2000")
    logger.info("Gate 9c OK: empty history returns None rather than raising")

    # --- END-TO-END through the DATABASE, not just the pure function ---
    # Gate 8 proves the arithmetic; this proves the wiring. A temporary
    # prior row is planted one day back with a deliberately DIFFERENT
    # regime, then run() must pick it up and the weighting must damp.
    probe_day = snapshot.as_of - timedelta(days=1)
    other = (
        MarketRegime.BEAR_TRENDING
        if snapshot.regime is not MarketRegime.BEAR_TRENDING
        else MarketRegime.BULL_TRENDING
    )
    # The probe slot is `as_of - 1 day`, which NORMAL DAILY OPERATION fills:
    # yesterday's run classified yesterday. The original guard just refused to
    # run in that case, which meant this gate silently stopped working as soon
    # as the pipeline had been used two days in a row (observed 2026-07-30).
    # Snapshot the real row, borrow the slot, put it back — the same fix
    # applied to test_multi_asset_phase8's decisions cleanup.
    async with session_scope() as session:
        preserved_rows = (
            await session.execute(
                select(MarketRegimeRow.__table__).where(
                    MarketRegimeRow.__table__.c.timestamp == probe_day
                )
            )
        ).mappings().all()
    preserved = [dict(row) for row in preserved_rows]
    if preserved:
        logger.info(
            "Gate 9d: real regime row at %s will be borrowed and restored",
            probe_day.date(),
        )
        async with session_scope() as session:
            await session.execute(
                text("DELETE FROM market_regimes WHERE timestamp = :t"),
                {"t": probe_day},
            )

    try:
        async with session_scope() as session:
            await session.execute(
                text(
                    "INSERT INTO market_regimes "
                    "(timestamp, regime, confidence, symbol, rule_version) "
                    "VALUES (:t, CAST(:r AS market_regime), 80, :s, :v)"
                ),
                {
                    "t": probe_day, "r": other.value,
                    "s": snapshot.symbol, "v": "test-probe",
                },
            )

        replayed = await detector.run()
        _require(
            replayed.previous_regime is other,
            f"run() should have read the planted {other.value} row, got "
            f"{replayed.previous_regime}",
        )
        _require(
            replayed.just_changed,
            "a different adjacent regime must register as a transition",
        )
        damped = DynamicWeightingEngine().evaluate(
            replayed.regime, replayed.previous_regime, replayed.confidence
        )
        _require(
            damped.is_transition,
            f"weighting must see the transition: {damped.summary()}",
        )
        expected_default = (
            replayed.confidence * TRANSITION_CONFIDENCE_FACTOR
            < MIN_CONFIDENCE_TO_APPLY
        )
        _require(
            damped.used_default is expected_default,
            f"damped confidence {damped.adjusted_confidence:.0f} vs bar "
            f"{MIN_CONFIDENCE_TO_APPLY} disagrees with used_default="
            f"{damped.used_default}",
        )
        logger.info(
            "Gate 9d OK: DB round-trip transition %s -> %s, confidence "
            "%d -> %.0f, weights %s",
            other.value, replayed.regime.value, replayed.confidence,
            damped.adjusted_confidence,
            "equal (damped below bar)" if damped.used_default else "regime-specific",
        )
    finally:
        # Always remove the probe, even if an assertion above failed...
        async with session_scope() as session:
            await session.execute(
                text(
                    "DELETE FROM market_regimes "
                    "WHERE timestamp = :t AND rule_version = 'test-probe'"
                ),
                {"t": probe_day},
            )
        # ...and always put the borrowed real row back. Note run() may have
        # re-persisted its own row at probe_day in the meantime, so clear the
        # slot before restoring rather than assuming it is empty.
        if preserved:
            async with session_scope() as session:
                await session.execute(
                    text("DELETE FROM market_regimes WHERE timestamp = :t"),
                    {"t": probe_day},
                )
                await session.execute(
                    MarketRegimeRow.__table__.insert(), preserved
                )
            logger.info(
                "Gate 9e OK: borrowed regime row at %s restored", probe_day.date()
            )
        async with session_scope() as session:
            leftover = (
                await session.execute(
                    text(
                        "SELECT count(*) FROM market_regimes "
                        "WHERE rule_version = 'test-probe'"
                    )
                )
            ).scalar_one()
        _require(leftover == 0, f"{leftover} test probe row(s) left behind")
        logger.info("Gate 9e OK: probe row cleaned up, no test data left behind")


def _panel(**scores: int) -> list[SubAgentResponse]:
    """Build a sub-agent panel with the given per-pillar scores."""
    return [
        SubAgentResponse(
            agent_name=name,
            signal=(
                Signal.BULLISH if score > 0
                else Signal.BEARISH if score < 0
                else Signal.NEUTRAL
            ),
            score=score,
            reasoning="Synthetic panel entry for deterministic weighting tests.",
        )
        for name, score in scores.items()
    ]


def gate_backward_compatibility() -> None:
    """No weights supplied => exactly the equal-weight average."""
    panel = _panel(
        technical=60, macro=-20, onchain=40, sentiment=0, liquidity_risk=-30
    )
    expected = (60 - 20 + 40 + 0 - 30) / 5.0

    fusion = MasterAgent.fuse(panel)
    _require(
        math.isclose(fusion.weighted_score, expected, abs_tol=1e-9),
        f"no-weights fusion gave {fusion.weighted_score}, expected {expected}",
    )
    _require(
        math.isclose(sum(fusion.effective_weights.values()), 1.0, abs_tol=1e-12),
        "effective weights must total 1.0",
    )
    _require(
        not fusion.is_dynamic,
        "equal weights must not be reported as a dynamic regime skew",
    )
    logger.info(
        "Gate 2a OK: no weights -> plain average %+.1f (pre-Phase-8 behaviour)",
        expected,
    )

    # Explicit equal weights must match the implicit path exactly.
    equal = DynamicWeightingEngine().weights_for(MarketRegime.UNDETERMINED)
    explicit = MasterAgent.fuse(panel, equal)
    _require(
        math.isclose(explicit.weighted_score, fusion.weighted_score, abs_tol=1e-12),
        "explicit equal weights must equal the implicit default",
    )
    logger.info("Gate 2b OK: explicit equal weights == implicit default")

    # A regime skew must actually move the number.
    bull = DynamicWeightingEngine().weights_for(MarketRegime.BULL_TRENDING, 90)
    skewed = MasterAgent.fuse(panel, bull, regime="BULL_TRENDING", regime_confidence=90)
    manual = sum(bull[p] * s for p, s in
                 {"technical": 60, "macro": -20, "onchain": 40,
                  "sentiment": 0, "liquidity_risk": -30}.items())
    _require(
        math.isclose(skewed.weighted_score, manual, abs_tol=1e-9),
        f"weighted fusion {skewed.weighted_score} != hand-computed {manual}",
    )
    _require(skewed.is_dynamic, "a real regime skew must report is_dynamic")
    logger.info(
        "Gate 2c OK: BULL weights move the panel score %+.1f -> %+.1f",
        fusion.weighted_score, skewed.weighted_score,
    )

    # Missing pillars: weights must be RENORMALISED, never silently < 1.
    partial = _panel(technical=80, macro=40, onchain=20)
    partial_fusion = MasterAgent.fuse(partial, bull)
    total = sum(partial_fusion.effective_weights.values())
    _require(
        math.isclose(total, 1.0, abs_tol=1e-12),
        f"renormalised weights total {total}, must be 1.0",
    )
    _require(
        len(partial_fusion.effective_weights) == 3,
        "only reporting pillars may carry weight",
    )
    _require(
        20 <= partial_fusion.weighted_score <= 80,
        f"renormalised score {partial_fusion.weighted_score} fell outside the "
        "range of the reporting pillars — weights were not renormalised",
    )
    logger.info(
        "Gate 2d OK: 3/5 pillars -> weights renormalised to 1.0, score %+.1f",
        partial_fusion.weighted_score,
    )


# ---------------------------------------------------------------------------
def _synthetic_history(
    days: int = 420,
    *,
    trend_pct_per_day: float = 0.0,
    volatility: float = 0.0,
    base: float = 30_000.0,
    fear_greed: float = 50.0,
    realized_vol: float = 0.5,
    volume: float = 1_000.0,
    rsi: float = 50.0,
    macd_sign: float = 0.0,
    tail_shock: float | None = None,
    tail_rsi: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Deterministic feature frame that drives a chosen regime."""
    index = pd.date_range(end=datetime(2026, 1, 1), periods=days, freq="D")
    steps = np.arange(days)
    close = base * (1.0 + trend_pct_per_day / 100.0) ** steps
    if volatility:
        # Deterministic zig-zag (no RNG) so the test never flakes.
        close = close * (1.0 + volatility * np.sin(steps * 1.3))

    frame = pd.DataFrame(
        {
            "close": close,
            "volume": np.full(days, volume),
            "macd": close * macd_sign,
            "rsi_14": np.full(days, rsi),
            "realized_vol_30d": np.full(days, realized_vol),
            "fear_greed_value": np.full(days, fear_greed),
        },
        index=index,
    )
    if tail_shock is not None:
        # Collapse the last 20 days by `tail_shock` to create a drawdown.
        frame.iloc[-20:, frame.columns.get_loc("close")] *= tail_shock
    if tail_rsi is not None:
        oversold, now = tail_rsi
        frame.iloc[-20:, frame.columns.get_loc("rsi_14")] = oversold
        frame.iloc[-1, frame.columns.get_loc("rsi_14")] = now
    return frame


def gate_synthetic_regimes() -> None:
    """Hand-built frames must produce each regime deterministically."""
    detector = RegimeDetector()

    cases = [
        (
            "BULL_TRENDING",
            _synthetic_history(
                trend_pct_per_day=0.30, fear_greed=75, macd_sign=0.02,
                volume=1_000.0, rsi=65,
            ),
        ),
        (
            "BEAR_TRENDING",
            _synthetic_history(
                trend_pct_per_day=-0.25, fear_greed=15, macd_sign=-0.02, rsi=35,
            ),
        ),
        (
            "SIDEWAYS_LOW_VOL",
            _synthetic_history(
                trend_pct_per_day=0.0, fear_greed=50, realized_vol=0.20, rsi=50,
            ),
        ),
    ]
    for expected, frame in cases:
        snapshot = detector.detect(frame)
        _require(
            snapshot.regime.value == expected,
            f"synthetic {expected}: got {snapshot.regime.value} "
            f"(scores {snapshot.scores})",
        )
        logger.info(
            "Gate 7 OK: synthetic %-18s -> %s (confidence %d)",
            expected, snapshot.regime.value, snapshot.confidence,
        )

    # SIDEWAYS_HIGH_VOL: same flat price path, volatility far above baseline.
    high_vol = _synthetic_history(trend_pct_per_day=0.0, realized_vol=0.50, rsi=50)
    high_vol.iloc[-40:, high_vol.columns.get_loc("realized_vol_30d")] = 1.30
    snapshot = detector.detect(high_vol)
    _require(
        snapshot.regime.value == "SIDEWAYS_HIGH_VOL",
        f"synthetic SIDEWAYS_HIGH_VOL: got {snapshot.regime.value} "
        f"(scores {snapshot.scores})",
    )
    logger.info(
        "Gate 7 OK: synthetic SIDEWAYS_HIGH_VOL  -> %s (confidence %d)",
        snapshot.regime.value, snapshot.confidence,
    )

    # A deep drawdown with RSI still pinned at its low is NOT a recovery —
    # this is the exact shape that used to mislabel the 2022 bear market.
    still_falling = _synthetic_history(
        trend_pct_per_day=-0.20, fear_greed=10, macd_sign=-0.02,
        tail_shock=0.70, tail_rsi=(18.0, 18.0),
    )
    snapshot = detector.detect(still_falling)
    _require(
        snapshot.regime.value != "CRASH_RECOVERY",
        "a drawdown with RSI pinned at its low must NOT be CRASH_RECOVERY "
        f"(got {snapshot.regime.value}, scores {snapshot.scores})",
    )
    logger.info(
        "Gate 7 OK: deep drawdown + RSI still at its low -> %s (NOT recovery)",
        snapshot.regime.value,
    )


# ---------------------------------------------------------------------------
async def gate_historical(detector: RegimeDetector, full: pd.DataFrame) -> None:
    """Known market dates must classify the way an analyst would name them."""
    failures: list[str] = []
    for iso, expected, label in HISTORICAL_EXPECTATIONS:
        as_of = pd.Timestamp(iso)
        window = full.loc[:as_of]
        if len(window) < MIN_HISTORY_DAYS:
            failures.append(
                f"{iso}: only {len(window)} days of history — run "
                f"scripts/backfill_history.py --since 2020-01-01"
            )
            continue
        snapshot = detector.detect(window.iloc[-500:])
        actual = snapshot.regime.value
        status = "OK " if actual == expected else "MISS"
        logger.info(
            "Gate 3 %s: %s %-34s expected %-17s got %-17s (conf %d)",
            status, iso, label, expected, actual, snapshot.confidence,
        )
        if actual != expected:
            failures.append(
                f"{iso} ({label}): expected {expected}, got {actual} "
                f"[scores {snapshot.scores}]"
            )
    _require(not failures, "historical regime mismatches:\n  - " + "\n  - ".join(failures))
    logger.info(
        "Gate 3 OK: all %d historical checkpoints classified correctly",
        len(HISTORICAL_EXPECTATIONS),
    )


async def gate_crash_recovery(detector: RegimeDetector, full: pd.DataFrame) -> None:
    """The CRASH_RECOVERY path must be reachable on real data."""
    window = full.loc[: pd.Timestamp(CRASH_RECOVERY_DATE)]
    _require(
        len(window) >= MIN_HISTORY_DAYS,
        f"not enough history to reach {CRASH_RECOVERY_DATE}",
    )
    snapshot = detector.detect(window.iloc[-500:])
    _require(
        snapshot.regime.value == "CRASH_RECOVERY",
        f"{CRASH_RECOVERY_DATE} (June 2021 China-ban crash bounce) should be "
        f"CRASH_RECOVERY, got {snapshot.regime.value} [{snapshot.scores}]",
    )
    evidence = snapshot.evidence
    _require(
        evidence["drawdown_from_ath_90d_pct"] < -20,
        "CRASH_RECOVERY requires a >20% drawdown from the 90-day high",
    )
    _require(
        evidence["rsi_14"] > evidence["rsi_min_30d"],
        "CRASH_RECOVERY requires RSI above its 30-day low",
    )
    logger.info(
        "Gate 4 OK: %s -> CRASH_RECOVERY (conf %d, drawdown %.1f%%, RSI %.1f "
        "up from %.1f)",
        CRASH_RECOVERY_DATE, snapshot.confidence,
        evidence["drawdown_from_ath_90d_pct"], evidence["rsi_14"],
        evidence["rsi_min_30d"],
    )


async def gate_performance() -> None:
    """A full load + classify must fit inside the 1-second budget."""
    detector = RegimeDetector()
    started = time.perf_counter()
    history = await detector.load_history()
    snapshot = detector.detect(history)
    elapsed = time.perf_counter() - started
    _require(
        elapsed < MAX_DETECTION_SECONDS,
        f"detection took {elapsed:.3f}s, budget is {MAX_DETECTION_SECONDS}s",
    )
    logger.info(
        "Gate 5 OK: load(%d rows) + detect = %.3fs (budget %.1fs) -> %s",
        len(history), elapsed, MAX_DETECTION_SECONDS, snapshot.regime.value,
    )


async def gate_persistence() -> None:
    """market_regimes must round-trip the verdict and stay idempotent."""
    detector = RegimeDetector()
    snapshot = await detector.run()

    async with session_scope() as session:
        row = (
            await session.execute(
                select(MarketRegimeRow).where(
                    MarketRegimeRow.timestamp == snapshot.as_of
                )
            )
        ).scalar_one_or_none()
    _require(row is not None, "regime was not persisted to market_regimes")
    _require(
        MarketRegime(row.regime) is snapshot.regime,
        f"stored regime {row.regime} != detected {snapshot.regime}",
    )
    _require(
        row.confidence == snapshot.confidence,
        f"stored confidence {row.confidence} != {snapshot.confidence}",
    )
    _require(row.regime_scores, "regime_scores must be persisted for auditability")
    _require(row.rule_version, "rule_version must be persisted")
    _require(
        row.close is not None and row.sma_200 is not None,
        "evidence columns must be persisted, not just the verdict",
    )
    logger.info(
        "Gate 6a OK: %s round-tripped with %d evidence scores and rule %s",
        snapshot.regime.value, len(row.regime_scores), row.rule_version,
    )

    # Idempotency: the same day must UPDATE, never duplicate.
    await detector.run()
    async with session_scope() as session:
        count = (
            await session.execute(
                text(
                    "SELECT count(*) FROM market_regimes WHERE timestamp = :t"
                ),
                {"t": snapshot.as_of},
            )
        ).scalar_one()
    _require(count == 1, f"re-running produced {count} rows for the same day")
    logger.info("Gate 6b OK: re-running the same day updates in place (1 row)")


async def gate_inference_wiring() -> None:
    """Run execute_inference() end-to-end with the LLM and writes stubbed.

    Rationale: the regime -> weighting -> fusion -> Master wiring lives in
    ``execute_inference``, which every other gate skips because it calls the
    model. A stale variable name there survives ``py_compile`` and only
    surfaces during a live run — which is exactly how a NameError reached a
    real daily run once. This exercises the whole orchestration offline.

    Both the model calls AND ``save_decision`` are stubbed, so the test
    neither spends tokens nor overwrites a genuine decision row.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_inference", Path(__file__).with_name("run_inference.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_inference"] = module          # dataclasses need this
    spec.loader.exec_module(module)

    canned_pillar = SubAgentResponse(
        agent_name="technical",
        signal=Signal.BEARISH,
        score=-40,
        reasoning="Stubbed pillar response for the offline wiring test.",
    )
    from src.agents.schemas import ConsolidatedAnalysis

    canned_consolidated = ConsolidatedAnalysis(
        technical_score=-40,
        onchain_score=-40,
        macro_score=-40,
        sentiment_score=-40,
        liquidity_risk_score=-40,
        reasoning_summary=(
            "Ringkasan konsolidasi stub untuk uji wiring offline."
        ),
    )
    canned_decision = module.MasterDecision(
        action=module.Action.HOLD,
        confidence_score=50,
        executive_summary=(
            "Ringkasan stub untuk uji wiring offline; tidak dipersist dan "
            "tidak memanggil model bahasa apa pun."
        ),
        risk_warning="Peringatan risiko stub untuk uji wiring offline.",
    )

    captured: dict = {}

    async def fake_consolidated(self, features, coin_dna=None, **kwargs):
        # Part 7 consolidated path: one call replaces the five specialists.
        captured.setdefault("coin_dna_seen", []).append(coin_dna)
        return canned_consolidated

    async def fake_analyze(self, features, coin_dna=None):
        # Signature must track SubAgent.analyze — Phase 8 Part 5 added
        # coin_dna, and this stub failing on it is precisely how this gate
        # reports an orchestration contract change.
        captured.setdefault("coin_dna_seen", []).append(coin_dna)
        return canned_pillar.model_copy(update={"agent_name": self.agent_name})

    async def fake_decide(self, analysis_date, sub_responses,
                          market_context=None, dynamic_weights=None, **kwargs):
        captured["dynamic_weights"] = dynamic_weights
        captured["market_context"] = market_context
        captured["fusion"] = kwargs.get("fusion")
        captured["coin_dna_summary"] = kwargs.get("coin_dna_summary")
        return canned_decision

    async def fake_save(*args, **kwargs):
        captured["saved"] = True

    from src.agents.consolidated_analyst import ConsolidatedAnalyst

    original = (
        module.SubAgent.analyze, module.MasterAgent.decide, module.save_decision,
    )
    original_consolidated = ConsolidatedAnalyst.analyze
    ConsolidatedAnalyst.analyze = fake_consolidated
    module.SubAgent.analyze = fake_analyze
    module.MasterAgent.decide = fake_decide
    module.save_decision = fake_save
    try:
        outcome = await module.execute_inference(print_output=False)
    finally:
        module.SubAgent.analyze, module.MasterAgent.decide = original[0], original[1]
        module.save_decision = original[2]
        ConsolidatedAnalyst.analyze = original_consolidated

    _require(
        outcome.decision.action is not None,
        "an action must be produced (composed deterministically in "
        "consolidated mode, or by the Master in panel mode)",
    )
    _require(outcome.fusion is not None, "execute_inference must attach a fusion")
    # Consolidated mode composes the decision in Python, so the Master is
    # not called. The weighting must still be applied to the fusion.
    _require(
        outcome.fusion.effective_weights,
        "the fusion must carry the effective pillar weights",
    )
    total = sum(outcome.fusion.effective_weights.values())
    _require(
        math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-12),
        f"effective weights total {total}, must be 1.0",
    )
    _require(
        math.isclose(
            outcome.fusion.weighted_score, -40.0, abs_tol=1e-9
        ),
        f"all pillars scored -40, so any weighting must fuse to -40, got "
        f"{outcome.fusion.weighted_score}",
    )
    # Part 5: every pillar must actually receive the Coin DNA dict, and the
    # Master must receive its narrative summary.
    seen = captured.get("coin_dna_seen", [])
    _require(
        seen and all(isinstance(entry, dict) for entry in seen),
        f"the analyst should receive a Coin DNA dict, got {seen!r}",
    )
    _require(
        outcome.decision.executive_summary,
        "the composed decision must carry an executive summary",
    )
    logger.info(
        "Gate 10 OK: execute_inference wiring intact — regime=%s, weights "
        "total 1.0, fused score %+.1f, Coin DNA reached 5/5 pillars + Master "
        "(no model call, nothing persisted)",
        outcome.fusion.regime, outcome.fusion.weighted_score,
    )


# ---------------------------------------------------------------------------
async def main() -> int:
    started = datetime.now(tz=UTC)
    try:
        logger.info("--- Weight table ---")
        gate_weight_table()
        logger.info("--- Backward compatibility ---")
        gate_backward_compatibility()
        logger.info("--- Transition damping ---")
        gate_transition_damping()
        logger.info("--- Synthetic regimes ---")
        gate_synthetic_regimes()

        detector = RegimeDetector(lookback_days=5000)
        full = await detector.load_history()
        logger.info(
            "Loaded %d days of real history (%s -> %s)",
            len(full), full.index.min().date(), full.index.max().date(),
        )
        logger.info("--- Historical regimes (Bull 2021 / Bear 2022) ---")
        await gate_historical(detector, full)
        logger.info("--- Crash recovery reachability ---")
        await gate_crash_recovery(detector, full)
        logger.info("--- Performance ---")
        await gate_performance()
        logger.info("--- Persistence ---")
        await gate_persistence()
        logger.info("--- Transition lookup (ordering & staleness) ---")
        await gate_transition_lookup()
        logger.info("--- Inference wiring (offline, no model call) ---")
        await gate_inference_wiring()
    except AssertionError as exc:
        logger.error("PHASE 8 PART 2 TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("PHASE 8 PART 2 TEST FAILED with an unexpected error")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "PHASE 8 PART 2 TEST PASSED (10 gates, %.1fs) — regime detection, "
        "dynamic weighting & transition damping ready", elapsed,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main()))

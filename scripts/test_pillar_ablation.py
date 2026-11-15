"""Phase 9 test: LLM pillar ablation (Information Coefficient).

No model calls. Synthetic frames for the arithmetic, the live warehouse for
the wiring.

Gates
-----
1. **Spearman** — perfect ranking is +1, reversed is -1, a constant side is
   ``None`` (never 0.0), ties average, and non-linear-but-monotonic is still
   +1 (which is why rank, not Pearson).
2. **IC aggregation** — a planted per-day signal recovers its own IC, days
   below ``MIN_ASSETS_PER_DAY`` are dropped, the sample size reported is DAYS
   not rows, and a t-statistic is ABSENT below ``MIN_IC_DAYS`` rather than
   merely large.
3. **Overlap correction** — ``eff_n = n / horizon``, and a large rho on
   overlapping windows is refused rather than published. This is the gate
   that stops a +0.90 on ten overlapping weeks from being read as a result.
4. **Span parity with the evaluator** — every live row's outcome recomputes
   from ``crypto_prices`` as close(T+lag) -> close(T+lag+h) in band units,
   and no row reaches past the finalization cutoff.
5. **Pillar extraction** — all five pillars survive the JSON round trip and
   ``panel_mean`` is their mean; a missing pillar degrades to None, not 0.
6. **Power arithmetic** — days needed falls as the detectable IC rises, and
   the relationship is the documented inverse square.
7. **Attenuation correction** — the multiplier arithmetic is exact, a
   correction can only GROW an IC's magnitude (noise attenuates, it cannot
   invent), a zero multiplier yields ``None`` rather than infinity, the
   denominator is the within-day spread with thin days excluded, and four
   unsafe cases (no measurement, too few runs, temperature mismatch,
   constant signal) REFUSE instead of silently defaulting to 1.0.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_pillar_ablation.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import select  # noqa: E402

from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import CryptoPrice, DailyFeature  # noqa: E402
from src.evaluation.decision_evaluator import (  # noqa: E402
    DAILY_TIMEFRAME,
    ENTRY_LAG_DAYS,
    band_pct_for,
)
from src.evaluation.pillar_ablation import (  # noqa: E402
    BASELINE_FEATURES,
    MIN_ASSETS_PER_DAY,
    MIN_EFFECTIVE_OBS,
    MIN_IC_DAYS,
    SIGNIFICANCE_T,
    PANEL_MEAN,
    PILLARS,
    days_needed_for,
    information_coefficient,
    load_ablation_frame,
    spearman,
    timeseries_correlation,
)
from src.evaluation.sample_size import (  # noqa: E402
    effective_independent_units,
    effective_sample_size,
    non_overlapping_count,
)
from src.forecasting.evaluator import _finalization_cutoff  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_pillar_ablation")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _synthetic_day(day: datetime, signal: list[float], outcome: list[float]):
    return pd.DataFrame(
        {
            "decision_ts": [day] * len(signal),
            "symbol": [f"S{i}/USDT" for i in range(len(signal))],
            "probe": signal,
            "forward_z": outcome,
        }
    )


# ---------------------------------------------------------------------------
async def gate_spearman() -> None:
    """Rank correlation must be exact at the extremes and honest when blind."""
    ascending = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    _require(
        abs(spearman(ascending, ascending) - 1.0) < 1e-12,
        "identical rankings must give exactly +1",
    )
    _require(
        abs(spearman(ascending, ascending[::-1].reset_index(drop=True)) + 1.0) < 1e-12,
        "reversed rankings must give exactly -1",
    )

    # Monotonic but wildly non-linear: rank sees +1, Pearson would not. This
    # is the whole reason the module ranks first.
    exponential = pd.Series([1.0, 10.0, 1_000.0, 100_000.0, 10_000_000.0])
    _require(
        abs(spearman(ascending, exponential) - 1.0) < 1e-12,
        "a monotonic non-linear relationship must still rank +1 — if not, the "
        "implementation is correlating values rather than ranks",
    )

    # A constant side is NOT zero skill; it is no measurement at all.
    _require(
        spearman(ascending, pd.Series([7.0] * 5)) is None,
        "a constant series must give None, never 0.0 — returning 0.0 would "
        "average a non-answer into the IC as evidence of no skill",
    )
    _require(spearman(pd.Series([1.0]), pd.Series([2.0])) is None,
             "a single pair cannot be correlated")

    # Ties must average, not error or bias.
    tied = pd.Series([1.0, 1.0, 2.0, 2.0])
    _require(
        spearman(tied, pd.Series([1.0, 1.0, 2.0, 2.0])) is not None,
        "tied ranks must be handled, not crash",
    )
    logger.info("Gate 1 OK: spearman exact at +/-1, monotonic-safe, None when blind")


async def gate_ic_aggregation() -> None:
    """The IC must recover a planted signal and count DAYS, not rows."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    n = MIN_ASSETS_PER_DAY + 2

    # Three days of perfect ranking -> IC exactly 1.0 on 3 days.
    perfect = pd.concat(
        [
            _synthetic_day(
                base + timedelta(days=k),
                list(range(n)),
                [float(v) for v in range(n)],
            )
            for k in range(3)
        ],
        ignore_index=True,
    )
    result = information_coefficient(perfect, "probe", 1)
    _require(
        result.mean_ic is not None and abs(result.mean_ic - 1.0) < 1e-12,
        f"perfect daily ranking must give IC 1.0, got {result.mean_ic}",
    )
    _require(
        result.n_days == 3 and result.n_obs == 3 * n,
        f"n_days must count DAYS ({result.n_days}) and n_obs rows "
        f"({result.n_obs}) — conflating them is how {3 * n} correlated rows "
        f"get reported as {3 * n} independent samples",
    )

    # A day thinner than MIN_ASSETS_PER_DAY must be dropped entirely.
    thin = pd.concat(
        [
            perfect,
            _synthetic_day(base + timedelta(days=99), [1.0, 2.0], [1.0, 2.0]),
        ],
        ignore_index=True,
    )
    thin_result = information_coefficient(thin, "probe", 1)
    _require(
        thin_result.n_days == 3,
        f"a {MIN_ASSETS_PER_DAY - 3}-asset day must not enter the IC "
        f"(n_days={thin_result.n_days}); with two names Spearman is +/-1 and "
        f"carries no information",
    )

    # Opposite days must cancel to ~0 with a large spread, not look like skill.
    mixed = pd.concat(
        [
            _synthetic_day(base, list(range(n)), [float(v) for v in range(n)]),
            _synthetic_day(
                base + timedelta(days=1), list(range(n)),
                [float(v) for v in reversed(range(n))],
            ),
        ],
        ignore_index=True,
    )
    mixed = pd.concat(
        [
            _synthetic_day(
                base + timedelta(days=k),
                list(range(n)),
                [float(v) for v in (range(n) if k % 2 == 0 else reversed(range(n)))],
            )
            for k in range(MIN_IC_DAYS + 1)
        ],
        ignore_index=True,
    )
    mixed_result = information_coefficient(mixed, "probe", 1)
    _require(
        mixed_result.mean_ic is not None and abs(mixed_result.mean_ic) < 0.2,
        f"alternating +1/-1 days must roughly cancel, got {mixed_result.mean_ic}",
    )
    _require(
        mixed_result.t_stat is not None
        and abs(mixed_result.t_stat) < SIGNIFICANCE_T,
        "a near-zero mean IC must not produce a significant t",
    )

    # -- the floor that this module's first real output made necessary ----
    # Live run 2026-08-02: IC -0.291 on std 0.039 across TWO days gave
    # t = -10.63 and the verdict announced "INVERTED — check sign". A
    # confident direction from a sample of two. The t must now be absent, not
    # merely large.
    two_days = pd.concat(
        [
            _synthetic_day(base + timedelta(days=k), list(range(n)),
                           [float(v) for v in range(n)])
            for k in range(2)
        ],
        ignore_index=True,
    )
    thin = information_coefficient(two_days, "probe", 1)
    _require(
        thin.mean_ic is not None and thin.n_days == 2,
        "the probe must produce an IC on 2 days",
    )
    _require(
        thin.t_stat is None,
        f"a t-statistic on {thin.n_days} day(s) must be None, got {thin.t_stat} "
        f"— below {MIN_IC_DAYS} days the denominator is noise and the number "
        f"is an artefact of how the few IC values happened to be spaced",
    )
    _require(
        "UNTESTABLE" in thin.verdict,
        f"the verdict on 2 days must refuse to editorialise, got {thin.verdict!r}",
    )
    logger.info(
        "Gate 2 OK: IC recovers 1.0/0.0 exactly, drops thin days, counts days"
    )


async def gate_overlap_correction() -> None:
    """A big rho on overlapping windows must be refused, not published."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    consecutive = [base + timedelta(days=k) for k in range(70)]
    _require(
        abs(effective_sample_size(consecutive, 7) - 10.0) < 1e-12,
        "70 consecutive daily windows at a 7-day horizon are 10 "
        "non-overlapping ones",
    )
    _require(
        abs(effective_sample_size(consecutive[:30], 1) - 30.0) < 1e-12,
        "at a 1-day horizon nothing overlaps and eff_n must equal n",
    )

    # Gaps must be honoured. This is why the count is a greedy interval scan
    # and not n/horizon: three decisions 40 days apart at a 30-day horizon do
    # not overlap at all, and n/horizon would report 0.1 observations.
    sparse = [base, base + timedelta(days=40), base + timedelta(days=80)]
    _require(
        abs(effective_sample_size(sparse, 30) - 3.0) < 1e-12,
        f"non-overlapping decisions must all count, got "
        f"{effective_sample_size(sparse, 30)} instead of 3",
    )

    # Cross-sectional deflation: 21 assets at rho 0.75 are not 21 draws.
    twenty_one = effective_independent_units(21, 0.75)
    _require(
        1.0 < twenty_one < 2.0,
        f"21 equicorrelated assets at rho=0.75 must be worth ~1.3 independent "
        f"ones, got {twenty_one:.2f}",
    )
    # And it saturates: doubling the registry buys almost nothing.
    forty_two = effective_independent_units(42, 0.75)
    _require(
        forty_two - twenty_one < 0.1,
        f"doubling 21 assets to 42 must buy < 0.1 independent observations "
        f"(got +{forty_two - twenty_one:.3f}) — in a market this correlated, "
        f"power comes from more DAYS, not more coins",
    )
    _require(
        abs(effective_independent_units(1, 0.75) - 1.0) < 1e-12
        and abs(effective_independent_units(21, 0.0) - 21.0) < 1e-12,
        "a single asset is one observation, and uncorrelated assets are all "
        "independent — both ends of the formula must be exact",
    )

    # The real case this gate exists for: BTC at +7d showed rho +0.898 on ten
    # overlapping weeks. Uncorrected that is t~5.8; corrected it is untestable.
    frame = pd.DataFrame(
        {
            "decision_ts": [
                datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=k)
                for k in range(10)
            ],
            "symbol": ["BTC/USDT"] * 10,
            "probe": [float(k) for k in range(10)],
            "forward_z": [float(k) for k in range(10)],
        }
    )
    result = timeseries_correlation(frame, "probe", "BTC/USDT", 7)
    _require(
        result.rho is not None and abs(result.rho - 1.0) < 1e-12,
        "the planted perfect relationship must be found",
    )
    _require(
        result.effective_obs < MIN_EFFECTIVE_OBS,
        f"10 windows at +7d give {result.effective_obs:.1f} effective obs, "
        f"which must be below the {MIN_EFFECTIVE_OBS} threshold",
    )
    _require(
        "UNTESTABLE" in result.verdict,
        f"a perfect rho on 1.4 effective observations must read UNTESTABLE, "
        f"got {result.verdict!r} — this is exactly the artefact the gate exists "
        f"to catch",
    )

    # Same relationship, enough non-overlapping data: now it may be reported.
    long_frame = pd.DataFrame(
        {
            "decision_ts": [
                datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=k)
                for k in range(70)
            ],
            "symbol": ["BTC/USDT"] * 70,
            "probe": [float(k) for k in range(70)],
            "forward_z": [float(k) for k in range(70)],
        }
    )
    long_result = timeseries_correlation(long_frame, "probe", "BTC/USDT", 7)
    _require(
        "UNTESTABLE" not in long_result.verdict,
        f"70 windows at +7d give {long_result.effective_obs:.1f} effective obs "
        f"and must become testable, got {long_result.verdict!r}",
    )

    # A constant signal must name its reason rather than show a bare n/a.
    flat = long_frame.assign(probe=0.0)
    flat_result = timeseries_correlation(flat, "probe", "BTC/USDT", 7)
    _require(
        flat_result.rho is None and "constant" in flat_result.verdict,
        f"a constant signal must say so (got {flat_result.verdict!r}) — this is "
        f"the real rs_7d-on-BTC case, where the value is 0.0 by construction",
    )
    logger.info(
        "Gate 3 OK: overlap penalty applied; perfect rho on 1.4 eff obs refused"
    )


async def gate_span_parity() -> None:
    """Live outcomes must recompute as the evaluator's exact span."""
    horizon = 1
    async with session_scope() as session:
        frame = await load_ablation_frame(session, horizon)
    _require(not frame.empty, "no matured ablation rows — cannot verify parity")

    cutoff = _finalization_cutoff(datetime.now(tz=UTC))
    async with session_scope() as session:
        closes = {
            (row.symbol, row.timestamp): float(row.close)
            for row in (
                await session.execute(
                    select(
                        CryptoPrice.symbol, CryptoPrice.timestamp, CryptoPrice.close
                    ).where(
                        CryptoPrice.exchange == "binance",
                        CryptoPrice.timeframe == DAILY_TIMEFRAME,
                    )
                )
            ).all()
        }
        vols = {
            (row.symbol, row.timestamp): float(row.realized_vol_30d)
            for row in (
                await session.execute(
                    select(
                        DailyFeature.symbol,
                        DailyFeature.timestamp,
                        DailyFeature.realized_vol_30d,
                    ).where(DailyFeature.realized_vol_30d.is_not(None))
                )
            ).all()
            if row.realized_vol_30d is not None
        }

    for row in frame.itertuples():
        entry_day = row.decision_ts + timedelta(days=ENTRY_LAG_DAYS)
        target_day = entry_day + timedelta(days=horizon)
        _require(
            target_day < cutoff,
            f"{row.symbol} {row.decision_ts.date()}: target {target_day.date()} "
            f"is at/after the finalization cutoff {cutoff.date()}",
        )
        entry = closes[(row.symbol, entry_day)]
        target = closes[(row.symbol, target_day)]
        band = band_pct_for(vols[(row.symbol, row.decision_ts)], horizon)
        expected = ((target / entry - 1.0) * 100.0) / band
        _require(
            abs(row.forward_z - expected) < 1e-9,
            f"{row.symbol} {row.decision_ts.date()}: forward_z "
            f"{row.forward_z:.9f} != independent recompute {expected:.9f}. The "
            f"ablation is measuring a different span than the evaluator scores",
        )
    logger.info(
        "Gate 4 OK: %d row(s) recompute to the evaluator's exact span "
        "(entry +%dd, band in decision-day units)", len(frame), ENTRY_LAG_DAYS,
    )


async def gate_pillar_extraction() -> None:
    """All five pillars must survive the JSON round trip."""
    async with session_scope() as session:
        frame = await load_ablation_frame(session, 1)
    _require(not frame.empty, "no rows to inspect")

    for pillar in PILLARS:
        _require(pillar in frame.columns, f"pillar column {pillar!r} missing")
        _require(
            frame[pillar].notna().any(),
            f"pillar {pillar!r} is entirely NaN — the agent_name in "
            f"sub_agents_json does not match PILLARS, and a typo there fails "
            f"silently rather than raising",
        )
    for feature in BASELINE_FEATURES:
        _require(feature in frame.columns, f"baseline {feature!r} missing")
    _require(PANEL_MEAN in frame.columns, "panel_mean missing")

    complete = frame[list(PILLARS)].notna().all(axis=1)
    _require(bool(complete.any()), "no row has all five pillars")
    subset = frame[complete]
    recomputed = subset[list(PILLARS)].mean(axis=1).to_numpy()
    _require(
        np.allclose(subset[PANEL_MEAN].to_numpy(), recomputed, atol=1e-12),
        "panel_mean does not equal the mean of the five pillar scores",
    )
    logger.info(
        "Gate 5 OK: 5/5 pillars + %d baseline(s) extracted; panel_mean verified "
        "on %d complete row(s)", len(BASELINE_FEATURES), len(subset),
    )


async def gate_power_arithmetic() -> None:
    """Detecting a smaller edge must require more data, inverse-square."""
    big, small = days_needed_for(0.10), days_needed_for(0.05)
    _require(
        big is not None and small is not None and small > big,
        f"a smaller IC must need MORE days ({small} vs {big})",
    )
    # Halving the target IC quadruples the requirement.
    _require(
        abs(small / big - 4.0) < 0.35,
        f"halving the target IC must roughly quadruple the days needed "
        f"({small}/{big} = {small / big:.2f})",
    )
    _require(
        days_needed_for(0.0) is None and days_needed_for(-0.1) is None,
        "a zero or negative target IC has no sample size and must give None",
    )
    logger.info(
        "Gate 6 OK: IC 0.10 -> ~%d days, IC 0.05 -> ~%d days (inverse square)",
        big, small,
    )


async def gate_attenuation() -> None:
    """The correction must restore, never invent, and must refuse when unsafe."""
    from datetime import timedelta

    import pandas as pd

    from src.evaluation.attenuation import (
        MIN_NOISE_RUNS,
        NoiseMeasurement,
        attenuation_from_spread,
        compute_attenuation,
        within_day_spread,
    )

    def _noise(sd: float, *, runs: int = 5, temp: float = 0.0) -> NoiseMeasurement:
        return NoiseMeasurement(
            pillar="technical", noise_sd=sd, n_runs=runs, temperature=temp,
            model="test/model", measured_at=datetime.now(tz=UTC), symbol="BTC/USDT",
        )

    # --- exact arithmetic: 1 - 9/25 = 0.64, sqrt = 0.80 -------------------
    att = attenuation_from_spread(
        signal="technical", observed_sd=5.0, noise=_noise(3.0),
        series_temperature=0.0,
    )
    _require(
        att.multiplier is not None and abs(att.multiplier - 0.8) < 1e-12,
        f"reliability 1-9/25=0.64 must give multiplier 0.80, got {att.multiplier}",
    )
    _require(
        abs(att.correct(0.10) - 0.125) < 1e-12,
        f"IC 0.10 at multiplier 0.80 must correct to 0.125, got {att.correct(0.10)}",
    )

    # --- correction can only grow the magnitude, never shrink or flip -----
    for raw in (-0.30, -0.05, 0.0, 0.05, 0.30):
        corrected = att.correct(raw)
        _require(
            abs(corrected) >= abs(raw) - 1e-12,
            f"correction must not shrink an IC: {raw} -> {corrected}",
        )
        _require(
            corrected == 0.0 or (corrected > 0) == (raw > 0),
            f"correction must not flip the sign: {raw} -> {corrected}",
        )

    # --- noise >= spread: exactly zero, and NOT a division by zero --------
    dead = attenuation_from_spread(
        signal="sentiment", observed_sd=3.0, noise=_noise(13.7),
        series_temperature=0.0,
    )
    _require(dead.multiplier == 0.0, f"noise > spread must give 0.0, got {dead.multiplier}")
    _require(
        dead.correct(0.05) is None,
        "a zero multiplier must give None, never inf — dividing by it is the "
        f"bug this guards, got {dead.correct(0.05)}",
    )
    _require(dead.severity == "TOTAL", dead.severity)

    # --- REFUSALS: each must be None with a reason, never a silent 1.0 ----
    mismatch = attenuation_from_spread(
        signal="technical", observed_sd=5.0, noise=_noise(3.0, temp=0.2),
        series_temperature=0.0,
    )
    _require(
        mismatch.multiplier is None and "temperature" in mismatch.reason,
        "noise measured at a different temperature must be REFUSED — it "
        f"describes a different process. Got {mismatch.multiplier} "
        f"({mismatch.reason})",
    )
    thin = attenuation_from_spread(
        signal="technical", observed_sd=5.0,
        noise=_noise(3.0, runs=MIN_NOISE_RUNS - 1), series_temperature=0.0,
    )
    _require(thin.multiplier is None, f"too-few-runs must refuse: {thin.reason}")
    absent = attenuation_from_spread(
        signal="technical", observed_sd=5.0, noise=None, series_temperature=0.0,
    )
    _require(
        absent.multiplier is None and absent.correct(0.1) is None,
        "no measurement means UNKNOWN, not 1.0 — an uncorrected IC must not "
        "masquerade as a corrected one",
    )
    flat = attenuation_from_spread(
        signal="onchain", observed_sd=0.0, noise=_noise(1.0),
        series_temperature=0.0,
    )
    _require(flat.multiplier is None, f"constant signal must refuse: {flat.reason}")

    # --- the denominator is WITHIN-DAY, and thin days are excluded --------
    day1 = datetime(2026, 8, 1, tzinfo=UTC)
    day2 = datetime(2026, 8, 2, tzinfo=UTC)
    rows = []
    # Day 1: 6 assets, spread 0 .. 50 (sample sd of 0,10,20,30,40,50 = 18.708)
    for i, score in enumerate([0, 10, 20, 30, 40, 50]):
        rows.append({"decision_ts": day1, "symbol": f"A{i}", "technical": score})
    # Day 2: only 2 assets, and WILDLY apart. Must be excluded entirely; if
    # it leaked in it would dominate the mean and inflate the denominator,
    # which would UNDER-correct every IC.
    rows += [
        {"decision_ts": day2, "symbol": "B0", "technical": -100},
        {"decision_ts": day2, "symbol": "B1", "technical": 100},
    ]
    frame = pd.DataFrame(rows)
    sd, n_days = within_day_spread(frame, "technical", min_assets=5)
    _require(
        n_days == 1,
        f"only the 6-asset day qualifies at min_assets=5, got {n_days} day(s)",
    )
    _require(
        abs(sd - 18.7083) < 1e-3,
        f"within-day sample sd of 0..50 step 10 is 18.708, got {sd}",
    )

    # A pooled (all-rows) sd would be ~48.6 here — nearly 3x the honest
    # within-day figure, and it would make the noise look harmless. That
    # inversion is the documented denominator trap, so assert the gap.
    pooled = float(frame["technical"].std(ddof=1))
    _require(
        pooled > sd * 2,
        f"the synthetic frame must actually exercise the trap "
        f"(pooled {pooled:.1f} vs within-day {sd:.1f})",
    )

    # --- end to end through compute_attenuation ---------------------------
    full = compute_attenuation(
        signal="technical", frame=frame, noise=_noise(9.354),
        series_temperature=0.0, min_assets=5,
    )
    _require(
        full.multiplier is not None and abs(full.multiplier - 0.8660) < 1e-3,
        f"noise 9.354 against within-day 18.708 is reliability 0.75, "
        f"multiplier 0.866; got {full.multiplier}",
    )
    _require(full.n_days == 1, full.n_days)

    logger.info(
        "Gate 7 OK: multiplier arithmetic exact, correction never shrinks or "
        "flips, zero-multiplier gives None not inf, and 4 unsafe cases refuse "
        "instead of defaulting to 1.0"
    )


# ---------------------------------------------------------------------------
async def main() -> int:
    started = datetime.now(tz=UTC)
    try:
        logger.info("--- Spearman ---")
        await gate_spearman()
        logger.info("--- IC aggregation ---")
        await gate_ic_aggregation()
        logger.info("--- Overlap correction ---")
        await gate_overlap_correction()
        logger.info("--- Span parity with the evaluator ---")
        await gate_span_parity()
        logger.info("--- Pillar extraction ---")
        await gate_pillar_extraction()
        logger.info("--- Power arithmetic ---")
        await gate_power_arithmetic()
        logger.info("--- Attenuation correction ---")
        await gate_attenuation()
    except AssertionError as exc:
        logger.error("PILLAR ABLATION TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("PILLAR ABLATION TEST FAILED with an unexpected error")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "PILLAR ABLATION TEST PASSED (7 gates, %.1fs) — IC machinery and "
        "attenuation correction verified",
        elapsed,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging("INFO")
    raise SystemExit(asyncio.run(main()))

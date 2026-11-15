"""Correct a measured IC for the model's own run-to-run noise (C5, 2026-08-09).

The problem, stated exactly
---------------------------
If an observed pillar score is the true score plus independent noise, then

    corr(observed, outcome) = corr(true, outcome) * sqrt(reliability)
    reliability             = var_true / (var_true + var_noise)
                            = 1 - var_noise / var_observed

So a measured IC is always SMALLER in magnitude than the real one. Noise
attenuates; it cannot manufacture. That asymmetry is what makes the
correction safe to report: a positive IC is real, and a null result is
ambiguous between "no signal" and "signal destroyed".

C1 established that the noise floor here is **irreducible** — at
``temperature=0`` the sampler is already argmax, so the residual variance
comes from MoE expert routing under batched inference, above the sampler
and out of reach of any client parameter. Correction is therefore not a
workaround for a bug that will be fixed; it is permanent arithmetic.

THE DENOMINATOR IS THE WHOLE GAME
---------------------------------
``var_observed`` must be measured over **the same population the
correlation is measured over**. The first version of this calculation used
the POOLED historical spread and produced a reassuring average multiplier
of 0.949. That was wrong and the error was flattering: pooled spread is
dominated by stable differences BETWEEN assets (BTC and PEPE always differ)
that a cross-sectional IC never sees. The IC ranks assets WITHIN one day,
so the denominator is the WITHIN-DAY cross-sectional variance. The honest
number was 0.690.

This module goes one step further than the version it replaces: the
within-day spread is computed from **the exact frame the IC was computed
on** — same horizon, same maturity cutoff, same >=5-assets-per-day filter —
rather than from all decisions ever stored. Two filters that disagree are
two populations, which is the same defect one layer down.

What this module REFUSES to do
------------------------------
* Correct across a temperature mismatch. Noise measured at 0.2 describes a
  different process than a series generated at 0. C1 already recorded a
  near-miss here: the 2026-08-04 and 2026-08-09 figures were compared even
  though they differ in temperature, day, inputs AND n.
* Correct with no measurement. An unknown noise floor is unknown, not zero
  — the same NULL-vs-0 discipline as the Part 4 feature contract.
* Invent a multiplier when the within-day spread is zero or absent. A
  signal that does not vary across assets has no cross-sectional ranking to
  attenuate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: A noise measurement older than this is reported as stale. Not a hard
#: refusal — a stale correction still beats no correction, and the caller is
#: told exactly how old it is — but C1 established the multiplier is not a
#: constant, so it must be re-measured, not trusted indefinitely.
STALE_AFTER_DAYS = 30

#: Below this many identical calls the noise sd says more about the accidental
#: spacing of a few numbers than about the process. Mirrors the reasoning
#: behind MIN_IC_DAYS: refuse at the same level of thinness on every path.
MIN_NOISE_RUNS = 3


@dataclass(slots=True, frozen=True)
class NoiseMeasurement:
    """One pillar's measured run-to-run noise, with its provenance."""

    pillar: str
    noise_sd: float
    n_runs: int
    temperature: float
    model: str
    measured_at: datetime
    symbol: str

    def age_days(self, now: datetime | None = None) -> float:
        return ((now or datetime.now(tz=UTC)) - self.measured_at).total_seconds() / 86400.0

    @property
    def is_usable(self) -> bool:
        return self.n_runs >= MIN_NOISE_RUNS and self.noise_sd >= 0.0


@dataclass(slots=True, frozen=True)
class Attenuation:
    """The correction for one signal, and why it is (or is not) applicable."""

    signal: str
    #: Within-day cross-sectional sd, from the IC's own frame.
    observed_sd: float | None
    noise_sd: float | None
    #: ``None`` when it could not be computed — never silently 1.0.
    multiplier: float | None
    reason: str
    n_days: int = 0

    @property
    def reliability(self) -> float | None:
        return None if self.multiplier is None else self.multiplier**2

    def correct(self, ic: float | None) -> float | None:
        """The IC this would be without the noise. ``None`` if not applicable."""
        if ic is None or self.multiplier is None or self.multiplier <= 0.0:
            return None
        return ic / self.multiplier

    @property
    def severity(self) -> str:
        """One word for how much damage the noise is doing."""
        if self.multiplier is None:
            return "unknown"
        if self.multiplier == 0.0:
            return "TOTAL"      # ranking is pure noise
        if self.multiplier < 0.70:
            return "severe"
        if self.multiplier < 0.85:
            return "notable"
        return "mild"


def within_day_spread(
    frame: pd.DataFrame,
    signal: str,
    *,
    min_assets: int,
    day_column: str = "decision_ts",
) -> tuple[float | None, int]:
    """Mean cross-sectional sd of ``signal``, averaged over qualifying days.

    THE denominator. Computed from the frame the IC used, so the population
    behind the correction and the population behind the correlation are the
    same set of rows — see the module docstring.

    Returns ``(sd, n_days)``; ``(None, n)`` when no day qualifies.
    """
    if frame.empty or signal not in frame.columns or day_column not in frame.columns:
        return None, 0

    per_day: list[float] = []
    for _, group in frame.groupby(day_column):
        values = group[signal].dropna()
        if len(values) < min_assets:
            continue
        # ddof=1: this is a sample of the assets scored that day, and with
        # ~8-21 assets the difference from the population sd is not noise-
        # level. Matches SQL's stddev_samp used by test_determinism.
        per_day.append(float(values.std(ddof=1)))

    usable = [sd for sd in per_day if np.isfinite(sd)]
    if not usable:
        return None, 0
    return float(np.mean(usable)), len(usable)


async def within_day_spread_from_decisions(
    session, *, min_assets: int
) -> dict[str, tuple[float, int]]:
    """Same denominator, but over EVERY stored decision instead of one frame.

    The broader population, for callers that have no horizon in hand —
    ``test_determinism`` reports a general indication of how badly the noise
    it just measured attenuates things, and it has no matured-return frame.

    Kept in this module rather than inlined at the call site so both
    denominators come from one place with one ``ddof`` and one
    ``min_assets`` rule. Two implementations of the same quantity is how
    two parts of a system start disagreeing — the reason
    ``compute_scoreboard()`` was extracted.

    Returns ``{pillar: (mean_within_day_sd, n_days)}``.
    """
    from sqlalchemy import text

    rows = (
        await session.execute(
            text(
                """
                -- Spread ACROSS ASSETS within each day, then averaged.
                -- HAVING mirrors MIN_ASSETS_PER_DAY: a day with two assets
                -- has no meaningful cross-sectional spread, and letting it
                -- in would distort the denominator.
                WITH per_day AS (
                    SELECT d.timestamp,
                           p.value ->> 'agent_name' AS pillar,
                           stddev_samp((p.value ->> 'score')::numeric) AS sd_day
                    FROM decisions d
                    CROSS JOIN LATERAL
                        jsonb_array_elements(d.sub_agents_json) AS p(value)
                    GROUP BY d.timestamp, p.value ->> 'agent_name'
                    HAVING count(*) >= :min_assets
                )
                SELECT pillar, avg(sd_day) AS sd, count(*) AS n
                FROM per_day
                WHERE sd_day IS NOT NULL
                GROUP BY pillar
                """
            ),
            {"min_assets": min_assets},
        )
    ).all()
    return {
        r._mapping["pillar"]: (float(r._mapping["sd"]), int(r._mapping["n"]))
        for r in rows
        if r._mapping["sd"] is not None
    }


def attenuation_from_spread(
    *, signal: str, observed_sd: float | None, noise: NoiseMeasurement | None,
    series_temperature: float, n_days: int = 0,
) -> Attenuation:
    """The arithmetic alone, for callers that already have the denominator."""
    if noise is None:
        return Attenuation(signal, observed_sd, None, None,
                           "no noise measurement on record", n_days)
    if not noise.is_usable:
        return Attenuation(
            signal, observed_sd, noise.noise_sd, None,
            f"noise measured from only {noise.n_runs} run(s) "
            f"(need >= {MIN_NOISE_RUNS})", n_days,
        )
    if not np.isclose(noise.temperature, series_temperature):
        return Attenuation(
            signal, observed_sd, noise.noise_sd, None,
            f"temperature mismatch: noise measured at {noise.temperature}, "
            f"series generated at {series_temperature}", n_days,
        )
    if observed_sd is None:
        return Attenuation(signal, None, noise.noise_sd, None,
                           "no cross-sectional spread available", n_days)
    if observed_sd <= 0.0:
        return Attenuation(
            signal, observed_sd, noise.noise_sd, None,
            "signal is constant across assets — no ranking to attenuate",
            n_days,
        )
    reliability = max(0.0, 1.0 - (noise.noise_sd**2) / (observed_sd**2))
    multiplier = reliability**0.5
    reason = (
        "noise >= cross-sectional spread — ranking carries no signal"
        if multiplier == 0.0 else "ok"
    )
    return Attenuation(signal, observed_sd, noise.noise_sd, multiplier,
                       reason, n_days)


def compute_attenuation(
    *,
    signal: str,
    frame: pd.DataFrame,
    noise: NoiseMeasurement | None,
    series_temperature: float,
    min_assets: int,
    day_column: str = "decision_ts",
) -> Attenuation:
    """Everything needed to report an IC raw AND corrected.

    ``series_temperature`` is the temperature the SCORES in ``frame`` were
    generated at. A mismatch against the noise measurement is refused rather
    than papered over.
    """
    observed_sd, n_days = within_day_spread(
        frame, signal, min_assets=min_assets, day_column=day_column
    )
    return attenuation_from_spread(
        signal=signal,
        observed_sd=observed_sd,
        noise=noise,
        series_temperature=series_temperature,
        n_days=n_days,
    )


async def load_latest_noise(session, *, model: str | None = None
                            ) -> dict[str, NoiseMeasurement]:
    """Most recent noise measurement per pillar, optionally pinned to a model.

    "Most recent" is per pillar rather than per run: a later run that only
    covered some pillars should refresh those and leave the rest alone,
    which is exactly what a partial re-measurement means.
    """
    from sqlalchemy import select

    from src.db.models import PillarNoiseMeasurement as PNM

    stmt = select(PNM).order_by(PNM.measured_at.desc())
    if model is not None:
        stmt = stmt.where(PNM.model == model)

    latest: dict[str, NoiseMeasurement] = {}
    for row in (await session.execute(stmt)).scalars():
        if row.pillar in latest:
            continue  # already have a newer one
        latest[row.pillar] = NoiseMeasurement(
            pillar=row.pillar,
            noise_sd=float(row.noise_sd),
            n_runs=int(row.n_runs),
            temperature=float(row.temperature),
            model=row.model,
            measured_at=row.measured_at,
            symbol=row.symbol,
        )
    return latest


async def save_noise_measurements(
    session,
    *,
    measured_at: datetime,
    symbol: str,
    model: str,
    temperature: float,
    seed: int | None,
    per_pillar_scores: dict[str, list[int]],
) -> int:
    """Persist one determinism run. Idempotent on (measured_at, symbol, pillar).

    Stores the raw scores too: the sd is a summary, and a later analysis may
    want to ask a different question of the same repeats.
    """
    from sqlalchemy.dialects.postgresql import insert

    from src.db.models import PillarNoiseMeasurement as PNM

    # `panel_mean` is the aggregate every downstream number is built from, so
    # it needs a noise floor too — and it must be MEASURED, not derived. The
    # obvious shortcut (combine the five pillar sds as if independent) is
    # wrong: all five come out of ONE model call, so their errors are
    # correlated by construction and the independent formula would understate
    # the noise. Averaging each run's five scores costs nothing extra: the
    # repeats are already in hand.
    per_pillar_scores = dict(per_pillar_scores)
    pillars = list(per_pillar_scores)  # captured BEFORE panel_mean is added
    lengths = {len(per_pillar_scores[p]) for p in pillars}
    if pillars and len(lengths) == 1 and lengths != {0}:
        n = lengths.pop()
        per_pillar_scores["panel_mean"] = [
            float(np.mean([per_pillar_scores[p][i] for p in pillars]))
            for i in range(n)
        ]

    rows = []
    for pillar, scores in per_pillar_scores.items():
        clean = [float(s) for s in scores if s is not None]
        if len(clean) < 2:
            continue  # sd of one number is not a number
        rows.append({
            "measured_at": measured_at,
            "pillar": pillar,
            "symbol": symbol,
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "n_runs": len(clean),
            "noise_sd": float(np.std(clean, ddof=1)),
            # Integer column, and panel_mean's scores are floats — round
            # rather than let the driver truncate silently.
            "score_range": round(max(clean) - min(clean)),
            "scores": clean,
        })
    if not rows:
        return 0

    stmt = insert(PNM).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_pillar_noise_run",
        set_={
            "n_runs": stmt.excluded.n_runs,
            "noise_sd": stmt.excluded.noise_sd,
            "score_range": stmt.excluded.score_range,
            "scores": stmt.excluded.scores,
            "temperature": stmt.excluded.temperature,
            "seed": stmt.excluded.seed,
            "model": stmt.excluded.model,
        },
    )
    await session.execute(stmt)
    return len(rows)

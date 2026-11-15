"""Do the LLM's pillar scores carry information? (Phase 9 ablation, 2026-07-31)

The one question nobody has asked yet
-------------------------------------
Every layer downstream of the model — regime weighting, ``weighted_score``,
the BUY/SELL threshold, Kelly — is deterministic arithmetic over five numbers
the LLM produced. Those five numbers are the ONLY place judgement enters the
pipeline, and nothing has ever measured whether they relate to what the price
subsequently did.

That makes "should we pay for a better model?" unanswerable and premature.
This module answers the prior question from data already on disk:
``decisions.sub_agents_json`` has stored every pillar score since 2026-07-10.
Zero API calls.

Why cross-sectional, not pooled
-------------------------------
A pooled correlation over all (day, asset) rows would be dominated by the
market: on 2026-07-30 all 21 coins moved largely together, so a pillar that
was merely bearish-on-everything would score well whenever the market fell.
With ``btc_corr_30d`` of 0.5-0.9 those 21 rows are nearer 3 observations than
21.

So the primary metric is the **Information Coefficient**: rank-correlate
signal against forward return WITHIN each day, then average across days. The
common market move cancels by construction, and the question sharpens to the
one that matters for a 21-asset book — *did the pillar rank the coins in the
right order?* The effective sample size is then the number of DAYS, which is
the honest denominator and a far smaller number than the row count.

Spearman, not Pearson: the claim being tested is monotonic ("a higher score
should mean a higher return"), not linear, and a single 40% altcoin day would
dominate a Pearson estimate.

Returns are measured in BAND UNITS
-----------------------------------
``forward_z = return_pct / band_pct``, the same volatility band the verdicts
use. Ranking raw returns across assets whose volatility differs by 3x would
mostly rank volatility. This asks whether the pillar picked the assets that
outperformed *their own risk*, which is also the yardstick Phase 9 already
grades decisions on.

Entry lag and finalization are inherited, not re-derived — an ablation that
measured a different span than the evaluator would answer a question nobody
is asking.

It is an ABLATION, not a correlation report
--------------------------------------------
Each LLM pillar is scored beside deterministic signals taken from the very
``daily_features`` row the model was shown. If ``momentum_score`` ranks the
book as well as the LLM's technical pillar, then that pillar is an expensive
way to compute something free — and that is worth knowing before spending
anything on a larger model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.evaluation.decision_evaluator import (
    DAILY_TIMEFRAME,
    ENTRY_LAG_DAYS,
    band_pct_for,
)
from src.evaluation.sample_size import effective_sample_size
from src.forecasting.evaluator import _finalization_cutoff

logger = logging.getLogger(__name__)

#: The five pillars, in the order they appear in reports. Names must match
#: ``SubAgentResponse.agent_name`` exactly — a typo here would silently
#: produce an all-NaN column rather than an error.
PILLARS: tuple[str, ...] = (
    "technical",
    "macro",
    "onchain",
    "sentiment",
    "liquidity_risk",
)

#: Deterministic comparators pulled from the SAME daily_features row the model
#: was shown. Without these the output is a correlation report; with them it
#: is an ablation. All three are scale-free enough to rank across assets —
#: `macd` is deliberately excluded because its magnitude tracks price level,
#: so ranking it cross-sectionally would mostly rank BTC above PEPE.
BASELINE_FEATURES: tuple[str, ...] = ("momentum_score", "rs_7d", "rsi_14")

#: Equal-weight mean of the five pillars. The regime-weighted score is NOT
#: stored on `decisions`, and reconstructing it needs the regime table plus
#: transition damping; the mean is a faithful enough stand-in for "what the
#: panel said overall" and is labelled as such wherever it appears.
PANEL_MEAN = "panel_mean"

#: A day needs at least this many assets for a cross-sectional rank to mean
#: anything. With 2 assets Spearman is +1 or -1 and carries no information.
MIN_ASSETS_PER_DAY = 5

#: |t| at which a mean IC stops being explainable as noise. Same threshold as
#: the null model's SIGNIFICANCE_Z, and subject to the same caveat: it
#: assumes independent days.
SIGNIFICANCE_T = 2.0

#: Days below which a t-statistic is not computed AT ALL.
#:
#: Added 2026-08-02, on this module's first real output. With ``n_days = 2``
#: the std comes from two numbers; when they happen to land close together
#: the t explodes. The live run produced ``IC = -0.291, std = 0.039,
#: t = -10.63`` and the verdict logic duly announced "INVERTED — check sign"
#: — a confident claim from a sample of two. A t-test on 2 observations has
#: 1 degree of freedom and effectively infinite variance.
#:
#: 5 matches ``MIN_EFFECTIVE_OBS`` deliberately: the cross-sectional and
#: time-series paths should refuse to answer at the same level of thinness,
#: or one of them becomes the loophole.
MIN_IC_DAYS = 5


def spearman(a: pd.Series, b: pd.Series) -> float | None:
    """Rank correlation, or ``None`` when it is undefined.

    Undefined means: fewer than two pairs, or either side constant. A
    constant signal has no ranking to be right or wrong about, and returning
    0.0 there would quietly average a non-answer into the IC as evidence of
    no skill.
    """
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 2:
        return None
    ranks = pair.rank(method="average")
    left, right = ranks.iloc[:, 0].to_numpy(), ranks.iloc[:, 1].to_numpy()
    if left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


@dataclass(slots=True, frozen=True)
class ICResult:
    """Information Coefficient for one signal at one horizon."""

    signal: str
    horizon_days: int
    mean_ic: float | None
    std_ic: float | None
    n_days: int
    n_obs: int

    @property
    def t_stat(self) -> float | None:
        """Mean IC over its own standard error across days.

        ``None`` below :data:`MIN_IC_DAYS` — not "large", not "small",
        ABSENT. Below that the denominator is estimated from a handful of
        numbers and the statistic says more about their accidental spacing
        than about any edge.

        **Assumes independent days, so even above the floor it is an upper
        bound on significance.** Consecutive days overlap in their return
        windows at any horizon above 1, exactly as documented for the null
        model's z.
        """
        if (
            self.mean_ic is None
            or self.std_ic is None
            or self.n_days < MIN_IC_DAYS
            or self.std_ic == 0
        ):
            return None
        return self.mean_ic / (self.std_ic / np.sqrt(self.n_days))

    @property
    def verdict(self) -> str:
        if self.mean_ic is None:
            return "no data"
        if self.n_days < MIN_IC_DAYS:
            # The sign is reported by the IC column; refusing to editorialise
            # on it here is the point. A direction from 2 days is a rumour.
            return f"UNTESTABLE (n_days={self.n_days}, butuh >= {MIN_IC_DAYS})"
        t = self.t_stat
        if t is None:
            return f"n_days={self.n_days} — cannot test"
        if abs(t) < SIGNIFICANCE_T:
            return "not significant"
        return "worth a closer look" if self.mean_ic > 0 else "INVERTED — check sign"


def days_needed_for(target_ic: float, ic_std: float = 0.20) -> int | None:
    """Independent days required to detect ``target_ic`` at |t| = 2.

    ``ic_std`` defaults to 0.20, a typical day-to-day spread for a
    cross-section of ~20 names. Solving ``t = ic*sqrt(n)/std >= 2`` gives
    ``n >= (2*std/ic)^2``.

    This is the number that turns "we do not know yet" into "here is when we
    could know". It is also optimistic for the same overlap reason as
    everything else here.
    """
    if target_ic <= 0:
        return None
    return int(np.ceil((SIGNIFICANCE_T * ic_std / target_ic) ** 2))


# ---------------------------------------------------------------------------
_FRAME_SQL = """
SELECT
    d.timestamp                AS decision_ts,
    d.symbol                   AS symbol,
    d.sub_agents_json          AS pillars,
    df.realized_vol_30d        AS realized_vol_30d,
    {feature_columns},
    e.close                    AS entry_close,
    t.close                    AS target_close
FROM decisions d
JOIN daily_features df
       ON df.symbol = d.symbol AND df.timestamp = d.timestamp
-- Exact-date joins and the same entry lag the evaluator uses. A row-offset
-- window would step over a missing candle; a zero lag would measure the
-- pre-2026-07-31 (contaminated) span.
JOIN crypto_prices e
       ON e.symbol    = d.symbol
      AND e.timestamp = d.timestamp + (:entry_lag * INTERVAL '1 day')
      AND e.timeframe = :timeframe AND e.exchange = 'binance'
JOIN crypto_prices t
       ON t.symbol    = d.symbol
      AND t.timestamp = d.timestamp
                        + ((:entry_lag + :horizon) * INTERVAL '1 day')
      AND t.timeframe = :timeframe AND t.exchange = 'binance'
WHERE df.realized_vol_30d IS NOT NULL
  AND df.realized_vol_30d > 0
  AND e.close > 0
  -- Never score against an unsettled candle: same cutoff as the evaluator.
  AND d.timestamp + ((:entry_lag + :horizon) * INTERVAL '1 day') < :cutoff
ORDER BY d.timestamp, d.symbol
"""


async def load_ablation_frame(
    session, horizon: int, *, now: datetime | None = None
) -> pd.DataFrame:
    """One row per (decision day, asset) with every signal and the outcome.

    Columns: ``decision_ts``, ``symbol``, ``forward_z``, one per pillar, one
    per baseline feature, plus :data:`PANEL_MEAN`.
    """
    cutoff = _finalization_cutoff(now or datetime.now(tz=UTC))
    columns = ", ".join(f"df.{name} AS {name}" for name in BASELINE_FEATURES)
    rows = (
        await session.execute(
            text(_FRAME_SQL.format(feature_columns=columns)),
            {
                "entry_lag": ENTRY_LAG_DAYS,
                "horizon": horizon,
                "timeframe": DAILY_TIMEFRAME,
                "cutoff": cutoff,
            },
        )
    ).all()
    if not rows:
        return pd.DataFrame()

    records: list[dict] = []
    for row in rows:
        entry, target = float(row.entry_close), float(row.target_close)
        band = band_pct_for(float(row.realized_vol_30d), horizon)
        if band <= 0:
            continue
        record: dict = {
            "decision_ts": row.decision_ts,
            "symbol": row.symbol,
            # Outcome in BAND UNITS — the same yardstick the verdicts use.
            "forward_z": ((target / entry - 1.0) * 100.0) / band,
        }
        scores = {
            str(item.get("agent_name", "")).strip().lower(): item.get("score")
            for item in (row.pillars or [])
        }
        for pillar in PILLARS:
            value = scores.get(pillar)
            record[pillar] = None if value is None else float(value)
        present = [record[p] for p in PILLARS if record[p] is not None]
        record[PANEL_MEAN] = float(np.mean(present)) if present else None
        for name in BASELINE_FEATURES:
            value = getattr(row, name)
            record[name] = None if value is None else float(value)
        records.append(record)

    return pd.DataFrame.from_records(records)


def information_coefficient(
    frame: pd.DataFrame, signal: str, horizon: int
) -> ICResult:
    """Mean daily cross-sectional rank correlation of ``signal`` vs outcome."""
    if frame.empty or signal not in frame.columns:
        return ICResult(signal, horizon, None, None, 0, 0)

    daily: list[float] = []
    observations = 0
    for _day, group in frame.groupby("decision_ts"):
        usable = group[[signal, "forward_z"]].dropna()
        if len(usable) < MIN_ASSETS_PER_DAY:
            continue
        value = spearman(usable[signal], usable["forward_z"])
        if value is None:
            continue
        daily.append(value)
        observations += len(usable)

    if not daily:
        return ICResult(signal, horizon, None, None, 0, 0)
    values = np.asarray(daily, dtype=float)
    return ICResult(
        signal=signal,
        horizon_days=horizon,
        mean_ic=float(values.mean()),
        # ddof=1: this is a sample of days, not the population of them.
        std_ic=float(values.std(ddof=1)) if len(values) > 1 else None,
        n_days=len(values),
        n_obs=observations,
    )


#: Non-overlapping observations below which a time-series rho is not testable
#: at all, no matter how large it looks.
MIN_EFFECTIVE_OBS = 5.0


@dataclass(slots=True, frozen=True)
class TimeSeriesResult:
    """Per-asset rank correlation over time, with its overlap penalty.

    ``effective_obs`` is computed once at construction from the ACTUAL
    decision dates via :func:`~src.evaluation.sample_size.effective_sample_size`
    — the same function the Kelly win-rate gate and the accuracy publication
    gate use. Dividing ``n / horizon`` would have been close enough here, but
    two parts of one system disagreeing about what a sample is was exactly
    the defect this shares a fix with.
    """

    signal: str
    symbol: str
    horizon_days: int
    rho: float | None
    n_obs: int
    #: Independent-observation equivalent, from the real dates (gaps honoured).
    effective_obs: float = 0.0
    #: Set when rho is undefined, so a bare "n/a" is never mistaken for a bug.
    reason: str = ""

    @property
    def t_stat(self) -> float | None:
        """Student t for a rank correlation, on the EFFECTIVE sample.

        ``t = rho * sqrt((n-2)/(1-rho^2))`` with n the overlap-corrected
        count, not the row count. Using the row count here is precisely how
        an overlapping-window artefact gets published as a finding.
        """
        if self.rho is None or abs(self.rho) >= 1.0:
            return None
        n = self.effective_obs
        if n <= 2:
            return None
        return self.rho * np.sqrt((n - 2) / (1 - self.rho**2))

    @property
    def verdict(self) -> str:
        if self.rho is None:
            return self.reason or "undefined"
        if self.effective_obs < MIN_EFFECTIVE_OBS:
            return (
                f"UNTESTABLE (only {self.effective_obs:.1f} non-overlapping obs)"
            )
        t = self.t_stat
        if t is None or abs(t) < SIGNIFICANCE_T:
            return "not significant"
        return "worth a closer look" if self.rho > 0 else "INVERTED — check sign"


def timeseries_correlation(
    frame: pd.DataFrame, signal: str, symbol: str, horizon: int
) -> TimeSeriesResult:
    """Per-asset rank correlation over time — the single-asset fallback.

    Cross-sectional IC needs several assets on the same day and therefore
    ignores every BTC-only day in the history. This uses them, at the cost of
    being a much weaker measure: consecutive daily windows overlap, so the
    result carries an explicit :attr:`TimeSeriesResult.effective_obs`.
    """
    if signal not in frame.columns:
        return TimeSeriesResult(
            signal, symbol, horizon, None, 0, 0.0, "signal missing"
        )
    subset = frame[frame["symbol"] == symbol][
        ["decision_ts", signal, "forward_z"]
    ].dropna()
    n = len(subset)
    effective = (
        effective_sample_size(subset["decision_ts"].tolist(), horizon)
        if n
        else 0.0
    )
    if n < 2:
        return TimeSeriesResult(
            signal, symbol, horizon, None, n, effective, "too few rows"
        )
    if subset[signal].std() == 0:
        # e.g. rs_7d is exactly 0.0 for BTC by construction — a constant has
        # no ranking to be right about. Naming it beats a bare n/a.
        return TimeSeriesResult(
            signal, symbol, horizon, None, n, effective,
            "signal constant for this asset",
        )
    rho = spearman(subset[signal], subset["forward_z"])
    return TimeSeriesResult(signal, symbol, horizon, rho, n, effective)

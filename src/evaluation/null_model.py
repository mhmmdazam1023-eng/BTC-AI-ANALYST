"""What a SKILL-FREE system would score on the same band (Phase 9, 2026-07-31).

Why this module exists
----------------------
``band_pct`` is exactly one sigma. Under any fat-tailed return distribution
roughly 70% of the mass sits inside +/-1 sigma, so a HOLD that knows nothing
is CORRECT about 70% of the time and a BUY/SELL is not even TESTABLE about
70% of the time. The first Phase 9 scoreboard read 76.5% at +1d and 100% at
+7d and looked like evidence; measured against 34,211 asset-days of history
the skill-free baseline for those same bands is 73.4% and 70.7%.

So the headline number was never wrong — it was unreadable. A hit rate means
nothing without the number it has to beat, and asking the reader to remember
that is not a control. This module computes the baseline so that
``_log_scoreboard`` and ``decision_accuracy_metrics`` can print it *next to*
every hit rate, structurally.

Unconditional per asset (user decision, 2026-07-31)
---------------------------------------------------
The baseline is estimated from EVERY day of that asset's history where a band
can be formed — not from the subset of days the system happened to emit a
decision on, and not reweighted to match the realised action mix by
resampling. The estimate answers "on this asset, at this horizon, how does
this band behave in general".

The one place the action mix does enter is :meth:`NullBaseline.blended`,
which combines the three per-action baselines using the counts of rows
actually being scored. That is not conditioning the estimate — it is making
the comparison legal. Comparing a HOLD-heavy scoreboard against a baseline
that assumes an even BUY/SELL/HOLD split would flatter or punish the system
for its action mix rather than its accuracy.

It mirrors the evaluator, entry lag included
--------------------------------------------
The baseline measures ``close(T + ENTRY_LAG_DAYS)`` to
``close(T + ENTRY_LAG_DAYS + horizon)``, with the band read from
``realized_vol_30d`` at T — the exact quantity
:class:`~src.evaluation.decision_evaluator.DecisionEvaluator` scores. Sampling
``close(T) -> close(T+horizon)`` instead would produce a baseline for the
PREVIOUS, buggy definition and quietly compare new numbers against an old
yardstick. Exact-date joins for the same reason the evaluator uses them: a
row-offset window function would silently step over a missing candle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import text

from src.evaluation.decision_evaluator import (
    DAILY_TIMEFRAME,
    DAYS_PER_YEAR,
    DECISION_HORIZONS,
    ENTRY_LAG_DAYS,
)
from src.forecasting.evaluator import _finalization_cutoff

logger = logging.getLogger(__name__)

#: Minimum historical samples before a baseline is reported at all. Well
#: below MIN_ACCURACY_SAMPLES on purpose: this is estimated from years of
#: price history rather than from decisions, so it matures far earlier — but
#: a baseline from 40 days of a freshly listed coin is still noise.
MIN_NULL_SAMPLES = 200


#: |z| at which an edge stops being explainable as noise. 2.0 ~ p<0.05
#: two-tailed under the normal approximation.
SIGNIFICANCE_Z = 2.0


def edge_z_score(observed: float, null: float, n: float) -> float | None:
    """How many standard errors the observed hit rate sits above the baseline.

    Normal approximation to the binomial. Deliberately simple — the point is
    to stop "+17.0pp edge" on a handful of rows from reading as a result, not
    to publish a p-value.

    ``n`` MUST be the independent-observation count from
    :func:`~src.evaluation.sample_size.effective_sample_size`, not a row
    count. Passing rows was the original sin this whole correction exists to
    undo: 30 consecutive daily decisions at a 30-day horizon share 29/30 of
    their return window, and 21 assets on one day carry ``btc_corr_30d`` of
    0.5-0.9. A float is expected precisely because an effective count is
    rarely a whole number.

    Even corrected, treat this as indicative. The normal approximation is
    poor at the small effective sizes this system currently has, and the
    equicorrelation deflation is itself an estimate.
    """
    if n <= 0 or not 0.0 < null < 1.0:
        return None
    standard_error = (null * (1.0 - null) / n) ** 0.5
    if standard_error <= 0:
        return None
    return (observed - null) / standard_error


@dataclass(slots=True, frozen=True)
class NullBaseline:
    """Skill-free outcome frequencies for one (asset, horizon).

    The three counts partition every sample exactly once, which is what makes
    them reusable per action: the band's three regions ARE the three verdicts.
    """

    symbol: str
    horizon_days: int
    samples: int
    inside: int  # |r| <= band -> a skill-free HOLD is CORRECT here
    above: int  # r >  band   -> a skill-free BUY  is CORRECT here
    below: int  # r < -band   -> a skill-free SELL is CORRECT here

    @property
    def decisive(self) -> int:
        """Samples where a BUY/SELL would have been testable at all."""
        return self.above + self.below

    @property
    def decisive_rate(self) -> float:
        """Fraction of days the band does NOT swallow. ~0.3 in practice."""
        return self.decisive / self.samples if self.samples else 0.0

    @property
    def hold_hit_rate(self) -> float:
        """What a HOLD with no information scores. HOLD is never NEUTRAL."""
        return self.inside / self.samples if self.samples else 0.0

    @property
    def buy_hit_rate(self) -> float | None:
        """What a decisive BUY with no information scores.

        Not 0.5 in general: an asset with positive drift puts more mass above
        the band than below it, so a coin-flip BUY beats a coin-flip SELL on
        the same tape. Assuming 0.5 would hand a trending asset a free edge.
        """
        return self.above / self.decisive if self.decisive else None

    @property
    def sell_hit_rate(self) -> float | None:
        """Mirror of :attr:`buy_hit_rate`."""
        return self.below / self.decisive if self.decisive else None

    def blended(self, *, buy_n: int, sell_n: int, hold_n: int) -> float | None:
        """Expected hit rate under zero skill for THIS mix of scored rows.

        ``buy_n``/``sell_n`` are DECISIVE counts (NEUTRAL rows are already
        excluded upstream); ``hold_n`` is every HOLD, which is always
        decisive. Returns ``None`` when a BUY or SELL is present but the
        history never left the band, since there is then no honest number to
        compare a directional call against.
        """
        total = buy_n + sell_n + hold_n
        if total <= 0:
            return None
        directional = self.buy_hit_rate, self.sell_hit_rate
        if (buy_n or sell_n) and any(rate is None for rate in directional):
            return None
        expected = hold_n * self.hold_hit_rate
        if buy_n:
            expected += buy_n * (self.buy_hit_rate or 0.0)
        if sell_n:
            expected += sell_n * (self.sell_hit_rate or 0.0)
        return expected / total

    def summary(self) -> str:
        buy = "n/a" if self.buy_hit_rate is None else f"{self.buy_hit_rate:.1%}"
        sell = "n/a" if self.sell_hit_rate is None else f"{self.sell_hit_rate:.1%}"
        return (
            f"{self.symbol} +{self.horizon_days}d null: HOLD {self.hold_hit_rate:.1%} "
            f"· BUY {buy} · SELL {sell} · decisive {self.decisive_rate:.1%} "
            f"(n={self.samples})"
        )


# ---------------------------------------------------------------------------
# One query for every (symbol, horizon). Per-asset round trips would be 63
# scans of the same two tables.
#
# The horizon list is interpolated from DECISION_HORIZONS rather than bound,
# because a VALUES list cannot be parameterised — the values are ints from a
# module constant, never user input, and are coerced with int() below.
_NULL_SQL = """
SELECT
    df.symbol,
    h.horizon AS horizon_days,
    count(*)                                        AS samples,
    count(*) FILTER (WHERE abs(t.close / e.close - 1) * 100
                           <= df.realized_vol_30d
                              * sqrt(h.horizon::numeric / :days_per_year)
                              * 100)                AS inside,
    count(*) FILTER (WHERE (t.close / e.close - 1) * 100
                           >  df.realized_vol_30d
                              * sqrt(h.horizon::numeric / :days_per_year)
                              * 100)                AS above,
    count(*) FILTER (WHERE (t.close / e.close - 1) * 100
                           < -(df.realized_vol_30d
                               * sqrt(h.horizon::numeric / :days_per_year)
                               * 100))              AS below
FROM daily_features df
CROSS JOIN (VALUES {horizons}) AS h(horizon)
-- Exact-date joins, never a row offset: crypto_prices can have a gap, and
-- lead(close, n) would step over it and silently measure the wrong span.
JOIN crypto_prices e
       ON e.symbol    = df.symbol
      AND e.timestamp = df.timestamp + (:entry_lag * INTERVAL '1 day')
      AND e.timeframe = :timeframe
      AND e.exchange  = 'binance'
JOIN crypto_prices t
       ON t.symbol    = df.symbol
      AND t.timestamp = df.timestamp
                        + ((:entry_lag + h.horizon) * INTERVAL '1 day')
      AND t.timeframe = :timeframe
      AND t.exchange  = 'binance'
WHERE df.realized_vol_30d IS NOT NULL
  AND df.realized_vol_30d > 0
  AND e.close > 0
  -- Same "settled" definition as the evaluator and the baseline comparator.
  -- Without it the baseline is estimated from marginally more data than the
  -- things it is compared against, and the cross-check in
  -- test_baseline_comparison stops being an exact equality.
  AND df.timestamp + ((:entry_lag + h.horizon) * INTERVAL '1 day') < :cutoff
GROUP BY df.symbol, h.horizon
"""


async def load_null_baselines(
    session,
    horizons: tuple[int, ...] = DECISION_HORIZONS,
    *,
    now: datetime | None = None,
) -> dict[tuple[str, int], NullBaseline]:
    """Baseline for every (symbol, horizon) with enough history, in one query."""
    values = ", ".join(f"({int(h)})" for h in horizons)
    rows = (
        await session.execute(
            text(_NULL_SQL.format(horizons=values)),
            {
                "days_per_year": DAYS_PER_YEAR,
                "entry_lag": ENTRY_LAG_DAYS,
                "timeframe": DAILY_TIMEFRAME,
                "cutoff": _finalization_cutoff(now or datetime.now(tz=UTC)),
            },
        )
    ).all()

    baselines: dict[tuple[str, int], NullBaseline] = {}
    thin = 0
    for row in rows:
        if row.samples < MIN_NULL_SAMPLES:
            thin += 1
            continue
        baselines[(row.symbol, row.horizon_days)] = NullBaseline(
            symbol=row.symbol,
            horizon_days=row.horizon_days,
            samples=int(row.samples),
            inside=int(row.inside),
            above=int(row.above),
            below=int(row.below),
        )
    if thin:
        logger.debug(
            "[null-model] %d (symbol, horizon) pair(s) below %d historical "
            "samples — no baseline published for them",
            thin, MIN_NULL_SAMPLES,
        )
    return baselines

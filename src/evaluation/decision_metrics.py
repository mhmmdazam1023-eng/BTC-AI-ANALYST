"""DecisionAccuracyEvaluator: per-asset decision scoreboard (Phase 9).

Aggregates :class:`~src.db.models.DecisionEvaluation` rows into
``decision_accuracy_metrics``, one row per (symbol, horizon) per run. This is
the layer that turns a log of verdicts into something the system can be
judged — and eventually calibrated — by.

Scope discipline for v1
-----------------------
This is a SCOREBOARD, not a feedback loop into the decision itself. Nothing
here changes ``BUY_THRESHOLD``, the regime weights, or how the agent scores a
pillar. The forecasting loop earned the right to self-correct because it only
ever shifts two bounded numbers (``daily_drift``/``vol_factor``, clamped at
+/-25%); rewriting decision logic from a scoreboard that has not yet proven
itself would have a far larger blast radius.

The ONE consumer is the Kelly sizer's win rate — see :func:`load_win_rate_for`
— and even there every existing conservatism layer still applies.

Brier here means something different from ``learning_metrics``
--------------------------------------------------------------
The forecast Brier scores ``prob_price_up`` against "did the price rise".
This one scores ``confidence_score / 100`` against "was the decision right":

    brier = mean((confidence/100 - was_correct)^2)

That is the question that actually matters for a decision engine — *when it
says 80, is it right 80% of the time?* — and unlike a directional
probability it is defined for HOLD too, so no action has to be dropped from
the calibration measure.

NEUTRAL verdicts are excluded from every metric. They record that the market
never moved enough to test the call, which is the absence of evidence, not
evidence of failure. They are counted separately in ``neutral_count`` so a
quiet market cannot masquerade as a small sample.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import Integer, case, func, select

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import (
    DecisionAccuracyMetric,
    DecisionEvaluation,
    DecisionVerdict,
)
from src.evaluation.decision_evaluator import DECISION_HORIZONS
from src.evaluation.null_model import NullBaseline, load_null_baselines
from src.evaluation.sample_size import describe, effective_sample_size

logger = logging.getLogger(__name__)

#: Trailing decisive evaluations per (symbol, horizon) behind each metric row.
ROLLING_WINDOW = 90

#: Minimum INDEPENDENT decisive evaluations before a metrics row is emitted.
#: Deliberately the same bar the Kelly sizer already uses for a win rate
#: (``MIN_WIN_RATE_SAMPLES``): a number too thin to size a position with is
#: also too thin to publish as this asset's accuracy.
#:
#: Compared against ``effective_sample_size`` since 2026-07-31. At +30d the
#: ``ROLLING_WINDOW`` of 90 daily rows is only ~3 non-overlapping windows, so
#: the old row-count test could have published an "n=90" accuracy built on
#: three observations.
MIN_ACCURACY_SAMPLES = 30

#: Horizon whose accuracy feeds position sizing. 30 days matches
#: ``scenario_engine.DEFAULT_HORIZON_DAYS``, so the win rate and the payoff
#: ratio Kelly combines describe the SAME holding period. Mixing a 1-day win
#: rate with a 30-day payoff would be a unit error that still produces a
#: plausible-looking number.
SIZING_HORIZON_DAYS = 30


class DecisionAccuracyEvaluator:
    """Computes per-(symbol, horizon) accuracy and persists it."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    async def evaluate_and_save(self) -> int:
        """Compute metrics for every (symbol, horizon). Returns rows saved."""
        async with session_scope() as session:
            symbols = (
                (
                    await session.execute(
                        select(DecisionEvaluation.symbol)
                        .distinct()
                        .order_by(DecisionEvaluation.symbol)
                    )
                )
                .scalars()
                .all()
            )
            if not symbols:
                logger.info(
                    "[decision-accuracy] no decision evaluations yet — nothing "
                    "to aggregate"
                )
                return 0

            # One query for every (symbol, horizon) baseline — see null_model.
            # Loaded before the loop so 63 pairs cost one scan, not 63.
            baselines = await load_null_baselines(session)

            now = datetime.now(tz=UTC)
            saved = 0
            thin = 0
            for symbol in symbols:
                for horizon in DECISION_HORIZONS:
                    metrics = await self._metrics_for(session, symbol, horizon)
                    if metrics is None:
                        continue
                    metrics.update(
                        self._null_columns(
                            baselines.get((symbol, horizon)), metrics
                        )
                    )
                    # Gate on INDEPENDENT observations, not rows. 90 daily
                    # rows at +30d are ~3 non-overlapping windows; publishing
                    # an accuracy from those as "n=90" is the misreading this
                    # threshold was written to prevent in the first place.
                    if metrics["effective_sample_size"] < MIN_ACCURACY_SAMPLES:
                        thin += 1
                        continue
                    session.add(
                        DecisionAccuracyMetric(
                            evaluated_at=now,
                            symbol=symbol,
                            horizon_days=horizon,
                            **metrics,
                        )
                    )
                    saved += 1
                    null_hit = metrics["null_hit_rate"]
                    edge = metrics["edge_vs_null"]
                    logger.info(
                        "[decision-accuracy] %-12s +%2dd n=%-3d (eff %.1f) "
                        "hit=%5.1f%% vs null %s (edge %s) conf=%5.1f%% "
                        "brier=%.4f (BUY %d/%d, SELL %d/%d, HOLD %d/%d)",
                        symbol, horizon, metrics["sample_size"],
                        metrics["effective_sample_size"],
                        (metrics["hit_rate"] or 0) * 100,
                        "  n/a" if null_hit is None else f"{null_hit * 100:5.1f}%",
                        "  n/a" if edge is None else f"{edge * 100:+5.1f}pp",
                        metrics["avg_confidence"] or 0,
                        metrics["brier_score"] or 0,
                        metrics["buy_correct"], metrics["buy_total"],
                        metrics["sell_correct"], metrics["sell_total"],
                        metrics["hold_correct"], metrics["hold_total"],
                    )

        if thin:
            logger.info(
                "[decision-accuracy] %d (symbol, horizon) pair(s) still below "
                "the %d-sample minimum — not published yet",
                thin, MIN_ACCURACY_SAMPLES,
            )
        if saved:
            logger.info("[decision-accuracy] saved %d metric row(s)", saved)
        return saved

    # ------------------------------------------------------------------
    @staticmethod
    def _null_columns(baseline: NullBaseline | None, metrics: dict) -> dict:
        """The four skill-free columns for one metrics row.

        Blended over THIS row's own action mix. A HOLD-heavy window and a
        BUY-heavy window have genuinely different baselines — comparing both
        against one average would credit or punish the system for its action
        mix rather than its accuracy.
        """
        if baseline is None:
            # No baseline is not zero edge; it is an unknown baseline. All
            # four stay NULL so nobody reads a missing estimate as "no edge",
            # the same NULL-vs-0 discipline as the Part 4 feature contract.
            return {
                "null_hit_rate": None,
                "edge_vs_null": None,
                "null_decisive_rate": None,
                "null_samples": None,
            }

        null_hit = baseline.blended(
            buy_n=metrics["buy_total"],
            sell_n=metrics["sell_total"],
            hold_n=metrics["hold_total"],
        )
        observed = metrics.get("hit_rate")
        return {
            "null_hit_rate": null_hit,
            "edge_vs_null": (
                None if null_hit is None or observed is None else observed - null_hit
            ),
            "null_decisive_rate": baseline.decisive_rate,
            "null_samples": baseline.samples,
        }

    # ------------------------------------------------------------------
    @staticmethod
    async def _metrics_for(session, symbol: str, horizon: int) -> dict | None:
        """Metrics for one (symbol, horizon), or None when there is nothing."""
        rows = (
            await session.execute(
                select(
                    DecisionEvaluation.decision_timestamp,
                    DecisionEvaluation.action,
                    DecisionEvaluation.confidence_score,
                    DecisionEvaluation.verdict,
                )
                .where(
                    DecisionEvaluation.symbol == symbol,
                    DecisionEvaluation.horizon_days == horizon,
                )
                .order_by(DecisionEvaluation.target_date.desc())
                .limit(ROLLING_WINDOW)
            )
        ).all()
        if not rows:
            return None

        neutral = sum(1 for r in rows if r.verdict is DecisionVerdict.NEUTRAL)
        decisive = [r for r in rows if r.verdict is not DecisionVerdict.NEUTRAL]
        if not decisive:
            # Every call in the window was untestable. Reported via the caller's
            # thin-sample path rather than as a 0% hit rate, which would be a
            # lie about the system's accuracy.
            return {
                "sample_size": 0,
                "effective_sample_size": 0.0,
                "neutral_count": neutral,
                "hit_rate": None,
                "brier_score": None,
                "avg_confidence": None,
                "buy_total": 0, "buy_correct": 0,
                "sell_total": 0, "sell_correct": 0,
                "hold_total": 0, "hold_correct": 0,
            }

        correct = np.array(
            [1.0 if r.verdict is DecisionVerdict.CORRECT else 0.0 for r in decisive]
        )
        confidence = np.array([float(r.confidence_score) for r in decisive])

        counts: dict[str, list[int]] = {
            "BUY": [0, 0], "SELL": [0, 0], "HOLD": [0, 0]
        }
        for row in decisive:
            action = (row.action or "").strip().upper()
            if action in counts:
                counts[action][0] += 1
                if row.verdict is DecisionVerdict.CORRECT:
                    counts[action][1] += 1

        return {
            "sample_size": len(decisive),
            # Per (symbol, horizon), so only temporal overlap applies. This is
            # the number the publication gate uses — see evaluate_and_save.
            "effective_sample_size": effective_sample_size(
                [r.decision_timestamp for r in decisive], horizon
            ),
            "neutral_count": neutral,
            "hit_rate": float(correct.mean()),
            # confidence is 0-100; the Brier probability is that over 100.
            "brier_score": float(np.mean((confidence / 100.0 - correct) ** 2)),
            "avg_confidence": float(confidence.mean()),
            "buy_total": counts["BUY"][0], "buy_correct": counts["BUY"][1],
            "sell_total": counts["SELL"][0], "sell_correct": counts["SELL"][1],
            "hold_total": counts["HOLD"][0], "hold_correct": counts["HOLD"][1],
        }


# ---------------------------------------------------------------------------
async def load_win_rate_for(symbol: str) -> tuple[float, str, int] | None:
    """Per-asset decision win rate for the Kelly sizer, or ``None``.

    Returns ``(win_rate, source, sample_size)`` built from BUY decisions only,
    at :data:`SIZING_HORIZON_DAYS`.

    Why BUY only: Kelly sizes a long position, and ``f* = (b*p - q)/b`` reads
    ``p`` as "probability THIS bet wins". Folding in HOLD and SELL accuracy
    would answer a different question — "how often is the system right in
    general" — and inflate or deflate ``p`` with outcomes that have nothing to
    do with the trade being sized.

    Returns ``None`` when there are fewer than
    :data:`~src.forecasting.kelly_criterion.MIN_WIN_RATE_SAMPLES` decisive BUY
    evaluations, so the caller keeps its existing conservative default rather
    than acting on a thin estimate.
    """
    # Local import: kelly_criterion may import this module in future, and the
    # constant is the single source of truth for "enough samples".
    from src.forecasting.kelly_criterion import MIN_WIN_RATE_SAMPLES

    async with session_scope() as session:
        # Rows, not aggregates: the DATES are needed to count how many of
        # these windows actually fail to overlap.
        rows = (
            await session.execute(
                select(
                    DecisionEvaluation.decision_timestamp,
                    DecisionEvaluation.verdict,
                ).where(
                    DecisionEvaluation.symbol == symbol,
                    DecisionEvaluation.horizon_days == SIZING_HORIZON_DAYS,
                    DecisionEvaluation.action == "BUY",
                    DecisionEvaluation.verdict != DecisionVerdict.NEUTRAL,
                )
            )
        ).all()

    total = len(rows)
    correct = sum(1 for r in rows if r.verdict is DecisionVerdict.CORRECT)

    # THE CORRECTION. Thirty consecutive daily BUY evaluations at +30d share
    # 29/30 of their price path and are ~1 observation, not 30. Gating on the
    # row count let a single independent observation set Kelly's `p`.
    # Per-asset, so only temporal overlap applies.
    effective = effective_sample_size(
        [r.decision_timestamp for r in rows], SIZING_HORIZON_DAYS
    )
    if effective < MIN_WIN_RATE_SAMPLES:
        logger.debug(
            "[decision-accuracy] %s: %s decisive BUY evaluation(s) at +%dd "
            "(need %d independent) — Kelly keeps its default",
            symbol, describe(total, effective), SIZING_HORIZON_DAYS,
            MIN_WIN_RATE_SAMPLES,
        )
        return None

    win_rate = correct / total
    logger.info(
        "[decision-accuracy] %s: empirical BUY win rate %.4f from %s "
        "evaluation(s) at +%dd",
        symbol, win_rate, describe(total, effective), SIZING_HORIZON_DAYS,
    )
    # The EFFECTIVE count is what gets stored. `win_rate_samples` exists so a
    # fraction from 6 observations cannot be misread as one from 600 — and
    # the raw count is precisely the misreading it was meant to prevent.
    return win_rate, "empirical_decision_accuracy", int(effective)

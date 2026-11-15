"""DynamicWeightingEngine: regime -> per-pillar trust weights (Phase 8 Part 2).

The five sub-agents are not equally informative in every market. Trend
indicators lead in a strong uptrend and whipsaw in a range; sentiment
extremes are noise in a grind and the single best signal in a capitulation.
This module turns the :class:`~src.db.models.MarketRegime` label into the
weights the Master Agent uses to fuse the panel.

Why integers
------------
Weights are declared in **percentage points that sum to exactly 100**, then
divided by 100 on the way out. Declaring 0.35/0.20/0.15/0.15/0.15 directly
in floats sums to 0.9999999999999999, so a "weights must total 1.0" check
would either fail or need a fudge factor. Integers make the invariant exact
by construction and the table trivial to eyeball.

Provenance of these numbers
---------------------------
The task referred to weights "sesuai diskusi sebelumnya", which is not in
this repository's history — so the table below was derived from the
principles the Master Agent's own system prompt already states (Krause
2026): BTC trades as a high-beta risk asset in normal regimes, so macro and
technical lead; in stress regimes liquidity and sentiment extremes lead;
on-chain is slow and structural, so it earns weight where structure matters
(ranges, capitulation) and less where short-horizon momentum rules.

**Treat them as a reviewable starting point, not settled truth.** Every
number lives in :data:`REGIME_WEIGHTS` and changing one is a one-line edit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.db.models import MarketRegime

logger = logging.getLogger(__name__)

#: The five pillars, in the order they are reported. Keys match each
#: sub-agent's ``agent_name`` exactly — the fusion step joins on these.
PILLARS: tuple[str, ...] = (
    "technical",
    "macro",
    "onchain",
    "sentiment",
    "liquidity_risk",
)

#: Weights in PERCENTAGE POINTS per regime; each row must sum to 100.
#: Rationale per row:
#: * BULL_TRENDING     — trend persists; technical leads. Sentiment runs hot
#:                       for months, so it carries little marginal news.
#: * BEAR_TRENDING     — downtrends are macro/liquidity driven (dollar,
#:                       rates, deleveraging); macro leads, technical
#:                       confirms.
#: * SIDEWAYS_LOW_VOL  — no trend to follow, so technical is the least
#:                       reliable it ever is; structure (on-chain) and the
#:                       macro backdrop decide what breaks the range.
#: * SIDEWAYS_HIGH_VOL — chop WITH volatility is positioning-driven:
#:                       liquidity/funding first, sentiment extremes second
#:                       (they mean-revert hard here).
#: * CRASH_RECOVERY    — forced deleveraging plus capitulation. Sentiment
#:                       and liquidity carry nearly all the information;
#:                       technical is unreliable across gaps and macro is
#:                       too slow to matter within the bounce.
REGIME_WEIGHTS: dict[MarketRegime, dict[str, int]] = {
    MarketRegime.BULL_TRENDING: {
        "technical": 35, "macro": 20, "onchain": 15,
        "sentiment": 15, "liquidity_risk": 15,
    },
    MarketRegime.BEAR_TRENDING: {
        "technical": 25, "macro": 30, "onchain": 15,
        "sentiment": 10, "liquidity_risk": 20,
    },
    MarketRegime.SIDEWAYS_LOW_VOL: {
        "technical": 20, "macro": 25, "onchain": 25,
        "sentiment": 15, "liquidity_risk": 15,
    },
    MarketRegime.SIDEWAYS_HIGH_VOL: {
        "technical": 20, "macro": 15, "onchain": 10,
        "sentiment": 25, "liquidity_risk": 30,
    },
    MarketRegime.CRASH_RECOVERY: {
        "technical": 15, "macro": 10, "onchain": 15,
        "sentiment": 30, "liquidity_risk": 30,
    },
}

#: Used for UNDETERMINED, for an unknown label, and whenever regime
#: detection failed — "regime tidak jelas" means trust nothing extra.
DEFAULT_WEIGHTS_PCT: dict[str, int] = {pillar: 20 for pillar in PILLARS}

#: Confidence below which a detected regime is not trusted enough to skew
#: the weights. A 40/100 read is barely better than a coin flip, and acting
#: on it would tilt a real decision on noise.
MIN_CONFIDENCE_TO_APPLY = 45

#: Confidence multiplier applied on the FIRST day a new regime appears.
#: Regime boundaries are the least reliable part of any classifier, and
#: flipping the whole panel's weighting on a one-day read invites whipsaw.
#: A fresh regime must therefore clear a higher bar to take effect:
#: MIN_CONFIDENCE_TO_APPLY / TRANSITION_CONFIDENCE_FACTOR = 45 / 0.7,
#: i.e. **65+** raw confidence on day one, versus 45 once it has persisted.
#:
#: Applies ONLY to moves between two definite regimes. Leaving
#: ``UNDETERMINED`` is the emergence of clarity, not a whipsaw — there was
#: no prior conviction to flip against — so it is exempt (user decision,
#: 2026-07-28). See :meth:`DynamicWeightingEngine.evaluate`.
TRANSITION_CONFIDENCE_FACTOR = 0.7


@dataclass(slots=True)
class WeightingDecision:
    """Weights actually applied, plus WHY — for logs, reports and audits."""

    weights: dict[str, float]
    regime: str | None
    raw_confidence: int | None
    adjusted_confidence: float | None
    is_transition: bool
    used_default: bool
    reason: str

    def summary(self) -> str:
        confidence = (
            "n/a" if self.raw_confidence is None else str(self.raw_confidence)
        )
        adjusted = (
            "" if not self.is_transition
            # One decimal: 64 * 0.7 = 44.8 must not render as "45 < 45".
            else f" -> {self.adjusted_confidence:.1f} after transition penalty"
        )
        return (
            f"{self.regime or 'UNDETERMINED'} (confidence {confidence}{adjusted}): "
            f"{self.reason}"
        )


class WeightingError(ValueError):
    """Raised when a weight table is malformed (must be caught at import/test)."""


def _validate_table() -> None:
    """Fail loudly at import if any row is malformed."""
    for regime, weights in (
        *REGIME_WEIGHTS.items(),
        (MarketRegime.UNDETERMINED, DEFAULT_WEIGHTS_PCT),
    ):
        if set(weights) != set(PILLARS):
            raise WeightingError(
                f"{regime}: pillars {sorted(weights)} != {sorted(PILLARS)}"
            )
        total = sum(weights.values())
        if total != 100:
            raise WeightingError(f"{regime}: weights sum to {total}, expected 100")
        if any(value < 0 for value in weights.values()):
            raise WeightingError(f"{regime}: negative weight in {weights}")


_validate_table()


class DynamicWeightingEngine:
    """Maps a regime (and its confidence) onto per-pillar weights."""

    def __init__(self, *, min_confidence: int = MIN_CONFIDENCE_TO_APPLY) -> None:
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    def weights_for(
        self,
        regime: MarketRegime | str | None,
        confidence: int | None = None,
    ) -> dict[str, float]:
        """Weights summing to exactly 1.0, ignoring regime transitions.

        Falls back to equal weights when the regime is missing, unknown,
        ``UNDETERMINED``, or detected with confidence below
        :attr:`min_confidence`. Use
        :meth:`get_weights_with_transition` when the previous day's regime
        is known — it additionally damps a regime seen for the first time.
        """
        return self.evaluate(regime, None, confidence).weights

    # ------------------------------------------------------------------
    def evaluate(
        self,
        current_regime: MarketRegime | str | None,
        previous_regime: MarketRegime | str | None = None,
        confidence: int | None = None,
    ) -> WeightingDecision:
        """Transition-aware weighting with the full rationale attached.

        A regime detected for the FIRST time today has its confidence
        multiplied by :data:`TRANSITION_CONFIDENCE_FACTOR` before being
        tested against :attr:`min_confidence`. Regime boundaries are where
        any classifier is least reliable; requiring a fresh regime to be
        clearly convincing (65+ instead of 45) stops the panel's weighting
        from flip-flopping on a single day's read.

        Three cases are deliberately NOT transitions:

        * ``previous_regime is None`` — no prior observation, so nothing
          changed. A cold start must not be penalised for a flip that never
          happened. (``RegimeDetector.previous_regime()`` also returns None
          when the last stored row is stale, for the same reason: a change
          across a multi-day gap is not a fresh transition.)
        * ``previous_regime is UNDETERMINED`` — moving from "I don't know"
          to a definite read is the **emergence of clarity**, not a
          whipsaw. There was no prior conviction to flip-flop against, so
          the normal bar applies (user decision, 2026-07-28). Only
          transitions BETWEEN definite regimes are damped.
        * identical regimes — the regime has persisted, so it earns the
          normal bar.
        """
        resolved = self._resolve(current_regime)
        previous = self._resolve(previous_regime)

        if resolved is None or resolved is MarketRegime.UNDETERMINED:
            return WeightingDecision(
                weights=self._to_fractions(DEFAULT_WEIGHTS_PCT),
                regime=None if resolved is None else resolved.value,
                raw_confidence=confidence,
                adjusted_confidence=None,
                is_transition=False,
                used_default=True,
                reason="no usable regime — equal weights",
            )

        # previous=None means "unknown", never "different".
        # previous=UNDETERMINED means the detector had no conviction to flip
        # away from, so gaining one is clarity rather than a regime change.
        # `resolved` cannot be UNDETERMINED here — that returned above.
        is_transition = (
            previous is not None
            and previous is not MarketRegime.UNDETERMINED
            and previous is not resolved
        )
        adjusted = (
            None
            if confidence is None
            else float(confidence)
            * (TRANSITION_CONFIDENCE_FACTOR if is_transition else 1.0)
        )

        if adjusted is not None and adjusted < self.min_confidence:
            reason = (
                f"fresh transition from {previous.value}; damped confidence "
                f"{adjusted:.1f} < {self.min_confidence} — equal weights until "
                "the regime persists"
                if is_transition
                else f"confidence {adjusted:.1f} < {self.min_confidence} — "
                "too weak a read to skew the panel"
            )
            decision = WeightingDecision(
                weights=self._to_fractions(DEFAULT_WEIGHTS_PCT),
                regime=resolved.value,
                raw_confidence=confidence,
                adjusted_confidence=adjusted,
                is_transition=is_transition,
                used_default=True,
                reason=reason,
            )
            logger.info("[weights] %s", decision.summary())
            return decision

        decision = WeightingDecision(
            weights=self._to_fractions(REGIME_WEIGHTS[resolved]),
            regime=resolved.value,
            raw_confidence=confidence,
            adjusted_confidence=adjusted,
            is_transition=is_transition,
            used_default=False,
            reason=(
                "regime weights applied"
                + (
                    f" despite a fresh transition from {previous.value} "
                    "(confidence cleared the higher bar)"
                    if is_transition
                    else ""
                )
            ),
        )
        logger.info(
            "[weights] %s -> %s",
            decision.summary(),
            ", ".join(f"{k}={v * 100:.0f}%" for k, v in decision.weights.items()),
        )
        return decision

    def get_weights_with_transition(
        self,
        current_regime: MarketRegime | str | None,
        previous_regime: MarketRegime | str | None = None,
        confidence: int | None = None,
    ) -> dict[str, float]:
        """Transition-aware weights. Thin wrapper over :meth:`evaluate`."""
        return self.evaluate(current_regime, previous_regime, confidence).weights

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve(regime: MarketRegime | str | None) -> MarketRegime | None:
        """Coerce a label to a MarketRegime; unknown strings -> None."""
        if regime is None:
            return None
        if isinstance(regime, MarketRegime):
            return regime
        try:
            return MarketRegime(str(regime))
        except ValueError:
            logger.warning(
                "[weights] unknown regime label %r — using equal weights", regime
            )
            return None

    @staticmethod
    def _to_fractions(percentage_points: dict[str, int]) -> dict[str, float]:
        """Percentage points -> fractions summing to exactly 1.0."""
        return {pillar: value / 100.0 for pillar, value in percentage_points.items()}

    # ------------------------------------------------------------------
    @staticmethod
    def weighted_score(
        scores: dict[str, int | float],
        weights: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Weighted average of the reporting pillars' conviction scores.

        Returns ``(weighted_score, effective_weights)``.

        **Renormalisation matters here.** Pillars are isolated and any of
        them can fail to report. If three of five report and the raw weights
        were used as-is, the total would be < 1 and the fused score would be
        silently dragged toward zero — reading as "neutral" when it is
        really "incomplete". So the weights of the pillars that DID report
        are renormalised to sum to 1.0, and the caller is told the panel was
        short (``pillars_reporting``) rather than having it hidden in the
        arithmetic.
        """
        reporting = {
            pillar: float(weights.get(pillar, 0.0))
            for pillar in scores
            if weights.get(pillar, 0.0) > 0
        }
        total = sum(reporting.values())
        if total <= 0:
            logger.warning(
                "[weights] no reporting pillar carries weight — falling back "
                "to a plain average of %d score(s)", len(scores),
            )
            if not scores:
                return 0.0, {}
            equal = 1.0 / len(scores)
            effective = {pillar: equal for pillar in scores}
            return float(sum(scores.values()) * equal), effective

        effective = {pillar: value / total for pillar, value in reporting.items()}
        weighted = sum(float(scores[pillar]) * share for pillar, share in effective.items())
        return float(weighted), effective

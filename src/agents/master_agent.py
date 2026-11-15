"""Master Decision Agent — the Chief Investment Officer (Phase 4 / 8).

Receives the five validated sub-agent responses, weighs confluence and
regime context, and produces the final Buy/Hold/Sell decision with a
confidence score, executive summary, and a mandatory risk warning.

Phase 8 Part 2 adds **regime-aware dynamic weighting**. The panel is fused
arithmetically in Python (:meth:`MasterAgent.fuse`) rather than asking the
model to average numbers: the weighted score is an auditable figure that
must be reproducible from the stored inputs, and language models are poor
and inconsistent arithmetic engines. The model still owns the judgement —
it receives the fused score as a quantitative anchor and decides the action,
the confidence, and the narrative around it.

Note that :class:`~src.agents.schemas.MasterDecision` deliberately did NOT
grow ``weighted_score``/``regime`` fields: ``BaseAgent.ask()`` embeds the
response model's JSON Schema in the prompt, so any field added there becomes
something the model is invited to invent. Computed values travel beside the
decision in :class:`PanelFusion` instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.schemas import MasterDecision, SubAgentResponse
from src.agents.weighting_engine import DynamicWeightingEngine

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PanelFusion:
    """Deterministic fusion of the sub-agent panel under a weight vector."""

    weighted_score: float
    effective_weights: dict[str, float]
    raw_weights: dict[str, float] = field(default_factory=dict)
    regime: str | None = None
    regime_confidence: int | None = None
    pillars_reporting: int = 0

    @property
    def is_dynamic(self) -> bool:
        """True when a real regime skewed the weights away from equal."""
        return self.regime is not None and len(set(self.raw_weights.values())) > 1

    def weights_line(self) -> str:
        """Compact ``technical 35% · macro 20% · ...`` rendering."""
        return " · ".join(
            f"{pillar} {share * 100:.0f}%"
            for pillar, share in sorted(
                self.effective_weights.items(), key=lambda kv: -kv[1]
            )
        )

    def summary(self) -> str:
        regime = self.regime or "UNDETERMINED"
        confidence = (
            "" if self.regime_confidence is None else f"/{self.regime_confidence}"
        )
        return (
            f"regime={regime}{confidence} weighted_score={self.weighted_score:+.1f} "
            f"[{self.weights_line()}]"
        )


class MasterAgent(BaseAgent):
    """Chief Investment Officer: synthesizes the five pillar signals."""

    agent_name = "master"
    default_system_prompt = (
        "You are the CHIEF INVESTMENT OFFICER of a Bitcoin research desk. "
        "Five specialist sub-agents (technical, macro, on-chain, sentiment, "
        "liquidity/derivatives risk) each provide a signal (BULLISH/BEARISH/"
        "NEUTRAL), a conviction score (-100..+100) and reasoning. Your job:\n"
        "1. REGIME FIRST (Krause 2026): in normal regimes BTC trades as a "
        "high-beta risk asset — weight macro and technical most; in stress "
        "regimes (volatility spikes, extreme sentiment, crowded funding) "
        "weight liquidity-risk and sentiment extremes more heavily. "
        "On-chain is slow/structural — lowest short-horizon weight.\n"
        "2. CONFLUENCE: strong agreement across pillars justifies a decisive "
        "BUY or SELL with high confidence; material disagreement or weak "
        "|scores| should produce HOLD with honest, lower confidence.\n"
        "3. STATISTICAL FORECAST (when provided in the market context): a "
        "Monte Carlo block-bootstrap simulation supplies, per horizon, the "
        "5th/50th/95th percentile price and P(price up). Treat it as an "
        "unconditional probabilistic BASE RATE from historical return "
        "dynamics — a sanity anchor, not a directional oracle. USE THE BAND "
        "WIDTH, NOT P(up): the p5-p95 width is the measurably calibrated "
        "part, while P(up) is the bootstrap pool's own compounding drift and "
        "is near-constant per horizon (+30d has never left 55.5-57.7% across "
        "every forecast made; it rises with horizon because drift "
        "accumulates, not because the outlook improves). A panel that "
        "disagrees with P(up) is NOT in conflict with anything — do not "
        "temper confidence on that basis, and do not describe it as a "
        "conflict in the summary.\n"
        "4. DETECTED REGIME & WEIGHTED SCORE (when provided): a rule-based "
        "detector classifies the market regime and sets the pillar weights, "
        "and the panel's weighted average score has ALREADY been computed "
        "for you. Do NOT recompute it — treat it as given. Use it as the "
        "quantitative anchor for your action, and say in the "
        "executive_summary WHICH regime was detected and WHICH pillars "
        "carry the most weight because of it. You may still disagree with "
        "the anchor when the reasoning warrants — if you do, say why "
        "explicitly.\n"
        "5. SECTOR ROTATION (when provided): per-sector 7-day relative "
        "strength versus BTC shows where capital is flowing across "
        "narratives. This is CONTEXT for a Bitcoin decision, not a signal "
        "about Bitcoin itself: broad altcoin outperformance signals "
        "risk-on appetite (often BTC-dominance falling), while capital "
        "fleeing every sector back toward BTC signals defensiveness. Note "
        "any sector average drawn from only one or two coins — it is that "
        "token's noise, not a narrative. Never let sector rotation "
        "override the pillar panel; use it to frame the environment.\n"
        "6. COIN DNA (when provided): the asset's own profile — volatility "
        "regime, beta and correlation to BTC, market-cap tier, turnover, "
        "supply utilisation and any SCHEDULED unlock. Use it to size your "
        "conviction, not to flip its direction: a high-beta small-cap "
        "deserves a wider risk warning at the same signal strength, and a "
        "large scheduled unlock is a concrete downside catalyst that belongs "
        "in the risk_warning with its date horizon. Where a field says data "
        "is unavailable, say so plainly rather than inventing a value.\n"
        "7. RISK DISCIPLINE: the risk_warning must name the specific metric "
        "that would invalidate the decision and the level to watch. Never "
        "omit it.\n"
        "8. This is decision support for a human investor, NOT financial "
        "advice, and you must not assume leverage.\n"
        "Write executive_summary and risk_warning in Bahasa Indonesia."
    )

    # ------------------------------------------------------------------
    @staticmethod
    def fuse(
        sub_responses: list[SubAgentResponse],
        dynamic_weights: dict[str, float] | None = None,
        *,
        regime: str | None = None,
        regime_confidence: int | None = None,
    ) -> PanelFusion:
        """Weighted average of the panel. Pure, deterministic, testable.

        ``dynamic_weights=None`` means equal weights — which is exactly the
        pre-Phase-8 behaviour, so callers that never learned about regimes
        keep getting the same arithmetic they always did.
        """
        scores = {
            response.agent_name: response.score for response in sub_responses
        }
        engine = DynamicWeightingEngine()
        raw = dynamic_weights or engine.weights_for(None)
        weighted, effective = engine.weighted_score(scores, raw)
        fusion = PanelFusion(
            weighted_score=weighted,
            effective_weights=effective,
            raw_weights=dict(raw),
            regime=regime,
            regime_confidence=regime_confidence,
            pillars_reporting=len(sub_responses),
        )
        logger.info("[master] panel fusion: %s", fusion.summary())
        return fusion

    # ------------------------------------------------------------------
    async def decide(
        self,
        analysis_date: str,
        sub_responses: list[SubAgentResponse],
        market_context: dict[str, Any] | None = None,
        dynamic_weights: dict[str, float] | None = None,
        *,
        regime: str | None = None,
        regime_confidence: int | None = None,
        fusion: PanelFusion | None = None,
        sector_context: str | None = None,
        coin_dna_summary: str | None = None,
    ) -> MasterDecision:
        """Fuse the sub-agent outputs into the final decision.

        Backward compatible: called without ``dynamic_weights``/``regime``/
        ``sector_context`` (as Phases 4-7 do) the panel is fused with equal
        weights and neither the regime nor the sector block is added to the
        prompt — the prompt stays byte-identical to the earlier version.
        """
        if fusion is None:
            fusion = self.fuse(
                sub_responses,
                dynamic_weights,
                regime=regime,
                regime_confidence=regime_confidence,
            )

        panel = [response.model_dump(mode="json") for response in sub_responses]
        prompt_parts = [
            f"Analysis date: {analysis_date}",
            f"Sub-agent panel ({len(panel)} of 5 pillars reporting):",
            json.dumps(panel, indent=2),
        ]
        if market_context:
            prompt_parts.append(
                "Key market context:\n" + json.dumps(market_context, indent=2, default=str)
            )

        # Only describe the regime when one was actually detected: without
        # it the prompt stays byte-identical to the Phase 4-7 version.
        if fusion.regime is not None:
            prompt_parts.append(
                "DETECTED MARKET REGIME (rule-based, not model-generated):\n"
                + json.dumps(
                    {
                        "regime": fusion.regime,
                        "regime_confidence": fusion.regime_confidence,
                        "pillar_weights_pct": {
                            pillar: round(share * 100, 1)
                            for pillar, share in fusion.effective_weights.items()
                        },
                        "panel_weighted_score": round(fusion.weighted_score, 2),
                        "scale": "-100 (max bearish) .. +100 (max bullish)",
                    },
                    indent=2,
                )
                + "\nThe weighted score above is already computed — use it as "
                "your anchor, and name the regime and the top-weighted "
                "pillars in the executive_summary."
            )

        # The asset's own profile, kept separate from the market-wide blocks
        # so "what kind of coin is this" never blurs into "what is the market
        # doing". Phase 8 Part 5.
        if coin_dna_summary:
            prompt_parts.append(
                "COIN DNA (rule-based profile of the asset under analysis):\n"
                + coin_dna_summary
            )

        # Sector rotation is cross-asset CONTEXT, kept separate from the
        # regime block so the model cannot confuse "where capital sits
        # across narratives" with "what state Bitcoin itself is in".
        if sector_context:
            prompt_parts.append(
                "CROSS-ASSET SECTOR ROTATION (rule-based, computed from "
                "per-coin relative strength vs BTC):\n" + sector_context
            )

        if len(panel) < 5:
            prompt_parts.append(
                "NOTE: some pillars failed to report this run — the weights "
                "shown were renormalised over the pillars that DID report. "
                "Lower your confidence accordingly and mention the gap in "
                "the summary."
            )
        prompt_parts.append(
            "Produce the final decision now (JSON only, per the schema)."
        )
        return await self.ask("\n\n".join(prompt_parts), MasterDecision)

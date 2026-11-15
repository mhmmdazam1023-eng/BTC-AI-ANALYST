"""Strict Pydantic schemas for all agent outputs (Phase 4).

Every model response MUST validate against one of these before anything
downstream may use it (project rule #3). Required fields are strict;
unknown extra keys from chatty reasoning models are ignored rather than
rejected, so harmless verbosity never triggers a needless retry.
"""

from __future__ import annotations

from enum import Enum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Signal(str, Enum):
    """Directional read from one analysis pillar."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Action(str, Enum):
    """Final portfolio action from the Master Agent."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class SubAgentResponse(BaseModel):
    """Output contract for every sub-agent (one analysis pillar)."""

    model_config = ConfigDict(extra="ignore", use_enum_values=False)

    agent_name: str = Field(
        description="Identifier of the sub-agent, e.g. 'technical'."
    )
    signal: Signal = Field(
        description="Directional signal: BULLISH, BEARISH, or NEUTRAL."
    )
    score: int = Field(
        ge=-100,
        le=100,
        description=(
            "Conviction score: -100 (maximum bearish) to +100 (maximum "
            "bullish); 0 is perfectly neutral. Must be consistent with "
            "'signal' (BULLISH > 0, BEARISH < 0, NEUTRAL near 0)."
        ),
    )
    reasoning: str = Field(
        min_length=20,
        description=(
            "Concise justification (3-6 sentences, in Bahasa Indonesia) "
            "citing the specific feature values that drove the signal."
        ),
    )


class ConsolidatedAnalysis(BaseModel):
    """All five pillars scored in ONE model call (Phase 8 Part 7).

    Replaces five separate sub-agent calls with one, cutting quota use per
    asset from 6 to 1 — the difference between 126 calls/day (impossible on
    OpenRouter's free 50/day tier) and 21.

    ``liquidity_risk_score`` vs ``liquidity_score``
    ----------------------------------------------
    The canonical field is ``liquidity_risk_score``, matching the pillar's
    ``agent_name``. ``liquidity_score`` is accepted as an alias because the
    original spec used it — but it is NOT the preferred name, because
    ``liquidity_score`` is ALREADY a Coin DNA feature (turnover: daily
    volume / circulating supply, typically ~0.002). Showing the model
    ``liquidity_score: 0.00197`` in the DNA block while asking it to emit a
    ``liquidity_score`` of -100..100 invites exactly the confusion the
    strict schema exists to prevent.
    """

    model_config = ConfigDict(
        extra="ignore", use_enum_values=False, populate_by_name=True
    )

    technical_score: int = Field(
        ge=-100, le=100, description="Price action & momentum conviction."
    )
    onchain_score: int = Field(
        ge=-100, le=100, description="Network fundamentals conviction."
    )
    macro_score: int = Field(
        ge=-100, le=100, description="Dollar, rates & equity context conviction."
    )
    sentiment_score: int = Field(
        ge=-100, le=100, description="Crowd psychology conviction (contrarian at extremes)."
    )
    liquidity_risk_score: int = Field(
        ge=-100,
        le=100,
        validation_alias=AliasChoices("liquidity_risk_score", "liquidity_score"),
        serialization_alias="liquidity_risk_score",
        description="Derivatives positioning & liquidity risk conviction.",
    )
    reasoning_summary: str = Field(
        min_length=20,
        description=(
            "Concise justification (3-6 sentences, in Bahasa Indonesia) "
            "covering all five pillars and citing concrete numbers."
        ),
    )

    def pillar_scores(self) -> dict[str, int]:
        """``{agent_name: score}`` keyed exactly as the sub-agents are.

        The keys must match ``SubAgent.agent_name`` so the existing
        regime-weighting and fusion machinery works unchanged.
        """
        return {
            "technical": self.technical_score,
            "macro": self.macro_score,
            "onchain": self.onchain_score,
            "sentiment": self.sentiment_score,
            "liquidity_risk": self.liquidity_risk_score,
        }


class MasterDecision(BaseModel):
    """Output contract for the Master Decision Agent (CIO)."""

    model_config = ConfigDict(extra="ignore", use_enum_values=False)

    action: Action = Field(description="Final decision: BUY, HOLD, or SELL.")
    confidence_score: int = Field(
        ge=0,
        le=100,
        description=(
            "0-100 confidence in the decision, driven by the degree of "
            "confluence between the sub-agents."
        ),
    )
    executive_summary: str = Field(
        min_length=50,
        description=(
            "Executive summary (in Bahasa Indonesia, 4-8 sentences): the "
            "decision, the confluence/disagreement across pillars, and the "
            "dominant drivers."
        ),
    )
    risk_warning: str = Field(
        min_length=20,
        description=(
            "Concrete risk warning (in Bahasa Indonesia): what would "
            "invalidate this decision and which metric to watch."
        ),
    )

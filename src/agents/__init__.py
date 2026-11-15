"""AI reasoning layer (Phase 4): base client, schemas, sub-agents, master.

Endpoint: OpenRouter (OpenAI-compatible) with the reasoning model
``nvidia/nemotron-3-ultra-550b-a55b:free``. All outputs are strict Pydantic
objects.
"""

from src.agents.base_agent import (
    BaseAgent,
    QwenConfigurationError,
    QwenResponseError,
)
from src.agents.master_agent import MasterAgent, PanelFusion
from src.agents.schemas import Action, MasterDecision, Signal, SubAgentResponse
from src.agents.utils import (
    DNA_FALLBACKS,
    format_dna_summary,
    format_dna_value,
    liquidity_dna_context,
    market_tier_context,
    onchain_dna_context,
    technical_dna_context,
)
from src.agents.weighting_engine import (
    DEFAULT_WEIGHTS_PCT,
    PILLARS,
    REGIME_WEIGHTS,
    DynamicWeightingEngine,
)
from src.agents.sub_agents import (
    ALL_SUB_AGENTS,
    LiquidityRiskAgent,
    MacroAgent,
    OnChainAgent,
    SentimentAgent,
    SubAgent,
    TechnicalAgent,
)

__all__ = [
    "BaseAgent",
    "QwenConfigurationError",
    "QwenResponseError",
    "Signal",
    "Action",
    "SubAgentResponse",
    "MasterDecision",
    "SubAgent",
    "TechnicalAgent",
    "MacroAgent",
    "OnChainAgent",
    "SentimentAgent",
    "LiquidityRiskAgent",
    "ALL_SUB_AGENTS",
    "MasterAgent",
    "PanelFusion",
    "DynamicWeightingEngine",
    "REGIME_WEIGHTS",
    "DEFAULT_WEIGHTS_PCT",
    "PILLARS",
    "format_dna_summary",
    "format_dna_value",
    "technical_dna_context",
    "liquidity_dna_context",
    "onchain_dna_context",
    "market_tier_context",
    "DNA_FALLBACKS",
]

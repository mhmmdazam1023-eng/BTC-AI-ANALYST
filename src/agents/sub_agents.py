"""The five analysis sub-agents (Phase 4).

Each sub-agent sees ONLY its pillar's slice of the ``daily_features`` row
(plus minimal price context) and returns a validated
:class:`~src.agents.schemas.SubAgentResponse`. System prompts are grounded
in the research corpus in the parent folder (Krause 2026; Lin et al. 2025;
Putra & Pimada 2026; Zhivkov et al. 2026; Bieganowski & Ślepaczuk 2026).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, ClassVar

from src.agents.base_agent import BaseAgent
from src.agents.schemas import SubAgentResponse
from src.agents.utils import (
    liquidity_dna_context,
    market_tier_context,
    onchain_dna_context,
    technical_dna_context,
)

#: Price context every pillar receives alongside its own columns.
_COMMON_CONTEXT = ("btc_close",)


class SubAgent(BaseAgent):
    """Base class for pillar sub-agents: slices features, asks, validates."""

    #: daily_features columns this agent is allowed to see.
    focus_columns: ClassVar[tuple[str, ...]] = ()

    #: Builds this pillar's "Coin DNA" paragraph (Phase 8 Part 5). None means
    #: the pillar takes no DNA context and its prompt stays unchanged.
    dna_context_builder: ClassVar[Callable[[dict[str, Any]], str] | None] = None

    async def analyze(
        self,
        features: dict[str, Any],
        coin_dna: dict[str, Any] | None = None,
    ) -> SubAgentResponse:
        """Analyze one day's feature slice and return the validated signal.

        ``coin_dna`` is optional on purpose: called without it — as every
        pre-Phase-8 caller does — the prompt is byte-identical to the
        original, so no existing behaviour shifts.
        """
        visible = {
            column: features[column]
            for column in (*_COMMON_CONTEXT, *self.focus_columns)
            if column in features
        }
        sections = [
            f"Analysis date: {features.get('date', 'unknown')}",
            "Feature values for your pillar (daily data):\n"
            + json.dumps(visible, indent=2, default=str),
        ]

        dna_block = self.build_dna_context(coin_dna)
        if dna_block:
            sections.append(dna_block)

        sections.append(
            f"Produce your assessment now. Remember: agent_name must be "
            f"'{self.agent_name}', the 'reasoning' field must be written in "
            "Bahasa Indonesia and must cite the concrete numbers above."
        )
        return await self.ask("\n\n".join(sections), SubAgentResponse)

    @classmethod
    def build_dna_context(cls, coin_dna: dict[str, Any] | None) -> str:
        """This pillar's DNA paragraph, or "" when it takes none.

        Exposed separately from :meth:`analyze` so the prompt text can be
        asserted in tests without calling the model.
        """
        if not coin_dna or cls.dna_context_builder is None:
            return ""
        return cls.dna_context_builder(coin_dna)


class TechnicalAgent(SubAgent):
    """Price action & momentum (RSI, MACD, Bollinger, ATR, realized vol)."""

    agent_name = "technical"
    # Volatility regime, trend quality and BTC sensitivity.
    dna_context_builder = staticmethod(technical_dna_context)
    focus_columns = (
        "btc_volume",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "atr_14",
        "realized_vol_7d",
        "realized_vol_30d",
    )
    default_system_prompt = (
        "You are the TECHNICAL ANALYSIS sub-agent of a Bitcoin research desk. "
        "Assess ONLY price action and momentum from the given daily features: "
        "RSI(14) (overbought >70, oversold <30), MACD line vs signal and the "
        "histogram's direction (momentum inflection), the close's position "
        "within the Bollinger(20,2) bands, ATR(14) and realized volatility "
        "7d vs 30d (expanding or compressing risk). Weigh confirmation vs "
        "divergence between indicators; do not consider macro, on-chain, or "
        "sentiment factors — other agents cover those."
    )


class MacroAgent(SubAgent):
    """Dollar, rates, and equity-market context (FRED + indices)."""

    agent_name = "macro"
    # Size/sector only: macro hits small caps harder than large ones.
    dna_context_builder = staticmethod(market_tier_context)
    focus_columns = (
        "dxy_value",
        "dxy_roc_7d",
        "dxy_roc_30d",
        "us10y_value",
        "us10y_roc_7d",
        "us10y_roc_30d",
        "sp500_close",
        "sp500_roc_7d",
        "nasdaq_close",
        "nasdaq_roc_7d",
    )
    default_system_prompt = (
        "You are the MACROECONOMIC sub-agent of a Bitcoin research desk. "
        "Research grounding: BTC usually behaves as a high-beta risk asset "
        "whose strongest daily driver is the Nasdaq (Krause 2026: beta ~0.86 "
        "normally, 2.0+ during shocks); a strengthening broad dollar (DXY) "
        "is a headwind (Lin et al. 2025), while the 10Y yield's effect is "
        "regime-dependent. Interpret the levels and 7d/30d rates of change "
        "of DXY, US10Y, S&P 500 and Nasdaq 100 for their implication on BTC "
        "over the coming days. Ignore on-chain/sentiment/technical factors."
    )


class OnChainAgent(SubAgent):
    """Network fundamentals (hash rate & transaction activity)."""

    agent_name = "onchain"
    # Supply utilisation and realised issuance.
    dna_context_builder = staticmethod(onchain_dna_context)
    focus_columns = (
        "hash_rate_ma_7d",
        "hash_rate_ma_30d",
        "tx_count_ma_7d",
        "tx_count_ma_30d",
    )
    default_system_prompt = (
        "You are the ON-CHAIN sub-agent of a Bitcoin research desk. Assess "
        "network fundamentals from hash rate and transaction-count moving "
        "averages (7d vs 30d): a 7d MA above the 30d MA signals expanding "
        "security/usage (miner confidence, adoption); a 7d MA meaningfully "
        "below the 30d MA signals contraction or miner capitulation. These "
        "are slow, structural signals — be conservative about how much they "
        "say about the NEXT FEW DAYS of price, and reflect that in a "
        "moderate |score| unless the divergence is extreme."
    )


class SentimentAgent(SubAgent):
    """Crowd psychology (Fear & Greed index)."""

    agent_name = "sentiment"
    # Size/sector only: speculative buckets swing harder on sentiment.
    dna_context_builder = staticmethod(market_tier_context)
    focus_columns = (
        "fear_greed_value",
        "fear_greed_roc_7d",
    )
    default_system_prompt = (
        "You are the SENTIMENT sub-agent of a Bitcoin research desk. You "
        "read the Crypto Fear & Greed index (0=Extreme Fear, 100=Extreme "
        "Greed) and its 7-day rate of change. Research grounding: sentiment "
        "is a short-horizon, partly CONTRARIAN signal (Putra & Pimada 2026): "
        "extreme fear (<25) with an IMPROVING trend often marks accumulation "
        "zones, while extreme greed (>75) flags froth; a falling index from "
        "mid-range signals deteriorating risk appetite. Combine the level "
        "AND the direction of change; do not consider other pillars."
    )


class LiquidityRiskAgent(SubAgent):
    """Derivatives positioning & liquidity risk (funding, OI, volatility)."""

    agent_name = "liquidity_risk"
    # Cap tier, turnover and scheduled supply pressure.
    dna_context_builder = staticmethod(liquidity_dna_context)
    focus_columns = (
        "btc_volume",
        "funding_rate",
        "funding_rate_change_1d",
        "open_interest",
        "oi_volume_ratio",
        "atr_14",
        "realized_vol_7d",
        "realized_vol_30d",
    )
    default_system_prompt = (
        "You are the LIQUIDITY & DERIVATIVES RISK sub-agent of a Bitcoin "
        "research desk. Research grounding: funding-rate dynamics and open "
        "interest are key liquidity-regime signals (Zhivkov et al. 2026), "
        "and crowded leveraged positioning amplifies cascades during stress "
        "(the Oct-2025 flash crash; Bieganowski & Ślepaczuk 2026). Read the "
        "8h funding rate (strongly positive = crowded longs, negative = "
        "crowded shorts / potential squeeze fuel), its 1-day change, open "
        "interest in BTC, and the OI/volume ratio (high = leverage stacked "
        "on thin turnover = fragile). Volatility context (ATR, realized vol) "
        "tells you how violent any unwind would be. Score the RISK-ADJUSTED "
        "directional implication for the next few days."
    )


#: The full roster used by the inference pipeline.
ALL_SUB_AGENTS: tuple[type[SubAgent], ...] = (
    TechnicalAgent,
    MacroAgent,
    OnChainAgent,
    SentimentAgent,
    LiquidityRiskAgent,
)

"""Consolidated Analyst — all five pillars in ONE model call (Phase 8 Part 7).

Why this exists
---------------
OpenRouter's free tier allows **50 model requests per DAY** (verified live,
2026-07-29). The Phase 4 architecture spends 6 calls per asset (5 pillars +
Master), so a 21-asset portfolio needs 126 — impossible. This agent scores
all five pillars in a single call, bringing the portfolio to 21 calls/day
with headroom to spare.

What is deliberately NOT delegated to the model
-----------------------------------------------
The model returns RAW PILLAR SCORES and nothing else. Every number derived
from them — the regime-weighted fusion, the action threshold, the
confidence — is computed in Python (:mod:`src.agents.decision_composer`).
Language models are unreliable and inconsistent arithmetic engines, and
these figures must be reproducible from the stored inputs months later.
This is the same separation the Master Agent has had since Part 2, applied
one level deeper.

Fidelity trade-off, stated plainly
----------------------------------
Five specialist prompts, each seeing only its own pillar, cannot be
perfectly reproduced by one generalist prompt seeing everything. Isolation
was a feature: it stopped the technical read from being coloured by the
macro read. Consolidation reintroduces exactly that cross-contamination.
The prompt below pushes back on it explicitly ("score each pillar on its
own evidence"), but the effect cannot be eliminated — only managed. That is
the price of fitting 21 assets into a 50-call budget, and it is a
deliberate, reversible choice: ``--mode panel`` still runs the original
five-agent path.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.schemas import ConsolidatedAnalysis, Signal, SubAgentResponse
from src.agents.usage_tracker import usage_context
from src.db.models import BASE_SYMBOL
from src.agents.utils import (
    liquidity_dna_context,
    market_tier_context,
    onchain_dna_context,
    technical_dna_context,
)

logger = logging.getLogger(__name__)

#: C2 (2026-08-09). ``prob_price_up`` reads like a directional call and is
#: not one — it is the compounding drift of the bootstrap pool, and it is
#: near-constant per horizon. Measured across all 22 daily forecasts:
#:
#:     horizon   mean    sd      range
#:     +7d       51.6%   4.17pp  44.5-55.3
#:     +30d      56.6%   0.54pp  55.5-57.7
#:     +60d      59.4%   0.53pp  58.3-60.4
#:     +180d     65.7%   0.58pp  64.7-66.7
#:
#: The monotone rise with horizon is the giveaway: a market view would not
#: care that the question was asked about 180 days instead of 30. The pool's
#: historical drift simply accumulates. At +30d it has never left a 2.2-point
#: band across every forecast ever made.
#:
#: The block already said "not a directional oracle". A qualitative label is
#: weaker than the number — a reader (human or model) still sees "56.7%" and
#: feels mildly bullish. So the number is stated.
#:
#: NOTE the exception, and state it honestly: +7d is the one horizon that
#: does move (sd 4.17pp). Short horizons have little accumulated drift, so
#: resampling noise dominates. That variation is noise, not signal either.
PROB_UP_IS_STRUCTURAL = (
    "HOW TO READ prob_up_pct — IT IS NOT A DIRECTIONAL FORECAST. It is the "
    "bootstrap pool's own historical drift, compounded over the horizon, and "
    "it is near-constant. Across every forecast this system has produced: "
    "+30d has stayed within 55.5-57.7% (sd 0.5pp) and +180d within "
    "64.7-66.7%. It rises with horizon because drift accumulates, not "
    "because the market looks better further out. A value near 57% at +30d "
    "is the DEFAULT, not evidence — it appears on bullish and bearish days "
    "alike. (+7d is noisier, sd 4.2pp, but that is resampling noise, also "
    "not signal.) Do NOT let prob_up_pct move any pillar score. The part of "
    "this forecast that is measurably calibrated is the WIDTH of the "
    "p5-p95 band, not its direction."
)

#: Feature columns handed to the consolidated agent, grouped by pillar so
#: the prompt can present them under headings the model can attribute.
#: Mirrors the five sub-agents' ``focus_columns`` exactly — no pillar gains
#: or loses evidence by being consolidated.
PILLAR_FEATURES: dict[str, tuple[str, ...]] = {
    "technical": (
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower", "atr_14",
        "realized_vol_7d", "realized_vol_30d",
    ),
    "macro": (
        "dxy_value", "dxy_roc_7d", "dxy_roc_30d",
        "us10y_value", "us10y_roc_7d", "us10y_roc_30d",
        "sp500_close", "sp500_roc_7d", "nasdaq_close", "nasdaq_roc_7d",
    ),
    "onchain": (
        "hash_rate_ma_7d", "hash_rate_ma_30d",
        "tx_count_ma_7d", "tx_count_ma_30d",
    ),
    "sentiment": ("fear_greed_value", "fear_greed_roc_7d"),
    "liquidity_risk": (
        "asset_volume", "funding_rate", "funding_rate_change_1d",
        "open_interest", "oi_volume_ratio",
    ),
}

#: Price context shown once at the top, not attributed to any pillar.
_COMMON_CONTEXT = ("asset_close", "btc_close")

#: Columns that always carry BITCOIN's reading, whatever asset is scored.
#:
#: ``on_chain_metrics`` and ``derivatives_metrics`` are Bitcoin-only tables,
#: so the feature engine copies Bitcoin's values onto every altcoin row —
#: verified 2026-07-29, byte-identical across all 21 symbols. Left unlabelled
#: the model reports Bitcoin's hash rate as if it were the altcoin's own
#: network activity (it did, for all 20 altcoins in that run). Naming them
#: costs nothing and stops the fabrication.
_BITCOIN_SOURCED_COLUMNS: frozenset[str] = frozenset(
    {
        "hash_rate_ma_7d", "hash_rate_ma_30d",
        "tx_count_ma_7d", "tx_count_ma_30d",
        "funding_rate", "funding_rate_change_1d",
        "open_interest", "oi_volume_ratio",
    }
)


class ConsolidatedAnalyst(BaseAgent):
    """Scores all five pillars in one call and returns a strict JSON object."""

    agent_name = "consolidated"
    default_system_prompt = (
        "You are a CRYPTOCURRENCY RESEARCH DESK scoring one asset across "
        "five independent pillars in a single pass. Return ONLY a JSON "
        "object matching the schema.\n\n"
        "SCORE EACH PILLAR ON ITS OWN EVIDENCE. This is the single most "
        "important instruction: five specialists would each see only their "
        "own data, and you must reproduce that discipline. Do NOT let a "
        "bearish macro read drag down the technical score, or a fearful "
        "sentiment reading colour the on-chain score. A day where technicals "
        "are strong and macro is weak MUST produce a positive technical "
        "score and a negative macro score — not two mediocre ones.\n\n"
        "Scale: -100 (maximum bearish) to +100 (maximum bullish); 0 is "
        "genuinely neutral. Use the full range — a pillar with strong, "
        "one-sided evidence deserves |score| > 50. Reserve small |scores| "
        "for genuinely mixed or absent evidence.\n\n"
        "PILLAR DEFINITIONS (research-grounded):\n"
        "1. TECHNICAL — RSI(14) (overbought >70, oversold <30), MACD line vs "
        "signal and histogram direction, position within Bollinger(20,2), "
        "ATR and realized vol 7d vs 30d (expanding or compressing risk). "
        "Weigh confirmation vs divergence.\n"
        "2. MACRO — BTC behaves as a high-beta risk asset whose strongest "
        "daily driver is the Nasdaq (Krause 2026: beta ~0.86 normally, 2.0+ "
        "in shocks); a strengthening broad dollar (DXY) is a headwind (Lin "
        "et al. 2025); the 10Y yield's effect is regime-dependent.\n"
        "3. ON-CHAIN — hash rate and transaction-count 7d vs 30d moving "
        "averages: 7d above 30d signals expanding security/usage, below "
        "signals contraction or miner capitulation. These are SLOW, "
        "structural signals — keep |score| moderate unless the divergence is "
        "extreme. For any asset OTHER than Bitcoin these are BITCOIN's "
        "network metrics — no per-altcoin on-chain feed exists here. Score "
        "them as ecosystem backdrop only, keep |score| small, and state in "
        "the summary that they describe Bitcoin's network, not the "
        "altcoin's.\n"
        "4. SENTIMENT — Fear & Greed level (0=Extreme Fear, 100=Extreme "
        "Greed) and its 7-day change. Partly CONTRARIAN (Putra & Pimada "
        "2026): extreme fear (<25) with an IMPROVING trend often marks "
        "accumulation; extreme greed (>75) flags froth. Combine level AND "
        "direction.\n"
        "5. LIQUIDITY RISK — funding rate (strongly positive = crowded "
        "longs, negative = squeeze fuel), its 1-day change, open interest, "
        "and OI/volume ratio (high = leverage on thin turnover = fragile). "
        "Crowded leveraged positioning amplifies cascades during stress "
        "(Zhivkov et al. 2026; Bieganowski & Slepaczuk 2026). Score the "
        "RISK-ADJUSTED directional implication. For any asset OTHER than "
        "Bitcoin, only 'asset_volume' belongs to that asset — funding rate, "
        "open interest and OI/volume are BITCOIN's. Lean on the asset's own "
        "volume and liquidity DNA, and attribute the rest to Bitcoin "
        "explicitly.\n\n"
        "COIN CHARACTERISTICS: when a Coin DNA block is supplied, use it to "
        "calibrate — a high-beta small-cap with a scheduled unlock warrants "
        "more caution at the same signal strength. Where DNA says data is "
        "unavailable, say so rather than inventing a value.\n\n"
        "Write 'reasoning_summary' in Bahasa Indonesia, 3-6 sentences, "
        "citing concrete numbers from the data and naming which pillars "
        "disagree. Do NOT compute a final action, weighted score, or "
        "position size — those are calculated deterministically downstream "
        "from your scores."
    )

    # ------------------------------------------------------------------
    async def analyze(
        self,
        features: dict[str, Any],
        coin_dna: dict[str, Any] | None = None,
        *,
        symbol: str | None = None,
        regime_context: str | None = None,
        sector_context: str | None = None,
        forecast_context: dict[str, Any] | None = None,
        intraday_context: str | None = None,
        intraday_forecast_context: dict[str, Any] | None = None,
    ) -> ConsolidatedAnalysis:
        """Score all five pillars for one asset in a single model call.

        ``intraday_context`` / ``intraday_forecast_context`` are Phase 8 Part 8
        and BITCOIN ONLY — the caller gates them, and both are ignored for any
        other symbol so an altcoin prompt can never receive Bitcoin's 4-hour
        tape labelled as its own (the same failure mode as the on-chain
        columns before they were labelled). With both omitted the prompt is
        byte-identical to the pre-Part-8 version.
        """
        asset = symbol or (coin_dna or {}).get("symbol", "the asset")

        common = {
            column: features[column]
            for column in _COMMON_CONTEXT
            if column in features
        }
        sections = [
            f"Asset under analysis: {asset}",
            f"Analysis date: {features.get('date', 'unknown')}",
            "Price context:\n" + json.dumps(common, indent=2, default=str),
        ]

        is_bitcoin = asset == BASE_SYMBOL
        for pillar, columns in PILLAR_FEATURES.items():
            visible = {
                column: features[column]
                for column in columns
                if column in features
            }
            header = f"{pillar.upper()} pillar data"
            # For altcoins, say plainly which numbers are Bitcoin's. Without
            # this the model narrates BTC's hash rate as the altcoin's own.
            borrowed = (
                sorted(set(visible) & _BITCOIN_SOURCED_COLUMNS)
                if not is_bitcoin
                else []
            )
            if borrowed:
                header += (
                    f" — WARNING: {', '.join(borrowed)} are BITCOIN's readings, "
                    f"NOT {asset}'s. This dataset has no per-altcoin on-chain "
                    "or derivatives feed. Treat them as ecosystem backdrop, "
                    f"keep |score| small, and never describe them as {asset}'s "
                    "own network or positioning data"
                )
            sections.append(
                f"{header}:\n" + json.dumps(visible, indent=2, default=str)
            )

        # The same per-pillar DNA blocks the specialists receive in Part 5,
        # concatenated. Reused rather than re-written so the two paths can
        # never drift apart.
        if coin_dna:
            dna_blocks = [
                builder(coin_dna)
                for builder in (
                    technical_dna_context,
                    liquidity_dna_context,
                    onchain_dna_context,
                    market_tier_context,
                )
            ]
            sections.append("\n\n".join(block for block in dna_blocks if block))

        if regime_context:
            sections.append(
                "MARKET REGIME (rule-based, not model-generated):\n" + regime_context
            )
        if sector_context:
            sections.append(
                "CROSS-ASSET SECTOR ROTATION (rule-based):\n" + sector_context
            )
        if forecast_context:
            # Bitcoin only — see run_inference. Carried here because in
            # consolidated mode no Master call happens, and dropping it would
            # silently discard the Phase 6 statistical base rate from the
            # decision entirely. It costs no extra request: it is prompt text.
            sections.append(
                "STATISTICAL FORECAST — Monte Carlo block-bootstrap "
                "(statistical_forecast_monte_carlo). Treat as an "
                "unconditional BASE RATE from historical return dynamics, a "
                "sanity anchor rather than a directional oracle; note any "
                "conflict with your pillar reads in the summary.\n"
                + PROB_UP_IS_STRUCTURAL
                + "\n"
                + json.dumps(forecast_context, indent=2, default=str)
            )

        # ---- Phase 8 Part 8: BTC intraday context (hard-gated to Bitcoin) ----
        # The gate is repeated here rather than trusted to the caller: this is
        # the exact shape of bug that let Bitcoin's hash rate be narrated as an
        # altcoin's own on-chain activity for a full 20-asset run.
        if is_bitcoin and intraday_context:
            sections.append(intraday_context)
        if is_bitcoin and intraday_forecast_context:
            sections.append(
                "NEAR-TERM STATISTICAL FORECAST — Monte Carlo block-bootstrap "
                "over 4-HOUR bars, anchored to the last CLOSED bar. Horizons "
                "are 4h/12h/24h, NOT days. Treat it exactly like the daily "
                "forecast: an unconditional base rate, not a directional "
                "oracle. Intraday direction is close to a coin flip — Brier "
                "~0.25 at all three horizons, the exact score of always "
                "guessing 50%. Measured across every intraday forecast made, "
                "P(up) has stayed inside 43.7-50.6% at 4h, 39.3-50.5% at 12h "
                "and 47.5-56.3% at 24h: it hugs 50% by construction. Read "
                "the BAND WIDTH, not the direction, and say so plainly "
                "rather than manufacture conviction:\n"
                + json.dumps(intraday_forecast_context, indent=2, default=str)
            )

        sections.append(
            "Produce the five pillar scores now (JSON only, per the schema). "
            "Remember: score each pillar on ITS OWN evidence."
        )

        # Cost attribution rides the context var rather than an `ask()`
        # argument. Widening `ask()` was tried and REVERTED: the stubs in
        # test_multi_asset_phase8 implement the real signature, so the extra
        # kwarg broke all three assets — exactly the failure mode this
        # indirection exists to prevent. Analytics must not be able to break
        # inference. `asset` above falls back to the literal "the asset" for
        # prompt prose, so attribution reads the raw value and stays NULL
        # rather than storing that phrase as a symbol.
        with usage_context(symbol or (coin_dna or {}).get("symbol")):
            analysis = await self.ask("\n\n".join(sections), ConsolidatedAnalysis)
        logger.info(
            "[consolidated] %s scored in 1 call: tech=%+d macro=%+d onchain=%+d "
            "sentiment=%+d liquidity=%+d",
            asset, analysis.technical_score, analysis.macro_score,
            analysis.onchain_score, analysis.sentiment_score,
            analysis.liquidity_risk_score,
        )
        return analysis

    # ------------------------------------------------------------------
    @staticmethod
    def to_panel(analysis: ConsolidatedAnalysis) -> list[SubAgentResponse]:
        """Convert one consolidated response into the 5-pillar panel.

        Everything downstream — regime weighting, ``PanelFusion``, the
        ``sub_agents_json`` audit trail — already speaks
        :class:`SubAgentResponse`. Adapting here means the consolidated path
        reuses all of it unchanged instead of growing a parallel
        implementation that could drift.
        """
        panel: list[SubAgentResponse] = []
        for agent_name, score in analysis.pillar_scores().items():
            signal = (
                Signal.BULLISH if score > 0
                else Signal.BEARISH if score < 0
                else Signal.NEUTRAL
            )
            panel.append(
                SubAgentResponse(
                    agent_name=agent_name,
                    signal=signal,
                    score=score,
                    reasoning=(
                        f"[Analisis konsolidasi, skor pilar {agent_name}: "
                        f"{score:+d}] {analysis.reasoning_summary}"
                    ),
                )
            )
        return panel

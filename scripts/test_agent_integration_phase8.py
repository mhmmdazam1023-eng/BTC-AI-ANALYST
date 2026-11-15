"""Phase 8 Part 5 test: Coin DNA injection into the agent prompts.

Gates
-----
1. **Prompt inspection** — with synthetic SOL DNA (high vol, beta 1.2,
   5% unlock in 10 days) the Technical prompt carries the beta and the
   Liquidity prompt carries both the shock size and the unlock horizon.
2. **Routing**          — each pillar receives ONLY its own slice: the
   Technical agent must not be told about unlocks, the Liquidity agent must
   not be told about beta, and Macro/Sentiment get size context only.
3. **NULL handling**    — with every DNA field NULL, no prompt contains
   "None", "NULL" or "nan", and the fallback text distinguishes "no unlock
   scheduled" from "data unavailable".
4. **BTC compatibility**— real Bitcoin DNA renders without error, shows
   beta 1.00 / large-cap, and is flagged as the reference asset so the
   model does not report the identity as a finding.
5. **Backward compat**  — omitting ``coin_dna`` leaves every prompt exactly
   as it was before Part 5, and ``analyze()`` still works.
6. **Live wiring**      — ``load_coin_dna`` returns real values from
   ``daily_features``, and the Master Agent receives the summary.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_agent_integration_phase8.py
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents import (  # noqa: E402
    ALL_SUB_AGENTS,
    LiquidityRiskAgent,
    MacroAgent,
    MasterAgent,
    OnChainAgent,
    SentimentAgent,
    TechnicalAgent,
    format_dna_summary,
    format_dna_value,
)
from src.agents.schemas import MasterDecision, Signal, SubAgentResponse  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine  # noqa: E402
from src.db.models import BASE_SYMBOL  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_agent_integration_phase8")

#: Synthetic profile from the Part 5 brief: high vol, beta 1.2, 5% unlock
#: in 10 days.
SOL_DNA = {
    "symbol": "SOL/USDT",
    "sector": "L1",
    "vol_regime": "HIGH",
    "vol_percentile_30d": 82.0,
    "btc_beta_30d": 1.2,
    "btc_corr_30d": 0.84,
    "trend_strength": 1.35,
    "momentum_score": 22.0,
    "liquidity_score": 0.00197,
    "market_cap_tier": "MID",
    "supply_shock_score_30d": 5.0,
    "days_to_next_unlock": 10,
    "supply_utilization": 0.9239,
    "inflation_rate_7d": 0.184,
}

#: Every field absent — the state most coins are in today.
EMPTY_DNA = {
    "symbol": "NEW/USDT",
    "sector": None,
    **{key: None for key in SOL_DNA if key not in ("symbol", "sector")},
}

#: Raw Python/SQL nulls that must NEVER reach a prompt.
#:
#: Matched on WORD BOUNDARIES, not as substrings. "nan" is a substring of
#: ordinary Indonesian words — "keyakinan", "penanganan", "pendanaan" — so a
#: naive `"nan" in prompt` check fails on correct output forever. This bit
#: on the first run of this very gate.
FORBIDDEN_TOKENS = ("None", "NULL", "null", "nan", "NaN", "NaT")
_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(token) for token in FORBIDDEN_TOKENS) + r")\b"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_clean(text: str, where: str) -> None:
    """Fail if a raw null token appears as a standalone word in ``text``."""
    found = _FORBIDDEN_RE.findall(text)
    _require(
        not found,
        f"{where} leaked raw null token(s) {sorted(set(found))}: {text}",
    )


# ---------------------------------------------------------------------------
def gate_prompt_inspection() -> None:
    """The brief's own assertions: beta in Technical, unlock in Liquidity."""
    technical = TechnicalAgent.build_dna_context(SOL_DNA)
    liquidity = LiquidityRiskAgent.build_dna_context(SOL_DNA)

    _require("Beta" in technical, f"Technical prompt lacks 'Beta': {technical}")
    _require(
        "1.2" in technical,
        f"Technical prompt lacks the beta value '1.2': {technical}",
    )
    _require(
        "SOL/USDT" in technical, "Technical prompt should name the symbol"
    )
    _require(
        "TINGGI" in technical,
        f"HIGH vol regime should render as 'TINGGI': {technical}",
    )
    logger.info("Gate 1 OK: Technical prompt carries Beta 1.2 and vol regime")

    _require(
        "5.0%" in liquidity,
        f"Liquidity prompt lacks the supply shock '5.0%': {liquidity}",
    )
    _require(
        "10 hari" in liquidity,
        f"Liquidity prompt lacks the unlock horizon '10 hari': {liquidity}",
    )
    _require(
        "mid-cap" in liquidity,
        f"Liquidity prompt lacks the cap tier: {liquidity}",
    )
    logger.info("Gate 1 OK: Liquidity prompt carries 5.0% shock and 10 hari unlock")

    onchain = OnChainAgent.build_dna_context(SOL_DNA)
    _require(
        "92.4%" in onchain,
        f"On-chain prompt should render utilisation as a percent: {onchain}",
    )
    _require("0.184%" in onchain, f"On-chain prompt lacks inflation: {onchain}")
    logger.info("Gate 1 OK: On-chain prompt carries utilisation 92.4% and inflation")


def gate_routing() -> None:
    """Each pillar gets ONLY its own slice."""
    technical = TechnicalAgent.build_dna_context(SOL_DNA)
    liquidity = LiquidityRiskAgent.build_dna_context(SOL_DNA)
    macro = MacroAgent.build_dna_context(SOL_DNA)
    sentiment = SentimentAgent.build_dna_context(SOL_DNA)

    _require(
        "unlock" not in technical.lower(),
        f"Technical must not receive unlock context: {technical}",
    )
    _require(
        "beta" not in liquidity.lower(),
        f"Liquidity must not receive beta context: {liquidity}",
    )
    for name, text in (("Macro", macro), ("Sentiment", sentiment)):
        _require(text, f"{name} should still receive size context")
        _require(
            "beta" not in text.lower() and "unlock" not in text.lower(),
            f"{name} should receive size/sector only, got: {text}",
        )
        _require(
            "mid-cap" in text and "L1" in text,
            f"{name} should name the cap tier and sector: {text}",
        )
    _require(
        macro == sentiment,
        "Macro and Sentiment share one builder and should produce identical text",
    )
    logger.info(
        "Gate 2 OK: routing correct — Technical(no unlocks), Liquidity(no beta), "
        "Macro/Sentiment(size+sector only)"
    )


def gate_null_handling() -> None:
    """All-NULL DNA must never leak None/NULL/nan into a prompt."""
    for agent_cls in ALL_SUB_AGENTS:
        text = agent_cls.build_dna_context(EMPTY_DNA)
        _require(text, f"{agent_cls.agent_name} produced an empty DNA block")
        _assert_clean(text, f"{agent_cls.agent_name} prompt")
    logger.info(
        "Gate 3 OK: all %d pillars render all-NULL DNA without leaking "
        "None/NULL/nan", len(ALL_SUB_AGENTS),
    )

    # The fallbacks must MEAN different things, not share one generic phrase.
    # Compared case-insensitively: a clause can legitimately open a sentence.
    liquidity = LiquidityRiskAgent.build_dna_context(EMPTY_DNA).lower()
    _require(
        "tidak ada unlock terjadwal" in liquidity,
        f"a NULL unlock must read as 'nothing scheduled': {liquidity}",
    )
    onchain = OnChainAgent.build_dna_context(EMPTY_DNA).lower()
    _require(
        "belum tersedia" in onchain,
        f"NULL supply data must read as 'not yet available': {onchain}",
    )
    _require(
        "tidak ada unlock terjadwal" not in onchain,
        "the 'nothing scheduled' phrasing must not be reused for missing data "
        "— they mean opposite things",
    )
    logger.info(
        "Gate 3 OK: 'no unlock scheduled' and 'data unavailable' stay distinct"
    )

    summary = format_dna_summary("NEW/USDT", EMPTY_DNA)
    _assert_clean(summary, "Master summary")
    logger.info("Gate 3 OK: Master summary clean on all-NULL DNA")

    # format_dna_value itself.
    _require(format_dna_value(None, "fallback") == "fallback", "None -> fallback")
    _require(
        format_dna_value(float("nan"), "fallback") == "fallback", "NaN -> fallback"
    )
    _require(format_dna_value(1.23456, decimals=2) == "1.23", "rounding to 2dp")
    _require(format_dna_value(5.0, decimals=1, suffix="%") == "5.0%", "percent suffix")
    _require(format_dna_value("HIGH", "fallback") == "HIGH", "labels pass through")
    _require(format_dna_value(10, decimals=0) == "10", "integers render cleanly")
    logger.info("Gate 3 OK: format_dna_value handles None/NaN/labels/rounding")


async def gate_btc_compatibility() -> None:
    """Real Bitcoin DNA must render sensibly and flag the identity."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_inference", Path(__file__).with_name("run_inference.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_inference"] = module
    spec.loader.exec_module(module)

    analysis_date, _ = await module.load_latest_features()
    dna = await module.load_coin_dna(BASE_SYMBOL, analysis_date)

    _require(dna.get("symbol") == BASE_SYMBOL, "DNA must carry the symbol")
    _require(
        dna.get("btc_beta_30d") is not None
        and abs(float(dna["btc_beta_30d"]) - 1.0) < 1e-6,
        f"BTC beta against itself must be 1.0, got {dna.get('btc_beta_30d')}",
    )
    _require(
        dna.get("market_cap_tier") == "LARGE",
        f"BTC must be LARGE cap, got {dna.get('market_cap_tier')}",
    )

    for agent_cls in ALL_SUB_AGENTS:
        _assert_clean(
            agent_cls.build_dna_context(dna), f"{agent_cls.agent_name} on BTC DNA"
        )

    technical = TechnicalAgent.build_dna_context(dna)
    _require("1.00" in technical, f"BTC beta should render as 1.00: {technical}")
    _require(
        "identitas matematis" in technical,
        "BTC's beta/corr of 1.00 must be flagged as an identity so the model "
        "does not report it as an analytical finding",
    )
    summary = format_dna_summary(BASE_SYMBOL, dna)
    _require(
        "acuan pasar" in summary,
        f"the Master summary should flag BTC as the reference: {summary}",
    )
    logger.info("Gate 4 OK: BTC DNA renders cleanly — %s", summary[:110] + "...")


async def gate_backward_compatibility() -> None:
    """No coin_dna -> prompts identical to the pre-Part-5 version."""
    for agent_cls in ALL_SUB_AGENTS:
        _require(
            agent_cls.build_dna_context(None) == "",
            f"{agent_cls.agent_name} must produce no DNA block when given None",
        )
        _require(
            agent_cls.build_dna_context({}) == "",
            f"{agent_cls.agent_name} must produce no DNA block for empty DNA",
        )
    logger.info("Gate 5 OK: no DNA -> no DNA block, for all 5 pillars")

    captured: dict = {}

    async def fake_ask(self, prompt, response_model):
        captured.setdefault(self.agent_name, []).append(prompt)
        if response_model is MasterDecision:
            return MasterDecision(
                action="HOLD",
                confidence_score=50,
                executive_summary=(
                    "Ringkasan stub untuk uji wiring Part 5; tidak memanggil "
                    "model bahasa apa pun."
                ),
                risk_warning="Peringatan risiko stub untuk uji wiring Part 5.",
            )
        return SubAgentResponse(
            agent_name=self.agent_name,
            signal=Signal.NEUTRAL,
            score=0,
            reasoning="Respons stub untuk uji injeksi konteks Coin DNA.",
        )

    from src.agents.base_agent import BaseAgent

    original = BaseAgent.ask
    BaseAgent.ask = fake_ask
    try:
        features = {"date": "2026-01-01", "btc_close": 65000.0, "rsi_14": 50.0}

        agent = TechnicalAgent()
        await agent.analyze(features)
        without = captured["technical"][-1]
        _require(
            "KARAKTERISTIK KOIN" not in without,
            "analyze() without coin_dna must not add a DNA block",
        )

        await agent.analyze(features, SOL_DNA)
        with_dna = captured["technical"][-1]
        _require(
            "KARAKTERISTIK KOIN" in with_dna,
            "analyze() with coin_dna must add the DNA block",
        )
        _require(
            without in with_dna.replace(
                TechnicalAgent.build_dna_context(SOL_DNA) + "\n\n", ""
            )
            or len(with_dna) > len(without),
            "the DNA block should ADD to the prompt, not replace its content",
        )
        logger.info(
            "Gate 5 OK: analyze() works with and without DNA (%d -> %d chars)",
            len(without), len(with_dna),
        )

        # Master Agent accepts the summary and puts it in the prompt.
        master = MasterAgent()
        panel = [
            SubAgentResponse(
                agent_name=name, signal=Signal.NEUTRAL, score=0,
                reasoning="Panel stub untuk uji wiring Master Agent.",
            )
            for name in ("technical", "macro", "onchain", "sentiment", "liquidity_risk")
        ]
        summary = format_dna_summary("SOL/USDT", SOL_DNA)
        await master.decide("2026-01-01", panel, coin_dna_summary=summary)
        master_prompt = captured["master"][-1]
        _require(
            summary in master_prompt,
            "the DNA summary did not reach the Master Agent prompt",
        )
        _require(
            "COIN DNA" in master_prompt,
            "the DNA block is missing its prompt heading",
        )

        captured["master"].clear()
        await master.decide("2026-01-01", panel)
        _require(
            "COIN DNA" not in captured["master"][-1],
            "omitting coin_dna_summary must leave the Master prompt unchanged",
        )
        logger.info(
            "Gate 5 OK: Master Agent takes coin_dna_summary; omitting it "
            "leaves the prompt clean"
        )
    finally:
        BaseAgent.ask = original


async def gate_live_wiring() -> None:
    """load_coin_dna must return real, populated values."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_inference", Path(__file__).with_name("run_inference.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_inference"] = module
    spec.loader.exec_module(module)

    analysis_date, _ = await module.load_latest_features()
    dna = await module.load_coin_dna(BASE_SYMBOL, analysis_date)
    populated = [
        key for key in module.DNA_FEATURE_COLUMNS if dna.get(key) is not None
    ]
    _require(
        len(populated) >= 8,
        f"only {len(populated)}/{len(module.DNA_FEATURE_COLUMNS)} DNA fields "
        f"populated for BTC — expected most of them: {populated}",
    )
    _require("sector" in dna, "DNA must carry the registry sector")

    # An unknown symbol must degrade, not raise.
    #
    # Asserted as "carries NO feature column" rather than the old
    # `== {"symbol": ...}` (updated 2026-08-09 for C2b). Exact-dict equality
    # conflated two different things: leaking a FEATURE for a symbol that has
    # none (a real bug) and carrying PROVENANCE about the absence (correct,
    # and the whole point of `has_tokenomics_data`). NOPE/USDT genuinely has
    # zero tokenomics rows, so False is the true answer, not a leak.
    missing = await module.load_coin_dna("NOPE/USDT", analysis_date)
    leaked = [c for c in module.DNA_FEATURE_COLUMNS if c in missing]
    _require(
        missing.get("symbol") == "NOPE/USDT" and not leaked,
        f"an unregistered symbol must carry no feature column, got {missing} "
        f"(leaked: {leaked})",
    )
    for agent_cls in ALL_SUB_AGENTS:
        _assert_clean(
            agent_cls.build_dna_context(missing),
            f"{agent_cls.agent_name} on unknown symbol",
        )
    logger.info(
        "Gate 6 OK: %d/%d DNA fields live for BTC; unknown symbol degrades "
        "cleanly", len(populated), len(module.DNA_FEATURE_COLUMNS),
    )


# ---------------------------------------------------------------------------
async def main() -> int:
    started = datetime.now(tz=UTC)
    try:
        logger.info("--- Prompt inspection ---")
        gate_prompt_inspection()
        logger.info("--- Per-pillar routing ---")
        gate_routing()
        logger.info("--- NULL handling ---")
        gate_null_handling()
        logger.info("--- BTC compatibility ---")
        await gate_btc_compatibility()
        logger.info("--- Backward compatibility ---")
        await gate_backward_compatibility()
        logger.info("--- Live wiring ---")
        await gate_live_wiring()
    except AssertionError as exc:
        logger.error("PHASE 8 PART 5 TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("PHASE 8 PART 5 TEST FAILED with an unexpected error")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "PHASE 8 PART 5 TEST PASSED (6 gates, %.1fs) — Coin DNA reaches the "
        "agents", elapsed,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main()))

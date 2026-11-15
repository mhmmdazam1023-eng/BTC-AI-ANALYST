"""Bitcoin AI Analyst — full inference run (Phase 4 + Phase 5 persistence).

Pipeline:
1. Load the LATEST row from ``daily_features`` (one simple query).
2. Run the 5 sub-agents CONCURRENTLY (``asyncio.gather``), each isolated:
   one failing pillar never kills the run (>=3 pillars required).
3. Feed the validated panel to the Master Agent (CIO).
4. **Persist the decision** into the ``decisions`` hypertable (upsert on the
   analysis day — re-runs update, never duplicate).
5. Print a clean, colour-coded terminal report.

``execute_inference()`` is importable by ``scripts/run_daily.py``.

Usage (from the project root)::

    python scripts/run_inference.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

from src.agents import (  # noqa: E402
    ALL_SUB_AGENTS,
    Action,
    DynamicWeightingEngine,
    MasterAgent,
    MasterDecision,
    PanelFusion,
    Signal,
    SubAgent,
    SubAgentResponse,
)
from src.agents.consolidated_analyst import ConsolidatedAnalyst  # noqa: E402
from src.agents.usage_tracker import usage_context  # noqa: E402
from src.agents.decision_composer import compose_decision  # noqa: E402
from src.agents.utils import format_dna_summary, format_intraday_context  # noqa: E402
from src.agents.weighting_engine import TRANSITION_CONFIDENCE_FACTOR  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import (  # noqa: E402
    BASE_SYMBOL,
    DAILY_FEATURE_CATEGORICAL_COLUMNS,
    DAILY_FEATURE_COLUMNS,
    CoinRegistry,
    DailyFeature,
    Decision,
    Forecast,
    IntradayForecast,
    TokenomicsEvent,
)
from src.features import (  # noqa: E402
    RegimeDetector,
    RegimeSnapshot,
    SectorSnapshot,
    SectorTracker,
)
from src.features.intraday_engine import IntradayFeatureEngine  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.timeframes import BAR_HOURS, bar_label  # noqa: E402

logger = logging.getLogger("scripts.run_inference")

#: Minimum reporting pillars for a meaningful Master decision.
MIN_PILLARS = 3


class InferenceError(RuntimeError):
    """Raised when a decision cannot be produced (no data / too few pillars)."""


@dataclass
class InferenceOutcome:
    """Everything one inference run produced (for run_daily reporting)."""

    symbol: str
    analysis_date: str
    decision: MasterDecision
    responses: list[SubAgentResponse]
    pillars_reporting: int
    #: Phase 8 Part 2 — None when regime detection was unavailable, in
    #: which case the panel was fused with equal weights.
    regime: RegimeSnapshot | None = None
    fusion: PanelFusion | None = None
    #: Phase 8 Part 3 — None when sector rotation was unavailable.
    sectors: SectorSnapshot | None = None

# --- ANSI palette (os.system("") enables VT processing on Windows) ---
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, RED, YELLOW, CYAN, MAGENTA = (
    "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[95m",
)

_SIGNAL_STYLE = {
    Signal.BULLISH: (GREEN, "▲"),
    Signal.BEARISH: (RED, "▼"),
    Signal.NEUTRAL: (YELLOW, "•"),
}
_ACTION_STYLE = {
    Action.BUY: (GREEN, "▲ BUY"),
    Action.SELL: (RED, "▼ SELL"),
    Action.HOLD: (YELLOW, "• HOLD"),
}
_WIDTH = 78


def _rule(char: str = "═") -> str:
    return char * _WIDTH


def _wrap(text: str, indent: int = 4) -> str:
    return textwrap.fill(
        " ".join(text.split()),
        width=_WIDTH - indent,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
    )


def _score_bar(score: int) -> str:
    """Render -100..+100 as a 21-slot bar centred on 0."""
    slots = 21
    center = slots // 2
    position = center + round(score / 100 * center)
    position = max(0, min(slots - 1, position))
    bar = "".join("█" if i == position else "─" for i in range(slots))
    colour = GREEN if score > 15 else RED if score < -15 else YELLOW
    return f"{colour}[{bar}]{RESET} {score:+d}"


# ---------------------------------------------------------------------------
async def load_latest_features(symbol: str = BASE_SYMBOL) -> tuple[str, dict]:
    """Fetch the newest ``daily_features`` row for ``symbol`` as (date, dict).

    The ``symbol`` filter is mandatory since Phase 8: ``daily_features``
    holds one row per (day, asset), so an unfiltered
    ``ORDER BY timestamp DESC LIMIT 1`` would hand the caller an arbitrary
    asset's feature vector. Defaults to Bitcoin so every pre-Part-6 caller
    keeps its exact behaviour.
    """
    async with session_scope() as session:
        row = (
            await session.execute(
                select(DailyFeature)
                .where(DailyFeature.symbol == symbol)
                .order_by(DailyFeature.timestamp.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    if row is None:
        raise InferenceError(
            f"daily_features has no {symbol} rows — run "
            "scripts/run_collectors.py first"
        )

    features: dict = {"date": row.timestamp.date().isoformat()}
    for column in DAILY_FEATURE_COLUMNS:
        value = getattr(row, column)
        if value is None:
            features[column] = None
        elif column in DAILY_FEATURE_CATEGORICAL_COLUMNS:
            # Phase 8 Part 4 added the first LABEL features (vol_regime,
            # market_cap_tier). float() on them would kill the daily run.
            features[column] = getattr(value, "value", str(value))
        else:
            features[column] = float(f"{float(value):.6g}")

    age_days = (datetime.now(tz=UTC) - row.timestamp).days
    if age_days > 2:
        logger.warning(
            "Latest features are %d days old (%s) — consider running the "
            "collectors first.", age_days, features["date"],
        )
    return features["date"], features


async def load_forecast_context(analysis_date: str) -> dict | None:
    """Latest Monte Carlo forecasts at/before the analysis date (Phase 6).

    Returns ``{"7d": {p5,p50,p95,prob_up_pct}, "30d": ...}`` or ``None``
    when no forecasts exist yet (inference then proceeds without them).
    """
    day = datetime.fromisoformat(analysis_date).replace(tzinfo=UTC)
    async with session_scope() as session:
        latest_date = (
            await session.execute(
                select(func.max(Forecast.forecast_date)).where(
                    Forecast.forecast_date <= day
                )
            )
        ).scalar_one_or_none()
        if latest_date is None:
            return None
        rows = (
            (
                await session.execute(
                    select(Forecast)
                    .where(Forecast.forecast_date == latest_date)
                    .order_by(Forecast.horizon_days)
                )
            )
            .scalars()
            .all()
        )
    context = {
        f"{row.horizon_days}d": {
            "p5": float(row.predicted_price_p5),
            "p50": float(row.predicted_price_p50),
            "p95": float(row.predicted_price_p95),
            "prob_up_pct": round(row.prob_price_up * 100, 1),
        }
        for row in rows
    }
    context["forecast_date"] = latest_date.date().isoformat()
    context["method"] = "Monte Carlo block-bootstrap (20k paths, 10-day blocks)"
    logger.info(
        "Statistical forecast context loaded (%s): 30d p50=%.0f, P(up)=%s%%",
        context["forecast_date"],
        context.get("30d", {}).get("p50", float("nan")),
        context.get("30d", {}).get("prob_up_pct", "n/a"),
    )
    return context


#: How many CLOSED 4h bars go into the BTC prompt. Six = one UTC day's worth
#: of bars, which is what the user asked for; the window rolls, so it usually
#: spans two calendar dates (hence the dated labels).
INTRADAY_CONTEXT_BARS = 6


def _as_float(value: object) -> float | None:
    """Best-effort float for prompt display; None when not numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def load_intraday_forecast_context() -> dict | None:
    """Latest 4H forecasts, keyed by horizon in HOURS (Phase 8 Part 8).

    Keys are ``"4h"`` / ``"12h"`` / ``"24h"`` — hours, not days, so a reader
    (human or model) cannot confuse them with the daily forecast block whose
    keys are ``"7d"`` / ``"30d"``. Returns ``None`` when the intraday engine
    has not run yet, and inference simply proceeds without the block.
    """
    async with session_scope() as session:
        latest_base = (
            await session.execute(
                select(func.max(IntradayForecast.base_at)).where(
                    IntradayForecast.symbol == BASE_SYMBOL
                )
            )
        ).scalar_one_or_none()
        if latest_base is None:
            return None
        rows = (
            (
                await session.execute(
                    select(IntradayForecast)
                    .where(
                        IntradayForecast.symbol == BASE_SYMBOL,
                        IntradayForecast.base_at == latest_base,
                    )
                    .order_by(IntradayForecast.horizon_bars)
                )
            )
            .scalars()
            .all()
        )
    if not rows:
        return None

    context: dict = {
        f"{row.horizon_bars * BAR_HOURS}h": {
            "target_bar": bar_label(row.target_at),
            "p5": float(row.predicted_price_p5),
            "p50": float(row.predicted_price_p50),
            "p95": float(row.predicted_price_p95),
            "prob_up_pct": round(row.prob_price_up * 100, 1),
        }
        for row in rows
    }
    context["anchor_bar"] = bar_label(latest_base)
    context["anchor_close"] = float(rows[0].base_price)
    context["method"] = (
        f"Monte Carlo block-bootstrap over 4h bars "
        f"({rows[0].pool_bars or 'n/a'}-bar pool, 12-bar blocks)"
    )
    logger.info(
        "Intraday 4H forecast context loaded (anchor %s): 24h p50=%s, P(up)=%s%%",
        context["anchor_bar"],
        context.get("24h", {}).get("p50", "n/a"),
        context.get("24h", {}).get("prob_up_pct", "n/a"),
    )
    return context


async def run_sub_agent(
    agent: SubAgent,
    features: dict,
    coin_dna: dict | None = None,
    errors: list[str] | None = None,
) -> SubAgentResponse | None:
    """Run one pillar fully isolated; None on failure (logged).

    ``errors`` collects the failure text. Isolation is right — one dead
    pillar must not kill the run — but swallowing the REASON hides
    account-level faults like an exhausted daily model quota, which the
    multi-asset loop needs to see so it can stop instead of repeating the
    same failure for every remaining asset.
    """
    try:
        response = await agent.analyze(features, coin_dna)
        logger.info(
            "Pillar '%s' reported: %s (score %+d)",
            agent.agent_name, response.signal.value, response.score,
        )
        return response
    except Exception as exc:  # noqa: BLE001 - isolation: one pillar must not kill the run
        if errors is not None:
            errors.append(f"{type(exc).__name__}: {exc}")
        logger.exception("Pillar '%s' FAILED — continuing without it", agent.agent_name)
        return None
    finally:
        await agent.close()


# ---------------------------------------------------------------------------
def print_report(
    analysis_date: str,
    features: dict,
    responses: list[SubAgentResponse],
    decision: MasterDecision,
    model_name: str,
    fusion: PanelFusion | None = None,
    transition_note: str = "",
    sectors: SectorSnapshot | None = None,
) -> None:
    print()
    print(f"{BOLD}{CYAN}{_rule()}{RESET}")
    title = "BITCOIN AI ANALYST — DAILY DECISION REPORT"
    print(f"{BOLD}{CYAN}{title.center(_WIDTH)}{RESET}")
    print(f"{DIM}{f'{analysis_date}  ·  BTC ${features['btc_close']:,.0f}  ·  model: {model_name}'.center(_WIDTH)}{RESET}")
    print(f"{BOLD}{CYAN}{_rule()}{RESET}")

    if fusion is not None and fusion.regime is not None:
        print(f"\n{BOLD}REZIM PASAR (deteksi rule-based){RESET}")
        print(_rule("─"))
        confidence = (
            "" if fusion.regime_confidence is None
            else f" (confidence {fusion.regime_confidence}/100)"
        )
        tag = "bobot dinamis" if fusion.is_dynamic else "bobot setara (default)"
        print(f"  {BOLD}{CYAN}{fusion.regime}{RESET}{confidence} — {tag}")
        if transition_note:
            print(f"{YELLOW}  {transition_note}{RESET}")
        print(f"{DIM}  Bobot pilar: {fusion.weights_line()}{RESET}")

    if sectors is not None and sectors.rankings:
        print(f"\n{BOLD}ROTASI SEKTOR (relative strength 7 hari vs BTC){RESET}")
        print(_rule("─"))
        for reading in sectors.rankings:
            seven = "n/a" if reading.avg_rs_7d is None else f"{reading.avg_rs_7d:+6.2f}%"
            thirty = "n/a" if reading.avg_rs_30d is None else f"{reading.avg_rs_30d:+6.2f}%"
            colour = (
                GREEN if reading.status.value == "IN_FLOW"
                else YELLOW if reading.status.value == "ACCUMULATING"
                else RED if reading.status.value == "OUT_OF_FLOW"
                else DIM
            )
            thin = f"{YELLOW} [tipis]{RESET}" if reading.is_thin else ""
            print(
                f"  {colour}#{reading.ranking} {reading.sector:<14s}"
                f"7d {seven:>8s}   30d {thirty:>8s}   "
                f"{reading.status.value:<12s}{RESET}"
                f"{DIM} n={reading.coin_count}/{reading.coin_count_30d}{RESET}{thin}"
            )
        print(f"{DIM}{_wrap(sectors.rotation_signal, indent=2)}{RESET}")

    print(f"\n{BOLD}PANEL SUB-AGENT ({len(responses)}/5 pilar melapor){RESET}")
    print(_rule("─"))
    for response in responses:
        colour, marker = _SIGNAL_STYLE[response.signal]
        weight = (fusion.effective_weights.get(response.agent_name) if fusion else None)
        weight_tag = "" if weight is None else f"{DIM} w={weight * 100:.0f}%{RESET}"
        header = (
            f"  {colour}{marker} {response.agent_name.upper():<16s}"
            f"{response.signal.value:<8s}{RESET} {_score_bar(response.score)}{weight_tag}"
        )
        print(header)
        print(f"{DIM}{_wrap(response.reasoning)}{RESET}\n")

    if fusion is not None:
        print(
            f"  {BOLD}Skor panel terbobot: {fusion.weighted_score:+.1f}{RESET}"
            f"{DIM}  (-100 sangat bearish .. +100 sangat bullish){RESET}"
        )

    colour, label = _ACTION_STYLE[decision.action]
    print(f"{BOLD}{colour}{_rule()}{RESET}")
    print(f"{BOLD}{colour}{f'KEPUTUSAN FINAL: {label}   (confidence {decision.confidence_score}/100)'.center(_WIDTH + len(colour) + len(RESET) - len(colour) - len(RESET))}{RESET}")
    print(f"{BOLD}{colour}{_rule()}{RESET}")

    print(f"\n{BOLD}RINGKASAN EKSEKUTIF{RESET}")
    print(_wrap(decision.executive_summary, indent=2))
    print(f"\n{BOLD}{MAGENTA}PERINGATAN RISIKO{RESET}")
    print(f"{MAGENTA}{_wrap(decision.risk_warning, indent=2)}{RESET}")
    print(f"\n{DIM}{_wrap('Output ini adalah dukungan keputusan berbasis data, bukan nasihat keuangan.', indent=2)}{RESET}")
    print(f"{BOLD}{CYAN}{_rule()}{RESET}\n")


# ---------------------------------------------------------------------------
# Phase 5: decision persistence (idempotent — same day updates, no duplicates)
# ---------------------------------------------------------------------------
async def save_decision(
    analysis_date: str,
    decision: MasterDecision,
    responses: list[SubAgentResponse],
    symbol: str = BASE_SYMBOL,
) -> None:
    """Upsert the decision into the ``decisions`` hypertable.

    Conflict target is the composite PK ``(timestamp, symbol)``: without
    ``symbol`` in it, a 21-coin run would upsert every asset onto one row
    and keep only whichever finished last.
    """
    day = datetime.fromisoformat(analysis_date).replace(tzinfo=UTC)
    payload = {
        "timestamp": day,
        "symbol": symbol,
        "action": decision.action.value,
        "confidence_score": decision.confidence_score,
        "executive_summary": decision.executive_summary,
        "risk_warning": decision.risk_warning,
        "sub_agents_json": [r.model_dump(mode="json") for r in responses],
    }
    statement = pg_insert(Decision.__table__).values(payload)
    update_set = {
        key: statement.excluded[key]
        for key in payload
        if key not in ("timestamp", "symbol")
    }
    update_set["ingested_at"] = func.now()
    statement = statement.on_conflict_do_update(
        index_elements=["timestamp", "symbol"], set_=update_set
    )
    async with session_scope() as session:
        await session.execute(statement)
    logger.info(
        "Decision persisted to 'decisions' (%s %s: %s, confidence %d) — "
        "re-runs on the same day update this row.",
        symbol, analysis_date, decision.action.value, decision.confidence_score,
    )


# ---------------------------------------------------------------------------
async def resolve_regime() -> RegimeSnapshot | None:
    """Detect today's market regime; None when unavailable (never raises).

    Regime detection is an ENHANCEMENT, not a dependency: if it fails, the
    decision must still be produced with equal weights exactly as it was
    before Phase 8 Part 2. Hence the broad catch.
    """
    try:
        return await RegimeDetector().run()
    except Exception as exc:  # noqa: BLE001 - must never block a decision
        logger.warning(
            "Regime detection unavailable (%s: %s) — falling back to equal "
            "weights", type(exc).__name__, exc,
        )
        return None


#: Coin DNA fields pulled from the latest daily_features row (Part 5).
DNA_FEATURE_COLUMNS: tuple[str, ...] = (
    "vol_regime",
    "vol_percentile_30d",
    "btc_beta_30d",
    "btc_corr_30d",
    "trend_strength",
    "momentum_score",
    "liquidity_score",
    "market_cap_tier",
    "supply_shock_score_30d",
    "days_to_next_unlock",
    "supply_utilization",
    "inflation_rate_7d",
)


async def load_coin_dna(symbol: str, analysis_date: str) -> dict:
    """Coin DNA for one symbol on one day, plus its registry sector.

    Values are returned RAW (numbers or None) — formatting for the prompt,
    including what a NULL means per field, is
    :mod:`src.agents.utils`' job. Never raises: a missing row yields a dict
    with only the symbol, and every downstream formatter degrades to its
    per-field fallback text.
    """
    day = datetime.fromisoformat(analysis_date).replace(tzinfo=UTC)
    dna: dict = {"symbol": symbol}
    try:
        async with session_scope() as session:
            row = (
                await session.execute(
                    select(DailyFeature).where(
                        DailyFeature.symbol == symbol,
                        DailyFeature.timestamp == day,
                    )
                )
            ).scalar_one_or_none()
            sector = (
                await session.execute(
                    select(CoinRegistry.sector).where(CoinRegistry.symbol == symbol)
                )
            ).scalar_one_or_none()
            # C2b (2026-08-09): does this symbol have ANY tokenomics row?
            #
            # A NULL `days_to_next_unlock` is ambiguous on its own and the
            # ambiguity was being resolved the flattering way. Part 5 renders
            # it as "tidak ada unlock terjadwal" — stated as fact, and BULLISH
            # for supply pressure. True when a schedule exists and holds
            # nothing upcoming; false when nobody ever loaded one, which was
            # the case for 20 of 21 assets. Same defect class as the
            # BTC-proxy on-chain columns: absence narrated as information.
            #
            # This flag is what lets the formatter tell the two apart. It is
            # NOT in DNA_FEATURE_COLUMNS on purpose — it is provenance about
            # a feature, not a feature, and adding it there would drag it
            # into the float() conversion and the populated-field count.
            has_tokenomics = (
                await session.execute(
                    select(func.count())
                    .select_from(TokenomicsEvent)
                    .where(TokenomicsEvent.symbol == symbol)
                )
            ).scalar_one() > 0
    except Exception as exc:  # noqa: BLE001 - DNA must never block a decision
        logger.warning(
            "Coin DNA unavailable for %s (%s: %s) — agents will run without it",
            symbol, type(exc).__name__, exc,
        )
        return dna

    if sector is not None:
        dna["sector"] = getattr(sector, "value", str(sector))
    dna["has_tokenomics_data"] = has_tokenomics
    if row is None:
        logger.warning(
            "No daily_features row for %s on %s — Coin DNA will be empty",
            symbol, analysis_date,
        )
        return dna

    for column in DNA_FEATURE_COLUMNS:
        value = getattr(row, column, None)
        if value is None:
            dna[column] = None
        elif column in DAILY_FEATURE_CATEGORICAL_COLUMNS:
            dna[column] = getattr(value, "value", str(value))
        elif column == "days_to_next_unlock":
            dna[column] = int(value)
        else:
            dna[column] = float(value)

    populated = sum(1 for key in DNA_FEATURE_COLUMNS if dna.get(key) is not None)
    logger.info(
        "Coin DNA for %s: %d/%d fields populated (regime=%s, tier=%s, beta=%s)",
        symbol, populated, len(DNA_FEATURE_COLUMNS),
        dna.get("vol_regime"), dna.get("market_cap_tier"), dna.get("btc_beta_30d"),
    )
    return dna


async def resolve_sectors() -> SectorSnapshot | None:
    """Compute today's sector rotation; None when unavailable (never raises).

    Like regime detection, sector rotation ENRICHES the decision but must
    never be able to prevent one. An empty registry, a failed collector or
    a bad join here costs the Master Agent some context, nothing more.
    """
    try:
        return await SectorTracker().run()
    except Exception as exc:  # noqa: BLE001 - must never block a decision
        logger.warning(
            "Sector rotation unavailable (%s: %s) — proceeding without "
            "cross-asset context", type(exc).__name__, exc,
        )
        return None


async def execute_inference(
    *,
    symbol: str = BASE_SYMBOL,
    print_output: bool = True,
    regime_snapshot: RegimeSnapshot | None = None,
    sector_snapshot: SectorSnapshot | None = None,
    detect_regime: bool = True,
    track_sectors: bool = True,
    use_consolidated: bool = True,
) -> InferenceOutcome:
    """Full inference for ONE asset: pillars -> Master -> persist.

    Importable entry point (used by run_daily.py). Raises
    :class:`InferenceError` when a decision cannot be produced.

    ``symbol`` defaults to Bitcoin so every pre-Part-6 caller is unchanged.
    ``regime_snapshot`` / ``sector_snapshot`` let the caller compute those
    ONCE and hand them in — they are market-wide, identical for every
    asset, and recomputing them per coin would be pure waste.
    ``detect_regime=False`` / ``track_sectors=False`` force the pre-Phase-8
    behaviour.
    """
    settings = get_settings()
    if regime_snapshot is None and detect_regime:
        regime_snapshot = await resolve_regime()
    if sector_snapshot is None and track_sectors:
        sector_snapshot = await resolve_sectors()
    analysis_date, features = await load_latest_features(symbol)
    logger.info(
        "Running inference for %s on %s (5 pillars, concurrent)",
        symbol, analysis_date,
    )

    # Phase 8 Part 5: each pillar receives the slice of Coin DNA it can act
    # on, so an altcoin is no longer analysed as if it were Bitcoin.
    coin_dna = await load_coin_dna(symbol, analysis_date)

    # Forecasts are BITCOIN-ONLY (see below). Resolved before the panel so
    # the consolidated agent can receive it too — in consolidated mode there
    # is no Master call to carry it otherwise.
    forecast_context: dict | None = None
    if symbol == BASE_SYMBOL:
        try:
            forecast_context = await load_forecast_context(analysis_date)
        except Exception as exc:  # noqa: BLE001 - forecast context is optional
            logger.warning("Forecast context unavailable (ignored): %s", exc)

    # Phase 8 Part 8: BTC intraday (4H) context. BITCOIN ONLY, for the same
    # reason forecasts are: there is no per-altcoin 4H layer, and handing an
    # altcoin Bitcoin's 4-hour tape would relabel BTC data as its own.
    # Isolated: any failure here leaves the daily decision path untouched.
    intraday_bars: list = []
    intraday_context: str | None = None
    intraday_forecast_context: dict | None = None
    if symbol == BASE_SYMBOL:
        try:
            intraday_bars = await IntradayFeatureEngine().recent_closed_bars(
                count=INTRADAY_CONTEXT_BARS
            )
            if intraday_bars:
                intraday_context = format_intraday_context(
                    intraday_bars,
                    live_close=_as_float(features.get("asset_close")),
                )
            intraday_forecast_context = await load_intraday_forecast_context()
        except Exception as exc:  # noqa: BLE001 - intraday context is optional
            logger.warning(
                "Intraday 4H context unavailable (ignored): %s: %s",
                type(exc).__name__, exc,
            )
            intraday_bars = []
            intraday_context = None
            intraday_forecast_context = None

    # --- Panel: five specialists, or one consolidated call (Part 7) ---
    consolidated_summary = ""
    if use_consolidated:
        analyst = ConsolidatedAnalyst()
        try:
            analysis = await analyst.analyze(
                features,
                coin_dna,
                symbol=symbol,
                regime_context=(
                    f"{regime_snapshot.regime.value} "
                    f"(confidence {regime_snapshot.confidence}/100)"
                    if regime_snapshot
                    else None
                ),
                sector_context=(
                    sector_snapshot.to_prompt_context() if sector_snapshot else None
                ),
                forecast_context=forecast_context,
                intraday_context=intraday_context,
                intraday_forecast_context=intraday_forecast_context,
            )
        finally:
            await analyst.close()
        responses = ConsolidatedAnalyst.to_panel(analysis)
        consolidated_summary = analysis.reasoning_summary
        pillar_errors = []
    else:
        agents = [agent_cls() for agent_cls in ALL_SUB_AGENTS]
        pillar_errors = []
        # Each gathered task copies the context at creation, so all five
        # pillar calls are attributed to this asset without widening
        # SubAgent.analyze() — see usage_tracker._usage_symbol.
        with usage_context(symbol):
            raw_results = await asyncio.gather(
                *(
                    run_sub_agent(agent, features, coin_dna, pillar_errors)
                    for agent in agents
                )
            )
        responses = [response for response in raw_results if response is not None]

    if len(responses) < MIN_PILLARS:
        detail = (
            f"only {len(responses)}/{len(agents)} pillars reported "
            f"(minimum {MIN_PILLARS})"
        )
        if pillar_errors:
            # Carry the underlying reason upward. Without it the caller sees
            # only "too few pillars" and cannot tell a transient hiccup from
            # an exhausted daily model quota — which decides whether running
            # the next 20 assets is worth attempting at all.
            detail += f" — pillar error: {pillar_errors[0]}"
        raise InferenceError(detail + " — decision aborted")

    market_context: dict = {
        "symbol": symbol,
        "asset_close": features.get("asset_close"),
        "btc_close": features["btc_close"],
        "realized_vol_30d": features["realized_vol_30d"],
        "fear_greed_value": features["fear_greed_value"],
    }
    # Forecasts are BITCOIN-ONLY (the `forecasts` table has no symbol column
    # and ForecastEngine is BTC-hardcoded). Attaching them to an altcoin's
    # prompt would present Bitcoin's Monte Carlo percentiles as that coin's
    # own — a wrong number wearing the right label. Altcoin forecasting is
    # deliberately out of scope for Part 6.
    if forecast_context:
        market_context["statistical_forecast_monte_carlo"] = forecast_context
    elif symbol != BASE_SYMBOL:
        logger.debug(
            "Skipping forecast context for %s — forecasting is BTC-only", symbol
        )

    # --- Phase 8 Part 2: regime -> weights -> deterministic panel fusion ---
    # Transition-aware: a regime seen for the first time today must clear a
    # higher confidence bar before it may skew the panel (see
    # DynamicWeightingEngine.evaluate).
    regime_label = regime_snapshot.regime.value if regime_snapshot else None
    regime_confidence = regime_snapshot.confidence if regime_snapshot else None
    weighting = DynamicWeightingEngine().evaluate(
        regime_label,
        regime_snapshot.previous_regime if regime_snapshot else None,
        regime_confidence,
    )
    fusion = MasterAgent.fuse(
        responses,
        weighting.weights,
        regime=regime_label,
        regime_confidence=regime_confidence,
    )
    if regime_snapshot is not None:
        market_context["market_regime"] = {
            "regime": regime_label,
            "confidence": regime_confidence,
            "as_of": regime_snapshot.as_of.date().isoformat(),
            "rule_version": regime_snapshot.rule_version,
            "weighting": weighting.reason,
        }
        if weighting.is_transition and regime_snapshot.previous_regime is not None:
            market_context["market_regime"]["changed_from"] = (
                regime_snapshot.previous_regime.value
            )

    sector_context = (
        sector_snapshot.to_prompt_context() if sector_snapshot else None
    )
    if use_consolidated:
        # No Master call: that would double the quota back to 2 calls per
        # asset. The action, confidence and narrative are computed in Python
        # from the panel the consolidated analyst produced.
        decision = compose_decision(
            symbol,
            analysis_date,
            responses,
            fusion,
            reasoning_summary=consolidated_summary,
            regime=fusion.regime,
            sector_signal=(
                sector_snapshot.rotation_signal if sector_snapshot else None
            ),
            # Part 8: BTC-only, and can ONLY lower confidence — never flip
            # the action. Empty for altcoins, which leaves this a no-op.
            intraday_bars=intraday_bars or None,
        ).decision
    else:
        master = MasterAgent()
        try:
            with usage_context(symbol):
                decision = await master.decide(
                    analysis_date, responses, market_context, weighting.weights,
                    fusion=fusion, sector_context=sector_context,
                    coin_dna_summary=format_dna_summary(symbol, coin_dna),
                )
        finally:
            await master.close()

    # Phase 5 rule: the decision MUST be persisted before it is reported.
    await save_decision(analysis_date, decision, responses, symbol)

    if print_output:
        transition_note = ""
        if weighting.is_transition and regime_snapshot is not None:
            previous = regime_snapshot.previous_regime
            transition_note = (
                f"Transisi rezim dari {previous.value if previous else '?'} "
                f"— confidence didiskon {int((1 - TRANSITION_CONFIDENCE_FACTOR) * 100)}%"
                + (
                    "; bobot setara dipakai sampai rezim bertahan."
                    if weighting.used_default
                    else "; tetap lolos ambang, bobot rezim dipakai."
                )
            )
        print_report(
            analysis_date, features, responses, decision, settings.qwen_model,
            fusion=fusion, transition_note=transition_note,
            sectors=sector_snapshot,
        )
    logger.info(
        "Inference complete: %s (confidence %d) | %s | weighting: %s",
        decision.action.value, decision.confidence_score, fusion.summary(),
        weighting.reason,
    )
    return InferenceOutcome(
        symbol=symbol,
        analysis_date=analysis_date,
        decision=decision,
        responses=responses,
        pillars_reporting=len(responses),
        regime=regime_snapshot,
        fusion=fusion,
        sectors=sector_snapshot,
    )


# ---------------------------------------------------------------------------
# Phase 8 Part 6: multi-asset orchestration
# ---------------------------------------------------------------------------
#: Pause between coins. Six model calls per asset (5 pillars + Master) times
#: 21 assets is ~126 requests; without a breather the free OpenRouter tier
#: starts refusing. Tunable via --delay.
DEFAULT_INTER_SYMBOL_DELAY = 2.0

#: Model calls consumed per asset, per mode (Phase 8 Part 7).
#: The free OpenRouter tier allows 50 requests/DAY, so this arithmetic is
#: the binding constraint on how many assets can be analysed at all.
CALLS_PER_ASSET_BY_MODE: dict[str, int] = {
    "consolidated": 1,  # one call scores all five pillars
    "panel": 6,         # 5 specialists + Master (the Phase 4 architecture)
    "hybrid": 1,        # altcoins consolidated; BTC costs 6 (added below)
}

#: Default mode. ``consolidated`` is the only one that fits 21 assets in a
#: 50-call day (21 calls, 29 to spare).
DEFAULT_INFERENCE_MODE = "consolidated"

#: Hard wall-clock ceiling for ONE asset, in seconds.
#:
#: Sized from measurement, not taste. On the 2026-07-31 21-asset run the mean
#: asset took 86s and the slowest healthy one 249s; 360s is ~4x the mean and
#: comfortably above the slowest legitimate case. Meanwhile the pathological
#: case measured 31 MINUTES (ONDO), because the retry budgets multiply: each
#: of ask()'s 3 validation rounds gets a fresh 4-attempt transport budget at
#: a 120s timeout.
#:
#: The trade-off is deliberate and worth stating: an asset that would have
#: succeeded at minute 20 now fails at minute 6. For a 21-asset scheduled job
#: that is the right side to err on — one sick asset must not be able to turn
#: a 28-minute run into an 8-hour one, and the loser can be re-run alone via
#: ``--symbols`` for one extra call.
PER_ASSET_TIMEOUT_SECONDS = 360.0

#: Backwards-compatible alias — the Phase 4 architecture's cost per asset.
CALLS_PER_ASSET = CALLS_PER_ASSET_BY_MODE["panel"]


def _uses_consolidated(mode: str, symbol: str) -> bool:
    """Whether ``symbol`` should be scored by the single consolidated call.

    ``hybrid`` keeps the full five-specialist panel plus the Master for
    Bitcoin — the asset with Monte Carlo forecasts to reconcile and the one
    the desk cares most about — and uses the cheap path for the altcoins.
    Cost: 6 + 20 = 26 calls/day instead of 21.
    """
    if mode == "panel":
        return False
    if mode == "hybrid":
        return symbol != BASE_SYMBOL
    return True


def estimate_calls(mode: str, symbol_count: int) -> int:
    """Model calls a run of ``symbol_count`` assets will consume."""
    if mode == "panel":
        return symbol_count * CALLS_PER_ASSET_BY_MODE["panel"]
    if mode == "hybrid" and symbol_count:
        return (symbol_count - 1) + CALLS_PER_ASSET_BY_MODE["panel"]
    return symbol_count

#: Substrings that identify a DAILY quota exhaustion rather than a
#: momentary throttle. Verified against OpenRouter's free tier
#: (2026-07-29): ``free-models-per-day``, limit 50/day.
#:
#: The distinction matters enormously. A per-minute throttle is worth
#: retrying — that is what the tenacity backoff is for. A per-DAY quota is
#: not: every remaining asset will fail identically, each burning ~3
#: minutes of backoff first. Twenty-one assets would spend an hour proving
#: the same thing. On detection the run stops and says so.
QUOTA_EXHAUSTED_MARKERS: tuple[str, ...] = (
    "free-models-per-day",
    "per-day",
    "daily limit",
    "quota exceeded",
)


def _is_daily_quota_exhausted(detail: str) -> bool:
    """True when the failure text indicates a DAILY cap, not a throttle."""
    lowered = detail.lower()
    return any(marker in lowered for marker in QUOTA_EXHAUSTED_MARKERS)


@dataclass
class MultiAssetOutcome:
    """Result of a full portfolio pass."""

    outcomes: list[InferenceOutcome]
    failures: dict[str, str]
    analysed: int = 0
    requested: int = 0
    #: Assets never attempted because the daily model quota ran out.
    skipped: list[str] = field(default_factory=list)
    quota_exhausted: bool = False

    @property
    def succeeded(self) -> int:
        return len(self.outcomes)

    def summary_line(self) -> str:
        actions = ", ".join(
            f"{outcome.symbol.split('/')[0]} {outcome.decision.action.value}"
            for outcome in self.outcomes
        )
        text = f"{self.succeeded}/{self.requested} assets analysed"
        if actions:
            text += f" — {actions}"
        if self.failures:
            text += f" | FAILED: {', '.join(sorted(self.failures))}"
        if self.skipped:
            text += (
                f" | SKIPPED {len(self.skipped)} (daily model quota exhausted)"
            )
        return text


async def load_active_symbols() -> list[str]:
    """Active registry symbols, Bitcoin first.

    BTC leads for the same reason it does in the collector: it is the
    reference every relative feature is built on, and if the run is cut
    short (rate limit, interruption) the asset that matters most is
    already done.
    """
    async with session_scope() as session:
        symbols = list(
            (
                await session.execute(
                    select(CoinRegistry.symbol)
                    .where(CoinRegistry.is_active.is_(True))
                    .order_by(CoinRegistry.symbol)
                )
            ).scalars().all()
        )
    if not symbols:
        logger.warning(
            "coins_registry has no active symbols — falling back to %s only",
            BASE_SYMBOL,
        )
        return [BASE_SYMBOL]
    ordered = [BASE_SYMBOL] + sorted(s for s in set(symbols) if s != BASE_SYMBOL)
    return ordered


async def execute_multi_asset_inference(
    *,
    symbols: list[str] | None = None,
    delay_seconds: float = DEFAULT_INTER_SYMBOL_DELAY,
    print_output: bool = True,
    regime_snapshot: RegimeSnapshot | None = None,
    sector_snapshot: SectorSnapshot | None = None,
    mode: str = "consolidated",
    asset_timeout: float = PER_ASSET_TIMEOUT_SECONDS,
) -> MultiAssetOutcome:
    """Run inference for every active asset, one at a time.

    Sequential on purpose. Each asset already fans out 5 concurrent pillar
    calls; running assets in parallel too would put ~100 requests in flight
    at once and trip the rate limit for all of them simultaneously.

    Per-asset failures are ISOLATED: a coin that fails is recorded and the
    run moves on, exactly like the collector stages. One altcoin's rate
    limit must never cost you the other twenty decisions.

    ``asset_timeout`` bounds each asset's wall clock — see
    :data:`PER_ASSET_TIMEOUT_SECONDS`. A timed-out asset is recorded as a
    failure like any other and can be re-run alone with ``--symbols``.

    Regime and sector context are market-wide, so they are resolved ONCE
    here and passed into every asset rather than recomputed 21 times.
    """
    requested = symbols if symbols is not None else await load_active_symbols()
    if regime_snapshot is None:
        regime_snapshot = await resolve_regime()
    if sector_snapshot is None:
        sector_snapshot = await resolve_sectors()

    result = MultiAssetOutcome(outcomes=[], failures={}, requested=len(requested))
    logger.info(
        "=== MULTI-ASSET INFERENCE: %d asset(s), mode=%s, %.1fs between "
        "each, ~%d model call(s) total (free tier: 50/day) ===",
        len(requested), mode, delay_seconds,
        estimate_calls(mode, len(requested)),
    )

    for position, symbol in enumerate(requested, start=1):
        logger.info("Processing %d/%d: %s ...", position, len(requested), symbol)
        try:
            # Hard wall-clock ceiling per asset. WITHOUT this a single asset
            # can consume half an hour: the retry budgets MULTIPLY, because
            # each of ask()'s 3 validation rounds gets a fresh 4-attempt
            # transport budget at a 120s timeout (12 requests = ~24 min in
            # theory, 31 min observed on ONDO 2026-07-31). Across 21 assets
            # that is an 8.5-hour worst case — unschedulable.
            #
            # This became reachable only once empty responses were made
            # retryable; before that they failed in ~21s. Retrying is the
            # right call, but it needs an outer bound, and bounding here
            # rather than by shrinking the inner budgets means it holds no
            # matter how those layers are tuned later.
            outcome = await asyncio.wait_for(
                execute_inference(
                    symbol=symbol,
                    print_output=print_output,
                    regime_snapshot=regime_snapshot,
                    sector_snapshot=sector_snapshot,
                    use_consolidated=_uses_consolidated(mode, symbol),
                ),
                timeout=asset_timeout,
            )
            result.outcomes.append(outcome)
            logger.info(
                "<<< %d/%d %s: %s (confidence %d)",
                position, len(requested), symbol,
                outcome.decision.action.value, outcome.decision.confidence_score,
            )
        except TimeoutError as exc:  # noqa: PERF203 - explicit, not incidental
            # asyncio.wait_for raises a bare TimeoutError with no message, so
            # say what actually happened and what to do about it. Handled
            # before the generic branch because that one would render it as
            # the useless string "TimeoutError: ".
            detail = (
                f"TimeoutError: exceeded the {asset_timeout:.0f}s per-asset "
                f"budget. The model kept retrying without producing a valid "
                f"response; re-run this symbol alone with "
                f"--symbols {symbol}"
            )
            result.failures[symbol] = detail
            result.analysed += 1
            logger.error(
                "<<< %d/%d %s TIMED OUT after %.0fs — moving on so one sick "
                "asset cannot stall the whole run (%s)",
                position, len(requested), symbol, asset_timeout, exc,
            )
            if position < len(requested):
                await asyncio.sleep(delay_seconds)
            continue
        except Exception as exc:  # noqa: BLE001 - per-asset isolation is the point
            detail = f"{type(exc).__name__}: {exc}"
            result.failures[symbol] = detail
            result.analysed += 1

            if _is_daily_quota_exhausted(detail):
                # Not a throttle — the day's model budget is gone. Every
                # remaining asset would fail identically after ~3 minutes of
                # pointless backoff each.
                result.quota_exhausted = True
                result.skipped = list(requested[position:])
                logger.error(
                    "<<< %d/%d %s FAILED: DAILY MODEL QUOTA EXHAUSTED. "
                    "Stopping — the remaining %d asset(s) would each burn "
                    "minutes of retry backoff to fail the same way. "
                    "Skipped: %s",
                    position, len(requested), symbol, len(result.skipped),
                    ", ".join(result.skipped) or "none",
                )
                break

            logger.exception(
                "<<< %d/%d %s FAILED — continuing with the remaining assets",
                position, len(requested), symbol,
            )
        else:
            result.analysed += 1

        # No point sleeping after the last asset.
        if delay_seconds > 0 and position < len(requested):
            await asyncio.sleep(delay_seconds)

    logger.info("=== MULTI-ASSET INFERENCE DONE: %s ===", result.summary_line())
    if result.failures:
        for symbol, detail in sorted(result.failures.items()):
            logger.error("    %s: %s", symbol, detail)
    if result.quota_exhausted:
        per_asset = CALLS_PER_ASSET_BY_MODE.get(mode, 1)
        budget = estimate_calls(mode, result.succeeded)
        logger.error(
            "    Daily model quota is gone (~%d call(s) used by the %d asset(s) "
            "that completed, mode=%s). At %d call(s) per asset a free "
            "OpenRouter key (50/day) covers about %d asset(s). Options: run a "
            "subset with --symbols, or switch --mode.",
            budget, result.succeeded, mode, per_asset, 50 // max(per_asset, 1),
        )
    return result


async def main(
    symbols: list[str] | None,
    all_assets: bool,
    delay: float,
    mode: str = DEFAULT_INFERENCE_MODE,
) -> int:
    try:
        if all_assets or symbols:
            result = await execute_multi_asset_inference(
                symbols=symbols, delay_seconds=delay, mode=mode
            )
            # Non-zero only when NOTHING succeeded: a partial run still
            # produced usable decisions and should not read as a failure.
            return 0 if result.succeeded else 1
        await execute_inference(
            use_consolidated=_uses_consolidated(mode, BASE_SYMBOL)
        )
        return 0
    except InferenceError as exc:
        logger.error("Inference failed: %s", exc)
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AI inference for one asset (default: BTC) or many."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyse every active symbol in coins_registry, sequentially.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Explicit symbols to analyse, e.g. --symbols BTC/USDT SOL/USDT.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_INTER_SYMBOL_DELAY,
        help=f"Seconds between assets (default: {DEFAULT_INTER_SYMBOL_DELAY}).",
    )
    parser.add_argument(
        "--mode",
        dest="inference_mode",
        choices=("consolidated", "panel", "hybrid"),
        default=DEFAULT_INFERENCE_MODE,
        help=(
            "consolidated: 1 call/asset (default — 21 assets fit a 50/day "
            "free tier); panel: 6 calls/asset (the Phase 4 five-specialist "
            "architecture); hybrid: panel for BTC, consolidated for the "
            "altcoins (26 calls)."
        ),
    )
    arguments = parser.parse_args()

    os.system("")  # enable ANSI escape processing on Windows consoles
    # Windows consoles often default to cp1252, which cannot render the
    # box-drawing/marker characters used in the report.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    setup_logging(get_settings().log_level)
    try:
        raise SystemExit(
            asyncio.run(
                main(
                    arguments.symbols,
                    arguments.all,
                    arguments.delay,
                    arguments.inference_mode,
                )
            )
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None

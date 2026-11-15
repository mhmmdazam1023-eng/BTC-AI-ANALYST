"""Phase 8 Part 6 test: multi-asset inference pipeline.

All model calls are STUBBED — this exercises orchestration, isolation and
persistence without spending a token or waiting on the API.

Gates
-----
1. **Keying**        — ``decisions`` is keyed by (timestamp, symbol) and
   refuses a duplicate pair. Without this a 21-coin run collapses to one
   row.
2. **Loop**          — running 3 symbols writes 3 rows for today, one per
   symbol, and Bitcoin is analysed FIRST.
3. **Context**       — the prompt for SOL carries SOL's DNA, not BTC's, and
   the Master receives a per-symbol summary.
4. **Isolation**     — when one asset raises, the others are still analysed
   AND persisted, and the failure is reported rather than swallowed.
5. **Forecast scope**— altcoins get NO forecast block (forecasts are
   BTC-only); Bitcoin still does.
6. **Backward compat**— ``execute_inference()`` with no symbol still means
   Bitcoin, and an empty registry falls back to Bitcoin alone.
7. **Bodyless response** — a reply carrying no JSON at all (OpenRouter
   keep-alive padding after an upstream stall) is retried rather than
   killing the asset, and an exhausted budget fails with a readable
   message that does not trip the daily-quota abort.

The gates run against the LATEST REAL feature date, because they exercise
the real feature loaders — there is no sentinel date. Several of them assert
on an empty ``decisions`` slate, so the suite deletes that date and then
restores whatever production rows it found first (``_snapshot_decisions`` /
``_restore_decisions``). Removing that snapshot destroys real decisions.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_multi_asset_phase8.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, text  # noqa: E402
from sqlalchemy.exc import DBAPIError, IntegrityError  # noqa: E402

from src.agents.schemas import (  # noqa: E402
    ConsolidatedAnalysis,
    MasterDecision,
    Signal,
    SubAgentResponse,
)
from src.config import get_settings  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import BASE_SYMBOL, DailyFeature, Decision  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("scripts.test_multi_asset_phase8")

#: Symbols exercised by the loop. All three must exist in the registry and
#: have features; they do after Parts 3-4.
TEST_SYMBOLS = ("BTC/USDT", "SOL/USDT", "TAO/USDT")

#: The asset made to fail in the isolation gate.
FAILING_SYMBOL = "TAO/USDT"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_run_inference():
    spec = importlib.util.spec_from_file_location(
        "run_inference", Path(__file__).with_name("run_inference.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_inference"] = module
    spec.loader.exec_module(module)
    return module


class _Stubs:
    """Swaps the model calls for deterministic stand-ins."""

    def __init__(self, module, fail_for: str | None = None) -> None:
        self.module = module
        self.fail_for = fail_for
        #: Message the injected failure raises; overridable per gate.
        self.failure_message = "synthetic failure injected"
        #: symbol -> the Master prompt it produced.
        self.master_prompts: dict[str, str] = {}
        #: symbol -> list of sub-agent prompts.
        self.pillar_prompts: dict[str, list[str]] = {}
        self._original_ask = None
        self._current_symbol: str | None = None

    def __enter__(self):
        from src.agents.base_agent import BaseAgent

        stubs = self

        async def fake_ask(agent_self, prompt, response_model):
            symbol = stubs._current_symbol or "?"
            # Consolidated mode (Part 7): ONE call returns all five pillar
            # scores. The Master no longer runs per asset, so this prompt is
            # the one carrying the per-symbol context.
            if response_model is ConsolidatedAnalysis:
                stubs.master_prompts[symbol] = prompt
                return ConsolidatedAnalysis(
                    technical_score=0,
                    onchain_score=0,
                    macro_score=0,
                    sentiment_score=0,
                    liquidity_risk_score=0,
                    reasoning_summary=(
                        "Ringkasan konsolidasi stub untuk uji pipeline; tidak "
                        "memanggil model bahasa apa pun."
                    ),
                )
            if response_model is MasterDecision:
                stubs.master_prompts[symbol] = prompt
                return MasterDecision(
                    action="HOLD",
                    confidence_score=50,
                    executive_summary=(
                        "Ringkasan stub untuk uji pipeline multi-aset; tidak "
                        "memanggil model bahasa apa pun."
                    ),
                    risk_warning="Peringatan risiko stub untuk uji multi-aset.",
                )
            stubs.pillar_prompts.setdefault(symbol, []).append(prompt)
            return SubAgentResponse(
                agent_name=agent_self.agent_name,
                signal=Signal.NEUTRAL,
                score=0,
                reasoning="Respons pilar stub untuk uji pipeline multi-aset.",
            )

        self._original_ask = BaseAgent.ask
        BaseAgent.ask = fake_ask

        # Wrap execute_inference so we know which symbol is in flight, and
        # so one symbol can be made to fail on demand.
        original_execute = self.module.execute_inference

        async def tracking_execute(*args, **kwargs):
            symbol = kwargs.get("symbol", BASE_SYMBOL)
            stubs._current_symbol = symbol
            if stubs.fail_for and symbol == stubs.fail_for:
                raise RuntimeError(f"{stubs.failure_message} for {symbol}")
            return await original_execute(*args, **kwargs)

        self._original_execute = original_execute
        self.module.execute_inference = tracking_execute
        return self

    def __exit__(self, *exc_info):
        from src.agents.base_agent import BaseAgent

        BaseAgent.ask = self._original_ask
        self.module.execute_inference = self._original_execute
        return False


async def _clear_decisions(day: datetime) -> None:
    async with session_scope() as session:
        await session.execute(
            text("DELETE FROM decisions WHERE timestamp = :t"), {"t": day}
        )


async def _resolve_analysis_day(module) -> datetime:
    """The date the gates operate on — the latest REAL feature date."""
    analysis_date, _ = await module.load_latest_features(BASE_SYMBOL)
    return datetime.fromisoformat(analysis_date).replace(tzinfo=UTC)


async def _snapshot_decisions(day: datetime) -> list[dict[str, object]]:
    """Copy every existing ``decisions`` row for ``day`` verbatim.

    Several gates assert on an empty slate ("exactly one BTC row"), so they
    have to delete this date — and this date is whatever a production run
    wrote today, not a sentinel. Without a snapshot the cleanup silently
    destroys real output: on 2026-07-29 it wiped all 20 decisions from a
    76-minute 21-asset run.
    """
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(Decision.__table__).where(
                    Decision.__table__.c.timestamp == day
                )
            )
        ).mappings().all()
    return [dict(row) for row in rows]


async def _restore_decisions(rows: list[dict[str, object]]) -> None:
    """Put the snapshotted rows back exactly as they were."""
    if not rows:
        return
    async with session_scope() as session:
        await session.execute(Decision.__table__.insert(), rows)


async def _decisions_on(day: datetime) -> dict[str, Decision]:
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(Decision).where(Decision.timestamp == day)
            )
        ).scalars().all()
    return {row.symbol: row for row in rows}


# ---------------------------------------------------------------------------
async def gate_keying() -> None:
    """decisions must be keyed by (timestamp, symbol)."""
    async with session_scope() as session:
        pk_def = (
            await session.execute(
                text(
                    "SELECT pg_get_constraintdef(con.oid) FROM pg_constraint con "
                    "JOIN pg_class rel ON rel.oid = con.conrelid "
                    "WHERE rel.relname='decisions' AND con.contype='p'"
                )
            )
        ).scalar_one()
    _require(
        "symbol" in pk_def and "timestamp" in pk_def,
        f"decisions PK is not composite: {pk_def} — a multi-asset run would "
        f"upsert every coin onto one row",
    )

    duplicates = None
    async with session_scope() as session:
        duplicates = (
            await session.execute(
                text(
                    "SELECT count(*) FROM (SELECT timestamp, symbol FROM decisions "
                    "GROUP BY 1,2 HAVING count(*) > 1) d"
                )
            )
        ).scalar_one()
    _require(duplicates == 0, f"{duplicates} duplicate (day, symbol) decisions")
    logger.info("Gate 1 OK: decisions PK %s, no duplicates", pk_def)

    # Every historical row must have been backfilled to Bitcoin.
    async with session_scope() as session:
        orphans = (
            await session.execute(
                text("SELECT count(*) FROM decisions WHERE symbol IS NULL")
            )
        ).scalar_one()
    _require(orphans == 0, f"{orphans} decision(s) have a NULL symbol")
    logger.info("Gate 1 OK: no decision rows left without a symbol")


async def gate_loop_and_context(module) -> datetime:
    """3 symbols -> 3 rows, BTC first, each prompt carrying its own DNA."""
    analysis_date, _ = await module.load_latest_features(BASE_SYMBOL)
    day = datetime.fromisoformat(analysis_date).replace(tzinfo=UTC)
    await _clear_decisions(day)

    with _Stubs(module) as stubs:
        result = await module.execute_multi_asset_inference(
            symbols=list(TEST_SYMBOLS),
            delay_seconds=0.0,
            print_output=False,
        )

    _require(
        result.succeeded == len(TEST_SYMBOLS),
        f"expected {len(TEST_SYMBOLS)} successes, got {result.succeeded} "
        f"(failures: {result.failures})",
    )
    _require(
        [o.symbol for o in result.outcomes] == list(TEST_SYMBOLS),
        f"assets should be analysed in order {TEST_SYMBOLS}, got "
        f"{[o.symbol for o in result.outcomes]}",
    )
    _require(
        result.outcomes[0].symbol == BASE_SYMBOL,
        "Bitcoin must be analysed FIRST so the headline decision survives a "
        "truncated run",
    )
    logger.info("Gate 2 OK: %s", result.summary_line())

    stored = await _decisions_on(day)
    for symbol in TEST_SYMBOLS:
        _require(symbol in stored, f"no decision row persisted for {symbol}")
        _require(
            stored[symbol].symbol == symbol,
            f"row for {symbol} carries symbol {stored[symbol].symbol}",
        )
    _require(
        len(stored) == len(TEST_SYMBOLS),
        f"expected {len(TEST_SYMBOLS)} rows for {day.date()}, found "
        f"{len(stored)}: {sorted(stored)}",
    )
    logger.info(
        "Gate 2 OK: %d distinct decision rows persisted for %s (%s)",
        len(stored), day.date(), ", ".join(sorted(stored)),
    )

    # --- context: SOL's prompt must carry SOL's DNA, not BTC's ---
    sol_prompt = stubs.master_prompts.get("SOL/USDT", "")
    btc_prompt = stubs.master_prompts.get(BASE_SYMBOL, "")
    _require(sol_prompt and btc_prompt, "missing captured Master prompts")
    _require(
        "SOL/USDT" in sol_prompt,
        f"SOL's Master prompt should name SOL: {sol_prompt[:200]}",
    )
    # The DNA block is headed "COIN DNA" in the Master prompt (panel mode)
    # and "KARAKTERISTIK KOIN" in the consolidated prompt (Part 7). Assert on
    # the content, not on which agent happens to carry it.
    _require(
        "COIN DNA" in sol_prompt or "KARAKTERISTIK KOIN" in sol_prompt,
        f"SOL's decision prompt lacks any Coin DNA block: {sol_prompt[:300]}",
    )
    _require(
        sol_prompt != btc_prompt,
        "SOL and BTC received IDENTICAL prompts — the per-symbol context is "
        "not being threaded through",
    )

    sol_dna = await module.load_coin_dna("SOL/USDT", analysis_date)
    btc_dna = await module.load_coin_dna(BASE_SYMBOL, analysis_date)
    _require(
        sol_dna.get("btc_beta_30d") != btc_dna.get("btc_beta_30d"),
        "SOL and BTC report the same beta — DNA is not per-symbol",
    )
    beta_text = f"{float(sol_dna['btc_beta_30d']):.2f}"
    _require(
        beta_text in sol_prompt,
        f"SOL's own beta {beta_text} is absent from its Master prompt",
    )
    logger.info(
        "Gate 3 OK: SOL prompt carries SOL beta %s (BTC's is %.2f) — context "
        "is per-symbol",
        beta_text, float(btc_dna["btc_beta_30d"]),
    )

    # Consolidated mode (Part 7): exactly ONE model call per asset, which
    # is the entire point — 21 assets must fit in a 50-call day.
    for symbol in TEST_SYMBOLS:
        _require(
            symbol in stubs.master_prompts,
            f"{symbol} produced no analyst prompt",
        )
        _require(
            not stubs.pillar_prompts.get(symbol),
            f"{symbol} also ran {len(stubs.pillar_prompts.get(symbol, []))} "
            f"separate pillar call(s) — consolidated mode must use exactly one",
        )
    _require(
        all(r.score is not None for o in result.outcomes for r in o.responses),
        "every consolidated response must expand to a scored 5-pillar panel",
    )
    _require(
        all(len(o.responses) == 5 for o in result.outcomes),
        f"each asset must yield a 5-pillar panel, got "
        f"{[len(o.responses) for o in result.outcomes]}",
    )
    logger.info(
        "Gate 3 OK: 1 model call per asset, each expanding to a 5-pillar panel"
    )
    return day


async def gate_isolation(module, day: datetime) -> None:
    """One failing asset must not cost the others their decisions."""
    await _clear_decisions(day)

    with _Stubs(module, fail_for=FAILING_SYMBOL):
        result = await module.execute_multi_asset_inference(
            symbols=list(TEST_SYMBOLS),
            delay_seconds=0.0,
            print_output=False,
        )

    _require(
        FAILING_SYMBOL in result.failures,
        f"{FAILING_SYMBOL} was made to fail but is not in failures: "
        f"{result.failures}",
    )
    _require(
        result.succeeded == len(TEST_SYMBOLS) - 1,
        f"expected {len(TEST_SYMBOLS) - 1} successes alongside the failure, "
        f"got {result.succeeded}",
    )
    _require(
        result.analysed == len(TEST_SYMBOLS),
        f"the loop should have ATTEMPTED all {len(TEST_SYMBOLS)} assets, "
        f"attempted {result.analysed} — it stopped early",
    )

    stored = await _decisions_on(day)
    survivors = [s for s in TEST_SYMBOLS if s != FAILING_SYMBOL]
    for symbol in survivors:
        _require(
            symbol in stored,
            f"{symbol} was lost when {FAILING_SYMBOL} failed — isolation broken",
        )
    _require(
        FAILING_SYMBOL not in stored,
        f"{FAILING_SYMBOL} failed but a decision row was written anyway",
    )
    logger.info(
        "Gate 4 OK: %s failed, %s still analysed and persisted",
        FAILING_SYMBOL, ", ".join(survivors),
    )


async def gate_quota_abort(module, day: datetime) -> None:
    """A DAILY quota error must stop the run, not grind through the rest.

    OpenRouter's free tier is 50 requests per DAY (verified 2026-07-29).
    At 6 calls per asset that covers ~8 assets. Once the budget is gone,
    every remaining asset fails identically after ~3 minutes of retry
    backoff — 21 assets would waste an hour proving it.
    """
    _require(
        module._is_daily_quota_exhausted(
            "RateLimitError: Error code: 429 - Rate limit exceeded: "
            "free-models-per-day. Add 10 credits to unlock 1000 free..."
        ),
        "the real OpenRouter daily-quota message must be recognised",
    )
    _require(
        not module._is_daily_quota_exhausted(
            "APITimeoutError: request timed out after 120s"
        ),
        "a plain timeout must NOT be mistaken for quota exhaustion",
    )
    _require(
        not module._is_daily_quota_exhausted("InferenceError: only 2/5 pillars"),
        "a pillar shortfall must NOT be mistaken for quota exhaustion",
    )
    logger.info("Gate 5 OK: daily-quota detection distinguishes cap from throttle")

    await _clear_decisions(day)
    all_symbols = [BASE_SYMBOL, "SOL/USDT", "TAO/USDT"]

    with _Stubs(module, fail_for="SOL/USDT") as stubs:
        stubs.failure_message = (
            "Error code: 429 - Rate limit exceeded: free-models-per-day."
        )
        result = await module.execute_multi_asset_inference(
            symbols=all_symbols, delay_seconds=0.0, print_output=False
        )

    _require(result.quota_exhausted, "the run should have flagged quota exhaustion")
    _require(
        "TAO/USDT" in result.skipped,
        f"assets after the quota error must be SKIPPED, got {result.skipped}",
    )
    _require(
        "TAO/USDT" not in result.failures,
        "a skipped asset is not a failure — it was never attempted",
    )
    _require(
        result.succeeded == 1 and result.outcomes[0].symbol == BASE_SYMBOL,
        "the assets analysed BEFORE the quota ran out must be kept",
    )
    stored = await _decisions_on(day)
    _require(
        BASE_SYMBOL in stored,
        "Bitcoin's decision must survive a later quota exhaustion",
    )
    logger.info(
        "Gate 5 OK: quota error stopped the run after %d asset(s); %d skipped "
        "instead of burning retry backoff",
        result.succeeded, len(result.skipped),
    )
    await _clear_decisions(day)


async def gate_forecast_scope(module, day: datetime) -> None:
    """Altcoins must not receive Bitcoin's forecast as their own."""
    await _clear_decisions(day)
    with _Stubs(module) as stubs:
        await module.execute_multi_asset_inference(
            symbols=[BASE_SYMBOL, "SOL/USDT"],
            delay_seconds=0.0,
            print_output=False,
        )

    btc_prompt = stubs.master_prompts[BASE_SYMBOL]
    sol_prompt = stubs.master_prompts["SOL/USDT"]
    marker = "statistical_forecast_monte_carlo"
    _require(
        marker not in sol_prompt,
        "SOL's prompt contains a Monte Carlo forecast — forecasts are "
        "BTC-only, so this would present Bitcoin's percentiles as SOL's",
    )
    async with session_scope() as session:
        forecasts = (
            await session.execute(text("SELECT count(*) FROM forecasts"))
        ).scalar_one()
    if forecasts:
        _require(
            marker in btc_prompt,
            "Bitcoin should still receive its forecast context",
        )
        logger.info(
            "Gate 6 OK: BTC gets the forecast block, SOL correctly does not"
        )
    else:
        logger.info(
            "Gate 6 OK: SOL gets no forecast block (forecasts table is empty, "
            "so BTC has none either)"
        )
    await _clear_decisions(day)


async def gate_backward_compatibility(module, day: datetime) -> None:
    """No symbol -> Bitcoin; empty registry -> Bitcoin alone."""
    await _clear_decisions(day)
    with _Stubs(module):
        outcome = await module.execute_inference(print_output=False)
    _require(
        outcome.symbol == BASE_SYMBOL,
        f"execute_inference() with no symbol must mean Bitcoin, got "
        f"{outcome.symbol}",
    )
    stored = await _decisions_on(day)
    _require(
        list(stored) == [BASE_SYMBOL],
        f"single-asset run should write exactly one BTC row, got {sorted(stored)}",
    )
    logger.info("Gate 7 OK: execute_inference() still defaults to Bitcoin")

    symbols = await module.load_active_symbols()
    _require(
        symbols[0] == BASE_SYMBOL,
        f"Bitcoin must lead the symbol order, got {symbols[:3]}",
    )
    _require(
        len(symbols) >= len(TEST_SYMBOLS),
        f"registry returned only {len(symbols)} active symbols",
    )
    logger.info(
        "Gate 7 OK: %d active symbols, Bitcoin first (%s ...)",
        len(symbols), ", ".join(symbols[:4]),
    )

    # An empty explicit list must still be handled — fall back, never crash.
    with _Stubs(module):
        empty = await module.execute_multi_asset_inference(
            symbols=[], delay_seconds=0.0, print_output=False
        )
    _require(
        empty.requested == 0 and empty.succeeded == 0,
        f"an empty symbol list should do nothing gracefully, got {empty}",
    )
    logger.info("Gate 7 OK: an empty symbol list is a clean no-op")
    await _clear_decisions(day)


async def gate_bodyless_response() -> None:
    """A response with no JSON body must be RETRIED, never fatal.

    Live failure 2026-07-29, ARB/USDT: OpenRouter pads long non-streaming
    requests with whitespace to hold the socket open. Poolside stalled, so
    the body arrived as padding alone. httpx's ``Response.json()`` is a bare
    ``json.loads`` and the OpenAI SDK has no handler for it, so a raw
    ``json.JSONDecodeError`` escaped the retry policy — which only knew
    about openai's own exception types — and killed the asset outright.
    """
    from src.agents.base_agent import (
        BaseAgent,
        EmptyChoicesError,
        EmptyResponseBodyError,
        QwenResponseError,
        _RETRYABLE_ERRORS,
    )

    _require(
        EmptyResponseBodyError in _RETRYABLE_ERRORS,
        "EmptyResponseBodyError must be retryable, else a single stalled "
        "upstream response permanently fails the asset",
    )
    _require(
        json.JSONDecodeError not in _RETRYABLE_ERRORS,
        "raw JSONDecodeError must NOT be retryable — it is a ValueError "
        "subclass and would also swallow genuine local decode bugs",
    )
    # The SECOND door into the same failure: body parses, but carries no
    # completion. Cost 3 of 21 assets on 2026-07-31 because only the first
    # door had been closed.
    _require(
        EmptyChoicesError in _RETRYABLE_ERRORS,
        "EmptyChoicesError must be retryable — a 200 with an empty 'choices' "
        "list is an upstream hiccup, and leaving it fatal loses whole assets",
    )
    _require(
        issubclass(EmptyChoicesError, QwenResponseError),
        "EmptyChoicesError must subclass QwenResponseError so existing "
        "handlers keep catching it",
    )

    def _padded_body() -> None:
        """Raise the exact error httpx produces on a keep-alive-padded body."""
        json.loads("    \n" * 580)

    class _StalledThenGood:
        """Mimics OpenRouter: padding, padding, ... then a real payload."""

        def __init__(self, failures: int) -> None:
            self.remaining = failures
            self.calls = 0
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        async def _create(self, **_kwargs):
            self.calls += 1
            if self.remaining > 0:
                self.remaining -= 1
                _padded_body()
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"ok": true}', reasoning=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

        async def close(self) -> None:
            return None

    # Keep the gate quick: 2 attempts => a single 1-second backoff.
    settings = get_settings().model_copy(update={"qwen_max_retries": 2})
    messages = [{"role": "user", "content": "ping"}]

    # (a) One stalled response, then success — the run must recover.
    agent = BaseAgent(settings=settings)
    stub = _StalledThenGood(failures=1)
    agent._client = stub
    content = await agent._chat_completion_raw(messages)
    _require(
        content == '{"ok": true}',
        f"a stalled first response must be retried into a good one, got {content!r}",
    )
    _require(
        stub.calls == 2,
        f"expected exactly 1 retry after the padded body, saw {stub.calls} call(s)",
    )
    logger.info("Gate 8 OK: a bodyless response is retried, not fatal")

    # (b) Every attempt stalled — fail with something a human can act on.
    agent = BaseAgent(settings=settings)
    stub = _StalledThenGood(failures=99)
    agent._client = stub
    try:
        await agent._chat_completion_raw(messages)
    except EmptyResponseBodyError as exc:
        detail = str(exc)
    except json.JSONDecodeError as exc:  # pragma: no cover - the old bug
        raise AssertionError(
            "a raw JSONDecodeError escaped the retry policy again — this is "
            f"the exact ARB/USDT regression: {exc}"
        ) from exc
    else:
        raise AssertionError("an endlessly stalled endpoint must raise")

    _require(
        stub.calls == 2,
        f"the retry budget is {settings.qwen_max_retries}, but the endpoint "
        f"was called {stub.calls} time(s)",
    )
    _require(
        "no JSON at all" in detail and "keep-alive padding" in detail,
        f"the failure must explain itself, not just echo json's position "
        f"report; got: {detail}",
    )
    _require(
        not _load_run_inference()._is_daily_quota_exhausted(detail),
        "a stalled upstream must NOT be mistaken for the daily quota cap — "
        "that would abort the whole remaining portfolio",
    )
    logger.info(
        "Gate 8 OK: exhausted retries raise a readable EmptyResponseBodyError "
        "that does not trip the quota abort"
    )


# ---------------------------------------------------------------------------
async def main() -> int:
    started = datetime.now(tz=UTC)
    module = _load_run_inference()
    day: datetime | None = None
    preserved: list[dict[str, object]] = []
    try:
        # The gates need an EMPTY decisions slate for the analysis date, but
        # that date is the latest real feature date — the very day a
        # production run writes to. Snapshot whatever is already there so the
        # cleanup below can put it back untouched.
        day = await _resolve_analysis_day(module)
        preserved = await _snapshot_decisions(day)
        if preserved:
            logger.warning(
                "%d real decision row(s) already exist for %s — preserving "
                "them across this test run",
                len(preserved), day.date(),
            )
        logger.info("--- decisions keying ---")
        await gate_keying()
        logger.info("--- Multi-asset loop & per-symbol context ---")
        day = await gate_loop_and_context(module)
        logger.info("--- Error isolation ---")
        await gate_isolation(module, day)
        logger.info("--- Daily quota abort ---")
        await gate_quota_abort(module, day)
        logger.info("--- Forecast scope ---")
        await gate_forecast_scope(module, day)
        logger.info("--- Backward compatibility ---")
        await gate_backward_compatibility(module, day)
        logger.info("--- Bodyless response resilience ---")
        await gate_bodyless_response()
    except AssertionError as exc:
        logger.error("PHASE 8 PART 6 TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("PHASE 8 PART 6 TEST FAILED with an unexpected error")
        return 1
    finally:
        # Never leave stub decisions behind in the real history.
        try:
            if day is not None:
                await _clear_decisions(day)
                await _restore_decisions(preserved)
                logger.info(
                    "Cleanup: removed stub rows for %s, restored %d real row(s)",
                    day.date(), len(preserved),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cleanup issue (ignored): %s", exc)
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "PHASE 8 PART 6 TEST PASSED (7 gates, %.1fs) — multi-asset inference "
        "pipeline ready", elapsed,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    raise SystemExit(asyncio.run(main()))

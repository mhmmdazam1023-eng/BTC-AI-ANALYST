"""C3 test: model usage / reasoning-token tracking.

No model calls — the OpenAI client is stubbed, so this costs nothing and
runs in well under a second.

Gates
-----
1. **Live field layout** — ``extract_usage`` reads ``reasoning_tokens`` from
   two levels down and ``cost`` from ``model_extra``, against an object
   shaped exactly like the response probed on 2026-08-09. Both are easy to
   miss by attribute access alone, and both are the point of the module.
2. **Degrades, never crashes** — a non-reasoning ``usage`` yields
   ``reasoning_tokens=None`` (NOT 0: "this model does not report it" is not
   "it reasoned for zero tokens" — the same NULL-vs-0 contract as the Part 4
   features), and a garbage object costs one field, not the row.
3. **Never breaks a decision** — with the DB unreachable, ``record_usage``
   returns normally. An analytics table is not worth failing inference over.
4. **WIRING** — ``_chat_completion_raw`` really does persist a row. Every
   other gate here tests a pure function; only this one proves the plumbing
   is connected. That distinction is what gate 10 of the regime suite exists
   to enforce, and it has caught a live-only break twice.
5. **Attribution under concurrency** — five tasks under ``asyncio.gather``
   each record their own symbol. Panel mode runs exactly that way, and a
   ContextVar that leaked between tasks would silently mis-attribute cost.
6. **Production path attribution** — consolidated mode (the default, and
   the only mode that fits 21 assets in 50 requests) attributes its call to
   its own asset. Gate 4 proves the writer works; this proves production
   reaches it.
7. **Stored invariant** — every row in the live table satisfies
   ``reasoning_tokens <= completion_tokens``. Reasoning is a SUBSET of
   completion; if that ever inverts, the column means something else and
   every cost figure derived from it is wrong.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_usage_tracking.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, func, select  # noqa: E402

from src.agents.base_agent import BaseAgent  # noqa: E402
from src.agents.usage_tracker import (  # noqa: E402
    current_usage_symbol,
    extract_usage,
    format_usage,
    record_usage,
    usage_context,
)
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import ModelUsage  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("test_usage_tracking")

#: Probe rows are tagged so cleanup can delete exactly what it wrote and
#: nothing else — the borrow-and-return discipline the regime suite learned
#: the hard way. Never a name a real agent could produce.
PROBE_AGENT = "__usage_probe__"


def _live_shaped_usage() -> SimpleNamespace:
    """A usage object with the exact shape probed from OpenRouter."""
    return SimpleNamespace(
        prompt_tokens=30,
        completion_tokens=41,
        total_tokens=71,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=34),
        prompt_tokens_details=SimpleNamespace(cached_tokens=12),
        # `cost` is NOT an SDK field; it arrives here.
        model_extra={"cost": "0.000123", "cost_details": {}},
    )


# ---------------------------------------------------------------------------
def gate_live_field_layout() -> None:
    fields = extract_usage(_live_shaped_usage())

    assert fields["prompt_tokens"] == 30, fields
    assert fields["completion_tokens"] == 41, fields
    assert fields["total_tokens"] == 71, fields
    assert fields["reasoning_tokens"] == 34, (
        f"reasoning_tokens lost — it lives two levels down at "
        f"usage.completion_tokens_details.reasoning_tokens: {fields}"
    )
    assert fields["cached_tokens"] == 12, fields
    assert fields["cost"] == Decimal("0.000123"), (
        f"cost lost — it is not an SDK field, it must be read from "
        f"model_extra: {fields}"
    )

    # The share is what makes the number readable; 34/41 = 83%.
    line = format_usage(fields, duration=1.5)
    assert "reasoning 34" in line and "83%" in line, line
    logger.info("  %s", line)
    logger.info("  OK: all six fields recovered, reasoning shown as 83%% of output")


# ---------------------------------------------------------------------------
def gate_degrades_never_crashes() -> None:
    # A non-reasoning model: no completion_tokens_details at all.
    plain = SimpleNamespace(
        prompt_tokens=10, completion_tokens=20, total_tokens=30,
        completion_tokens_details=None, prompt_tokens_details=None,
        model_extra={},
    )
    fields = extract_usage(plain)
    assert fields["reasoning_tokens"] is None, (
        "absent reasoning must be NULL, not 0 — 'not reported' and 'reasoned "
        f"for zero tokens' are different facts: {fields}"
    )
    assert fields["cached_tokens"] is None, fields
    assert fields["cost"] is None, fields
    assert fields["completion_tokens"] == 20, fields

    # Garbage in one field must cost that field only.
    junk = SimpleNamespace(
        prompt_tokens="not-a-number", completion_tokens=7, total_tokens=None,
        completion_tokens_details=SimpleNamespace(reasoning_tokens="?"),
        prompt_tokens_details=None,
        model_extra={"cost": "not-a-decimal"},
    )
    fields = extract_usage(junk)
    assert fields["prompt_tokens"] is None, fields
    assert fields["completion_tokens"] == 7, fields
    assert fields["reasoning_tokens"] is None, fields
    assert fields["cost"] is None, fields

    assert extract_usage(None) == {}, "no usage at all must yield {}"
    assert format_usage({}) == "usage unavailable"
    logger.info("  OK: missing/garbage fields degrade to NULL, never raise")


# ---------------------------------------------------------------------------
async def gate_never_breaks_a_decision() -> None:
    """A dead database must cost the measurement, not the run."""
    import src.db.engine as engine_module

    original = engine_module.session_scope

    def _explode(*_args, **_kwargs):
        raise RuntimeError("simulated database outage")

    engine_module.session_scope = _explode
    try:
        await record_usage(
            agent_name=PROBE_AGENT,
            model="test/model",
            usage=_live_shaped_usage(),
            symbol="BTC/USDT",
            duration_seconds=1.0,
        )
    finally:
        engine_module.session_scope = original
    logger.info("  OK: record_usage swallowed a DB outage and returned normally")


# ---------------------------------------------------------------------------
class _StubAgent(BaseAgent):
    """BaseAgent with the network replaced, everything else intact."""

    agent_name = PROBE_AGENT


class _StubCompletions:
    def __init__(self, content: str = '{"ok": true}') -> None:
        self.calls: list[dict] = []
        self._content = content

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content, reasoning=None),
                    finish_reason="stop",
                )
            ],
            usage=_live_shaped_usage(),
            model_extra={},
        )


def _stub_client(agent: BaseAgent, content: str = '{"ok": true}') -> _StubCompletions:
    completions = _StubCompletions(content)
    async def _close() -> None:
        return None

    agent._client = SimpleNamespace(  # noqa: SLF001 - deliberate test seam
        chat=SimpleNamespace(completions=completions), close=_close
    )
    return completions


async def gate_wiring() -> None:
    agent = _StubAgent()
    _stub_client(agent)

    content = await agent._chat_completion_raw(  # noqa: SLF001
        [{"role": "user", "content": "probe"}],
        usage_symbol="BTC/USDT",
    )
    assert content == '{"ok": true}', content

    async with session_scope() as session:
        row = (
            await session.execute(
                select(ModelUsage)
                .where(ModelUsage.agent_name == PROBE_AGENT)
                .order_by(ModelUsage.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    assert row is not None, (
        "no usage row was written — _chat_completion_raw is not calling "
        "record_usage; the tracking is dead in production even though every "
        "pure-function gate above passes"
    )
    assert row.symbol == "BTC/USDT", row.symbol
    assert row.reasoning_tokens == 34, row.reasoning_tokens
    assert row.prompt_tokens == 30 and row.completion_tokens == 41
    assert row.cost == Decimal("0.000123"), row.cost
    assert row.duration_seconds is not None and row.duration_seconds >= 0.0
    logger.info(
        "  OK: row persisted (symbol=%s prompt=%d completion=%d reasoning=%d)",
        row.symbol, row.prompt_tokens, row.completion_tokens, row.reasoning_tokens,
    )


# ---------------------------------------------------------------------------
async def gate_concurrent_attribution() -> None:
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PEPE/USDT", "TAO/USDT"]

    # Scope the assertion to rows THIS gate writes. Gate 4 already logged a
    # BTC/USDT probe, and counting it here would test the previous gate's
    # work as if it were this one's — the same "the sample measured is not
    # the sample claimed" error this whole measurement layer exists to stop.
    async with session_scope() as session:
        watermark = (await session.execute(
            select(func.coalesce(func.max(ModelUsage.id), 0))
        )).scalar_one()

    async def one(symbol: str) -> None:
        with usage_context(symbol):
            await asyncio.sleep(0)  # force interleaving
            assert current_usage_symbol() == symbol, (
                f"context leaked between tasks: saw {current_usage_symbol()} "
                f"inside the {symbol} task"
            )
            agent = _StubAgent()
            _stub_client(agent)
            await agent._chat_completion_raw(  # noqa: SLF001
                [{"role": "user", "content": symbol}]
            )

    # Set an outer symbol too: if a task's reset leaked outward, the outer
    # value would come back wrong.
    with usage_context("OUTER/USDT"):
        await asyncio.gather(*(one(s) for s in symbols))
        assert current_usage_symbol() == "OUTER/USDT", current_usage_symbol()

    async with session_scope() as session:
        rows = (
            await session.execute(
                select(ModelUsage.symbol)
                .where(ModelUsage.agent_name == PROBE_AGENT)
                .where(ModelUsage.id > watermark)
            )
        ).scalars().all()

    assert sorted(rows) == sorted(symbols), (
        f"expected one row per symbol, got {sorted(rows)}"
    )
    logger.info("  OK: 5 concurrent tasks each recorded their own symbol")


# ---------------------------------------------------------------------------
async def gate_production_path_attributes() -> None:
    """The DEFAULT path — consolidated mode — must attribute its own asset.

    Gate 4 drives ``_chat_completion_raw`` directly with an explicit symbol.
    That proves the writer works, not that production reaches it: consolidated
    mode is the only mode that fits 21 assets in 50 requests, and it sets the
    symbol through ``usage_context`` in ``analyze()``. Delete that ``with``
    and every production row silently records ``symbol=NULL`` — a whole day of
    per-asset cost data lost to a change no other gate objects to.
    """
    from src.agents.consolidated_analyst import ConsolidatedAnalyst

    analyst = ConsolidatedAnalyst()
    analyst.agent_name = PROBE_AGENT
    _stub_client(
        analyst,
        '{"technical_score": 10, "onchain_score": -5, "macro_score": 0, '
        '"sentiment_score": 5, "liquidity_risk_score": -10, '
        '"reasoning_summary": "Probe response for the C3 usage-tracking '
        'gate; long enough to satisfy the schema minimum."}',
    )
    try:
        async with session_scope() as session:
            watermark = (await session.execute(
                select(func.coalesce(func.max(ModelUsage.id), 0))
            )).scalar_one()

        await analyst.analyze(
            {"date": "2026-08-09", "asset_close": 1.0},
            {"symbol": "ONDO/USDT"},
            symbol="ONDO/USDT",
        )
    finally:
        await analyst.close()

    async with session_scope() as session:
        symbols = (await session.execute(
            select(ModelUsage.symbol).where(ModelUsage.id > watermark)
        )).scalars().all()

    assert symbols == ["ONDO/USDT"], (
        "consolidated mode did not attribute its call to the asset "
        f"(got {symbols}) — usage_context is missing from analyze()"
    )
    logger.info("  OK: consolidated mode attributed its call to ONDO/USDT")


# ---------------------------------------------------------------------------
async def gate_stored_invariant() -> None:
    async with session_scope() as session:
        total = (await session.execute(
            select(func.count()).select_from(ModelUsage)
        )).scalar_one()
        violations = (await session.execute(
            select(func.count()).select_from(ModelUsage).where(
                ModelUsage.reasoning_tokens > ModelUsage.completion_tokens
            )
        )).scalar_one()

    assert violations == 0, (
        f"{violations} row(s) have reasoning_tokens > completion_tokens — "
        "reasoning is a SUBSET of completion, so this column no longer means "
        "what every cost figure derived from it assumes"
    )
    logger.info("  OK: %d row(s) stored, 0 violate reasoning <= completion", total)


# ---------------------------------------------------------------------------
async def _cleanup() -> None:
    async with session_scope() as session:
        result = await session.execute(
            delete(ModelUsage).where(ModelUsage.agent_name == PROBE_AGENT)
        )
    logger.info("Cleanup: removed %d probe row(s)", result.rowcount or 0)


async def main() -> int:
    started = datetime.now(tz=UTC)
    try:
        logger.info("--- Live field layout ---")
        gate_live_field_layout()
        logger.info("--- Degrades, never crashes ---")
        gate_degrades_never_crashes()
        logger.info("--- Never breaks a decision ---")
        await gate_never_breaks_a_decision()
        logger.info("--- Wiring (row really lands) ---")
        await gate_wiring()
        logger.info("--- Concurrent attribution ---")
        await gate_concurrent_attribution()
        logger.info("--- Production path attribution ---")
        await gate_production_path_attributes()
        logger.info("--- Stored invariant ---")
        await gate_stored_invariant()
    except AssertionError as exc:
        logger.error("USAGE TRACKING TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("USAGE TRACKING TEST FAILED unexpectedly")
        return 1
    finally:
        try:
            await _cleanup()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cleanup issue (probe rows may remain): %s", exc)
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "USAGE TRACKING TEST PASSED (7 gates, %.1fs) — reasoning_tokens is "
        "read, persisted, and correctly attributed", elapsed,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging("INFO")
    raise SystemExit(asyncio.run(main()))

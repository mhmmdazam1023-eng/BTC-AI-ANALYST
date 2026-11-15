"""Record what each model call cost (C3, 2026-08-09).

The other half of "is this model worth it?". The ablation measures whether
the pillar scores carry information; this measures what producing them
costs. Neither number answers the question alone.

Design constraints, in priority order
-------------------------------------
1. **It must never break a decision.** A failed usage write is a lost
   measurement, not a lost run. Every path here swallows its own errors and
   logs them — an analytics table is not worth failing inference over.
2. **It must not slow the pipeline.** One small insert per call, against a
   local Postgres, on ~21 calls a day. Cheap enough to do inline; not worth
   the complexity of batching or a background queue.
3. **It must record what was actually returned, not what the schema
   promises.** The field layout below was read off a live response
   (2026-08-09), not copied from the OpenAI docs — ``cost`` and
   ``cost_details`` arrive in ``model_extra`` because the SDK has no field
   for them, and would be silently dropped by attribute access alone.

The trap this module exists to avoid
------------------------------------
``reasoning_tokens`` lives at ``usage.completion_tokens_details.reasoning_tokens``
— two levels deep, absent on non-reasoning models, and easy to miss. It is
also the DOMINANT term: measured over the first real 21-asset run,
**24,175 of 40,189 output tokens (60.2%) were reasoning**. Reading only
``completion_tokens`` understates the visible output by ~2.5x, and output
length is what drives both the free tier's wall-clock and any future bill.

(The initial probe suggested 83%, from a 41-token toy response. It was not
representative — a reminder that one sample is one sample, the same lesson
the wall-clock estimates in CLAUDE.md record.)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)


#: Which asset the current call belongs to.
#:
#: A ContextVar rather than a parameter threaded through every agent, for one
#: concrete reason: of the three ``ask()`` call sites, only
#: ``ConsolidatedAnalyst.analyze()`` knows the symbol. The five pillar agents
#: and the Master do not, and widening their signatures is precisely the change
#: that broke gate 10 of ``test_regime_phase8.py`` twice (its stubs implement
#: the old signature, so production passing a new kwarg blows up only in a real
#: run). Attribution is analytics; it must not be able to break inference.
#:
#: Safe under ``asyncio.gather``: each task captures a copy of the context at
#: creation, so the five concurrent pillar calls all read the value set before
#: the gather, and nothing they do leaks back out.
_usage_symbol: ContextVar[str | None] = ContextVar("usage_symbol", default=None)


@contextmanager
def usage_context(symbol: str | None) -> Iterator[None]:
    """Attribute every model call made inside this block to ``symbol``."""
    token = _usage_symbol.set(symbol)
    try:
        yield
    finally:
        _usage_symbol.reset(token)


def current_usage_symbol() -> str | None:
    """The symbol set by the enclosing :func:`usage_context`, if any."""
    return _usage_symbol.get()


def extract_usage(usage: Any) -> dict[str, Any]:
    """Flatten a provider ``usage`` object into plain columns.

    Pure and defensive: every field is optional, because a provider that
    omits one should cost us that one number and nothing else. Returns an
    empty dict when there is no usage at all.
    """
    if usage is None:
        return {}

    def _int(value: Any) -> int | None:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    extra = getattr(usage, "model_extra", None) or {}

    # Two levels down, and absent entirely on non-reasoning models.
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = _int(getattr(details, "reasoning_tokens", None)) if details else None

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached = _int(getattr(prompt_details, "cached_tokens", None)) if prompt_details else None

    # `cost` is NOT an SDK field — it arrives in model_extra. Attribute
    # access alone returns None and the number is lost.
    raw_cost = getattr(usage, "cost", None)
    if raw_cost is None:
        raw_cost = extra.get("cost")
    try:
        cost = None if raw_cost is None else Decimal(str(raw_cost))
    except (InvalidOperation, TypeError, ValueError):
        cost = None

    return {
        "prompt_tokens": _int(getattr(usage, "prompt_tokens", None)),
        "completion_tokens": _int(getattr(usage, "completion_tokens", None)),
        "reasoning_tokens": reasoning,
        "cached_tokens": cached,
        "total_tokens": _int(getattr(usage, "total_tokens", None)),
        "cost": cost,
    }


def format_usage(fields: dict[str, Any], duration: float | None = None) -> str:
    """One-line human summary, with reasoning shown as a share of output.

    The share is the point. "completion=1394" hides that 838 of those were
    reasoning; "completion=1394 (reasoning 838, 60%)" does not.
    """
    if not fields:
        return "usage unavailable"
    completion = fields.get("completion_tokens")
    reasoning = fields.get("reasoning_tokens")
    parts = [f"prompt={fields.get('prompt_tokens')}"]
    if completion is not None and reasoning is not None and completion > 0:
        parts.append(
            f"completion={completion} (reasoning {reasoning}, "
            f"{100.0 * reasoning / completion:.0f}%)"
        )
    else:
        parts.append(f"completion={completion}")
    if fields.get("cached_tokens"):
        parts.append(f"cached={fields['cached_tokens']}")
    if duration is not None:
        parts.append(f"{duration:.1f}s")
    return " ".join(parts)


async def record_usage(
    *,
    agent_name: str,
    model: str,
    usage: Any,
    symbol: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    """Persist one call's usage. NEVER raises — see constraint 1 above."""
    fields = extract_usage(usage)
    if not fields:
        return
    try:
        # Imported here, not at module scope: `src.agents` must not pull in
        # the DB layer just by being imported, and the agents package is
        # loaded by prompt-only tests that have no database.
        from src.db.engine import session_scope
        from src.db.models import ModelUsage

        async with session_scope() as session:
            session.add(
                ModelUsage(
                    timestamp=datetime.now(tz=UTC),
                    agent_name=agent_name,
                    model=model,
                    symbol=symbol,
                    duration_seconds=duration_seconds,
                    **fields,
                )
            )
    except Exception as exc:  # noqa: BLE001 - analytics must never break a run
        logger.warning(
            "[usage] could not record usage for '%s' (%s: %s) — the decision "
            "is unaffected, only this measurement is lost",
            agent_name, type(exc).__name__, exc,
        )

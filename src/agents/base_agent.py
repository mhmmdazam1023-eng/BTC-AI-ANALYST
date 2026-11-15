"""Base client for the AI reasoning engine (reasoning-model aware).

Current endpoint: **OpenRouter** with
``nvidia/nemotron-3-ultra-550b-a55b:free`` (swapped 2026-07-31) — a
REASONING model (user decision 2026-07-10). Two mandatory adaptations:

1. Every ``chat.completions.create()`` call sends
   ``extra_body={"reasoning": {"enabled": True}}``.
2. Reasoning models sometimes ignore ``response_format`` or wrap JSON in
   prose/markdown, so :meth:`_extract_json_object` is a **battle-hardened
   parser**: direct parse -> markdown-fence extraction -> balanced-JSON
   scan via ``json.JSONDecoder.raw_decode`` from every ``{`` — and only if
   all of that fails does the self-correction loop re-prompt the model.

Determinism (C1 cutover, 2026-08-05)
------------------------------------
Every call sends a fixed ``seed`` and ``temperature=0`` (both from
``Settings``, both verified present in the model's ``supported_parameters``
on OpenRouter). Before this, ``temperature`` was hardcoded to 0.2 and the
five pillar scores were measurably non-deterministic: 0/5 identical across
5 calls with byte-identical inputs, and the sentiment pillar's sampling
noise (sd 13.69) EXCEEDED its within-day cross-asset spread (12.61) —
making its cross-sectional Information Coefficient structurally zero.

The pillar scores are the only non-deterministic input in the pipeline;
everything downstream is arithmetic over them. So this is not tidiness, it
is a precondition for the ablation measuring signal rather than sampling.

Everything else follows the project rules: OpenAI SDK only, strict JSON
mode requested (with graceful downgrade if the endpoint rejects the
parameter), Pydantic validation of every response, tenacity exponential
backoff on 429/5xx/timeouts, specific logging on every failure path.
"""

from __future__ import annotations

import json
import logging
import re
from time import perf_counter
from typing import Any, ClassVar, TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.usage_tracker import (
    current_usage_symbol,
    extract_usage,
    format_usage,
    record_usage,
)
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class QwenConfigurationError(RuntimeError):
    """Raised when the AI client is misconfigured (e.g. missing API key)."""


class QwenResponseError(RuntimeError):
    """Raised when the model returns an empty/unparseable/invalid response."""


class EmptyResponseBodyError(QwenResponseError):
    """The HTTP body carried no JSON at all — not even a malformed object.

    Categorically different from "the model wrote bad JSON": there is nothing
    to re-prompt, because no payload ever arrived. OpenRouter pads long
    non-streaming requests with whitespace to keep the socket alive; when the
    upstream provider stalls, that padding is all that comes back.

    The OpenAI SDK has no handler for this — it hands the body to httpx's
    ``Response.json()``, a bare ``json.loads`` — so it surfaces as a raw
    ``json.JSONDecodeError``. Observed 2026-07-29 on ARB/USDT, whose retry
    budget had already been drained by three consecutive Poolside 429s.
    """


class EmptyChoicesError(QwenResponseError):
    """HTTP 200, valid JSON, but ``choices`` is empty — no completion at all.

    Sibling of :class:`EmptyResponseBodyError`, one layer further in: the body
    parsed fine, it just contains no answer. OpenRouter returns this shape when
    an upstream provider fails *after* the gateway has already committed to a
    200 — the payload carries an ``error`` object instead of a completion, and
    the OpenAI SDK models that as a ``ChatCompletion`` with an empty
    ``choices`` list.

    Retryable for the same reason: nothing about the prompt caused it, so
    re-prompting is pointless but a fresh request usually succeeds. Observed
    2026-07-31 on ETH, ONDO and TAO — 3 of 21 assets in one run, each failing
    ~21s in, while the other 18 went through untouched.
    """


#: Errors worth retrying — transient network / server / rate-limit issues.
#: Authentication or bad-request errors are NOT retried (fail fast).
_RETRYABLE_ERRORS = (
    openai.APIConnectionError,   # includes APITimeoutError
    openai.RateLimitError,       # HTTP 429 — exponential backoff prevents bans
    openai.InternalServerError,  # HTTP 5xx
    EmptyResponseBodyError,      # bodyless response — see the class docstring
    EmptyChoicesError,           # 200 with no completion — see above
)


class BaseAgent:
    """Async reasoning-model client with strict-JSON output via Pydantic.

    Sub-agents subclass this, set :attr:`agent_name` /
    :attr:`default_system_prompt`, and call :meth:`ask` with a Pydantic
    ``response_model`` describing the exact JSON they expect.
    """

    agent_name: str = "base"
    default_system_prompt: str = (
        "You are a rigorous financial analysis agent for Bitcoin. "
        "You always respond with a single valid JSON object and nothing else."
    )

    #: Mandatory for reasoning models on OpenRouter (project rule 2026-07-10).
    reasoning_extra_body: ClassVar[dict[str, Any]] = {"reasoning": {"enabled": True}}

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.system_prompt = system_prompt or self.default_system_prompt
        # None -> settings (0.0 since the C1 cutover). Previously hardcoded
        # to 0.2 here, which meant the value could not be changed without
        # editing code. An explicit argument still wins, for tests that
        # deliberately want sampling.
        self.temperature = (
            self.settings.qwen_temperature if temperature is None else temperature
        )
        self.seed = self.settings.qwen_seed
        #: Some endpoints/models reject response_format; downgraded at runtime.
        self._json_mode_supported = True

        api_key = self.settings.qwen_api_key.get_secret_value()
        if not api_key:
            raise QwenConfigurationError(
                "QWEN_API_KEY is not set. Fill in your OpenRouter "
                "(OpenAI-compatible) API key in .env."
            )

        # SDK-level retries are disabled: tenacity owns the retry policy so
        # backoff behaviour is uniform and observable in our logs.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.settings.qwen_base_url,
            timeout=self.settings.qwen_timeout_seconds,
            max_retries=0,
        )
        logger.debug(
            "Agent '%s' initialised (model=%s, base_url=%s, temperature=%s, "
            "seed=%s)",
            self.agent_name,
            self.settings.qwen_model,
            self.settings.qwen_base_url,
            self.temperature,
            self.seed if self.seed is not None else "DISABLED (sampling)",
        )

    # ------------------------------------------------------------------
    # Low-level call with retry + response validation
    # ------------------------------------------------------------------
    async def _chat_completion_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        usage_symbol: str | None = None,
    ) -> str:
        """Call the model and return the raw text content.

        Retries transient errors with exponential backoff (1s -> 2s -> 4s...,
        capped at 30s) up to ``settings.qwen_max_retries`` attempts. If the
        endpoint rejects ``response_format`` (some reasoning models do), the
        call is repeated once without it and the downgrade is remembered.
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.settings.qwen_max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(_RETRYABLE_ERRORS),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                started = perf_counter()
                create_kwargs: dict[str, Any] = {
                    "model": self.settings.qwen_model,
                    "messages": messages,
                    "temperature": (
                        self.temperature if temperature is None else temperature
                    ),
                    # MANDATORY for reasoning models (project rule):
                    "extra_body": self.reasoning_extra_body,
                }
                # Determinism (C1, 2026-08-05). Sent as a first-class SDK
                # parameter, not inside extra_body: `seed` is part of the
                # OpenAI schema, and burying it in extra_body would hide it
                # from the SDK's own validation. Omitted entirely when None
                # so disabling it in .env restores the old sampling exactly,
                # rather than sending `seed=null` and hoping the provider
                # reads that as "no seed".
                if self.seed is not None:
                    create_kwargs["seed"] = self.seed
                if self._json_mode_supported:
                    create_kwargs["response_format"] = {"type": "json_object"}

                try:
                    response = await self._client.chat.completions.create(**create_kwargs)
                except openai.BadRequestError as exc:
                    # Graceful downgrade when the endpoint rejects JSON mode;
                    # the robust parser + Pydantic still enforce strictness.
                    message = str(exc).lower()
                    if self._json_mode_supported and "response_format" in message:
                        logger.warning(
                            "Agent '%s': endpoint rejected response_format — "
                            "retrying without JSON mode (robust parser active).",
                            self.agent_name,
                        )
                        self._json_mode_supported = False
                        create_kwargs.pop("response_format", None)
                        response = await self._client.chat.completions.create(
                            **create_kwargs
                        )
                    else:
                        logger.error(
                            "Agent '%s': bad request to model endpoint: %s",
                            self.agent_name, exc,
                        )
                        raise
                except openai.AuthenticationError:
                    logger.error(
                        "Agent '%s': authentication failed — check QWEN_API_KEY.",
                        self.agent_name,
                    )
                    raise
                except json.JSONDecodeError as exc:
                    # Raised by httpx's bare Response.json() and passed
                    # through the SDK untouched, so it arrives here as a
                    # plain ValueError subclass that tenacity would NOT
                    # retry. Re-raise it as the retryable, self-describing
                    # EmptyResponseBodyError instead.
                    logger.warning(
                        "Agent '%s': response body carried no JSON (%s) — "
                        "upstream stall, will retry with backoff.",
                        self.agent_name, exc,
                    )
                    raise EmptyResponseBodyError(
                        f"Agent '{self.agent_name}': the endpoint returned a "
                        f"body with no JSON at all ({exc}). The upstream "
                        f"provider for '{self.settings.qwen_model}' stalled "
                        "and sent only keep-alive padding — re-prompting "
                        "cannot help, only a fresh request."
                    ) from exc
                except _RETRYABLE_ERRORS as exc:
                    logger.warning(
                        "Agent '%s': transient model error (%s), will retry with backoff.",
                        self.agent_name, type(exc).__name__,
                    )
                    raise

                # ---- Response structure validation ----
                if not response.choices:
                    # Dig out the provider's own error before giving up on it.
                    # OpenRouter puts an `error` object in the 200 body when an
                    # upstream provider fails; the SDK has no field for it, so
                    # it lands in `model_extra` and is silently lost unless we
                    # go looking. Without this the log said only "no choices",
                    # which names the symptom and hides the cause.
                    provider_error = getattr(response, "error", None)
                    if provider_error is None:
                        provider_error = (getattr(response, "model_extra", None) or {}).get(
                            "error"
                        )
                    detail = (
                        f" Provider reported: {provider_error}"
                        if provider_error
                        else " No error detail was included in the payload."
                    )
                    logger.warning(
                        "Agent '%s': response carried no choices — transient "
                        "upstream failure, will retry with backoff.%s",
                        self.agent_name, detail,
                    )
                    raise EmptyChoicesError(
                        f"Agent '{self.agent_name}': the endpoint returned a "
                        f"200 with an empty 'choices' list — the upstream "
                        f"provider for '{self.settings.qwen_model}' failed "
                        f"after the gateway had already responded.{detail}"
                    )
                choice = response.choices[0]
                content = choice.message.content
                if not content or not content.strip():
                    # Reasoning models occasionally leave `content` empty and
                    # put everything into the `reasoning` field — salvage it.
                    reasoning_text = getattr(choice.message, "reasoning", None)
                    if reasoning_text and str(reasoning_text).strip():
                        logger.warning(
                            "Agent '%s': empty content, salvaging from the "
                            "reasoning field (finish_reason=%s).",
                            self.agent_name, choice.finish_reason,
                        )
                        content = str(reasoning_text)
                    else:
                        raise QwenResponseError(
                            f"Agent '{self.agent_name}': model returned empty "
                            f"content (finish_reason={choice.finish_reason})."
                        )
                # C3 (2026-08-09). Was logger.debug — invisible at the
                # project's INFO level — and reasoning_tokens was never read
                # at all, despite being ~83% of the output on this model.
                # Now logged at INFO and persisted, so the cost half of
                # "is this model worth it?" survives the run.
                usage = getattr(response, "usage", None)
                if usage is not None:
                    elapsed = perf_counter() - started
                    logger.info(
                        "Agent '%s': completion ok (%s)",
                        self.agent_name,
                        format_usage(extract_usage(usage), elapsed),
                    )
                    await record_usage(
                        agent_name=self.agent_name,
                        model=self.settings.qwen_model,
                        usage=usage,
                        symbol=usage_symbol or current_usage_symbol(),
                        duration_seconds=elapsed,
                    )
                return content

        raise QwenResponseError(f"Agent '{self.agent_name}': retry loop exhausted.")

    # ------------------------------------------------------------------
    # Battle-hardened JSON extraction (reasoning models are messy)
    # ------------------------------------------------------------------
    @classmethod
    def _extract_json_object(cls, text: str) -> dict[str, Any] | None:
        """Extract the first JSON object from arbitrary model output.

        Strategy, in order:
        1. Direct ``json.loads`` on the (fence-stripped) text.
        2. Every ```` ```json ... ``` ```` / ```` ``` ... ``` ```` block.
        3. ``json.JSONDecoder.raw_decode`` starting at every ``{`` in the
           text — parses a balanced JSON value and ignores trailing prose,
           which covers outputs like ``Here is my analysis: {...} Hope it helps``.

        Returns ``None`` when no parseable JSON object exists.
        """
        if not text:
            return None
        candidates: list[str] = []

        stripped = cls._strip_code_fences(text)
        candidates.append(stripped)

        # All fenced blocks (with or without a language tag).
        for block in re.findall(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", text, re.DOTALL):
            candidates.append(block.strip())

        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Balanced-object scan over the raw text (max 50 start points).
        decoder = json.JSONDecoder()
        search_space = text
        starts = [i for i, ch in enumerate(search_space) if ch == "{"][:50]
        for start in starts:
            try:
                parsed, _ = decoder.raw_decode(search_space[start:])
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    # ------------------------------------------------------------------
    # High-level helper: prompt -> validated Pydantic object
    # ------------------------------------------------------------------
    async def ask(
        self,
        user_prompt: str,
        response_model: type[T],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_validation_retries: int = 2,
    ) -> T:
        """Send a prompt and return the response parsed into ``response_model``.

        The JSON Schema is embedded in the system prompt. Parsing order:
        robust extraction (:meth:`_extract_json_object`) -> Pydantic
        validation -> up to ``max_validation_retries`` self-correction
        rounds where the errors are fed back to the model.
        """
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        full_system_prompt = (
            f"{system_prompt or self.system_prompt}\n\n"
            "Your final answer MUST be a single JSON object that conforms "
            f"exactly to this JSON Schema:\n{schema}\n"
            "Do not wrap the JSON in markdown fences or add any text around it."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        for validation_attempt in range(1, max_validation_retries + 2):
            raw = await self._chat_completion_raw(messages, temperature=temperature)

            extracted = self._extract_json_object(raw)
            if extracted is None:
                last_error = QwenResponseError("no JSON object found in model output")
                logger.warning(
                    "Agent '%s': no JSON found in output (attempt %d/%d); "
                    "first 200 chars: %r",
                    self.agent_name, validation_attempt,
                    max_validation_retries + 1, raw[:200],
                )
                messages.append({"role": "assistant", "content": raw[:4000]})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply contained no parseable JSON "
                            "object. Respond again with ONLY the JSON object "
                            "matching the schema — no prose, no markdown."
                        ),
                    }
                )
                continue

            try:
                parsed = response_model.model_validate(extracted)
                logger.info(
                    "Agent '%s': response validated against %s (attempt %d)",
                    self.agent_name, response_model.__name__, validation_attempt,
                )
                return parsed
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "Agent '%s': response failed %s validation (attempt %d/%d): %s",
                    self.agent_name, response_model.__name__,
                    validation_attempt, max_validation_retries + 1, exc.errors(),
                )
                messages.append(
                    {"role": "assistant", "content": json.dumps(extracted)[:4000]}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous JSON failed schema validation with "
                            f"these errors:\n{exc}\n"
                            "Return ONLY a corrected JSON object that conforms "
                            "to the schema."
                        ),
                    }
                )

        raise QwenResponseError(
            f"Agent '{self.agent_name}': response never validated against "
            f"{response_model.__name__} after {max_validation_retries + 1} attempts."
        ) from last_error

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_code_fences(content: str) -> str:
        """Defensively strip markdown code fences wrapping the whole payload."""
        text = content.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        return text.strip()

    async def close(self) -> None:
        """Release the underlying HTTP resources (call at shutdown)."""
        await self._client.close()
        logger.debug("Agent '%s': client closed", self.agent_name)

"""Centralised application configuration.

All secrets and environment-dependent values are loaded from environment
variables (or a local ``.env`` file) via ``pydantic-settings``. Nothing
sensitive is ever hard-coded. Access the singleton through
:func:`get_settings` so the ``.env`` file is only parsed once.

Example
-------
>>> from src.config import get_settings
>>> settings = get_settings()
>>> settings.async_database_url
'postgresql+asyncpg://btc_analyst:***@localhost:5432/bitcoin_ai'
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Absolute project root so `.env` loads regardless of the caller's working
#: directory (critical for Windows Task Scheduler, which sets its own CWD).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings, populated from environment variables / `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # PostgreSQL / TimescaleDB
    # ------------------------------------------------------------------
    postgres_user: str = "btc_analyst"
    postgres_password: SecretStr = SecretStr("change_me")
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "bitcoin_ai"

    # ------------------------------------------------------------------
    # AI reasoning endpoint (OpenAI-compatible; currently OpenRouter with
    # nvidia/nemotron-3-ultra-550b-a55b:free — a reasoning model, per user
    # decision 2026-07-10; model swapped 2026-07-18 (laguna-m.1 -> laguna-xs)
    # and again 2026-07-31 (laguna-xs -> Nemotron 3 Ultra). Field names keep
    # the legacy QWEN_ prefix; QWEN_MODEL_NAME is an alias of QWEN_MODEL.
    #
    # NOTE on strict JSON: neither this model nor its predecessor advertises
    # `response_format` support on OpenRouter (verified against
    # /api/v1/models, 2026-07-31). BaseAgent already handles that — it
    # downgrades gracefully and relies on the battle-hardened parser plus
    # Pydantic validation, which is the path this system has in fact been
    # running on all along.
    # ------------------------------------------------------------------
    qwen_base_url: str = "https://openrouter.ai/api/v1"
    qwen_api_key: SecretStr = SecretStr("")
    qwen_model: str = Field(
        default="nvidia/nemotron-3-ultra-550b-a55b:free",
        validation_alias=AliasChoices("QWEN_MODEL", "QWEN_MODEL_NAME"),
    )
    qwen_timeout_seconds: float = Field(default=120.0, gt=0)
    qwen_max_retries: int = Field(default=4, ge=1, le=10)

    # --- Determinism (C1, 2026-08-05) ---------------------------------
    # `seed` and `temperature` are BOTH in this model's
    # `supported_parameters` on /api/v1/models (verified 2026-08-05):
    #   ['include_reasoning','max_tokens','reasoning','reasoning_effort',
    #    'seed','temperature','tool_choice','tools','top_p']
    # Note what is NOT there: `response_format` — consistent with the
    # long-standing downgrade path documented above. Same verification
    # discipline: never assume a parameter is honoured, check the listing.
    #
    # WHY this exists: at temperature=0.2 the five pillar scores were
    # measured moving on IDENTICAL inputs — 0/5 identical across 5 calls
    # (scripts/test_determinism.py, BTC 2026-08-04). Sentiment's sampling
    # noise (sd 13.69) EXCEEDED its within-day cross-asset spread (12.61),
    # making its cross-sectional IC structurally zero. macro, the
    # highest-weighted pillar in the current regime, was attenuated 25%.
    #
    # A fixed seed is NOT enough on its own: providers may ignore it, and
    # the same seed with temperature>0 still samples. Both are set, and
    # C6 re-runs the determinism test to prove the provider honours it —
    # a provider that silently drops `seed` looks identical to one that
    # respects it, and only that test tells them apart.
    qwen_seed: int | None = Field(
        default=20260808,
        description=(
            "Fixed sampling seed for reproducible pillar scores. Set to "
            "null/empty in .env to disable (restores non-deterministic "
            "sampling). The value itself is arbitrary — the cutover date — "
            "what matters is that it never changes, because changing it "
            "fragments the IC measurement series exactly like a prompt edit."
        ),
    )
    qwen_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature. 0.0 since the C1 cutover (2026-08-05); "
            "was 0.2. Raising it re-introduces the sampling noise that made "
            "the sentiment pillar unmeasurable — do not raise it without "
            "re-running scripts/test_determinism.py and accepting the "
            "attenuation."
        ),
    )

    # ------------------------------------------------------------------
    # External data APIs (used from Phase 2 onwards)
    # ------------------------------------------------------------------
    fred_api_key: SecretStr = SecretStr("")
    glassnode_api_key: SecretStr = SecretStr("")
    #: Public market-data host for Binance spot. Defaults to Binance's
    #: official public-data mirror because api.binance.com is ISP-blocked
    #: in Indonesia. Set to "" in .env to use ccxt's default host.
    binance_public_api_url: str = "https://data-api.binance.vision/api/v3"

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    log_level: str = "INFO"
    environment: str = "development"

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @property
    def async_database_url(self) -> str:
        """SQLAlchemy URL for the asyncpg driver (runtime)."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:"
            f"{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """SQLAlchemy URL for the psycopg2 driver (utility scripts / migrations)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:"
            f"{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()

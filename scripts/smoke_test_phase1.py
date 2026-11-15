"""Phase 1 smoke test: syntax, imports, and offline behavioural checks.

Run from the project root::

    python scripts/smoke_test_phase1.py

Requires no network and no running database.
"""

from __future__ import annotations

import compileall
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    # 1) Syntax check on the whole tree
    ok = compileall.compile_dir("src", quiet=1) and compileall.compile_file(
        "scripts/init_db.py", quiet=1
    )
    assert ok, "syntax errors found"

    # 2) Import smoke test
    from src.agents.base_agent import BaseAgent, QwenConfigurationError
    from src.config import get_settings
    from src.db import models

    # 3) Behavioural checks (offline)
    settings = get_settings()
    assert settings.async_database_url.startswith("postgresql+asyncpg://")
    assert settings.sync_database_url.startswith("postgresql+psycopg2://")
    assert len(models.HYPERTABLES) == 5
    assert set(models.Base.metadata.tables) == {t for t, _ in models.HYPERTABLES}

    # BaseAgent must fail fast with a clear error when QWEN_API_KEY is missing.
    if not settings.qwen_api_key.get_secret_value():
        try:
            BaseAgent()
            raise AssertionError("expected QwenConfigurationError")
        except QwenConfigurationError:
            pass

    print(
        "SMOKE TEST OK | python",
        sys.version.split()[0],
        "| tables:",
        sorted(models.Base.metadata.tables),
    )


if __name__ == "__main__":
    main()

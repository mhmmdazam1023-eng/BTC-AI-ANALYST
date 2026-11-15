"""One line: are there enough independent samples to trust an edge yet?

Reuses the SAME functions production already calls — never a second copy of
the sample-size math. Two thresholds:

* **IC** — ``measurement_snapshots.ic_n_days`` (written by Stage A6 via
  :func:`~src.evaluation.pillar_ablation.information_coefficient`) against
  :func:`~src.evaluation.pillar_ablation.days_needed_for`.
* **Kelly** — per-asset, per-symbol, using
  :func:`~src.evaluation.sample_size.effective_sample_size` on that asset's
  decisive BUY evaluations at :data:`SIZING_HORIZON_DAYS`, against
  :data:`MIN_WIN_RATE_SAMPLES`. This is the SAME calculation
  ``load_win_rate_for()`` makes internally — not a reimplementation, a
  direct call to the same overlap-correction utility it calls.

Why not just read ``position_recommendations.win_rate_samples``
-----------------------------------------------------------------
That column is NOT what it looks like once the null-baseline fallback fires.
``position_pipeline.py`` overwrites ``win_rate_source`` to
``"default_null_baseline"`` when a per-asset estimate is unavailable, but
does NOT overwrite ``win_rate_samples`` alongside it — the stored number is
left over from ``ForecastEvaluation`` (the BTC-only Monte Carlo table),
completely unrelated to the asset it is stored against. Caught 2026-08-09
while building this script: AAVE/USDT showed "17 samples" that grew ~1/day
in lockstep with every OTHER asset's row, which is the signature of a global
counter, not a per-asset one. Reading it directly would have shipped a
readiness line that was confidently wrong.

Meant to be tailed from the scheduled-run log, not read as a report: one
line per threshold, every run, so the day it crosses reads as loud as
every other line around it.

Usage::

    python scripts/check_sample_readiness.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import CoinRegistry, DecisionEvaluation, DecisionVerdict  # noqa: E402
from src.db.models import MeasurementSnapshot  # noqa: E402
from src.evaluation.decision_metrics import SIZING_HORIZON_DAYS  # noqa: E402
from src.evaluation.pillar_ablation import days_needed_for  # noqa: E402
from src.evaluation.sample_size import effective_sample_size  # noqa: E402
from src.forecasting.kelly_criterion import MIN_WIN_RATE_SAMPLES  # noqa: E402

#: The IC this project has been treating as the smallest one worth acting on
#: (CLAUDE.md: "days_needed_for(0.05)"). Not a new decision — restating the
#: same target so the printed line is self-contained.
TARGET_IC = 0.05


async def best_kelly_readiness(session) -> tuple[float, str | None]:
    """(effective independent decisive BUY evaluations, symbol) — the max
    across all active assets, computed the same way load_win_rate_for() does.
    """
    symbols = (
        await session.execute(
            select(CoinRegistry.symbol).where(CoinRegistry.is_active)
        )
    ).scalars().all()

    best_effective, best_symbol = 0.0, None
    for symbol in symbols:
        rows = (
            await session.execute(
                select(DecisionEvaluation.decision_timestamp).where(
                    DecisionEvaluation.symbol == symbol,
                    DecisionEvaluation.horizon_days == SIZING_HORIZON_DAYS,
                    DecisionEvaluation.action == "BUY",
                    DecisionEvaluation.verdict != DecisionVerdict.NEUTRAL,
                )
            )
        ).scalars().all()
        if not rows:
            continue
        effective = effective_sample_size(list(rows), SIZING_HORIZON_DAYS)
        if effective > best_effective:
            best_effective, best_symbol = effective, symbol
    return best_effective, best_symbol


async def main() -> int:
    needed_days = days_needed_for(TARGET_IC)  # pure function, no I/O

    async with session_scope() as session:
        best_ic_days = (
            await session.execute(select(func.max(MeasurementSnapshot.ic_n_days)))
        ).scalar_one_or_none() or 0
        best_kelly_effective, best_kelly_symbol = await best_kelly_readiness(session)

    ic_ready = needed_days is not None and best_ic_days >= needed_days
    kelly_ready = best_kelly_effective >= MIN_WIN_RATE_SAMPLES

    ic_part = (
        f"IC: {best_ic_days}/{needed_days} independent day(s) "
        f"{'>= threshold — READY' if ic_ready else 'so far'}"
    )
    kelly_part = (
        f"Kelly: {best_kelly_effective:.1f}/{MIN_WIN_RATE_SAMPLES} effective "
        f"decisive BUY obs at +{SIZING_HORIZON_DAYS}d "
        f"(best: {best_kelly_symbol or '-'}) "
        f"{'>= threshold — READY' if kelly_ready else 'so far'}"
    )

    print(f"SAMPLE READINESS | {ic_part} | {kelly_part}")
    if ic_ready or kelly_ready:
        print(
            "SAMPLE READINESS | >>> At least one threshold has been crossed. "
            "Read CLAUDE.md \"BACKLOG PENGUKURAN\" before treating any edge "
            "as real — crossing the sample-size floor is necessary, not "
            "sufficient (z >= 2 after overlap correction still applies). <<<"
        )

    await dispose_engine()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

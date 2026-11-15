"""One daily row answering "is this working?" (Phase 9, 2026-07-31).

Three measurement layers landed in one day — the null model, the baseline
comparator and the pillar ablation — and all three were report-only scripts
somebody had to remember to run. A measurement nobody looks at cannot change
a decision, so this composes all three into a single row per (day, horizon)
that ``run_daily`` writes automatically.

**The point is the trend, not the day.** Any single day's ``edge_vs_null`` on
thirteen decisive evaluations is noise. Whether it drifts toward zero or away
from it over months is the question, and that cannot be reconstructed later
from a log nobody kept.

Composition, not reimplementation
---------------------------------
Every number here comes from the module that owns it:
:meth:`DecisionEvaluator.compute_scoreboard` for the system and its null,
:func:`compare_baselines` for the dumb strategies,
:func:`information_coefficient` for the panel IC. Nothing is recomputed
"equivalently" — that is how two parts of one system start disagreeing.

Missing stays NULL
------------------
An IC that cannot be computed yet (no day has enough assets with matured
returns) is NULL, never 0.0. The same NULL-vs-0 discipline as the Part 4
feature contract: zero IC is a measurement, absent IC is not.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.engine import session_scope
from src.db.models import MeasurementSnapshot
from src.evaluation.baseline_strategies import compare_baselines
from src.evaluation.decision_evaluator import DECISION_HORIZONS, DecisionEvaluator
from src.evaluation.pillar_ablation import (
    PANEL_MEAN,
    information_coefficient,
    load_ablation_frame,
)

logger = logging.getLogger(__name__)


async def load_decision_grid(
    session, horizon: int
) -> set[tuple[datetime, str]]:
    """The exact ``(decision day, symbol)`` pairs the system was scored on.

    Read from ``decision_evaluations`` rather than ``decisions``: a decision
    with no matured candle was never scored, so including it would hand the
    baseline days the system was not measured on.

    ALL verdicts, including NEUTRAL. Each strategy then drops its own
    NEUTRALs by its own rule — which is what makes the comparison fair.
    Pre-filtering to the system's decisive rows would instead condition the
    baseline on the system's outcome: those rows are decisive *because* the
    price left the band, and always_hold is wrong exactly there.
    """
    from sqlalchemy import select

    from src.db.models import DecisionEvaluation

    rows = (
        await session.execute(
            select(
                DecisionEvaluation.decision_timestamp,
                DecisionEvaluation.symbol,
            ).where(DecisionEvaluation.horizon_days == horizon)
        )
    ).all()
    return {(day, symbol) for day, symbol in rows}


async def build_snapshot(
    horizon: int, *, now: datetime | None = None
) -> dict | None:
    """Every headline number for one horizon, or ``None`` if nothing to say."""
    reference = now or datetime.now(tz=UTC)
    board = await DecisionEvaluator.compute_scoreboard()
    row = board.get(horizon)

    async with session_scope() as session:
        # HEAD-TO-HEAD, NOT FULL HISTORY (fixed 2026-08-09, C5).
        #
        # This used to call compare_baselines() with no grid at all, so
        # `baseline_best_hit_rate` came from every scoreable asset-day ever
        # (~34.7k rows) while `system_hit_rate` came from the handful the
        # system actually decided and that matured. Subtracting one from the
        # other produced `edge_vs_baseline` between two different
        # populations — the exact defect the rest of this layer exists to
        # prevent, sitting inside the summary row meant to be trusted.
        # Measured on 2026-08-08 at +1d: it reported +6.4pp where the
        # same-grid answer is +4.0pp.
        #
        # `pairs` is the exact (day, symbol) set, not days x symbols: the
        # cross-product over-includes on any day the system did not cover
        # every asset, which is most of the early history.
        pairs = await load_decision_grid(session, horizon)
        baselines = await compare_baselines(
            session,
            horizon,
            days=sorted({day for day, _ in pairs}) or None,
            symbols=sorted({sym for _, sym in pairs}) or None,
            pairs=pairs or None,
            now=reference,
        )
        frame = await load_ablation_frame(session, horizon, now=reference)

    # Strongest baseline that day. Usually always_hold — HOLD wins ~70% from
    # band geometry alone, and that is the bar, not buy_and_hold's.
    best_name: str | None = None
    best_rate: float | None = None
    for name, result in baselines.items():
        if result.hit_rate is None:
            continue
        if best_rate is None or result.hit_rate > best_rate:
            best_name, best_rate = name, result.hit_rate

    ic = information_coefficient(frame, PANEL_MEAN, horizon) if not frame.empty else None

    system_hit = row.hit_rate if row else None
    payload = {
        "timestamp": reference.replace(
            hour=0, minute=0, second=0, microsecond=0
        ),
        "horizon_days": horizon,
        "system_hit_rate": system_hit,
        "system_decisive": row.decisive if row else 0,
        "system_effective_n": row.effective_n if row else None,
        "null_hit_rate": row.null_hit_rate if row else None,
        "edge_vs_null": row.edge_vs_null if row else None,
        "baseline_best_name": best_name,
        "baseline_best_hit_rate": best_rate,
        "edge_vs_baseline": (
            None
            if system_hit is None or best_rate is None
            else system_hit - best_rate
        ),
        "ic_panel_mean": ic.mean_ic if ic else None,
        "ic_n_days": ic.n_days if ic and ic.mean_ic is not None else None,
    }
    if payload["system_decisive"] == 0 and best_rate is None:
        # Nothing has matured and no baseline could be scored either. Writing
        # a row of NULLs would only add noise to the trend.
        return None
    return payload


async def save_snapshots(
    horizons: tuple[int, ...] = DECISION_HORIZONS,
    *,
    now: datetime | None = None,
) -> int:
    """Compute and upsert one row per horizon. Returns rows written."""
    payloads = []
    for horizon in horizons:
        try:
            payload = await build_snapshot(horizon, now=now)
        except Exception:  # noqa: BLE001 - one horizon must not sink the rest
            logger.exception(
                "[snapshot] +%dd failed — continuing with the other horizons",
                horizon,
            )
            continue
        if payload is not None:
            payloads.append(payload)

    if not payloads:
        logger.info("[snapshot] nothing measurable yet — no rows written")
        return 0

    async with session_scope() as session:
        statement = pg_insert(MeasurementSnapshot.__table__).values(payloads)
        # Re-running the same day must refresh, not duplicate or refuse: the
        # daily job can legitimately run twice, and the later run has strictly
        # more matured evaluations behind it.
        update = {
            key: statement.excluded[key]
            for key in payloads[0]
            if key not in ("timestamp", "horizon_days")
        }
        update["computed_at"] = datetime.now(tz=UTC)
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=["timestamp", "horizon_days"], set_=update
            )
        )

    for payload in payloads:
        logger.info(
            "[snapshot] +%2dd: hit %s | null %s | edge %s | best baseline %s %s "
            "| vs baseline %s | IC(panel) %s",
            payload["horizon_days"],
            _pct(payload["system_hit_rate"]),
            _pct(payload["null_hit_rate"]),
            _pp(payload["edge_vs_null"]),
            payload["baseline_best_name"] or "n/a",
            _pct(payload["baseline_best_hit_rate"]),
            _pp(payload["edge_vs_baseline"]),
            "n/a" if payload["ic_panel_mean"] is None
            else f"{payload['ic_panel_mean']:+.3f} ({payload['ic_n_days']}d)",
        )
    return len(payloads)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _pp(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}pp"

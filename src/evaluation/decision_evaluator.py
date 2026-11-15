"""DecisionEvaluator: was the BUY/HOLD/SELL right? (Phase 9)

Covers ALL 21 assets. Nothing here is Bitcoin-locked — ``decisions`` has
carried a ``symbol`` since Phase 8 Part 6 and ``crypto_prices`` always has —
which is why this closes the gap the Monte Carlo layer structurally cannot.

The three design decisions (user, 2026-07-31)
---------------------------------------------
1. **Three horizons: 1, 7 and 30 days.** One decision, three verdicts. A BUY
   can be wrong tomorrow and right in a month, and *which horizon the system
   has edge at* is the thing worth measuring.
2. **Correctness is a volatility-scaled band per asset**, not raw direction::

       band_pct = realized_vol_30d * sqrt(horizon_days / 365) * 100

   ``realized_vol_30d`` is ANNUALISED (same convention Part 4 had to fix in
   ``trend_strength``), so the square-root-of-time factor puts it on the
   horizon's own scale. A FIXED band was rejected: across assets whose
   volatility differs by an order of magnitude, a flat 2% would make the
   noisiest assets look the most "accurate".
3. **HOLD is judged, not excluded.** Correct when the price stayed inside the
   band — "nothing worth acting on happened" — wrong when a big move was
   missed in either direction. Symmetric and refutable: HOLD cannot always
   win, which the "HOLD is right whenever price falls" alternative would have
   allowed.

Correctness rules
-----------------
====== ============================ =========================== ============
Action CORRECT                      WRONG                       else
====== ============================ =========================== ============
BUY    ``return > +band``           ``return < -band``          NEUTRAL
SELL   ``return < -band``           ``return > +band``          NEUTRAL
HOLD   ``|return| <= band``         ``|return| > band``         —
====== ============================ =========================== ============

NEUTRAL is not a soft failure. It marks "the market did not move enough for
this call to be testable", and it is excluded from hit-rate on BOTH sides of
the ratio — the same treatment ``panel_agreement`` gives a pillar scoring 0.

Entry lag: the trade is entered at close(T+1), NOT close(T)
-----------------------------------------------------------
Fixed 2026-07-31 (user decision, "Opsi A"). Until then the return was
measured from ``close(decision_day)``, and that price did not exist when the
decision was written.

The decision for day T is produced from ``daily_features[T]``, which is built
from day T's candle **while that candle is still forming** — the daily path
deliberately depends on the running candle (see the ``drop_unclosed``
discussion in Part 8). Measured against the live table, decisions were being
written 5 to 20 hours before the close they were then scored from: the
2026-07-31 run wrote at 13:07-14:14 UTC for a candle that only settled at
2026-08-01 00:00 UTC, 9.8 hours later.

So the old entry price contained hours of price action that had not happened
yet, on the SAME bar the features were read from. That is not neutral noise —
it is an error correlated with whatever the model just keyed on. Sized
against BTC's own +1d band it was worth ~64% of the band being tested, which
made the +1d verdict unreadable in either direction.

:data:`ENTRY_LAG_DAYS` moves the entry to the first candle the decision could
not have seen any part of::

    entry_date  = decision_timestamp + ENTRY_LAG_DAYS
    target_date = entry_date + horizon_days
    return_pct  = close(target_date) / close(entry_date) - 1

The band is still read at the DECISION date — see :meth:`_load_vols`. That
asymmetry is deliberate and is the whole point: the *entry* must be something
the trader could still act on, while the *band* must describe what was
knowable when the call was made.

Finalization
------------
Reuses :data:`~src.forecasting.evaluator.FINALIZATION_BUFFER_DAYS` rather
than inventing a second definition of "settled", and additionally requires
the EXACT daily candle to exist for both endpoints. A missing candle
postpones the evaluation to a later run; it is never replaced by a nearby
day's price. That substitution is the precise bug fixed on 2026-07-25.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import and_, or_, select, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import (
    CryptoPrice,
    DailyFeature,
    Decision,
    DecisionEvaluation,
    DecisionVerdict,
)
from src.forecasting.evaluator import FINALIZATION_BUFFER_DAYS, _finalization_cutoff

logger = logging.getLogger(__name__)

#: Horizons in CALENDAR DAYS (user decision 2026-07-31). Named ``_DAYS`` and
#: kept distinct from the intraday ``HORIZONS_BARS`` so the two unit systems
#: can never be confused at a call site.
DECISION_HORIZONS: tuple[int, ...] = (1, 7, 30)

#: Trading days per year — matches the FeatureEngine's annualisation of
#: ``realized_vol_30d``. Changing one without the other silently rescales
#: every band.
DAYS_PER_YEAR = 365

#: Daily timeframe. Every ``crypto_prices`` read MUST filter this (multi-asset
#: rule 6): BTC now holds both '1d' and '4h' rows.
DAILY_TIMEFRAME = "1d"

#: Candles between the decision day and the day the position is entered.
#:
#: MUST stay >= 1. The decision's own features are computed from day T's
#: candle while it is still forming, so entering at ``close(T)`` means
#: entering on a bar the decision partly saw. 1 is the smallest lag that
#: guarantees the entry bar opened after the decision existed.
#:
#: Raising it above 1 is a legitimate conservatism knob (it models a trader
#: who acts a day late), but it costs one extra calendar day of maturity per
#: evaluation and shrinks an already thin sample. Do not raise it without a
#: reason written down here.
ENTRY_LAG_DAYS = 1


def band_pct_for(realized_vol_30d: float, horizon_days: int) -> float:
    """Volatility band in PERCENT for one asset at one horizon.

    ``realized_vol_30d`` is a fraction (0.315 = 31.5% annualised), so the
    result is multiplied by 100 to match ``return_pct``.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    return realized_vol_30d * math.sqrt(horizon_days / DAYS_PER_YEAR) * 100.0


def classify_verdict(action: str, return_pct: float, band_pct: float) -> DecisionVerdict:
    """Apply the correctness rules. Pure function — the heart of Phase 9."""
    normalised = (action or "").strip().upper()
    if normalised == "BUY":
        if return_pct > band_pct:
            return DecisionVerdict.CORRECT
        if return_pct < -band_pct:
            return DecisionVerdict.WRONG
        return DecisionVerdict.NEUTRAL
    if normalised == "SELL":
        if return_pct < -band_pct:
            return DecisionVerdict.CORRECT
        if return_pct > band_pct:
            return DecisionVerdict.WRONG
        return DecisionVerdict.NEUTRAL
    if normalised == "HOLD":
        return (
            DecisionVerdict.CORRECT
            if abs(return_pct) <= band_pct
            else DecisionVerdict.WRONG
        )
    raise ValueError(f"unknown action {action!r} — expected BUY/SELL/HOLD")


@dataclass(slots=True)
class EvaluationRun:
    """What one pass over the pending decisions actually did."""

    evaluated: int = 0
    postponed_price: int = 0
    postponed_band: int = 0
    unfinalized: int = 0

    def summary(self) -> str:
        return (
            f"{self.evaluated} evaluated"
            f" | {self.postponed_price} postponed (candle missing)"
            f" | {self.postponed_band} postponed (no volatility band)"
            f" | {self.unfinalized} not yet matured"
        )


@dataclass(slots=True, frozen=True)
class ScoreboardRow:
    """The system's own numbers at one horizon, beside its baseline.

    Returned rather than logged so the daily snapshot and the log line are
    computed ONCE, by one implementation. Two callers deriving a blended null
    separately is exactly how they drift apart.
    """

    horizon_days: int
    total: int
    decisive: int
    correct: int
    neutral: int
    #: Rows that had a null baseline and could therefore be compared at all.
    comparable: int
    null_hit_rate: float | None
    effective_n: float

    @property
    def hit_rate(self) -> float | None:
        return self.correct / self.decisive if self.decisive else None

    @property
    def edge_vs_null(self) -> float | None:
        if self.hit_rate is None or self.null_hit_rate is None:
            return None
        return self.hit_rate - self.null_hit_rate


@dataclass(slots=True)
class _PendingEvaluation:
    """One (decision, horizon) awaiting a verdict, with both its dates.

    Carrying ``entry`` and ``target`` explicitly rather than deriving them at
    each use site is the point: the two dates are now DIFFERENT offsets from
    ``decision.timestamp`` (+``ENTRY_LAG_DAYS`` and +``ENTRY_LAG_DAYS+horizon``),
    and a bare tuple made it easy to reach for the wrong one.
    """

    decision: Decision
    horizon: int
    entry: datetime
    target: datetime


class DecisionEvaluator:
    """Scores matured decisions across every asset and horizon."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    async def evaluate_matured(self, now: datetime | None = None) -> int:
        """Evaluate every matured, unevaluated (decision, horizon). Returns count."""
        reference = now or datetime.now(tz=UTC)
        cutoff = _finalization_cutoff(reference)
        logger.info(
            "[decision-eval] finalization cutoff %s (>= %d full day(s) old); "
            "horizons %s; entry lag +%dd (entry is close(T+%d), never close(T))",
            cutoff.date(), FINALIZATION_BUFFER_DAYS,
            ", ".join(f"+{h}d" for h in DECISION_HORIZONS),
            ENTRY_LAG_DAYS, ENTRY_LAG_DAYS,
        )

        run = EvaluationRun()
        async with session_scope() as session:
            # Only decisions old enough for the SHORTEST horizon to have
            # matured are worth loading at all. The entry lag pushes that
            # boundary a further ENTRY_LAG_DAYS back: a decision now needs
            # ENTRY_LAG_DAYS + horizon days of settled candles, not horizon.
            #
            # STRICTLY earlier than, matching the `target >= cutoff` test
            # below. With `<=` a decision landing exactly on this date was
            # loaded every run only to be rejected as unmatured — harmless,
            # but it made the load filter and the maturity predicate disagree,
            # and `unfinalized` counted rows that were never candidates.
            newest_evaluable = cutoff - timedelta(
                days=min(DECISION_HORIZONS) + ENTRY_LAG_DAYS
            )
            decisions = (
                (
                    await session.execute(
                        select(Decision)
                        .where(Decision.timestamp < newest_evaluable)
                        .order_by(Decision.timestamp, Decision.symbol)
                    )
                )
                .scalars()
                .all()
            )
            if not decisions:
                logger.info("[decision-eval] no decisions old enough to evaluate yet")
                return 0

            # Everything already scored, so re-runs do no work.
            done = {
                (row.decision_timestamp, row.symbol, row.horizon_days)
                for row in (
                    await session.execute(
                        select(
                            DecisionEvaluation.decision_timestamp,
                            DecisionEvaluation.symbol,
                            DecisionEvaluation.horizon_days,
                        )
                    )
                ).all()
            }

            # Build the work list first, then fetch prices/bands in BULK.
            # A per-row lookup would be 21 assets x 3 horizons x N days of
            # history in separate round trips.
            pending: list[_PendingEvaluation] = []
            for decision in decisions:
                # The bar the trade is entered on. First candle the decision
                # could not have seen any part of — see the module docstring.
                entry = decision.timestamp + timedelta(days=ENTRY_LAG_DAYS)
                for horizon in DECISION_HORIZONS:
                    key = (decision.timestamp, decision.symbol, horizon)
                    if key in done:
                        continue
                    target = entry + timedelta(days=horizon)
                    if target >= cutoff:
                        run.unfinalized += 1
                        continue
                    pending.append(
                        _PendingEvaluation(decision, horizon, entry, target)
                    )

            if not pending:
                logger.info(
                    "[decision-eval] nothing new to score (%d not yet matured)",
                    run.unfinalized,
                )
                # Still print where the system stands. The scoreboard is a
                # status report, not a report of work done — and on most days
                # nothing matures, so returning early here meant the hit rate
                # and its baseline were invisible exactly when nobody was
                # looking at fresh output.
                await self._log_scoreboard()
                return 0

            prices = await self._load_prices(session, pending)
            bands = await self._load_vols(session, pending)

            payload: list[dict] = []
            for item in pending:
                decision = item.decision
                # base is the ENTRY candle's close, not the decision day's.
                base = prices.get((decision.symbol, item.entry))
                actual = prices.get((decision.symbol, item.target))
                if base is None or actual is None or base <= 0:
                    run.postponed_price += 1
                    continue

                realized_vol = bands.get((decision.symbol, decision.timestamp))
                if realized_vol is None or realized_vol <= 0:
                    run.postponed_band += 1
                    continue

                return_pct = float((actual - base) / base * 100)
                band = band_pct_for(realized_vol, item.horizon)
                verdict = classify_verdict(decision.action, return_pct, band)

                payload.append(
                    {
                        "decision_timestamp": decision.timestamp,
                        "symbol": decision.symbol,
                        "horizon_days": item.horizon,
                        "entry_date": item.entry,
                        "target_date": item.target,
                        "action": decision.action,
                        "confidence_score": decision.confidence_score,
                        "base_price": base,
                        "actual_price": actual,
                        "return_pct": Decimal(str(round(return_pct, 8))),
                        "band_pct": Decimal(str(round(band, 8))),
                        "verdict": verdict,
                    }
                )

            if payload:
                statement = (
                    pg_insert(DecisionEvaluation.__table__)
                    .values(payload)
                    .on_conflict_do_nothing(constraint="uq_decision_evaluation")
                )
                await session.execute(statement)
                run.evaluated = len(payload)

        if run.postponed_band:
            logger.warning(
                "[decision-eval] %d evaluation(s) postponed: realized_vol_30d "
                "missing for the decision day, so no band can be computed",
                run.postponed_band,
            )
        if run.postponed_price:
            logger.warning(
                "[decision-eval] %d evaluation(s) postponed: exact daily candle "
                "not collected yet (never substituted with a nearby day)",
                run.postponed_price,
            )
        logger.info("[decision-eval] %s", run.summary())
        await self._log_scoreboard()
        return run.evaluated

    # ------------------------------------------------------------------
    @staticmethod
    async def _load_prices(
        session, pending: list[_PendingEvaluation]
    ) -> dict[tuple[str, datetime], Decimal]:
        """Every (symbol, date) close needed by this batch, in one query.

        Note which dates are fetched: the ENTRY and the TARGET. The decision
        day's own close is deliberately not among them any more — it is the
        price the trade could not have been placed at.
        """
        wanted: set[tuple[str, datetime]] = set()
        for item in pending:
            wanted.add((item.decision.symbol, item.entry))
            wanted.add((item.decision.symbol, item.target))
        if not wanted:
            return {}

        rows = (
            await session.execute(
                select(
                    CryptoPrice.symbol, CryptoPrice.timestamp, CryptoPrice.close
                ).where(
                    CryptoPrice.exchange == "binance",
                    # Multi-asset rule 6: BTC holds '1d' AND '4h' rows now.
                    CryptoPrice.timeframe == DAILY_TIMEFRAME,
                    tuple_(CryptoPrice.symbol, CryptoPrice.timestamp).in_(wanted),
                )
            )
        ).all()
        return {(row.symbol, row.timestamp): row.close for row in rows}

    # ------------------------------------------------------------------
    @staticmethod
    async def _load_vols(
        session, pending: list[_PendingEvaluation]
    ) -> dict[tuple[str, datetime], float]:
        """``realized_vol_30d`` per (symbol, decision day), in one query.

        Read at the DECISION date — not the entry date, and not the target
        date. The band has to describe what was knowable when the call was
        made, or the evaluation would be judging the decision against
        information it never had.

        This is the one place the entry lag is deliberately NOT applied, and
        the asymmetry is the design: the entry moved forward because a trader
        cannot transact in the past, while the band stayed put because a
        forecast cannot be graded on a yardstick from the future.
        """
        wanted = {(item.decision.symbol, item.decision.timestamp) for item in pending}
        if not wanted:
            return {}

        rows = (
            await session.execute(
                select(
                    DailyFeature.symbol,
                    DailyFeature.timestamp,
                    DailyFeature.realized_vol_30d,
                ).where(
                    # Multi-asset rule 1: daily_features MUST be symbol-filtered.
                    tuple_(DailyFeature.symbol, DailyFeature.timestamp).in_(wanted)
                )
            )
        ).all()
        return {
            (row.symbol, row.timestamp): float(row.realized_vol_30d)
            for row in rows
            if row.realized_vol_30d is not None
        }

    # ------------------------------------------------------------------
    @staticmethod
    async def compute_scoreboard() -> dict[int, ScoreboardRow]:
        """Hit rate per horizon beside its skill-free baseline.

        The baseline is not decoration. ``band_pct`` is exactly one sigma, so
        a HOLD that knows nothing scores ~70%; the first Phase 9 scoreboard
        read 76.5% and 100% and was mistaken for evidence.
        """
        from sqlalchemy import func

        # Local import: null_model imports this module's constants, so a
        # top-level import would be circular.
        from src.evaluation.null_model import load_null_baselines
        from src.evaluation.sample_size import effective_sample_size

        async with session_scope() as session:
            # Grouped down to (symbol, action) because the baseline differs
            # per asset AND per action — an asset that drifted up gives a
            # skill-free BUY a better score than a skill-free SELL on the
            # same tape, and averaging that away would hide it.
            rows = (
                await session.execute(
                    select(
                        DecisionEvaluation.horizon_days,
                        DecisionEvaluation.symbol,
                        DecisionEvaluation.action,
                        DecisionEvaluation.verdict,
                        func.count(),
                    ).group_by(
                        DecisionEvaluation.horizon_days,
                        DecisionEvaluation.symbol,
                        DecisionEvaluation.action,
                        DecisionEvaluation.verdict,
                    )
                )
            ).all()
            # Dates and per-day breadth for the overlap correction. The
            # scoreboard pools assets AND days, so BOTH dependencies apply.
            #
            # DECISIVE rows only. NEUTRAL evaluations are excluded from the
            # hit rate, so letting them into its effective sample size would
            # overstate the n every significance test divides by. Caught by
            # the Stage A6 gate, which found effective_n = 14 against 13
            # decisive rows.
            spread = (
                await session.execute(
                    select(
                        DecisionEvaluation.horizon_days,
                        DecisionEvaluation.decision_timestamp,
                        func.count(DecisionEvaluation.symbol.distinct()),
                    )
                    .where(DecisionEvaluation.verdict != DecisionVerdict.NEUTRAL)
                    .group_by(
                        DecisionEvaluation.horizon_days,
                        DecisionEvaluation.decision_timestamp,
                    )
                )
            ).all()
            baselines = await load_null_baselines(session)

        dates_by_horizon: dict[int, list] = {}
        breadth_by_horizon: dict[int, list[int]] = {}
        for horizon, moment, assets in spread:
            dates_by_horizon.setdefault(horizon, []).append(moment)
            breadth_by_horizon.setdefault(horizon, []).append(assets)

        per_horizon: dict[int, dict[str, float]] = {}
        for horizon, symbol, action, verdict, count in rows:
            slot = per_horizon.setdefault(
                horizon,
                {"total": 0, "neutral": 0, "correct": 0, "decisive": 0,
                 "expected": 0.0, "unbaselined": 0},
            )
            slot["total"] += count
            if verdict is DecisionVerdict.NEUTRAL:
                slot["neutral"] += count
                continue
            slot["decisive"] += count
            if verdict is DecisionVerdict.CORRECT:
                slot["correct"] += count

            baseline = baselines.get((symbol, horizon))
            rate = None
            if baseline is not None:
                rate = {
                    "BUY": baseline.buy_hit_rate,
                    "SELL": baseline.sell_hit_rate,
                    "HOLD": baseline.hold_hit_rate,
                }.get((action or "").strip().upper())
            if rate is None:
                slot["unbaselined"] += count
            else:
                slot["expected"] += count * rate

        board: dict[int, ScoreboardRow] = {}
        for horizon, slot in per_horizon.items():
            decisive = slot["decisive"]
            # Only rows that HAVE a baseline may enter the comparison.
            comparable = decisive - slot["unbaselined"]
            dates = dates_by_horizon.get(horizon, [])
            breadth = breadth_by_horizon.get(horizon, [1])
            board[horizon] = ScoreboardRow(
                horizon_days=horizon,
                total=slot["total"],
                decisive=decisive,
                correct=slot["correct"],
                neutral=slot["neutral"],
                comparable=comparable,
                null_hit_rate=(
                    slot["expected"] / comparable if comparable > 0 else None
                ),
                effective_n=effective_sample_size(
                    dates, horizon, assets_per_date=sum(breadth) / len(breadth)
                ),
            )
        return board

    # ------------------------------------------------------------------
    @classmethod
    async def _log_scoreboard(cls) -> None:
        """Print the scoreboard, never a hit rate on its own."""
        from src.evaluation.null_model import SIGNIFICANCE_Z, edge_z_score

        board = await cls.compute_scoreboard()
        for horizon in sorted(board):
            row = board[horizon]
            hit = (row.hit_rate or 0.0) * 100
            if row.null_hit_rate is None:
                verdict_text = "null n/a (no baseline yet)"
            else:
                null_hit = row.null_hit_rate * 100
                verdict_text = f"null {null_hit:.1f}% | edge {hit - null_hit:+.1f}pp"
                if hit <= null_hit:
                    verdict_text += " (NO EDGE — the band explains this)"
                else:
                    # A positive edge on a thin sample is the easiest number
                    # in this whole system to over-read, so it never prints
                    # unqualified — and the n it is tested against is the
                    # INDEPENDENT count, not the row count.
                    effective = row.effective_n
                    z = edge_z_score(hit / 100, null_hit / 100, effective)
                    if z is None or effective < 2:
                        verdict_text += (
                            f" (UNTESTABLE — {row.comparable} rows are only "
                            f"{effective:.1f} independent obs)"
                        )
                    elif z < SIGNIFICANCE_Z:
                        verdict_text += (
                            f" (z={z:.1f} — NOT significant; {row.comparable} "
                            f"rows ~ {effective:.1f} independent)"
                        )
                    else:
                        verdict_text += (
                            f" (z={z:.1f} on {effective:.1f} independent obs "
                            f"— worth a closer look)"
                        )
            logger.info(
                "[decision-eval] SCOREBOARD +%2dd: %d evaluation(s), %d decisive "
                "| hit-rate %.1f%% | %s | %d NEUTRAL (market inside band)",
                horizon, row.total, row.decisive, hit, verdict_text, row.neutral,
            )

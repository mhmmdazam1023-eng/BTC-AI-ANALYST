"""ForecastEvaluator: score expired forecasts against realized prices (Phase 6).

CRITICAL CORRECTNESS RULE (fix 2026-07-18): a forecast is only evaluated once
its target day's daily candle is **finalized**. A daily candle for day D is
still forming until D ends (23:59:59 UTC) and is only stored with its true
close after a subsequent collector run. Evaluating "today's" forecast against
an intraday, still-moving price produces a wrong ``actual_price`` that never
self-corrects (one evaluation per forecast). Therefore we require:

    target_date < CURRENT_DATE - INTERVAL '1 day'     (finalization guard)

i.e. the target day must be at least one full day in the past — its candle has
closed AND been captured by a later daily collection run. See
:data:`FINALIZATION_BUFFER_DAYS` / :meth:`is_data_finalized`.

For each finalized, not-yet-evaluated forecast:

* ``actual_price``        — BTC daily ``close`` from ``crypto_prices`` at the
                            EXACT ``target_date`` (no earlier-day fallback; a
                            missing candle postpones evaluation instead of
                            grabbing the wrong day's price).
* ``price_error_pct``     — (actual - p50) / p50 * 100.
* ``directional_correct`` — the forecast's predicted DIRECTION
                            (``prob_price_up > 0.5``, ties broken by p50 vs
                            base) matched the realized direction
                            (actual vs ``base_price``).

Idempotent: one evaluation per forecast (unique constraint, DO NOTHING on
conflict). Ends each run with a running scoreboard (MAPE + hit-rate).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import Integer, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import CryptoPrice, Forecast, ForecastEvaluation

logger = logging.getLogger(__name__)

#: A target day is only evaluated once it is at least this many full days in
#: the past, guaranteeing its daily candle has closed and been collected.
#: With value 1 the guard is exactly ``target_date < CURRENT_DATE - 1 day``.
FINALIZATION_BUFFER_DAYS = 1


def _finalization_cutoff(now: datetime) -> datetime:
    """UTC-midnight cutoff; forecasts with ``target_date < cutoff`` are final.

    ``cutoff = today_00:00 - FINALIZATION_BUFFER_DAYS days`` which, for the
    default buffer of 1, equals ``CURRENT_DATE - INTERVAL '1 day'``.
    """
    today_midnight = now.astimezone(UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return today_midnight - timedelta(days=FINALIZATION_BUFFER_DAYS)


class ForecastEvaluator:
    """Evaluates finalized, expired forecasts and maintains the scoreboard."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    async def _exact_close(
        self, session: AsyncSession, target_date: datetime
    ) -> Decimal | None:
        """Daily ``close`` for the EXACT target_date candle, else ``None``.

        No earlier-day fallback: BTC trades every calendar day, so a missing
        candle means the data is not collected yet — we postpone rather than
        substitute a different day's price.
        """
        close = (
            await session.execute(
                select(CryptoPrice.close).where(
                    CryptoPrice.exchange == "binance",
                    CryptoPrice.symbol == "BTC/USDT",
                    CryptoPrice.timeframe == "1d",
                    CryptoPrice.timestamp == target_date,
                )
            )
        ).scalar_one_or_none()
        if close is None or close <= 0:
            return None
        return close

    # ------------------------------------------------------------------
    async def is_data_finalized(
        self,
        target_date: datetime,
        *,
        session: AsyncSession | None = None,
    ) -> bool:
        """Whether ``target_date`` is safe to evaluate.

        Returns ``True`` only when BOTH hold:
        * ``target_date`` is at least ``FINALIZATION_BUFFER_DAYS`` full days in
          the past (its daily candle has closed), and
        * a valid closing candle for ``target_date`` exists in
          ``crypto_prices``.

        Pass an open ``session`` to reuse it; otherwise one is opened.
        """
        cutoff = _finalization_cutoff(datetime.now(tz=UTC))
        if target_date >= cutoff:
            return False
        if session is not None:
            return await self._exact_close(session, target_date) is not None
        async with session_scope() as own_session:
            return await self._exact_close(own_session, target_date) is not None

    # ------------------------------------------------------------------
    async def evaluate_expired_forecasts(self) -> int:
        """Evaluate finalized, not-yet-evaluated forecasts. Returns count."""
        cutoff = _finalization_cutoff(datetime.now(tz=UTC))
        logger.info(
            "[evaluator] finalization cutoff: only target_date < %s "
            "(>= %d full day(s) old) is evaluated",
            cutoff.date(), FINALIZATION_BUFFER_DAYS,
        )
        async with session_scope() as session:
            due = (
                (
                    await session.execute(
                        select(Forecast)
                        .outerjoin(
                            ForecastEvaluation,
                            ForecastEvaluation.forecast_id == Forecast.id,
                        )
                        .where(
                            # Finalization guard — NOT `<= now`: excludes
                            # today's (and yesterday's) still-settling candle.
                            Forecast.target_date < cutoff,
                            ForecastEvaluation.id.is_(None),
                        )
                        .order_by(Forecast.target_date)
                    )
                )
                .scalars()
                .all()
            )

            if not due:
                logger.info(
                    "[evaluator] no finalized, unevaluated forecasts — nothing to do"
                )
                return 0
            logger.info(
                "[evaluator] %d finalized forecast(s) awaiting evaluation", len(due)
            )

            evaluated = 0
            skipped = 0
            for forecast in due:
                # Defensive re-check (query already filtered by the cutoff):
                # confirm the exact closing candle is actually present.
                actual = await self._exact_close(session, forecast.target_date)
                if actual is None:
                    skipped += 1
                    logger.warning(
                        "[evaluator] forecast id=%d (+%dd from %s): finalized but "
                        "no close candle at %s yet — retrying on a later run",
                        forecast.id, forecast.horizon_days,
                        forecast.forecast_date.date(), forecast.target_date.date(),
                    )
                    continue

                p50 = forecast.predicted_price_p50
                base = forecast.base_price
                error_pct = (actual - p50) / p50 * Decimal(100)
                predicted_up = (
                    forecast.prob_price_up > 0.5
                    if forecast.prob_price_up != 0.5
                    else p50 > base
                )
                actual_up = actual > base
                correct = bool(predicted_up == actual_up)

                statement = (
                    pg_insert(ForecastEvaluation.__table__)
                    .values(
                        forecast_id=forecast.id,
                        target_date=forecast.target_date,
                        actual_price=actual,
                        price_error_pct=error_pct.quantize(Decimal("0.00000001")),
                        directional_correct=correct,
                    )
                    .on_conflict_do_nothing(constraint="uq_evaluation_per_forecast")
                )
                await session.execute(statement)
                evaluated += 1
                logger.info(
                    "[evaluator] id=%d %s+%dd: p50=%.0f actual(close)=%.0f "
                    "err=%+.2f%% direction=%s",
                    forecast.id, forecast.forecast_date.date(), forecast.horizon_days,
                    p50, actual, error_pct, "CORRECT" if correct else "WRONG",
                )

            if skipped:
                logger.warning(
                    "[evaluator] %d forecast(s) postponed (close candle missing)", skipped
                )

        await self._log_scoreboard()
        return evaluated

    # ------------------------------------------------------------------
    async def _log_scoreboard(self) -> None:
        """Running accuracy scoreboard across ALL evaluations."""
        async with session_scope() as session:
            row = (
                await session.execute(
                    select(
                        func.count(ForecastEvaluation.id),
                        func.avg(func.abs(ForecastEvaluation.price_error_pct)),
                        func.avg(ForecastEvaluation.directional_correct.cast(Integer)),
                    )
                )
            ).one()
        total, mape, hit_rate = row
        if total:
            logger.info(
                "[evaluator] SCOREBOARD: %d evaluations | MAPE %.2f%% | "
                "directional hit-rate %.1f%%",
                total, float(mape or 0), float(hit_rate or 0) * 100,
            )

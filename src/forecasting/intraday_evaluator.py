"""IntradayForecastEvaluator: score matured 4H forecasts (Phase 8 Part 8).

Mirrors :mod:`src.forecasting.evaluator` in shape and in its correctness rule,
with the finalization guard expressed in BARS instead of days.

The finalization rule
---------------------
A forecast targets the bar opening at ``target_at``. That bar has finished
forming once ``target_at + 4h <= now``, i.e.::

    target_at < floor_to_bar(now)

This is the *same predicate* the collector uses to drop forming bars — one
helper, :func:`src.timeframes.is_bar_closed`, is the only definition of
"closed" in the codebase. The daily evaluator needed a separate
``FINALIZATION_BUFFER_DAYS`` constant because a daily candle is only captured
by a *subsequent* collector run; here the bar-boundary predicate already
encodes that, and the belt-and-braces check below covers capture.

Belt and braces: the predicate says the bar *should* be final, and then the
exact candle must actually EXIST in ``crypto_prices``. A missing candle
postpones the evaluation to a later run rather than substituting a nearby
price — the precise mistake fixed on 2026-07-25.

Because ``run_daily.py`` runs once a day while horizons are 4h/12h/24h, a
single run normally matures SEVERAL forecasts at once. That is expected, not
a backlog.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import Integer, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import (
    INTRADAY_TIMEFRAME,
    CryptoPrice,
    IntradayForecast,
    IntradayForecastEvaluation,
)
from src.timeframes import bar_label, describe_horizon, floor_to_bar

logger = logging.getLogger(__name__)


class IntradayForecastEvaluator:
    """Evaluates matured 4H forecasts and maintains the intraday scoreboard."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    async def _exact_bar_close(
        self, session: AsyncSession, symbol: str, target_at: datetime
    ) -> Decimal | None:
        """Close of the EXACT 4h bar opening at ``target_at``, else ``None``.

        No "nearest bar" fallback. BTC trades continuously, so a missing bar
        means it has not been collected yet; postponing is correct, guessing
        is not.
        """
        close = (
            await session.execute(
                select(CryptoPrice.close).where(
                    CryptoPrice.exchange == "binance",
                    CryptoPrice.symbol == symbol,
                    CryptoPrice.timeframe == INTRADAY_TIMEFRAME,
                    CryptoPrice.timestamp == target_at,
                )
            )
        ).scalar_one_or_none()
        if close is None or close <= 0:
            return None
        return close

    # ------------------------------------------------------------------
    async def evaluate_matured(self, now: datetime | None = None) -> int:
        """Evaluate matured, not-yet-evaluated 4H forecasts. Returns count."""
        reference = now or datetime.now(tz=UTC)
        cutoff = floor_to_bar(reference)
        logger.info(
            "[4h-evaluator] maturity cutoff: only target_at < %s "
            "(bar fully closed) is evaluated",
            bar_label(cutoff),
        )

        async with session_scope() as session:
            due = (
                (
                    await session.execute(
                        select(IntradayForecast)
                        .outerjoin(
                            IntradayForecastEvaluation,
                            IntradayForecastEvaluation.forecast_id
                            == IntradayForecast.id,
                        )
                        .where(
                            # Maturity guard — the target bar must be CLOSED.
                            IntradayForecast.target_at < cutoff,
                            IntradayForecastEvaluation.id.is_(None),
                        )
                        .order_by(IntradayForecast.target_at)
                    )
                )
                .scalars()
                .all()
            )

            if not due:
                logger.info(
                    "[4h-evaluator] no matured, unevaluated 4H forecasts — "
                    "nothing to do"
                )
                return 0
            logger.info(
                "[4h-evaluator] %d matured forecast(s) awaiting evaluation", len(due)
            )

            evaluated = 0
            postponed = 0
            for forecast in due:
                actual = await self._exact_bar_close(
                    session, forecast.symbol, forecast.target_at
                )
                if actual is None:
                    postponed += 1
                    logger.warning(
                        "[4h-evaluator] id=%d (%s +%s): bar closed but no "
                        "candle stored at %s yet — retrying on a later run",
                        forecast.id, forecast.symbol,
                        describe_horizon(forecast.horizon_bars),
                        bar_label(forecast.target_at),
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
                    pg_insert(IntradayForecastEvaluation.__table__)
                    .values(
                        forecast_id=forecast.id,
                        target_at=forecast.target_at,
                        actual_price=actual,
                        price_error_pct=error_pct.quantize(Decimal("0.00000001")),
                        directional_correct=correct,
                    )
                    .on_conflict_do_nothing(
                        constraint="uq_intraday_evaluation_per_forecast"
                    )
                )
                await session.execute(statement)
                evaluated += 1
                logger.info(
                    "[4h-evaluator] id=%d +%-14s %s: p50=%.2f actual=%.2f "
                    "err=%+.2f%% direction=%s",
                    forecast.id, describe_horizon(forecast.horizon_bars),
                    bar_label(forecast.target_at), p50, actual, error_pct,
                    "CORRECT" if correct else "WRONG",
                )

            if postponed:
                logger.warning(
                    "[4h-evaluator] %d forecast(s) postponed (target bar candle "
                    "not collected yet)", postponed,
                )

        await self._log_scoreboard()
        return evaluated

    # ------------------------------------------------------------------
    async def _log_scoreboard(self) -> None:
        """Running accuracy per horizon across ALL intraday evaluations."""
        async with session_scope() as session:
            rows = (
                await session.execute(
                    select(
                        IntradayForecast.horizon_bars,
                        func.count(IntradayForecastEvaluation.id),
                        func.avg(func.abs(IntradayForecastEvaluation.price_error_pct)),
                        func.avg(
                            IntradayForecastEvaluation.directional_correct.cast(Integer)
                        ),
                    )
                    .join(
                        IntradayForecast,
                        IntradayForecast.id
                        == IntradayForecastEvaluation.forecast_id,
                    )
                    .group_by(IntradayForecast.horizon_bars)
                    .order_by(IntradayForecast.horizon_bars)
                )
            ).all()

        if not rows:
            return
        for horizon_bars, total, mape, hit_rate in rows:
            logger.info(
                "[4h-evaluator] SCOREBOARD +%-14s %d evaluation(s) | "
                "MAPE %.2f%% | directional hit-rate %.1f%%",
                describe_horizon(horizon_bars), total,
                float(mape or 0), float(hit_rate or 0) * 100,
            )

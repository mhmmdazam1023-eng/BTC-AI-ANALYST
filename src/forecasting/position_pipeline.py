"""Scenario -> Kelly -> ``position_recommendations`` (Phase 8 Part 7).

Ties the scenario engine and the position sizer to the day's decisions and
persists the result. MODEL OUTPUT, not advice.

Per-asset isolation, as everywhere else in this pipeline: a scenario or
sizing failure for one asset is recorded and the run continues. One coin
with a missing volatility estimate must not cost the other twenty their
sizing.

Sizing follows the DECISION, never overrides it. Only assets the Master
Agent marked BUY are sized; HOLD and SELL are written with a zero position
and the reason recorded, so the table always reflects the full book rather
than only its long side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import DailyFeature, Decision, PositionRecommendation
from src.forecasting.kelly_criterion import (
    PositionSizer,
    PositionSizing,
    portfolio_exposure,
)
from src.forecasting.scenario_engine import (
    ScenarioEngine,
    ScenarioSet,
    horizon_band_multiplier,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PositionRun:
    """Outcome of one portfolio sizing pass."""

    as_of: datetime
    sizings: list[PositionSizing] = field(default_factory=list)
    scenarios: dict[str, ScenarioSet] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    exposure: dict[str, Any] = field(default_factory=dict)

    def summary_line(self) -> str:
        live = [s for s in self.sizings if s.adjusted_fraction > 0]
        text = (
            f"{len(self.sizings)} asset(s) sized, {len(live)} with a non-zero "
            f"position, total {self.exposure.get('total_pct', 0):.1f}%"
        )
        if self.exposure.get("over_allocated"):
            text += " [OVER-ALLOCATED >100%]"
        if self.failures:
            text += f" | FAILED: {', '.join(sorted(self.failures))}"
        return text


class PositionPipeline:
    """Builds and persists position recommendations for the day's decisions."""

    def __init__(
        self,
        *,
        horizon_days: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.scenario_engine = (
            ScenarioEngine(horizon_days=horizon_days)
            if horizon_days
            else ScenarioEngine()
        )
        self.sizer = PositionSizer()

    # ------------------------------------------------------------------
    async def load_decisions(self, day: datetime | None = None) -> list[Decision]:
        """All decisions for one day (defaults to the most recent day)."""
        async with session_scope() as session:
            if day is None:
                day = (
                    await session.execute(select(func.max(Decision.timestamp)))
                ).scalar_one_or_none()
                if day is None:
                    return []
            return list(
                (
                    await session.execute(
                        select(Decision)
                        .where(Decision.timestamp == day)
                        .order_by(Decision.symbol)
                    )
                ).scalars().all()
            )

    async def load_coin_dna(self, symbol: str, day: datetime) -> dict:
        """The DNA fields the sizing overlays actually read.

        Queried straight from ``daily_features`` rather than reused from the
        inference script: a package must not import from ``scripts/``, and
        sizing needs only three of the twelve DNA fields. Missing values
        stay None — an overlay must never fire on absent data.
        """
        async with session_scope() as session:
            row = (
                await session.execute(
                    select(
                        DailyFeature.vol_regime,
                        DailyFeature.days_to_next_unlock,
                        DailyFeature.supply_shock_score_30d,
                        DailyFeature.btc_beta_30d,
                        # Added 2026-07-31: continuous Kelly divides the edge
                        # by the horizon variance, so the sizer needs the
                        # volatility itself, not just its LOW/NORMAL/HIGH bucket.
                        DailyFeature.realized_vol_30d,
                        DailyFeature.ewma_vol_30d,
                    )
                    .where(
                        DailyFeature.symbol == symbol,
                        DailyFeature.timestamp <= day,
                    )
                    .order_by(DailyFeature.timestamp.desc())
                    .limit(1)
                )
            ).first()
        if row is None:
            logger.warning(
                "[position] no features for %s at %s — sizing proceeds with "
                "no DNA overlays", symbol, day.date(),
            )
            return {"symbol": symbol}

        vol_regime, unlock_days, supply_shock, beta, realized_vol, ewma_vol = row
        return {
            "symbol": symbol,
            "vol_regime": (
                None if vol_regime is None else getattr(vol_regime, "value", str(vol_regime))
            ),
            "days_to_next_unlock": (
                None if unlock_days is None else int(unlock_days)
            ),
            "supply_shock_score_30d": (
                None if supply_shock is None else float(supply_shock)
            ),
            "btc_beta_30d": None if beta is None else float(beta),
            "realized_vol_30d": (
                None if realized_vol is None else float(realized_vol)
            ),
            "ewma_vol_30d": None if ewma_vol is None else float(ewma_vol),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _horizon_variance(coin_dna: dict, horizon_days: int) -> float | None:
        """Variance of the horizon return — the ``sigma^2`` in ``f* = mu/sigma^2``.

        Prefers ``ewma_vol_30d`` (2026-08-02) and falls back to
        ``realized_vol_30d``: the EWMA estimate covers better at every horizon
        measured, and a smaller sigma^2 here means a LARGER position, so a
        stale over-estimate is the safe direction to fall back in.

        The volatility is ANNUALISED (the same convention Part 4 had to fix in
        ``trend_strength`` and Phase 9 uses for the band), so the annual
        VARIANCE is scaled by ``horizon/365``: variance grows linearly in time
        where standard deviation grows with its square root. Scaling by
        ``sqrt`` here instead would be wrong by a factor of ~3.5 at +30d and
        would still look entirely plausible.

        The sqrt-time correction from ``horizon_band_multiplier`` is applied
        to the STANDARD DEVIATION, hence squared here — variance scales with
        the square of any factor applied to sigma. Forgetting the square
        would under-widen by the square root of the correction.
        """
        vol = coin_dna.get("ewma_vol_30d") or coin_dna.get("realized_vol_30d")
        if vol is None:
            return None
        try:
            annualised = float(vol)
        except (TypeError, ValueError):
            return None
        if annualised <= 0:
            return None
        multiplier = horizon_band_multiplier(horizon_days)
        return (annualised**2) * (horizon_days / 365.0) * (multiplier**2)

    # ------------------------------------------------------------------
    async def run(self, day: datetime | None = None) -> PositionRun:
        """Size every asset that has a decision today, and persist."""
        decisions = await self.load_decisions(day)
        if not decisions:
            logger.warning(
                "[position] no decisions to size — run the inference stage first"
            )
            return PositionRun(as_of=day or datetime.now(tz=UTC))

        as_of = decisions[0].timestamp
        run = PositionRun(as_of=as_of)

        # Portfolio-wide fallback, resolved once. Phase 9 lets each asset
        # override this with its OWN decision accuracy inside the loop below;
        # this remains the answer for assets that have not accumulated enough
        # decisive BUY evaluations yet.
        fallback = await self.sizer.load_win_rate()
        logger.info(
            "[position] sizing %d asset(s) for %s | fallback win rate %.4f "
            "(%s, n=%d); per-asset decision accuracy is preferred where available",
            len(decisions), as_of.date(), fallback[0], fallback[1], fallback[2],
        )

        regimes = await self._load_regime(as_of)

        for decision in decisions:
            symbol = decision.symbol
            try:
                coin_dna = await self.load_coin_dna(symbol, as_of)
                coin_dna.setdefault("regime", regimes)
                scenarios = await self.scenario_engine.build(symbol, coin_dna)
                # Phase 9: this asset's own BUY accuracy when it has enough
                # decisive evaluations, otherwise the portfolio fallback.
                win_rate, win_rate_source, win_rate_samples = (
                    await self.sizer.load_win_rate(symbol)
                )
                horizon = self.scenario_engine.horizon_days
                # The two inputs continuous Kelly needs (2026-07-31): what
                # "no skill" scores on THIS asset, and the scale to divide the
                # edge by. Both come from the asset itself — neither is
                # assumed, and either being absent means it is not sized.
                null_win_rate = await self.sizer.load_null_win_rate(
                    symbol, horizon
                )
                if win_rate_source.startswith("default") and null_win_rate is not None:
                    # No measured skill: p defaults to the null so the excess
                    # return is EXACTLY zero rather than an assumed 0.50
                    # manufacturing an edge or a deficit.
                    win_rate = null_win_rate
                    win_rate_source = "default_null_baseline"
                sizing = self.sizer.size(
                    symbol,
                    decision.action,
                    scenarios,
                    coin_dna,
                    win_rate=win_rate,
                    win_rate_source=win_rate_source,
                    win_rate_samples=win_rate_samples,
                    null_win_rate=null_win_rate,
                    horizon_variance=self._horizon_variance(coin_dna, horizon),
                )
                run.sizings.append(sizing)
                run.scenarios[symbol] = scenarios
            except Exception as exc:  # noqa: BLE001 - per-asset isolation
                run.failures[symbol] = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "[position] %s FAILED — continuing with the rest", symbol
                )

        run.exposure = portfolio_exposure(run.sizings)
        if run.exposure.get("over_allocated"):
            logger.warning(
                "[position] TOTAL EXPOSURE %.1f%% EXCEEDS 100%%. Kelly sizes "
                "each asset independently against cash, so per-asset caps do "
                "not bound the book. These assets are also highly correlated, "
                "so the true concentration is worse than the sum suggests.",
                run.exposure["total_pct"],
            )

        await self.save(run)
        logger.info("[position] %s", run.summary_line())
        return run

    async def _load_regime(self, as_of: datetime) -> str | None:
        """Market regime for the day, used to tilt scenario probabilities."""
        from src.db.models import MarketRegimeRow

        try:
            async with session_scope() as session:
                row = (
                    await session.execute(
                        select(MarketRegimeRow.regime)
                        .where(MarketRegimeRow.timestamp <= as_of)
                        .order_by(MarketRegimeRow.timestamp.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
            return None if row is None else getattr(row, "value", str(row))
        except Exception as exc:  # noqa: BLE001 - regime tilt is optional
            logger.warning("[position] regime unavailable (%s) — no tilt", exc)
            return None

    # ------------------------------------------------------------------
    async def save(self, run: PositionRun) -> int:
        """Idempotent upsert into ``position_recommendations``."""
        if not run.sizings:
            return 0

        def dec(value: float | None, places: int = 8) -> Decimal | None:
            if value is None:
                return None
            return Decimal(repr(round(float(value), places)))

        payload = []
        for sizing in run.sizings:
            scenarios = run.scenarios.get(sizing.symbol)
            payload.append(
                {
                    "timestamp": run.as_of,
                    "symbol": sizing.symbol,
                    "action": sizing.action,
                    "kelly_fraction": dec(sizing.kelly_fraction),
                    "adjusted_fraction": dec(sizing.adjusted_fraction),
                    "position_size_pct": dec(sizing.position_size_pct, 4),
                    "win_rate": dec(sizing.win_rate, 6),
                    "win_rate_source": sizing.win_rate_source,
                    "win_rate_samples": sizing.win_rate_samples,
                    "payoff_ratio": dec(sizing.payoff_ratio, 6),
                    "scenario_source": scenarios.source if scenarios else None,
                    "scenario_probs": scenarios.as_dict() if scenarios else None,
                    "risk_factors": sizing.risk_factors,
                }
            )

        statement = pg_insert(PositionRecommendation.__table__).values(payload)
        update_set = {
            column: statement.excluded[column]
            for column in payload[0]
            if column not in ("timestamp", "symbol")
        }
        update_set["ingested_at"] = func.now()
        statement = statement.on_conflict_do_update(
            index_elements=["timestamp", "symbol"], set_=update_set
        )
        async with session_scope() as session:
            await session.execute(statement)
        logger.info(
            "[position] upserted %d recommendation(s) for %s",
            len(payload), run.as_of.date(),
        )
        return len(payload)

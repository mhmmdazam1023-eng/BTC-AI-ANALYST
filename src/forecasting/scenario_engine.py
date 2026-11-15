"""ScenarioEngine: price distribution -> three weighted scenarios (Part 7).

Turns a forward price distribution into Bull / Base / Bear scenarios with
probabilities and magnitudes, which the Kelly sizer then consumes.

Where the distribution comes from
---------------------------------
Two sources, and which one was used is ALWAYS recorded on the output:

* ``monte_carlo`` — the Phase 6 block-bootstrap forecast. Bitcoin only:
  the ``forecasts`` table has no ``symbol`` column and ``ForecastEngine`` is
  BTC-hardcoded.
* ``realized_vol_analytic`` — a lognormal band built from the asset's own
  ``realized_vol_30d``. Used for the other twenty assets, which have no
  Monte Carlo at all.

The second is a WEAKER estimate and is labelled as such rather than blended
in silently. It assumes lognormal returns with zero drift; the bootstrap
makes no such assumption and preserves volatility clustering. Reading a
0.25 sizing off an analytic band as if it came from 20,000 simulated paths
would be a category error, so the provenance travels with the number all
the way into ``position_recommendations.scenario_source``.

Probability weighting
---------------------
The base rates are 25/50/25. Regime and supply pressure tilt them:

* ``BULL_TRENDING``   -> shift weight from bear to bull.
* ``BEAR_TRENDING``   -> shift weight from bull to bear.
* ``CRASH_RECOVERY``  -> widen BOTH tails at the expense of the base case;
  a reflex bounce is genuinely two-sided, not directional.
* ``supply_shock_score_30d`` above the threshold -> shift weight to bear,
  scaled by how large the unlock is. Supply arriving is a one-directional
  pressure, so it never tilts bullish.

Probabilities are renormalised to sum to exactly 1.0 after every tilt.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import select

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import BASE_SYMBOL, DailyFeature, Forecast

logger = logging.getLogger(__name__)

#: Horizon the sizing is built around. 30 days matches the rebalancing
#: cadence a daily system realistically supports, and it is long enough for
#: a scheduled unlock inside the 30-day window to actually matter.
DEFAULT_HORIZON_DAYS = 30

#: Empirical correction to square-root-of-time band scaling (2026-08-02).
#:
#: `sigma_h = sigma_annual * sqrt(h/365)` assumes returns aggregate like
#: independent Gaussian steps. Crypto returns do neither: volatility clusters
#: and tails are fat, so a multi-day return is more dispersed than sqrt-time
#: predicts, and increasingly so as the horizon grows.
#:
#: Measured across ALL 21 assets, 34.8k asset-days, using EWMA volatility.
#: `k` is the factor that makes a nominal 90% band actually cover 90%:
#:     h        1      3      7     14     30     60
#:     coverage 91.0%  90.6%  89.8%  88.7%  87.0%  86.2%
#:     k        0.957  0.978  1.009  1.063  1.160  1.242
#: Square-root-of-time would mean k = 1.0 everywhere. It does not.
#:
#: A single Hurst-style exponent was tried and REJECTED: the implied H drifts
#: 0.507 -> 0.498 -> 0.441 -> 0.380 across those horizons, so no one exponent
#: fits. A measured table is less elegant and more honest.
#:
#: These are calibrated to EWMA volatility. Feeding realized_vol_30d through
#: them will over-widen, because that estimator is already ~2pp narrower.
_HORIZON_BAND_MULTIPLIER: dict[int, float] = {
    1: 0.957, 3: 0.978, 7: 1.009, 14: 1.063, 30: 1.160, 60: 1.242,
}


def horizon_band_multiplier(horizon_days: int) -> float:
    """Empirical sqrt-time correction, linearly interpolated between anchors.

    Clamped outside the measured range rather than extrapolated: the trend is
    real but there is no evidence it continues linearly to 180 days, and
    inventing a number there would be the kind of plausible-looking guess
    this table exists to replace.
    """
    anchors = sorted(_HORIZON_BAND_MULTIPLIER)
    if horizon_days <= anchors[0]:
        return _HORIZON_BAND_MULTIPLIER[anchors[0]]
    if horizon_days >= anchors[-1]:
        return _HORIZON_BAND_MULTIPLIER[anchors[-1]]
    for low, high in zip(anchors, anchors[1:], strict=False):
        if low <= horizon_days <= high:
            span = high - low
            weight = (horizon_days - low) / span
            return (
                _HORIZON_BAND_MULTIPLIER[low] * (1 - weight)
                + _HORIZON_BAND_MULTIPLIER[high] * weight
            )
    return 1.0  # pragma: no cover - unreachable, anchors cover the range

#: Unweighted base rates before any regime or supply tilt.
BASE_PROB_BULL = 0.25
BASE_PROB_BASE = 0.50
BASE_PROB_BEAR = 0.25

#: How much probability mass a strong regime read moves between tails.
REGIME_TILT = 0.10

#: Supply shock (% of circulating supply in the next 30 days) above which
#: the bear case is weighted more heavily, and the cap on that tilt.
SUPPLY_SHOCK_THRESHOLD_PCT = 5.0
SUPPLY_SHOCK_MAX_TILT = 0.15

#: Trading days per year, matching the FeatureEngine's annualisation.
TRADING_DAYS_PER_YEAR = 365

#: Percentile bands each scenario is measured over.
BULL_BAND = (75, 95)
BASE_BAND = (40, 60)
BEAR_BAND = (5, 25)


class ScenarioEngineError(RuntimeError):
    """Raised when no usable distribution can be built for an asset."""


@dataclass(slots=True)
class ScenarioSet:
    """Three weighted scenarios for one asset at one horizon."""

    symbol: str
    as_of: datetime
    horizon_days: int
    source: str

    prob_bull: float
    prob_base: float
    prob_bear: float

    #: Expected RETURN in each scenario, as a fraction of the base price.
    expected_return_bull: float
    expected_return_base: float
    #: Positive number describing the size of the loss (a 17% fall -> 0.17).
    expected_drawdown_bear: float

    base_price: float | None = None
    tilts: dict[str, Any] = field(default_factory=dict)

    @property
    def payoff_ratio(self) -> float | None:
        """Bull upside divided by bear downside.

        ``None`` when the bear case shows no downside at all — Kelly would
        divide by zero, and "cannot lose" is never a real input.
        """
        if self.expected_drawdown_bear <= 0:
            return None
        return self.expected_return_bull / self.expected_drawdown_bear

    @property
    def expected_value(self) -> float:
        """Probability-weighted return. Negative means don't take the trade."""
        return (
            self.prob_bull * self.expected_return_bull
            + self.prob_base * self.expected_return_base
            - self.prob_bear * self.expected_drawdown_bear
        )

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "horizon_days": self.horizon_days,
            "prob_bull": round(self.prob_bull, 6),
            "prob_base": round(self.prob_base, 6),
            "prob_bear": round(self.prob_bear, 6),
            "expected_return_bull": round(self.expected_return_bull, 6),
            "expected_return_base": round(self.expected_return_base, 6),
            "expected_drawdown_bear": round(self.expected_drawdown_bear, 6),
            "payoff_ratio": (
                None if self.payoff_ratio is None else round(self.payoff_ratio, 6)
            ),
            "expected_value": round(self.expected_value, 6),
            "tilts": self.tilts,
        }

    def summary(self) -> str:
        payoff = "n/a" if self.payoff_ratio is None else f"{self.payoff_ratio:.2f}"
        return (
            f"{self.symbol} [{self.source}] "
            f"bull {self.prob_bull:.0%}/{self.expected_return_bull:+.1%} · "
            f"base {self.prob_base:.0%} · "
            f"bear {self.prob_bear:.0%}/-{self.expected_drawdown_bear:.1%} · "
            f"payoff {payoff} · EV {self.expected_value:+.2%}"
        )


class ScenarioEngine:
    """Builds Bull/Base/Bear scenarios for one asset."""

    def __init__(
        self,
        *,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.horizon_days = horizon_days

    # ------------------------------------------------------------------
    async def load_monte_carlo(self, symbol: str) -> Forecast | None:
        """Latest Monte Carlo forecast, or None when the asset has none.

        Only Bitcoin ever has one — ``forecasts`` carries no symbol column,
        so anything else would be reading BTC's numbers under another name.
        """
        if symbol != BASE_SYMBOL:
            return None
        async with session_scope() as session:
            return (
                await session.execute(
                    select(Forecast)
                    .where(Forecast.horizon_days == self.horizon_days)
                    .order_by(Forecast.forecast_date.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

    async def load_features(self, symbol: str) -> DailyFeature | None:
        async with session_scope() as session:
            return (
                await session.execute(
                    select(DailyFeature)
                    .where(DailyFeature.symbol == symbol)
                    .order_by(DailyFeature.timestamp.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

    # ------------------------------------------------------------------
    @staticmethod
    def _band_returns(
        p5: float, p50: float, p95: float, base_price: float
    ) -> tuple[float, float, float]:
        """Representative return for each percentile band.

        The forecast gives three points, so the interior percentiles are
        interpolated on a normal quantile scale (z(95) = 1.645) rather than
        linearly — a price distribution's tail is not linear in percentile,
        and linear interpolation would understate both tails.
        """
        z95 = 1.6448536269514722
        z75, z60, z40, z25 = (
            0.6744897501960817,
            0.2533471031357997,
            -0.2533471031357997,
            -0.6744897501960817,
        )
        upper_sigma = (p95 - p50) / z95
        lower_sigma = (p50 - p5) / z95

        def price_at(z: float) -> float:
            return p50 + z * (upper_sigma if z >= 0 else lower_sigma)

        bull = (np.mean([price_at(z75), price_at(z95)]) / base_price) - 1.0
        base = (np.mean([price_at(z40), price_at(z60)]) / base_price) - 1.0
        bear = (np.mean([price_at(z5) for z5 in (-z95, z25)]) / base_price) - 1.0
        return float(bull), float(base), float(bear)

    def _from_monte_carlo(self, forecast: Forecast) -> tuple[float, float, float, float]:
        base_price = float(forecast.base_price)
        bull, base, bear = self._band_returns(
            float(forecast.predicted_price_p5),
            float(forecast.predicted_price_p50),
            float(forecast.predicted_price_p95),
            base_price,
        )
        return bull, base, bear, base_price

    def _from_realized_vol(
        self, close: float, realized_vol_30d: float
    ) -> tuple[float, float, float, float]:
        """Lognormal band from annualised volatility, zero drift.

        Zero drift is deliberate. Extrapolating recent momentum into the
        forecast would manufacture exactly the edge Kelly is being asked to
        size, and would do it from the same data the panel already scored.

        The sigma is scaled by :func:`horizon_band_multiplier` — square root
        of time alone is measurably too narrow here (see that function).
        """
        sigma = (
            realized_vol_30d
            * math.sqrt(self.horizon_days / TRADING_DAYS_PER_YEAR)
            * horizon_band_multiplier(self.horizon_days)
        )
        z95 = 1.6448536269514722
        p50 = close
        p95 = close * math.exp(z95 * sigma)
        p5 = close * math.exp(-z95 * sigma)
        bull, base, bear = self._band_returns(p5, p50, p95, close)
        return bull, base, bear, close

    # ------------------------------------------------------------------
    @staticmethod
    def apply_tilts(
        regime: str | None, supply_shock_pct: float | None
    ) -> tuple[float, float, float, dict]:
        """Base rates tilted by regime and scheduled supply pressure."""
        bull, base, bear = BASE_PROB_BULL, BASE_PROB_BASE, BASE_PROB_BEAR
        tilts: dict[str, Any] = {}

        if regime == "BULL_TRENDING":
            bull, bear = bull + REGIME_TILT, bear - REGIME_TILT
            tilts["regime"] = {"regime": regime, "bull_tilt": REGIME_TILT}
        elif regime == "BEAR_TRENDING":
            bull, bear = bull - REGIME_TILT, bear + REGIME_TILT
            tilts["regime"] = {"regime": regime, "bear_tilt": REGIME_TILT}
        elif regime == "CRASH_RECOVERY":
            # Genuinely two-sided: a reflex bounce can extend or fail.
            bull, base, bear = (
                bull + REGIME_TILT / 2,
                base - REGIME_TILT,
                bear + REGIME_TILT / 2,
            )
            tilts["regime"] = {"regime": regime, "both_tails_widened": REGIME_TILT / 2}
        elif regime is not None:
            tilts["regime"] = {"regime": regime, "tilt": 0.0}

        if supply_shock_pct is not None and supply_shock_pct > SUPPLY_SHOCK_THRESHOLD_PCT:
            # Scale with size, capped. Supply hitting the market is a
            # one-directional pressure — this never tilts bullish.
            excess = supply_shock_pct - SUPPLY_SHOCK_THRESHOLD_PCT
            tilt = min(SUPPLY_SHOCK_MAX_TILT, excess / 100.0 * 3.0)
            bull, bear = bull - tilt, bear + tilt
            tilts["supply_shock"] = {
                "supply_shock_pct": supply_shock_pct,
                "bear_tilt": round(tilt, 6),
            }

        # Never let a tilt drive a probability negative, then renormalise so
        # the three always sum to exactly 1.0.
        bull, base, bear = max(bull, 0.01), max(base, 0.01), max(bear, 0.01)
        total = bull + base + bear
        return bull / total, base / total, bear / total, tilts

    # ------------------------------------------------------------------
    async def build(
        self, symbol: str, coin_dna: dict | None = None
    ) -> ScenarioSet:
        """Build the scenario set for one asset. Raises on no usable data."""
        coin_dna = coin_dna or {}
        forecast = await self.load_monte_carlo(symbol)

        if forecast is not None:
            bull, base, bear, base_price = self._from_monte_carlo(forecast)
            source = "monte_carlo"
            as_of = forecast.forecast_date
        else:
            features = await self.load_features(symbol)
            if features is None or features.asset_close is None:
                raise ScenarioEngineError(
                    f"no features for {symbol} — cannot build scenarios"
                )
            # EWMA first (2026-08-02): better calibrated at every horizon.
            # realized_vol_30d remains the fallback so a row written before
            # the column existed still sizes rather than raising — but the
            # SOURCE records which one was used, because the horizon
            # multipliers are calibrated to EWMA and are ~2pp too wide for
            # the equally-weighted estimate.
            volatility = features.ewma_vol_30d or features.realized_vol_30d
            if volatility is None:
                raise ScenarioEngineError(
                    f"{symbol} has no ewma_vol_30d or realized_vol_30d — "
                    f"cannot size a distribution without a volatility estimate"
                )
            bull, base, bear, base_price = self._from_realized_vol(
                float(features.asset_close), float(volatility)
            )
            source = (
                "ewma_vol_analytic"
                if features.ewma_vol_30d is not None
                else "realized_vol_analytic"
            )
            as_of = features.timestamp

        prob_bull, prob_base, prob_bear, tilts = self.apply_tilts(
            coin_dna.get("regime"), coin_dna.get("supply_shock_score_30d")
        )

        scenarios = ScenarioSet(
            symbol=symbol,
            as_of=as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC),
            horizon_days=self.horizon_days,
            source=source,
            prob_bull=prob_bull,
            prob_base=prob_base,
            prob_bear=prob_bear,
            expected_return_bull=bull,
            expected_return_base=base,
            # Stored as a POSITIVE magnitude; a bear band that is somehow
            # above the base price yields 0 downside, which payoff_ratio
            # then reports as None rather than as a free lunch.
            expected_drawdown_bear=max(0.0, -bear),
            base_price=base_price,
            tilts=tilts,
        )
        logger.info("[scenario] %s", scenarios.summary())
        return scenarios

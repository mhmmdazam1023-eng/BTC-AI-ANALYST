"""PositionSizer: Kelly-criterion position sizing (Phase 8 Part 7).

Produces MODEL OUTPUT, not advice. Every number here is arithmetic applied
to this system's own estimates; it knows nothing about the reader's
finances, tax position, or risk tolerance, and the estimates feeding it are
themselves uncertain. Treat the output as one input to a human decision.

The formula, and why the obvious one was wrong (rewritten 2026-07-31)
---------------------------------------------------------------------
This module used binary Kelly, ``f* = (b*p - q) / b``, with ``b`` taken from
the scenario band. Two defects, one root cause.

**Defect 1 — ``p`` and ``b`` described different bets.** Binary Kelly is
derived for a wager with exactly two outcomes: win ``b`` per unit staked
with probability ``p``, lose the unit otherwise. Here ``p`` was the
probability a BUY is scored CORRECT, which by the Phase 9 rule *requires*
``|return| > 1 sigma`` — it conditions on the tail. ``b`` meanwhile averaged
the WHOLE distribution, including the middle 40% that ``p`` discards. The
bet Kelly was sizing did not exist.

**Defect 2 — position size rose with volatility.** With the analytic band
``p95 = S*exp(z*sigma)``, ``p5 = S*exp(-z*sigma)``, the payoff ratio
simplifies exactly::

    b = (e^{z*sigma} - 1) / (1 - e^{-z*sigma}) = e^{z*sigma}

A pure function of volatility, carrying no market information. With ``p``
pinned at 0.50 that made ``f* = 0.5*(1 - 1/b)``, so the noisiest asset drew
the largest position: ONDO 16.5% against BNB 5.5%, purely for being 2.9x
more volatile. That is risk management inverted.

**The fix: continuous Kelly on a measured edge.** For a continuous return
distribution the growth-optimal fraction is ``f* = mu / sigma^2``, and the
two defects vanish together because both inputs now describe the same
horizon-length holding period:

* ``sigma^2`` is the variance of the horizon return, so ``f*`` **falls** as
  volatility rises — the correct direction.
* ``mu`` is the expected EXCESS return, and it is zero unless directional
  skill has actually been measured.

**``mu`` is measured against the asset's own null, not against 0.50.** A
skill-free BUY does not win half the time: an asset that drifted up puts
more mass above the band than below it, and BTC's measured skill-free BUY
rate at +30d is **60.9%**. Scoring 55% there is negative edge, which the old
formula would have sized positively. So::

    mu = (p - p_null) * (gain + loss)

Moving probability mass from the loss outcome to the win outcome by
``p - p_null`` is worth exactly that much expected return. When no win rate
has passed the sample gate, ``p`` defaults to ``p_null`` and ``mu`` is
**exactly zero** — no measurement, no position. Not "0.50", which would
manufacture an edge or a deficit out of an assumption.

Ours are still estimates, not known probabilities. Full Kelly on a
mis-estimated edge overbets badly and asymmetrically: overshooting ``p`` by
a little can turn a growth-optimal bet into a ruinous one, because the
penalty for betting too much is far steeper than for betting too little.
Everything below exists to lean against that.

Layers of conservatism, in the order they apply
-----------------------------------------------
1. **Direction gate** — only a BUY is sized. HOLD and SELL are zero. This
   system does not model shorts and must not assume leverage, so there is
   no such thing as a "negative position" here.
2. **Baseline gate** — no null baseline for the asset means no way to know
   what "no skill" looks like there, so nothing is sized. An unknown
   baseline is not a zero baseline.
3. **Win-rate floor** — below :data:`MIN_WIN_RATE_SAMPLES` INDEPENDENT
   observations (overlap-corrected) the empirical win rate is discarded for
   the null, giving zero edge. Six evaluations that happened to be 5-for-6
   is not an 83% edge; the 95% interval on that runs from roughly 36% to
   99%.
4. **Negative-edge gate** — ``f*`` at or below zero means no edge. Zero.
5. **Fractional Kelly** — multiply by :data:`KELLY_FRACTION` (0.5). Half
   Kelly keeps about three quarters of the growth rate for roughly half the
   volatility, and is the standard practitioner setting.
6. **Volatility overlay** — HIGH volatility regime halves it again. Now a
   second-order correction rather than a patch over the wrong sign: the
   ``sigma^2`` denominator already does the first-order work.
7. **Event overlay** — a scheduled unlock inside
   :data:`UNLOCK_BLACKOUT_DAYS` forces zero. A dated supply event is a
   binary risk that a volatility estimate does not capture.
8. **Per-asset cap** — :data:`MAX_POSITION_FRACTION` (25%).

``payoff_ratio`` is still recorded on every row. It is DESCRIPTIVE only —
the asymmetry of the scenario band — and no longer feeds the sizing.

A NULL never triggers an overlay. "No unlock date on file" is not "an
unlock is imminent", and conflating them would zero out every asset whose
tokenomics nobody has entered yet.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import ForecastEvaluation
from src.forecasting.scenario_engine import ScenarioSet

logger = logging.getLogger(__name__)

#: Fractional-Kelly multiplier. 0.5 = "half Kelly".
KELLY_FRACTION = 0.5

#: Hard ceiling for a single asset, as a fraction of the portfolio.
MAX_POSITION_FRACTION = 0.25

#: Minimum INDEPENDENT observations before an empirical win rate is trusted
#: at all. Below this the default is used — see the module docstring.
#:
#: Compared against ``effective_sample_size``, not a row count (2026-07-31).
#: Thirty consecutive daily BUY evaluations at a 30-day horizon share 29/30
#: of their price path and are worth ~1 observation; gating on rows let that
#: single observation set Kelly's ``p``. Reaching 30 independent ones needs
#: roughly 30 x 30 = 900 calendar days of BUY signals on one asset, which is
#: the honest cost of the estimate this constant claims to protect.
MIN_WIN_RATE_SAMPLES = 30

#: Conservative stand-in when history is too thin. A coin flip: it grants
#: no edge of its own, so any resulting position comes purely from an
#: asymmetric payoff rather than from claimed predictive skill.
DEFAULT_WIN_RATE = 0.50

#: HIGH volatility regime multiplier.
HIGH_VOL_MULTIPLIER = 0.5

#: A scheduled unlock within this many days forces the position to zero.
UNLOCK_BLACKOUT_DAYS = 14

#: Only this action is ever sized.
SIZEABLE_ACTIONS = ("BUY",)


@dataclass(slots=True)
class PositionSizing:
    """One asset's sizing, with every step that produced it."""

    symbol: str
    action: str
    kelly_fraction: float
    adjusted_fraction: float
    win_rate: float
    win_rate_source: str
    win_rate_samples: int
    payoff_ratio: float | None
    risk_factors: dict[str, Any] = field(default_factory=dict)

    @property
    def position_size_pct(self) -> float:
        return self.adjusted_fraction * 100.0

    def summary(self) -> str:
        payoff = "n/a" if self.payoff_ratio is None else f"{self.payoff_ratio:.2f}"
        return (
            f"{self.symbol} {self.action}: f*={self.kelly_fraction:.4f} -> "
            f"{self.adjusted_fraction:.4f} ({self.position_size_pct:.2f}%) "
            f"[p={self.win_rate:.2f} n={self.win_rate_samples} "
            f"({self.win_rate_source}), b={payoff}]"
        )


class PositionSizer:
    """Kelly sizing with mandatory conservatism layers."""

    def __init__(
        self,
        *,
        kelly_fraction: float = KELLY_FRACTION,
        max_position: float = MAX_POSITION_FRACTION,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if not 0 < kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if not 0 < max_position <= 1:
            raise ValueError("max_position must be in (0, 1]")
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position

    # ------------------------------------------------------------------
    async def load_win_rate(self, symbol: str | None = None) -> tuple[float, str, int]:
        """Win rate for the Kelly formula: ``(win_rate, source, sample_size)``.

        Two sources, in order of preference (Phase 9):

        1. **Per-asset DECISION accuracy** — how often a BUY on THIS asset
           actually worked out, at the 30-day horizon. Requires ``symbol``.
        2. **Bitcoin forecast direction accuracy** — the original Phase 8
           source, used for every asset when (1) is unavailable.

        Source (1) is preferred because source (2) answers a different
        question. ``f* = (b*p - q)/b`` reads ``p`` as "probability THIS bet
        wins"; the forecast hit rate measures whether a Monte Carlo P50 for
        BITCOIN pointed the right way, which is neither this asset nor this
        bet. It was a stand-in until decision-level evaluation existed, and
        it necessarily gave all 21 assets the same number.

        Below :data:`MIN_WIN_RATE_SAMPLES` either way, the empirical value is
        REPORTED but not USED — the caller gets the default and the real
        sample size, so a thin estimate can never look authoritative.
        """
        if symbol:
            # Local import: src.evaluation imports MIN_WIN_RATE_SAMPLES from
            # this module, so a top-level import would be circular.
            from src.evaluation.decision_metrics import load_win_rate_for

            try:
                per_asset = await load_win_rate_for(symbol)
            except Exception as exc:  # noqa: BLE001 - never block sizing
                logger.warning(
                    "Per-asset win-rate lookup failed for %s (%s: %s) — "
                    "falling back to the forecast-derived rate",
                    symbol, type(exc).__name__, exc,
                )
                per_asset = None
            if per_asset is not None:
                return per_asset

        try:
            async with session_scope() as session:
                total = (
                    await session.execute(
                        select(func.count()).select_from(ForecastEvaluation)
                    )
                ).scalar_one()
                correct = (
                    await session.execute(
                        select(func.count())
                        .select_from(ForecastEvaluation)
                        .where(ForecastEvaluation.directional_correct.is_(True))
                    )
                ).scalar_one()
        except Exception as exc:  # noqa: BLE001 - sizing must never be blocked
            logger.warning(
                "Win-rate lookup failed (%s: %s) — using the conservative "
                "default %.2f", type(exc).__name__, exc, DEFAULT_WIN_RATE,
            )
            return DEFAULT_WIN_RATE, "default_lookup_failed", 0

        if total < MIN_WIN_RATE_SAMPLES:
            empirical = (correct / total) if total else float("nan")
            # Deliberately does NOT name the value that will be used. The
            # caller (position_pipeline) substitutes the asset's OWN null
            # baseline here, which ranges 0.390-0.667 across the registry —
            # so a log line announcing "using 0.50" was simply false, and
            # anyone reading it would have believed a flat 0.50 went into
            # Kelly. Provenance lives in `win_rate_source` on the stored row.
            logger.info(
                "[kelly] only %d forecast evaluation(s) (need %d) — the "
                "empirical %.2f is DISCARDED; the caller substitutes a "
                "skill-free default. A sample this small carries almost no "
                "information.",
                total, MIN_WIN_RATE_SAMPLES,
                0.0 if math.isnan(empirical) else empirical,
            )
            return DEFAULT_WIN_RATE, "default_insufficient_data", total

        win_rate = correct / total
        logger.info(
            "[kelly] empirical win rate %.4f from %d evaluation(s)",
            win_rate, total,
        )
        return win_rate, "empirical", total

    # ------------------------------------------------------------------
    @staticmethod
    async def load_null_win_rate(symbol: str, horizon_days: int) -> float | None:
        """Skill-free BUY win rate for this asset and horizon, or ``None``.

        This is the number ``p`` has to BEAT. Returning ``None`` when the
        asset lacks enough price history is deliberate: sizing then refuses
        rather than falling back to 0.50, because 0.50 is not the skill-free
        rate for anything that drifts (BTC +30d is 60.9%).
        """
        # Local import: null_model pulls constants from src.evaluation, which
        # in turn reaches into this module for MIN_WIN_RATE_SAMPLES.
        from src.evaluation.null_model import load_null_baselines

        try:
            async with session_scope() as session:
                baselines = await load_null_baselines(session, (horizon_days,))
        except Exception as exc:  # noqa: BLE001 - sizing must never be blocked
            logger.warning(
                "Null-baseline lookup failed for %s (%s: %s) — that asset will "
                "not be sized", symbol, type(exc).__name__, exc,
            )
            return None

        baseline = baselines.get((symbol, horizon_days))
        if baseline is None:
            logger.debug(
                "[kelly] no null baseline for %s at +%dd — not sizing",
                symbol, horizon_days,
            )
            return None
        return baseline.buy_hit_rate

    # ------------------------------------------------------------------
    @staticmethod
    def excess_return(
        win_rate: float, null_win_rate: float, gain: float, loss: float
    ) -> float:
        """Expected EXCESS return over the horizon: ``(p - p_null)*(gain+loss)``.

        Under the null a decisive BUY returns ``p_null*gain -
        (1-p_null)*loss``; with skill ``p`` it returns ``p*gain -
        (1-p)*loss``. The difference is exactly ``(p - p_null)*(gain + loss)``
        — the value of moving that much probability mass from the losing
        outcome to the winning one.

        ``p_null`` rather than 0.50 is the whole point. A skill-free BUY on a
        drifting asset wins far more than half its decisive cases (BTC +30d:
        60.9%), so 0.50 would credit the drift to the model.
        """
        return (win_rate - null_win_rate) * (gain + loss)

    @staticmethod
    def continuous_kelly(excess_return: float, variance: float) -> float:
        """``f* = mu / sigma^2`` — growth-optimal fraction, continuous case.

        The binary ``(b*p - q)/b`` it replaced required ``p`` and ``b`` to
        describe the same two-outcome wager, which they never did here. This
        form needs only an expected excess return and a variance over the
        SAME holding period, which is what the scenario engine and
        ``realized_vol_30d`` actually provide.

        Zero excess return gives exactly zero, and the fraction shrinks as
        variance grows. Both were false of the previous formula.
        """
        if variance <= 0:
            return 0.0
        return excess_return / variance

    # ------------------------------------------------------------------
    def size(
        self,
        symbol: str,
        action: str,
        scenarios: ScenarioSet,
        coin_dna: dict | None = None,
        *,
        win_rate: float = DEFAULT_WIN_RATE,
        win_rate_source: str = "default_insufficient_data",
        win_rate_samples: int = 0,
        null_win_rate: float | None = None,
        horizon_variance: float | None = None,
    ) -> PositionSizing:
        """Full sizing pipeline for one asset. Pure — no I/O.

        ``null_win_rate`` is the asset's skill-free BUY rate at this horizon
        and ``horizon_variance`` the variance of its horizon return. Both are
        REQUIRED to size: without the first there is no way to know what "no
        skill" looks like on this asset, and without the second no scale to
        divide the edge by.
        """
        coin_dna = coin_dna or {}
        risk: dict[str, Any] = {}

        # Descriptive only since 2026-07-31 — recorded, never sized from.
        payoff = scenarios.payoff_ratio
        if payoff is None:
            risk["no_payoff"] = "bear scenario shows no downside; cannot size"
            return self._zero(
                symbol, action, 0.0, win_rate, win_rate_source, win_rate_samples,
                payoff, risk,
            )

        gain = scenarios.expected_return_bull
        loss = scenarios.expected_drawdown_bear
        risk["inputs"] = {
            "win_rate": round(win_rate, 6),
            "null_win_rate": (
                None if null_win_rate is None else round(null_win_rate, 6)
            ),
            "expected_gain": round(gain, 6),
            "expected_loss": round(loss, 6),
            "horizon_variance": (
                None if horizon_variance is None else round(horizon_variance, 8)
            ),
            "payoff_ratio_descriptive": round(payoff, 6),
            "scenario_source": scenarios.source,
            "expected_value": round(scenarios.expected_value, 6),
        }

        # (1) Direction gate — only a BUY is ever sized.
        if action not in SIZEABLE_ACTIONS:
            risk["direction_gate"] = (
                f"action is {action}; only {'/'.join(SIZEABLE_ACTIONS)} is sized "
                f"(no shorting, no leverage)"
            )
            return self._zero(
                symbol, action, 0.0, win_rate, win_rate_source, win_rate_samples,
                payoff, risk,
            )

        # (2) Baseline gate. An unknown null is not a zero null: without it
        # there is no way to tell skill from the asset's own drift, and
        # assuming 0.50 is exactly the error this rewrite removed.
        if null_win_rate is None:
            risk["no_null_baseline"] = (
                "no skill-free baseline for this asset/horizon — cannot "
                "separate edge from drift, so nothing is sized"
            )
            return self._zero(
                symbol, action, 0.0, win_rate, win_rate_source, win_rate_samples,
                payoff, risk,
            )
        if horizon_variance is None or horizon_variance <= 0:
            risk["no_variance"] = (
                f"horizon variance {horizon_variance} is unusable; f* = mu/sigma^2 "
                f"has no scale to divide by"
            )
            return self._zero(
                symbol, action, 0.0, win_rate, win_rate_source, win_rate_samples,
                payoff, risk,
            )

        mu = self.excess_return(win_rate, null_win_rate, gain, loss)
        raw = self.continuous_kelly(mu, horizon_variance)
        risk["inputs"]["excess_return"] = round(mu, 8)
        risk["inputs"]["raw_kelly"] = round(raw, 6)

        # (3) Negative or zero edge -> no bet. Kelly's own answer, not an
        # override. p == p_null lands here exactly, which is the intended
        # resting state until skill is actually measured.
        if raw <= 0:
            risk["no_edge"] = (
                f"f* = mu/sigma^2 = {raw:.6f} <= 0 at p={win_rate:.4f} vs "
                f"null {null_win_rate:.4f} (excess return {mu:+.6f})"
            )
            return self._zero(
                symbol, action, raw, win_rate, win_rate_source, win_rate_samples,
                payoff, risk,
            )

        fraction = raw

        # (3) Fractional Kelly — always, never full.
        fraction *= self.kelly_fraction
        risk["fractional_kelly"] = {
            "multiplier": self.kelly_fraction,
            "after": round(fraction, 6),
        }

        # (4) Volatility overlay.
        if coin_dna.get("vol_regime") == "HIGH":
            fraction *= HIGH_VOL_MULTIPLIER
            risk["high_volatility"] = {
                "multiplier": HIGH_VOL_MULTIPLIER,
                "after": round(fraction, 6),
            }

        # (5) Event overlay. A NULL is NOT an imminent unlock.
        unlock_days = coin_dna.get("days_to_next_unlock")
        if unlock_days is not None and 0 <= unlock_days < UNLOCK_BLACKOUT_DAYS:
            risk["unlock_blackout"] = {
                "days_to_next_unlock": unlock_days,
                "threshold": UNLOCK_BLACKOUT_DAYS,
                "after": 0.0,
            }
            return self._zero(
                symbol, action, raw, win_rate, win_rate_source, win_rate_samples,
                payoff, risk,
            )

        # (6) Per-asset cap.
        if fraction > self.max_position:
            risk["position_cap"] = {
                "cap": self.max_position,
                "before": round(fraction, 6),
            }
            fraction = self.max_position

        sizing = PositionSizing(
            symbol=symbol,
            action=action,
            kelly_fraction=raw,
            adjusted_fraction=fraction,
            win_rate=win_rate,
            win_rate_source=win_rate_source,
            win_rate_samples=win_rate_samples,
            payoff_ratio=payoff,
            risk_factors=risk,
        )
        logger.info("[kelly] %s", sizing.summary())
        return sizing

    # ------------------------------------------------------------------
    @staticmethod
    def _zero(
        symbol: str,
        action: str,
        raw: float,
        win_rate: float,
        win_rate_source: str,
        win_rate_samples: int,
        payoff: float | None,
        risk: dict,
    ) -> PositionSizing:
        sizing = PositionSizing(
            symbol=symbol,
            action=action,
            kelly_fraction=raw,
            adjusted_fraction=0.0,
            win_rate=win_rate,
            win_rate_source=win_rate_source,
            win_rate_samples=win_rate_samples,
            payoff_ratio=payoff,
            risk_factors=risk,
        )
        logger.info("[kelly] %s -> ZERO (%s)", symbol, ", ".join(risk))
        return sizing


def portfolio_exposure(sizings: list[PositionSizing]) -> dict[str, Any]:
    """Total exposure across a set of sizings.

    Kelly sizes each asset AGAINST CASH, independently. Nothing in the
    per-asset formula knows the other twenty exist, so twenty-one assets
    capped at 25% each can sum to 525% — leverage this system explicitly
    forbids. The per-asset cap alone does not bound the portfolio; this
    reports the total so an over-allocated book is visible rather than
    implied. Crypto assets are also highly correlated (see ``btc_corr_30d``,
    typically 0.5-0.9 here), so naive summing understates the true risk
    concentration further.
    """
    total = sum(s.adjusted_fraction for s in sizings)
    non_zero = [s for s in sizings if s.adjusted_fraction > 0]
    return {
        "total_fraction": round(total, 6),
        "total_pct": round(total * 100, 4),
        "positions": len(non_zero),
        "over_allocated": total > 1.0,
        "largest": (
            max(non_zero, key=lambda s: s.adjusted_fraction).symbol
            if non_zero
            else None
        ),
    }

"""Deterministic decision from a scored panel (Phase 8 Part 7).

In consolidated mode the Master Agent does NOT run per asset — that would
double the quota back to 2 calls/asset. The final BUY/HOLD/SELL, the
confidence, and the narrative are therefore composed here, in Python.

Everything in this module is a pure function of its inputs. Given the same
panel, weights and regime it always returns the same decision, and any
figure can be re-derived months later from the stored ``sub_agents_json``.
That is the property the Master Agent could never offer.

What is lost without the CIO, stated plainly
--------------------------------------------
The Master Agent did more than pick an action. It reconciled the panel
against the Monte Carlo forecast, weighed regime and sector context in
prose, and wrote a risk warning naming the specific metric that would
invalidate the call. None of that is reproduced here: this module writes an
honest, mechanical summary of what the numbers say, and labels it as such.

Bitcoin can keep the full treatment — ``--mode hybrid`` runs the five
specialists plus the Master for BTC and the consolidated path for the
altcoins, at a cost of 5 extra calls per day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.agents.master_agent import PanelFusion
from src.agents.schemas import Action, MasterDecision, SubAgentResponse

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime coupling
    from src.features.intraday_engine import IntradayBar

logger = logging.getLogger(__name__)

#: Weighted-score thresholds for the action. Deliberately symmetric and
#: wide: a |score| below 20 out of 100 is not a directional edge, it is
#: noise, and the honest answer there is HOLD.
BUY_THRESHOLD = 20.0
SELL_THRESHOLD = -20.0

#: Weighted |score| at which the "strength" component saturates. 50/100 is
#: already a strong panel consensus; demanding 100 would mean confidence
#: could never reach its top band in practice.
STRENGTH_SATURATION = 50.0

#: Confidence blend: how much comes from the size of the weighted score
#: versus how much of the panel actually agrees with its direction.
_STRENGTH_WEIGHT = 0.55
_AGREEMENT_WEIGHT = 0.45

#: Confidence is capped below 100 for a mechanically-composed decision.
#: A deterministic rule cannot notice the thing nobody put in the features,
#: and a flat 100 would misrepresent that.
MAX_MECHANICAL_CONFIDENCE = 90

#: Pillars reporting below this fraction of the full five trigger an
#: explicit confidence haircut.
_FULL_PANEL = 5


@dataclass(slots=True)
class ComposedDecision:
    """A decision plus the arithmetic that produced it."""

    decision: MasterDecision
    weighted_score: float
    agreement: float
    action: Action
    confidence: int
    #: Phase 8 Part 8 — set only when 4H context was supplied (BTC).
    intraday: "IntradayOverlay | None" = None

    def summary(self) -> str:
        text = (
            f"{self.action.value} (confidence {self.confidence}/100) from "
            f"weighted score {self.weighted_score:+.1f}, panel agreement "
            f"{self.agreement * 100:.0f}%"
        )
        if self.intraday is not None and self.intraday.applied:
            text += (
                f" | 4H overlay {self.intraday.stance}: confidence "
                f"{self.intraday.confidence_before} -> "
                f"{self.intraday.confidence_after}"
            )
        return text


def panel_agreement(responses: list[SubAgentResponse], weighted_score: float) -> float:
    """Fraction of reporting pillars whose sign matches the fused score.

    Pillars scoring exactly 0 are excluded from both sides of the ratio:
    a neutral pillar neither confirms nor contradicts a direction, and
    counting it as disagreement would understate a genuine consensus.
    """
    directional = [r for r in responses if r.score != 0]
    if not directional or weighted_score == 0:
        return 0.0
    target = 1 if weighted_score > 0 else -1
    matching = sum(1 for r in directional if (1 if r.score > 0 else -1) == target)
    return matching / len(directional)


def decide_action(weighted_score: float) -> Action:
    """BUY / SELL / HOLD from the weighted score."""
    if weighted_score >= BUY_THRESHOLD:
        return Action.BUY
    if weighted_score <= SELL_THRESHOLD:
        return Action.SELL
    return Action.HOLD


def compute_confidence(
    weighted_score: float, agreement: float, pillars_reporting: int
) -> int:
    """0-100 confidence from score magnitude, agreement and panel completeness."""
    strength = min(abs(weighted_score) / STRENGTH_SATURATION, 1.0)
    raw = _STRENGTH_WEIGHT * strength + _AGREEMENT_WEIGHT * agreement
    confidence = raw * MAX_MECHANICAL_CONFIDENCE

    # An incomplete panel cannot be as trustworthy as a full one.
    if pillars_reporting < _FULL_PANEL:
        confidence *= pillars_reporting / _FULL_PANEL
    return max(0, min(MAX_MECHANICAL_CONFIDENCE, int(round(confidence))))


def _describe_panel(responses: list[SubAgentResponse]) -> tuple[str, str]:
    """(bullish description, bearish description) of the panel."""
    bullish = [r for r in responses if r.score > 0]
    bearish = [r for r in responses if r.score < 0]
    render = lambda group: ", ".join(  # noqa: E731 - local formatting shorthand
        f"{r.agent_name} {r.score:+d}" for r in sorted(group, key=lambda x: -abs(x.score))
    )
    return render(bullish) or "tidak ada", render(bearish) or "tidak ada"


def compose_decision(
    symbol: str,
    analysis_date: str,
    responses: list[SubAgentResponse],
    fusion: PanelFusion,
    *,
    reasoning_summary: str = "",
    regime: str | None = None,
    sector_signal: str | None = None,
    intraday_bars: list["IntradayBar"] | None = None,
) -> ComposedDecision:
    """Build the final decision deterministically. Pure function.

    ``intraday_bars`` (Phase 8 Part 8, Bitcoin only) applies the 4-hour
    overlay: it can only lower ``confidence``, never change ``action``. Omit it
    and every value returned is identical to the pre-Part-8 behaviour.
    """
    weighted = fusion.weighted_score
    agreement = panel_agreement(responses, weighted)
    action = decide_action(weighted)
    confidence = compute_confidence(weighted, agreement, len(responses))

    # The overlay reads `action` to pick which conflict test applies, and
    # returns only a confidence. `action` is NOT reassigned anywhere below.
    overlay = (
        evaluate_intraday_overlay(action, confidence, intraday_bars)
        if intraday_bars
        else None
    )
    if overlay is not None:
        confidence = overlay.confidence_after

    bullish, bearish = _describe_panel(responses)
    regime_text = regime or "tidak terdeteksi"

    summary_parts = [
        f"Keputusan {action.value} untuk {symbol} pada {analysis_date} dihitung "
        f"secara deterministik dari skor panel terbobot {weighted:+.1f} "
        f"(-100..+100), dengan kesepakatan panel {agreement * 100:.0f}% dari "
        f"{len(responses)} pilar yang melapor.",
        f"Rezim pasar terdeteksi: {regime_text}; bobot pilar yang dipakai: "
        f"{fusion.weights_line()}.",
        f"Pilar bullish: {bullish}. Pilar bearish: {bearish}.",
    ]
    if sector_signal:
        summary_parts.append(f"Konteks rotasi sektor: {sector_signal}")
    if overlay is not None and overlay.stance != "unavailable":
        summary_parts.append(f"Overlay 4 jam: {overlay.reason}")
    if reasoning_summary:
        summary_parts.append(f"Ringkasan analis: {reasoning_summary}")
    summary_parts.append(
        "Catatan: ringkasan ini disusun secara mekanis dari skor pilar, bukan "
        "oleh Master Agent (CIO) — tidak ada rekonsiliasi naratif terhadap "
        "forecast Monte Carlo."
    )

    threshold_text = (
        f"di atas +{BUY_THRESHOLD:.0f}" if action is Action.BUY
        else f"di bawah {SELL_THRESHOLD:.0f}" if action is Action.SELL
        else f"di antara {SELL_THRESHOLD:.0f} dan +{BUY_THRESHOLD:.0f}"
    )
    flip_note = (
        f"Keputusan ini berbalik bila skor terbobot bergerak keluar dari zona "
        f"{threshold_text}."
    )
    risk_parts = [
        flip_note,
        f"Pantau pilar dengan bobot terbesar pada rezim ini "
        f"({fusion.weights_line()}) — perubahan tanda pada pilar tersebut yang "
        f"paling mungkin membalikkan skor.",
    ]
    if len(responses) < _FULL_PANEL:
        risk_parts.append(
            f"Hanya {len(responses)}/{_FULL_PANEL} pilar melapor, sehingga "
            f"confidence sudah dipotong secara proporsional."
        )
    if overlay is not None and overlay.stance == "conflicts":
        risk_parts.append(
            f"Tape 4 jam sedang bertentangan dengan arah harian, sehingga "
            f"confidence sudah dipotong dari {overlay.confidence_before} ke "
            f"{overlay.confidence_after}. Arah keputusan tetap "
            f"{action.value} — 4 jam mengatur TIMING, bukan arah."
        )
    risk_parts.append(
        "Output ini adalah dukungan keputusan berbasis data, bukan nasihat "
        "keuangan."
    )

    decision = MasterDecision(
        action=action,
        confidence_score=confidence,
        executive_summary=" ".join(summary_parts),
        risk_warning=" ".join(risk_parts),
    )
    composed = ComposedDecision(
        decision=decision,
        weighted_score=weighted,
        agreement=agreement,
        action=action,
        confidence=confidence,
        intraday=overlay,
    )
    logger.info("[composer] %s: %s", symbol, composed.summary())
    return composed


# ---------------------------------------------------------------------------
# Phase 8 Part 8 — BTC intraday (4H) overlay
#
# "1D is King, 4H is Adviser." The overlay may reduce confidence or counsel
# patience; it can NEVER change the action.
#
# That invariant is enforced STRUCTURALLY, not by an assertion: the function
# below does not accept or return an :class:`Action` at all. There is no code
# path through it that could flip BUY into SELL, so no future edit can
# accidentally introduce one. Precedent for the shape: the regime layer
# dampens confidence via MIN_CONFIDENCE_TO_APPLY / TRANSITION_CONFIDENCE_FACTOR
# without ever choosing an action either.
# ---------------------------------------------------------------------------

#: 4H RSI above/below which entry timing is considered stretched. Standard
#: RSI thresholds; on 4h bars they describe ~2.3 days of tape, so they flag
#: "stretched right now", not "stretched this month".
INTRADAY_OVERBOUGHT = 70.0
INTRADAY_OVERSOLD = 30.0

#: Confidence multiplier when the 4-hour tape contradicts the daily call.
#: 0.75 is chosen to MATTER without being able to reverse anything: a SELL at
#: confidence 72 becomes a SELL at 54, still a SELL. Flipping would require
#: changing the action, which this module cannot do.
INTRADAY_CONFLICT_FACTOR = 0.75

#: Bars needed to judge the window's direction. With one bar there is no
#: "direction of the window", only a single close.
INTRADAY_MIN_BARS = 2


@dataclass(slots=True)
class IntradayOverlay:
    """Result of the 4H overlay: a confidence adjustment plus its reason."""

    stance: str  # 'supports' | 'conflicts' | 'neutral' | 'unavailable'
    factor: float
    confidence_before: int
    confidence_after: int
    reason: str

    @property
    def applied(self) -> bool:
        return self.confidence_after != self.confidence_before


def _window_direction(bars: list[Any]) -> int:
    """+1 if the 4H window closed higher than it opened, -1 lower, 0 flat."""
    closes = [bar.close for bar in bars if bar.close is not None]
    if len(closes) < INTRADAY_MIN_BARS:
        return 0
    if closes[-1] > closes[0]:
        return 1
    if closes[-1] < closes[0]:
        return -1
    return 0


def evaluate_intraday_overlay(
    action: Action,
    confidence: int,
    bars: list["IntradayBar"] | None,
) -> IntradayOverlay:
    """Judge whether the 4-hour tape supports or contradicts the daily call.

    ``action`` is READ ONLY — it selects which conflict test applies and is
    never returned. HOLD is always neutral: a non-directional call has no
    direction for the intraday tape to contradict.

    Two independent conflict triggers, either sufficient:

    * **Momentum conflict** — the last closed bar's MACD and the direction of
      the whole window BOTH oppose the daily action. Requiring both keeps a
      single red 4-hour candle from counting as a conflict.
    * **Stretched timing** — 4H RSI is overbought while the daily call is BUY,
      or oversold while it is SELL. This is the "don't buy the top, wait for
      the pullback" case, and it fires even when momentum still agrees,
      because agreeing momentum at RSI 75 is exactly what a local top looks
      like.
    """
    if not bars or len(bars) < INTRADAY_MIN_BARS:
        return IntradayOverlay(
            stance="unavailable",
            factor=1.0,
            confidence_before=confidence,
            confidence_after=confidence,
            reason=(
                "Konteks 4 jam tidak tersedia (butuh minimal "
                f"{INTRADAY_MIN_BARS} bar closed) — confidence tidak diubah."
            ),
        )

    if action is Action.HOLD:
        return IntradayOverlay(
            stance="neutral",
            factor=1.0,
            confidence_before=confidence,
            confidence_after=confidence,
            reason=(
                "Keputusan harian HOLD tidak punya arah yang bisa dikonfirmasi "
                "atau dibantah oleh tape 4 jam — overlay tidak diterapkan."
            ),
        )

    last = bars[-1]
    daily_direction = 1 if action is Action.BUY else -1
    window = _window_direction(bars)

    macd_direction = 0
    if last.macd_state == "bullish":
        macd_direction = 1
    elif last.macd_state == "bearish":
        macd_direction = -1

    momentum_conflict = (
        macd_direction != 0
        and window != 0
        and macd_direction == window
        and macd_direction != daily_direction
    )

    rsi = last.rsi_14
    stretched = False
    stretch_note = ""
    if rsi is not None:
        if action is Action.BUY and rsi >= INTRADAY_OVERBOUGHT:
            stretched = True
            stretch_note = (
                f"RSI 4 jam {rsi:.1f} sudah overbought (>= "
                f"{INTRADAY_OVERBOUGHT:.0f}) sementara keputusan harian BUY — "
                f"timing masuk buruk, pertimbangkan menunggu koreksi."
            )
        elif action is Action.SELL and rsi <= INTRADAY_OVERSOLD:
            stretched = True
            stretch_note = (
                f"RSI 4 jam {rsi:.1f} sudah oversold (<= "
                f"{INTRADAY_OVERSOLD:.0f}) sementara keputusan harian SELL — "
                f"timing keluar buruk, potensi bounce jangka pendek."
            )

    if momentum_conflict or stretched:
        adjusted = max(0, int(round(confidence * INTRADAY_CONFLICT_FACTOR)))
        reasons: list[str] = []
        if momentum_conflict:
            reasons.append(
                f"MACD 4 jam {last.macd_state} dan arah window 4 jam "
                f"{'naik' if window > 0 else 'turun'} keduanya berlawanan "
                f"dengan keputusan harian {action.value}."
            )
        if stretch_note:
            reasons.append(stretch_note)
        reasons.append(
            f"Confidence dipotong {int((1 - INTRADAY_CONFLICT_FACTOR) * 100)}% "
            f"({confidence} -> {adjusted}); ARAH keputusan TIDAK diubah — "
            f"tape 4 jam mengatur timing, bukan arah."
        )
        return IntradayOverlay(
            stance="conflicts",
            factor=INTRADAY_CONFLICT_FACTOR,
            confidence_before=confidence,
            confidence_after=adjusted,
            reason=" ".join(reasons),
        )

    return IntradayOverlay(
        stance="supports",
        factor=1.0,
        confidence_before=confidence,
        confidence_after=confidence,
        reason=(
            f"Tape 4 jam (MACD {last.macd_state}, RSI "
            f"{'n/a' if rsi is None else f'{rsi:.1f}'}) tidak bertentangan "
            f"dengan keputusan harian {action.value} — confidence dipertahankan. "
            f"Konfirmasi tidak menaikkan confidence: 6 bar intraday bukan bukti "
            f"tambahan atas tesis harian."
        ),
    )

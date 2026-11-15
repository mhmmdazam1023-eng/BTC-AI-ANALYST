"""RegimeDetector: rule-based market-regime classification (Phase 8 Part 2).

Reads Bitcoin's engineered features and answers one question: *what kind of
market is this?* The answer drives how much each analysis pillar is trusted
(see :mod:`src.agents.weighting_engine`).

Deliberately rule-based, not learned — same reason feature engineering is
deterministic (project rule since 2026-07-08): a regime label that changes
the weighting of a real decision must be explainable, reproducible, and
auditable months later. Every input is persisted alongside the verdict.

The five regimes and their rules
--------------------------------
======================  ====================================================
Regime                  Conditions
======================  ====================================================
``BULL_TRENDING``       close > SMA200, volume elevated, F&G > 50, MACD > 0
``BEAR_TRENDING``       close < SMA200, volume elevated, F&G < 30, MACD < 0
``SIDEWAYS_LOW_VOL``    close within +/-5% of SMA50, realized vol < 1y mean
``SIDEWAYS_HIGH_VOL``   close within +/-5% of SMA50, realized vol > 1y mean
``CRASH_RECOVERY``      drawdown > 20% from the 90d high, RSI rising from <30
======================  ====================================================

Plus :attr:`MarketRegime.UNDETERMINED` when nothing scores above
:data:`MIN_REGIME_STRENGTH` — an honest "I don't know" that makes the
weighting engine fall back to equal weights.

Two design decisions worth knowing
----------------------------------
1. **Window.** The task specifies "30 days of BTC data", but SMA200 needs
   200 days, the 90-day ATH needs 90, and the 1-year volatility baseline
   needs 365. So :data:`DEFAULT_LOOKBACK_DAYS` of history is loaded to build
   the reference levels, and the classification is evaluated **on the latest
   day** using 30-day aggregates. Same intent, arithmetically possible.

2. **Soft scoring, hard rules.** Each condition returns a score in [0, 1]
   via :func:`_ramp` (0.5 exactly at the threshold, 1.0 once comfortably
   past it) rather than a bare boolean. A regime one dollar above SMA200 is
   not as convincing as one 30% above, and confidence must say so. The
   winning regime is still chosen by the rules; only the *confidence* is
   graded.

``CRASH_RECOVERY`` takes precedence over ``BEAR_TRENDING`` when it scores
at least :data:`CRASH_PRIORITY_FLOOR`. Both are usually "true" in a
capitulation — deep below SMA200, fearful, negative MACD — but only
CRASH_RECOVERY carries the actionable nuance (forced deleveraging plus a
reflex bounce), and it implies very different pillar weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import BASE_SYMBOL, DailyFeature, MarketRegime, MarketRegimeRow

logger = logging.getLogger(__name__)

#: Ruleset identity, persisted with every row. Bump it whenever a threshold
#: or condition changes so historical rows stay interpretable.
RULE_VERSION = "v1"

#: History loaded to build reference levels. 365 (volatility baseline) plus
#: margin so every rolling window is fully warmed at the evaluated day.
DEFAULT_LOOKBACK_DAYS = 500

#: Minimum rows required to classify at all (SMA200 + a little slack).
MIN_HISTORY_DAYS = 210

#: Below this winning score the detector refuses to label the market.
MIN_REGIME_STRENGTH = 0.50

#: How recent the previous classification must be for a regime change to
#: count as a FRESH transition. Beyond this the daily job evidently did not
#: run, and the new regime has had days to establish itself.
TRANSITION_LOOKBACK_DAYS = 3

#: CRASH_RECOVERY outranks the trend regimes once it scores this well.
CRASH_PRIORITY_FLOOR = 0.60

#: Only these leaders can be overridden by CRASH_RECOVERY — a capitulation
#: reads as a downtrend, but a quiet range after a drawdown is a base.
_TRENDING_REGIMES = frozenset(
    {MarketRegime.BULL_TRENDING.value, MarketRegime.BEAR_TRENDING.value}
)

#: Half-width of the "sideways" band around SMA50, in percent.
SIDEWAYS_BAND_PCT = 5.0

#: Drawdown from the 90-day high that qualifies as a crash, in percent.
CRASH_DRAWDOWN_PCT = 20.0

#: RSI level the market must have climbed back to for "RSI naik dari < 30"
#: to count as an actual recovery. Without this the condition is satisfied
#: for most of a bear market: price is always >20% off the 90-day high and
#: RSI is forever dipping under 30 and ticking up a point. Verified against
#: June 2022 (RSI 20.3, sitting AT its 30-day low = still crashing, NOT
#: recovering) and Nov 2022 post-FTX.
RSI_RECOVERY_LEVEL = 40.0

#: The recovery leg must itself be at least this convincing before
#: CRASH_RECOVERY may win or override — see _score_regimes.
CRASH_RECOVERY_LEG_FLOOR = 0.50

#: "Current volume" is smoothed over this many days before being compared
#: with the 30-day average. A single day's volume is far too noisy to flip
#: a regime on; set to 1 for the literal one-day reading.
VOLUME_FAST_WINDOW = 7
VOLUME_SLOW_WINDOW = 30

#: Confidence blend: how much comes from rule strength vs how cleanly the
#: winner separated from the runner-up.
_STRENGTH_WEIGHT = 0.65
_MARGIN_WEIGHT = 0.35
#: Margin at which the separation term saturates.
_MARGIN_SATURATION = 0.30


class RegimeDetectionError(RuntimeError):
    """Raised when the regime cannot be determined (e.g. too little history)."""


def _ramp(value: float, threshold: float, width: float) -> float:
    """Soft indicator: 0.5 at ``threshold``, 1.0 at ``threshold + width``.

    Returns 0.0 at ``threshold - width``. A missing or non-finite input —
    including a non-finite *width*, which happens when the baseline it is
    derived from is still NaN — scores 0.0, so an absent feature simply
    fails its condition instead of poisoning the whole mean with NaN.
    """
    if value is None or width is None:
        return 0.0
    if not np.isfinite(value) or not np.isfinite(width) or width <= 0:
        return 0.0
    return float(np.clip(0.5 + (value - threshold) / (2.0 * width), 0.0, 1.0))


def _log_value(evidence: dict[str, float | None], key: str) -> float:
    """Evidence value for log formatting; missing -> NaN (0.0 stays 0.0)."""
    raw = evidence.get(key)
    return float("nan") if raw is None else float(raw)


@dataclass(slots=True)
class RegimeSnapshot:
    """One day's classification plus the evidence that produced it."""

    as_of: datetime
    regime: MarketRegime
    confidence: int
    symbol: str = BASE_SYMBOL
    evidence: dict[str, float | None] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    rule_version: str = RULE_VERSION
    #: The regime stored for the previous run, captured BEFORE this snapshot
    #: was persisted. ``None`` when there is no recent prior observation —
    #: which is NOT the same as "no change", see :meth:`just_changed`.
    previous_regime: MarketRegime | None = None
    previous_as_of: datetime | None = None

    @property
    def is_determined(self) -> bool:
        return self.regime is not MarketRegime.UNDETERMINED

    @property
    def just_changed(self) -> bool:
        """True when a RECENT prior observation carried a different label.

        This reports the FACT that the label moved. Whether that movement
        deserves a confidence penalty is policy, and belongs to
        :meth:`~src.agents.weighting_engine.DynamicWeightingEngine.evaluate`
        — which additionally exempts moves out of ``UNDETERMINED``. Keeping
        the fact and the policy apart means the detector never has to know
        how the weights are computed.

        Deliberately False when :attr:`previous_regime` is None: a first
        run, or a run after a long gap, has no established prior state, so
        nothing changed — treating "unknown" as "changed" would penalise
        every cold start.
        """
        return self.previous_regime is not None and self.previous_regime is not self.regime

    @property
    def clarified_from_undetermined(self) -> bool:
        """True when the detector went from having no opinion to having one."""
        return self.previous_regime is MarketRegime.UNDETERMINED and self.is_determined

    def summary(self) -> str:
        """One-line human-readable verdict (used in logs and prompts)."""
        if self.clarified_from_undetermined:
            change = " [clarified from UNDETERMINED]"
        elif self.just_changed and self.previous_regime is not None:
            change = f" [transition from {self.previous_regime.value}]"
        else:
            change = ""
        return (
            f"{self.regime.value} (confidence {self.confidence}/100) "
            f"as of {self.as_of.date()}{change}"
        )


class RegimeDetector:
    """Classifies the market regime from ``daily_features``."""

    def __init__(
        self,
        *,
        symbol: str = BASE_SYMBOL,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.symbol = symbol
        self.lookback_days = lookback_days

    # ------------------------------------------------------------------
    # 1) LOAD
    # ------------------------------------------------------------------
    async def load_history(self, as_of: datetime | None = None) -> pd.DataFrame:
        """Feature history for :attr:`symbol`, oldest -> newest.

        ``as_of`` truncates the series, which is what makes historical
        backtests (e.g. "classify 2021-11-09") free of look-ahead bias.
        """
        query = (
            select(
                DailyFeature.timestamp,
                DailyFeature.asset_close,
                DailyFeature.asset_volume,
                DailyFeature.macd,
                DailyFeature.rsi_14,
                DailyFeature.realized_vol_30d,
                DailyFeature.fear_greed_value,
            )
            .where(DailyFeature.symbol == self.symbol)
            .order_by(DailyFeature.timestamp.desc())
            .limit(self.lookback_days)
        )
        if as_of is not None:
            query = query.where(DailyFeature.timestamp <= as_of)

        async with session_scope() as session:
            rows = (await session.execute(query)).all()

        frame = pd.DataFrame(
            rows,
            columns=[
                "timestamp", "close", "volume", "macd",
                "rsi_14", "realized_vol_30d", "fear_greed_value",
            ],
        )
        if frame.empty:
            raise RegimeDetectionError(
                f"no {self.symbol} rows in daily_features — run the collectors first"
            )
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], utc=True
        ).dt.tz_localize(None)
        frame = frame.set_index("timestamp").sort_index()
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        return frame

    # ------------------------------------------------------------------
    # 2) CLASSIFY (pure — no I/O, unit-testable with synthetic frames)
    # ------------------------------------------------------------------
    def detect(self, history: pd.DataFrame) -> RegimeSnapshot:
        """Classify the LAST row of ``history``. Pure function."""
        if len(history) < MIN_HISTORY_DAYS:
            raise RegimeDetectionError(
                f"need >= {MIN_HISTORY_DAYS} days of history to compute SMA200, "
                f"got {len(history)} — backfill more candles "
                f"(scripts/backfill_history.py)"
            )

        evidence = self._build_evidence(history)
        scores = self._score_regimes(evidence)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # CRASH_RECOVERY outranks a TRENDING leader when clearly present: in
        # a capitulation both labels are true (deep below SMA200, fearful,
        # negative MACD), but only this one implies the sentiment/liquidity
        # -heavy weighting a reflex bounce needs.
        #
        # It deliberately does NOT override a SIDEWAYS leader. A market
        # chopping quietly on its 50-day average after a drawdown is a BASE,
        # not an active crash — and promoting a 0.79 crash score over a 1.00
        # sideways score would misread exactly that situation.
        crash_score = scores[MarketRegime.CRASH_RECOVERY.value]
        if (
            crash_score >= CRASH_PRIORITY_FLOOR
            and best_label in _TRENDING_REGIMES
        ):
            logger.info(
                "[regime] CRASH_RECOVERY (%.2f) takes precedence over trending "
                "regime %s (%.2f)", crash_score, best_label, best_score,
            )
            runner_up_score = best_score
            best_label, best_score = MarketRegime.CRASH_RECOVERY.value, crash_score

        if best_score < MIN_REGIME_STRENGTH:
            regime = MarketRegime.UNDETERMINED
            confidence = int(round(100 * best_score))
            logger.info(
                "[regime] no regime cleared the %.2f strength floor (best: %s "
                "at %.2f) — reporting UNDETERMINED so weighting falls back to "
                "equal weights", MIN_REGIME_STRENGTH, best_label, best_score,
            )
        else:
            regime = MarketRegime(best_label)
            margin = best_score - runner_up_score
            confidence = int(
                round(
                    100
                    * (
                        _STRENGTH_WEIGHT * best_score
                        + _MARGIN_WEIGHT
                        * float(np.clip(margin / _MARGIN_SATURATION, 0.0, 1.0))
                    )
                )
            )
        confidence = max(0, min(100, confidence))

        snapshot = RegimeSnapshot(
            as_of=history.index[-1].to_pydatetime().replace(tzinfo=UTC),
            regime=regime,
            confidence=confidence,
            symbol=self.symbol,
            evidence=evidence,
            scores={key: round(value, 4) for key, value in scores.items()},
        )
        logger.info(
            "[regime] %s | close=%.0f vs SMA200 %+.1f%% vs SMA50 %+.1f%% | "
            "vol_ratio=%.2f F&G=%.0f MACD=%+.2f%% | rv30=%.3f vs 1y %.3f | "
            "dd90=%+.1f%% RSI=%.0f (min30 %.0f)",
            snapshot.summary(),
            _log_value(evidence, "close"),
            _log_value(evidence, "pct_vs_sma_200"),
            _log_value(evidence, "pct_vs_sma_50"),
            _log_value(evidence, "volume_ratio"),
            _log_value(evidence, "fear_greed_value"),
            _log_value(evidence, "macd_pct"),
            _log_value(evidence, "realized_vol_30d"),
            _log_value(evidence, "vol_baseline_1y"),
            _log_value(evidence, "drawdown_from_ath_90d_pct"),
            _log_value(evidence, "rsi_14"),
            _log_value(evidence, "rsi_min_30d"),
        )
        return snapshot

    # ------------------------------------------------------------------
    @staticmethod
    def _build_evidence(history: pd.DataFrame) -> dict[str, float | None]:
        """Reference levels and readings at the latest day."""
        close = history["close"]
        volume = history["volume"]

        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        volume_fast = volume.rolling(VOLUME_FAST_WINDOW).mean()
        volume_slow = volume.rolling(VOLUME_SLOW_WINDOW).mean()
        ath_90 = close.rolling(90).max()
        # 1-year mean of realized vol. min_periods lets a shorter (but still
        # substantial) history produce a usable baseline instead of NaN.
        vol_baseline = history["realized_vol_30d"].rolling(365, min_periods=180).mean()
        rsi_min_30 = history["rsi_14"].rolling(30).min()

        def last(series: pd.Series) -> float | None:
            value = series.iloc[-1]
            return None if pd.isna(value) else float(value)

        close_now = last(close)
        sma_50_now = last(sma_50)
        sma_200_now = last(sma_200)
        macd_now = last(history["macd"])
        ath_now = last(ath_90)
        volume_fast_now = last(volume_fast)
        volume_slow_now = last(volume_slow)

        def pct_vs(level: float | None) -> float | None:
            if close_now is None or level is None or level == 0:
                return None
            return (close_now / level - 1.0) * 100.0

        return {
            "close": close_now,
            "sma_50": sma_50_now,
            "sma_200": sma_200_now,
            "pct_vs_sma_50": pct_vs(sma_50_now),
            "pct_vs_sma_200": pct_vs(sma_200_now),
            # MACD normalised by price so the threshold is scale-free:
            # a raw MACD of 500 means something very different at $10k and $100k.
            "macd_pct": (
                None
                if macd_now is None or not close_now
                else macd_now / close_now * 100.0
            ),
            "volume_ratio": (
                None
                if not volume_slow_now or volume_fast_now is None
                else volume_fast_now / volume_slow_now
            ),
            "fear_greed_value": last(history["fear_greed_value"]),
            "realized_vol_30d": last(history["realized_vol_30d"]),
            "vol_baseline_1y": last(vol_baseline),
            "drawdown_from_ath_90d_pct": (
                None
                if close_now is None or not ath_now
                else (close_now / ath_now - 1.0) * 100.0
            ),
            "rsi_14": last(history["rsi_14"]),
            "rsi_min_30d": last(rsi_min_30),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _score_regimes(evidence: dict[str, float | None]) -> dict[str, float]:
        """Soft score in [0, 1] for each candidate regime."""

        def value(key: str) -> float:
            raw = evidence.get(key)
            return float("nan") if raw is None else float(raw)

        vs_200 = value("pct_vs_sma_200")
        vs_50 = value("pct_vs_sma_50")
        macd_pct = value("macd_pct")
        volume_ratio = value("volume_ratio")
        fear_greed = value("fear_greed_value")
        realized_vol = value("realized_vol_30d")
        vol_baseline = value("vol_baseline_1y")
        drawdown = value("drawdown_from_ath_90d_pct")
        rsi = value("rsi_14")
        rsi_min = value("rsi_min_30d")

        # Volatility comparisons are relative to the baseline's own size, so
        # the same rule works whether BTC is at 40% or 90% annualised vol.
        vol_width = abs(vol_baseline) * 0.10 if np.isfinite(vol_baseline) else float("nan")
        # Distance INSIDE the sideways band: positive when within +/-5%.
        band_slack = SIDEWAYS_BAND_PCT - abs(vs_50) if np.isfinite(vs_50) else float("nan")

        elevated_volume = _ramp(volume_ratio, 1.0, 0.15)
        in_band = _ramp(band_slack, 0.0, SIDEWAYS_BAND_PCT / 2.0)

        bull = np.mean([
            _ramp(vs_200, 0.0, 5.0),
            elevated_volume,
            _ramp(fear_greed, 50.0, 15.0),
            _ramp(macd_pct, 0.0, 0.5),
        ])
        bear = np.mean([
            _ramp(-vs_200, 0.0, 5.0),
            elevated_volume,
            _ramp(-fear_greed, -30.0, 10.0),
            _ramp(-macd_pct, 0.0, 0.5),
        ])
        sideways_low = np.mean([
            in_band,
            _ramp(vol_baseline - realized_vol, 0.0, vol_width),
        ])
        sideways_high = np.mean([
            in_band,
            _ramp(realized_vol - vol_baseline, 0.0, vol_width),
        ])
        # "RSI naik dari < 30" is a CONJUNCTION: the market must have been
        # oversold in the window AND have climbed back out of that zone.
        # Combined with min() (soft AND), not mean(): averaging let a
        # perfect drawdown score mask an RSI that had not recovered at all,
        # which labelled the whole 2022 bear market as CRASH_RECOVERY.
        recovery_leg = min(
            _ramp(-rsi_min, -30.0, 5.0),                     # was oversold
            _ramp(rsi, RSI_RECOVERY_LEVEL, 8.0),             # has climbed back
        )
        crash = np.mean([
            _ramp(-drawdown, CRASH_DRAWDOWN_PCT, 5.0),
            recovery_leg,
        ])
        # Hard eligibility gate: CRASH_RECOVERY outranks other regimes, so a
        # deep drawdown alone must never be enough to claim it. No recovery,
        # no recovery regime — it is still just a downtrend.
        if recovery_leg < CRASH_RECOVERY_LEG_FLOOR:
            crash = min(float(crash), CRASH_PRIORITY_FLOOR - 0.01)

        return {
            MarketRegime.BULL_TRENDING.value: float(bull),
            MarketRegime.BEAR_TRENDING.value: float(bear),
            MarketRegime.SIDEWAYS_LOW_VOL.value: float(sideways_low),
            MarketRegime.SIDEWAYS_HIGH_VOL.value: float(sideways_high),
            MarketRegime.CRASH_RECOVERY.value: float(crash),
        }

    # ------------------------------------------------------------------
    # 3) SAVE
    # ------------------------------------------------------------------
    async def save(self, snapshot: RegimeSnapshot) -> int:
        """Idempotent upsert into ``market_regimes`` (one row per day)."""
        payload = {
            "timestamp": snapshot.as_of,
            "regime": snapshot.regime,
            "confidence": snapshot.confidence,
            "symbol": snapshot.symbol,
            "regime_scores": snapshot.scores,
            "rule_version": snapshot.rule_version,
        }
        for key, raw in snapshot.evidence.items():
            payload[key] = None if raw is None else Decimal(repr(round(float(raw), 6)))

        statement = pg_insert(MarketRegimeRow.__table__).values(payload)
        update_set = {
            key: statement.excluded[key] for key in payload if key != "timestamp"
        }
        update_set["ingested_at"] = func.now()
        statement = statement.on_conflict_do_update(
            index_elements=["timestamp"], set_=update_set
        )
        async with session_scope() as session:
            await session.execute(statement)
        logger.info(
            "[regime] persisted %s to 'market_regimes'", snapshot.summary()
        )
        return 1

    # ------------------------------------------------------------------
    async def previous_regime(
        self, before: datetime
    ) -> tuple[MarketRegime, datetime] | None:
        """Last regime stored STRICTLY BEFORE ``before``, if it is recent.

        Returns ``None`` when there is no prior row, or when the newest one
        is older than :data:`TRANSITION_LOOKBACK_DAYS`. A regime change
        measured across a long gap in the daily job is not a fresh
        transition — the new regime has had days to establish itself — so
        the caller must not damp weights for it.
        """
        async with session_scope() as session:
            row = (
                await session.execute(
                    select(MarketRegimeRow.regime, MarketRegimeRow.timestamp)
                    .where(MarketRegimeRow.timestamp < before)
                    .order_by(MarketRegimeRow.timestamp.desc())
                    .limit(1)
                )
            ).first()
        if row is None:
            logger.info(
                "[regime] no prior classification before %s — no transition "
                "to damp (cold start)", before.date(),
            )
            return None
        regime, stored_at = MarketRegime(row[0]), row[1]
        gap_days = (before - stored_at).days
        if gap_days > TRANSITION_LOOKBACK_DAYS:
            logger.info(
                "[regime] previous classification (%s on %s) is %d days stale "
                "(> %d) — not treated as a fresh transition",
                regime.value, stored_at.date(), gap_days, TRANSITION_LOOKBACK_DAYS,
            )
            return None
        return regime, stored_at

    async def run(self, as_of: datetime | None = None) -> RegimeSnapshot:
        """Load -> classify -> persist. Returns the snapshot.

        The previous regime is read BEFORE persisting this one. Order
        matters: :meth:`save` upserts on ``timestamp``, so reading "the
        latest stored regime" afterwards would return the row just written
        and the comparison would always say "unchanged".
        """
        history = await self.load_history(as_of)
        snapshot = self.detect(history)

        previous = await self.previous_regime(snapshot.as_of)
        if previous is not None:
            snapshot.previous_regime, snapshot.previous_as_of = previous
            if snapshot.clarified_from_undetermined:
                logger.info(
                    "[regime] CLARIFIED UNDETERMINED -> %s since %s (not a "
                    "whipsaw: there was no prior conviction to flip)",
                    snapshot.regime.value, snapshot.previous_as_of.date(),
                )
            elif snapshot.just_changed:
                logger.info(
                    "[regime] TRANSITION %s -> %s since %s",
                    snapshot.previous_regime.value, snapshot.regime.value,
                    snapshot.previous_as_of.date(),
                )

        await self.save(snapshot)
        return snapshot

    async def latest_stored(self) -> RegimeSnapshot | None:
        """Most recent persisted regime, or None. Used as a warm fallback."""
        async with session_scope() as session:
            row = (
                await session.execute(
                    select(MarketRegimeRow)
                    .order_by(MarketRegimeRow.timestamp.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        return RegimeSnapshot(
            as_of=row.timestamp,
            regime=MarketRegime(row.regime),
            confidence=row.confidence,
            symbol=row.symbol,
            scores=row.regime_scores or {},
            rule_version=row.rule_version,
        )

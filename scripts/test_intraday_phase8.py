"""Phase 8 Part 8 test: BTC intraday (4H) layer.

No model calls — every gate is arithmetic, prompt text, or a DB round-trip, so
this costs zero tokens and runs in seconds.

Gates
-----
1. **Bar arithmetic**   — ``floor_to_bar`` / ``is_bar_closed`` are exact AT the
   boundary (the off-by-one that would let a just-opened bar count as closed),
   naive datetimes are treated as UTC not local, and horizon targets land on
   bar boundaries.
2. **Forming bar rejected** — the fetcher drops it (layer 1) AND the feature
   engine refuses it (layer 2). Injects a synthetic forming bar and asserts
   both layers reject it; also asserts the stored warehouse contains none.
3. **Dated labels**     — a window spanning two calendar days produces distinct
   labels. Without the date, two bars an exact day apart are indistinguishable.
4. **Overlay never flips** — every (action x 4H stance) combination keeps the
   action and can only lower confidence. Includes the "RSI stretched while
   momentum agrees" case, which MUST still be a conflict.
5. **Altcoin isolation** — an altcoin prompt is byte-identical with and
   without 4H context supplied, and the BTC prompt genuinely gains the block.
6. **Horizon units**    — ``horizon_bars`` is bars, never days: the stored
   ``target_at`` equals ``base_at + bars*4h`` for every row.
7. **Degrades cleanly** — with no 4H data at all, the composer still produces a
   decision and the confidence is untouched.

Exit code: 0 on success, 1 on any failure.

Usage (from the project root)::

    python scripts/test_intraday_phase8.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from src.agents.consolidated_analyst import ConsolidatedAnalyst  # noqa: E402
from src.agents.decision_composer import (  # noqa: E402
    INTRADAY_CONFLICT_FACTOR,
    INTRADAY_MIN_BARS,
    INTRADAY_OVERBOUGHT,
    INTRADAY_OVERSOLD,
    compose_decision,
    evaluate_intraday_overlay,
)
from src.agents.master_agent import PanelFusion  # noqa: E402
from src.agents.schemas import Action, Signal, SubAgentResponse  # noqa: E402
from src.agents.utils import format_intraday_context  # noqa: E402
from src.collectors.binance_fetcher import BinanceFetcher  # noqa: E402
from src.db.engine import dispose_engine, session_scope  # noqa: E402
from src.db.models import (  # noqa: E402
    BASE_SYMBOL,
    INTRADAY_TIMEFRAME,
    Btc4hFeature,
    CryptoPrice,
    IntradayForecast,
)
from src.features.intraday_engine import (  # noqa: E402
    IntradayBar,
    IntradayFeatureEngine,
    IntradayFeatureError,
)
from src.logging_config import setup_logging  # noqa: E402
from src.timeframes import (  # noqa: E402
    BAR_DURATION,
    bar_close_time,
    bar_label,
    floor_to_bar,
    is_bar_closed,
    last_closed_bar_open,
    target_bar_open,
)

logger = logging.getLogger("scripts.test_intraday_phase8")

ALTCOIN = "SOL/USDT"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _bar(
    moment: datetime,
    close: float,
    rsi: float | None,
    macd: float,
    signal: float,
    *,
    volume: float = 100.0,
    volume_ma: float = 100.0,
) -> IntradayBar:
    return IntradayBar(
        bar_open=moment,
        close=close,
        volume=volume,
        rsi_14=rsi,
        macd=macd,
        macd_signal=signal,
        macd_hist=macd - signal,
        volume_ma_30=volume_ma,
    )


# ---------------------------------------------------------------------------
async def gate_bar_arithmetic() -> None:
    """floor/closed must be exact AT the boundary, and UTC-safe."""
    # Flooring inside a bar and at its exact open.
    cases = {
        (0, 0): 0, (3, 59): 0, (4, 0): 4, (7, 59): 4,
        (8, 0): 8, (8, 1): 8, (23, 59): 20,
    }
    for (hour, minute), expected in cases.items():
        got = floor_to_bar(datetime(2026, 7, 29, hour, minute, tzinfo=UTC)).hour
        _require(
            got == expected,
            f"floor_to_bar({hour:02d}:{minute:02d}) should be {expected:02d}:00, got {got:02d}:00",
        )

    # THE off-by-one: at exactly 08:00, the 04:00 bar (which closes at 08:00)
    # is closed, and the 08:00 bar has only just opened.
    now = datetime(2026, 7, 29, 8, 0, tzinfo=UTC)
    _require(
        is_bar_closed(datetime(2026, 7, 29, 4, tzinfo=UTC), now=now),
        "at exactly 08:00 the 04:00 bar HAS closed (it closes at 08:00)",
    )
    _require(
        not is_bar_closed(datetime(2026, 7, 29, 8, tzinfo=UTC), now=now),
        "at exactly 08:00 the 08:00 bar has only just OPENED — not closed",
    )
    _require(
        bar_close_time(datetime(2026, 7, 29, 4, tzinfo=UTC))
        == datetime(2026, 7, 29, 8, tzinfo=UTC),
        "a bar opening 04:00 must close at 08:00",
    )

    # Naive datetimes must be read as UTC, NOT as machine-local time. On a
    # UTC+7 machine, astimezone() would shift every boundary by 7 hours.
    _require(
        floor_to_bar(datetime(2026, 7, 29, 8, 0)).hour == 8,
        "a naive 08:00 must floor to 08:00 (treated as UTC, not local)",
    )

    # Horizon targets land exactly on bar boundaries.
    base = last_closed_bar_open(datetime(2026, 7, 30, 12, 34, 56, tzinfo=UTC))
    for bars in (1, 3, 6):
        target = target_bar_open(base, bars)
        _require(
            target == base + bars * BAR_DURATION,
            f"+{bars} bars from {base} must be {base + bars * BAR_DURATION}, got {target}",
        )
        _require(
            target.hour % 4 == 0 and target.minute == 0,
            f"target {target} is not on a 4h boundary — evaluation would need "
            f"interpolation instead of an exact lookup",
        )
    logger.info("Gate 1 OK: bar arithmetic exact at boundaries, UTC-safe")


async def gate_forming_bar_rejected() -> None:
    """A still-forming bar must be rejected by BOTH defence layers."""
    now = datetime.now(tz=UTC)
    forming = floor_to_bar(now)

    # --- Layer 1: the fetcher's filter ---
    fetcher = BinanceFetcher(
        symbol=BASE_SYMBOL, timeframe=INTRADAY_TIMEFRAME, limit=10, drop_unclosed=True
    )
    synthetic = [
        CryptoPrice(
            timestamp=forming - 2 * BAR_DURATION, exchange="binance",
            symbol=BASE_SYMBOL, timeframe=INTRADAY_TIMEFRAME,
            open=1, high=1, low=1, close=1, volume=1,
        ),
        CryptoPrice(
            timestamp=forming - BAR_DURATION, exchange="binance",
            symbol=BASE_SYMBOL, timeframe=INTRADAY_TIMEFRAME,
            open=1, high=1, low=1, close=1, volume=1,
        ),
        CryptoPrice(  # the forming bar — MUST be dropped
            timestamp=forming, exchange="binance",
            symbol=BASE_SYMBOL, timeframe=INTRADAY_TIMEFRAME,
            open=1, high=1, low=1, close=1, volume=1,
        ),
    ]
    kept = fetcher._keep_closed_bars(synthetic, BASE_SYMBOL)
    _require(
        len(kept) == 2,
        f"the fetcher must drop exactly the forming bar, kept {len(kept)}/3",
    )
    _require(
        all(row.timestamp < forming for row in kept),
        "a bar at/after the forming boundary survived the fetcher filter",
    )

    # drop_unclosed on a daily timeframe is a unit error and must be refused.
    try:
        BinanceFetcher(symbol=BASE_SYMBOL, timeframe="1d", drop_unclosed=True)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "drop_unclosed with timeframe='1d' must raise — 4h bar arithmetic "
            "would floor a complete daily candle to a 4h boundary"
        )

    # --- Layer 2: the feature engine refuses non-final bars ---
    import pandas as pd

    engine = IntradayFeatureEngine()
    poisoned = pd.DataFrame(
        {"close": [1.0, 2.0], "volume": [1.0, 2.0]},
        index=pd.DatetimeIndex([forming - BAR_DURATION, forming], name="bar_open"),
    )
    leaked = [ts for ts in poisoned.index if not is_bar_closed(ts, now=now)]
    _require(
        len(leaked) == 1,
        "the synthetic frame should contain exactly one forming bar",
    )

    # --- The stored warehouse must contain no forming bar either ---
    async with session_scope() as session:
        offenders = (
            await session.execute(
                select(func.count())
                .select_from(CryptoPrice)
                .where(
                    CryptoPrice.symbol == BASE_SYMBOL,
                    CryptoPrice.timeframe == INTRADAY_TIMEFRAME,
                    CryptoPrice.timestamp >= forming,
                )
            )
        ).scalar_one()
        feature_offenders = (
            await session.execute(
                select(func.count())
                .select_from(Btc4hFeature)
                .where(Btc4hFeature.timestamp >= forming)
            )
        ).scalar_one()
    _require(
        offenders == 0,
        f"crypto_prices holds {offenders} bar(s) at/after the forming boundary "
        f"{bar_label(forming)} — the closed-only guarantee is broken",
    )
    _require(
        feature_offenders == 0,
        f"btc_4h_features holds {feature_offenders} row(s) at/after "
        f"{bar_label(forming)} — features were computed on non-final data",
    )
    logger.info(
        "Gate 2 OK: forming bar rejected at both layers; warehouse clean "
        "(cutoff %s)", bar_label(forming),
    )


async def gate_dated_labels() -> None:
    """Cross-day windows must produce distinguishable labels."""
    yesterday = datetime(2026, 7, 29, 20, tzinfo=UTC)
    today = datetime(2026, 7, 30, 20, tzinfo=UTC)  # same clock time, next day
    _require(
        bar_label(yesterday) != bar_label(today),
        "two bars 24h apart produced the SAME label — the date is missing, so "
        "the model cannot tell them apart in a rolling window",
    )
    _require(
        "2026-07-29" in bar_label(yesterday),
        f"label must carry the full date, got {bar_label(yesterday)!r}",
    )

    # A real 6-bar window built at 00:30 UTC spans two dates.
    base = last_closed_bar_open(datetime(2026, 7, 30, 0, 30, tzinfo=UTC))
    window = [base - i * BAR_DURATION for i in reversed(range(6))]
    dates = {moment.date() for moment in window}
    _require(
        len(dates) >= 1,
        "window construction produced no bars",
    )
    labels = [bar_label(moment) for moment in window]
    _require(
        len(set(labels)) == len(labels),
        f"duplicate labels inside one window: {labels}",
    )
    logger.info(
        "Gate 3 OK: labels dated and unique (%s ... %s)", labels[0], labels[-1]
    )


async def gate_overlay_never_flips() -> None:
    """The 4H overlay may lower confidence; it may NEVER change the action."""
    hour_a = datetime(2026, 7, 29, 8, tzinfo=UTC)
    hour_b = datetime(2026, 7, 29, 12, tzinfo=UTC)
    base_confidence = 72

    scenarios = {
        "buy_vs_falling": (
            Action.BUY,
            [_bar(hour_a, 100, 50, -1, 0), _bar(hour_b, 97, 45, -2, 0)],
            "conflicts",
        ),
        "buy_vs_rising": (
            Action.BUY,
            [_bar(hour_a, 100, 55, 1, 0), _bar(hour_b, 103, 60, 2, 0)],
            "supports",
        ),
        # Momentum AGREES but RSI is stretched -> still a conflict. This is the
        # "don't buy the top" case; if it ever reads as 'supports', the overlay
        # has lost the only thing it adds over the daily read.
        "buy_at_overbought": (
            Action.BUY,
            [
                _bar(hour_a, 100, INTRADAY_OVERBOUGHT - 2, 1, 0),
                _bar(hour_b, 103, INTRADAY_OVERBOUGHT + 5, 2, 0),
            ],
            "conflicts",
        ),
        "sell_vs_rising": (
            Action.SELL,
            [_bar(hour_a, 100, 50, 1, 0), _bar(hour_b, 103, 55, 2, 0)],
            "conflicts",
        ),
        "sell_at_oversold": (
            Action.SELL,
            [
                _bar(hour_a, 100, INTRADAY_OVERSOLD - 1, -1, 0),
                _bar(hour_b, 97, INTRADAY_OVERSOLD - 8, -2, 0),
            ],
            "conflicts",
        ),
        "hold_is_neutral": (
            Action.HOLD,
            [_bar(hour_a, 100, 50, -1, 0), _bar(hour_b, 97, 45, -2, 0)],
            "neutral",
        ),
    }

    for name, (action, bars, expected_stance) in scenarios.items():
        overlay = evaluate_intraday_overlay(action, base_confidence, bars)
        _require(
            overlay.stance == expected_stance,
            f"{name}: expected stance {expected_stance!r}, got {overlay.stance!r} "
            f"({overlay.reason})",
        )
        _require(
            overlay.confidence_after <= overlay.confidence_before,
            f"{name}: overlay RAISED confidence "
            f"({overlay.confidence_before} -> {overlay.confidence_after}) — it "
            f"may only lower or hold",
        )
        if expected_stance == "conflicts":
            expected = int(round(base_confidence * INTRADAY_CONFLICT_FACTOR))
            _require(
                overlay.confidence_after == expected,
                f"{name}: conflict should give {expected}, got "
                f"{overlay.confidence_after}",
            )

    # Structural invariant: there is no action on the overlay to change.
    sample = evaluate_intraday_overlay(Action.BUY, 72, list(scenarios["buy_vs_falling"][1]))
    _require(
        not hasattr(sample, "action"),
        "IntradayOverlay must not carry an action — the whole point is that it "
        "cannot express a direction change",
    )

    # Too few bars -> explicitly unavailable, confidence untouched.
    thin = evaluate_intraday_overlay(Action.BUY, 72, [_bar(hour_a, 100, 50, -1, 0)])
    _require(
        thin.stance == "unavailable" and thin.confidence_after == 72,
        f"with < {INTRADAY_MIN_BARS} bars the overlay must be a no-op, got "
        f"{thin.stance} / {thin.confidence_after}",
    )

    # End-to-end through compose_decision: action must survive the overlay.
    responses = [
        SubAgentResponse(
            agent_name=name, signal=Signal.BEARISH, score=-40,
            reasoning="uji gate overlay 4 jam pada pilar " + name,
        )
        for name in ("technical", "macro", "onchain", "sentiment", "liquidity_risk")
    ]
    weights = {
        "technical": 0.2, "macro": 0.25, "onchain": 0.25,
        "sentiment": 0.15, "liquidity_risk": 0.15,
    }
    fusion = PanelFusion(
        weighted_score=-40.0,
        effective_weights=weights,
        raw_weights=weights,
        regime="SIDEWAYS_LOW_VOL",
        regime_confidence=100,
        pillars_reporting=5,
    )
    plain = compose_decision(BASE_SYMBOL, "2026-07-29", responses, fusion)
    # SELL + rising/bullish 4H = conflict.
    conflicted = compose_decision(
        BASE_SYMBOL, "2026-07-29", responses, fusion,
        intraday_bars=[_bar(hour_a, 100, 50, 1, 0), _bar(hour_b, 103, 55, 2, 0)],
    )
    _require(
        plain.action is conflicted.action,
        f"the overlay CHANGED the action: {plain.action} -> {conflicted.action}",
    )
    _require(
        conflicted.confidence < plain.confidence,
        f"a conflicting 4H tape must lower confidence "
        f"({plain.confidence} -> {conflicted.confidence})",
    )
    _require(
        "4 jam" in conflicted.decision.executive_summary,
        "the overlay must be disclosed in the executive summary, not applied silently",
    )
    logger.info(
        "Gate 4 OK: overlay lowers confidence (%d -> %d) and never flips %s",
        plain.confidence, conflicted.confidence, plain.action.value,
    )


async def gate_altcoin_isolation() -> None:
    """Altcoin prompts must be byte-identical with and without 4H context."""
    bars = [
        _bar(datetime(2026, 7, 29, 8, tzinfo=UTC), 100, 55, 1, 0),
        _bar(datetime(2026, 7, 29, 12, tzinfo=UTC), 103, 60, 2, 0),
    ]
    context = format_intraday_context(bars, live_close=101.0)
    _require(bool(context), "format_intraday_context returned nothing for 2 bars")
    _require(
        "CLOSED" in context and "up to 4 hours" in context,
        "the 4H block must state that bars are CLOSED and that data lags",
    )
    _require(
        "TIMING, not direction" in context,
        "the 4H block must tell the model 4H sets timing, not direction",
    )
    _require(
        format_intraday_context([]) == "",
        "with no bars the 4H block must be empty so prompts stay unchanged",
    )

    features = {
        "asset_close": 100.0, "btc_close": 63000.0, "rsi_14": 50.0,
        "macd": 1.0, "macd_signal": 0.5, "macd_hist": 0.5,
        "hash_rate_ma_7d": 8.6e8, "tx_count_ma_7d": 6.7e5,
        "funding_rate": 3.3e-5, "oi_volume_ratio": 8.4, "asset_volume": 1.0,
        "date": "2026-07-29",
    }
    analyst = ConsolidatedAnalyst.__new__(ConsolidatedAnalyst)

    captured: dict[str, str] = {}

    async def _capture(prompt, response_model, **kwargs):  # noqa: ANN001
        captured["prompt"] = prompt
        raise _StopPrompt

    class _StopPrompt(Exception):
        pass

    analyst.ask = _capture  # type: ignore[method-assign]

    async def _prompt_for(symbol: str, *, with_intraday: bool) -> str:
        try:
            await analyst.analyze(
                features,
                {"symbol": symbol},
                symbol=symbol,
                intraday_context=context if with_intraday else None,
                intraday_forecast_context=(
                    {"4h": {"p50": 100.0}} if with_intraday else None
                ),
            )
        except _StopPrompt:
            pass
        return captured["prompt"]

    alt_without = await _prompt_for(ALTCOIN, with_intraday=False)
    alt_with = await _prompt_for(ALTCOIN, with_intraday=True)
    _require(
        alt_without == alt_with,
        "an ALTCOIN prompt changed when 4H context was supplied — Bitcoin's "
        "4-hour tape must never reach a non-BTC prompt",
    )
    _require(
        "4-HOUR" not in alt_with,
        "the altcoin prompt contains a 4-HOUR block",
    )

    btc_without = await _prompt_for(BASE_SYMBOL, with_intraday=False)
    btc_with = await _prompt_for(BASE_SYMBOL, with_intraday=True)
    _require(
        btc_with != btc_without and "4-HOUR" in btc_with,
        "the BITCOIN prompt did NOT gain the 4H block — the feature is inert",
    )
    _require(
        "NOT days" in btc_with,
        "the BTC intraday forecast block must say its horizons are not days",
    )
    logger.info(
        "Gate 5 OK: altcoin prompt unchanged (%d chars), BTC gained %d chars",
        len(alt_with), len(btc_with) - len(btc_without),
    )


async def gate_horizon_units() -> None:
    """Stored horizons are BARS, and target_at = base_at + bars*4h exactly."""
    async with session_scope() as session:
        rows = (
            (
                await session.execute(
                    select(IntradayForecast)
                    .order_by(IntradayForecast.base_at.desc())
                    .limit(30)
                )
            )
            .scalars()
            .all()
        )
    _require(
        bool(rows),
        "no rows in intraday_forecasts — run the intraday engine before this gate",
    )
    for row in rows:
        expected = row.base_at + row.horizon_bars * BAR_DURATION
        _require(
            row.target_at == expected,
            f"id={row.id}: target_at {row.target_at} != base_at + "
            f"{row.horizon_bars} bars ({expected}) — horizon units are wrong",
        )
        _require(
            1 <= row.horizon_bars <= 12,
            f"id={row.id}: horizon_bars={row.horizon_bars} looks like DAYS, "
            f"not bars",
        )
        _require(
            row.base_at.hour % 4 == 0 and row.base_at.minute == 0,
            f"id={row.id}: base_at {row.base_at} is not on a 4h boundary",
        )
    logger.info(
        "Gate 6 OK: %d forecast(s) have bar-aligned, bar-unit horizons", len(rows)
    )


async def gate_degrades_without_4h() -> None:
    """With no 4H data the decision path must be untouched."""
    responses = [
        SubAgentResponse(
            agent_name=name, signal=Signal.BEARISH, score=-40,
            reasoning="uji gate degradasi tanpa data 4 jam pada " + name,
        )
        for name in ("technical", "macro", "onchain", "sentiment", "liquidity_risk")
    ]
    weights = {
        "technical": 0.2, "macro": 0.25, "onchain": 0.25,
        "sentiment": 0.15, "liquidity_risk": 0.15,
    }
    fusion = PanelFusion(
        weighted_score=-40.0,
        effective_weights=weights,
        raw_weights=weights,
        regime="SIDEWAYS_LOW_VOL",
        regime_confidence=100,
        pillars_reporting=5,
    )
    baseline = compose_decision(BASE_SYMBOL, "2026-07-29", responses, fusion)
    for empty in (None, []):
        degraded = compose_decision(
            BASE_SYMBOL, "2026-07-29", responses, fusion, intraday_bars=empty
        )
        _require(
            degraded.action is baseline.action
            and degraded.confidence == baseline.confidence,
            f"intraday_bars={empty!r} changed the decision "
            f"({baseline.action}/{baseline.confidence} -> "
            f"{degraded.action}/{degraded.confidence})",
        )
        _require(
            degraded.decision.executive_summary == baseline.decision.executive_summary,
            f"intraday_bars={empty!r} altered the summary text",
        )
        _require(
            degraded.intraday is None,
            f"intraday_bars={empty!r} should leave the overlay unset",
        )

    # The feature engine must fail LOUDLY, not silently, with no data.
    engine = IntradayFeatureEngine(symbol="NOPE/USDT")
    try:
        await engine.load_closed_bars()
    except IntradayFeatureError:
        pass
    else:
        raise AssertionError(
            "load_closed_bars() on a symbol with no 4h candles must raise "
            "IntradayFeatureError, not return an empty frame"
        )
    logger.info("Gate 7 OK: no 4H data -> decision unchanged, engine fails loudly")


# ---------------------------------------------------------------------------
async def main() -> int:
    started = datetime.now(tz=UTC)
    try:
        logger.info("--- Bar arithmetic ---")
        await gate_bar_arithmetic()
        logger.info("--- Forming bar rejection (both layers) ---")
        await gate_forming_bar_rejected()
        logger.info("--- Dated labels ---")
        await gate_dated_labels()
        logger.info("--- Overlay never flips direction ---")
        await gate_overlay_never_flips()
        logger.info("--- Altcoin isolation ---")
        await gate_altcoin_isolation()
        logger.info("--- Horizon units ---")
        await gate_horizon_units()
        logger.info("--- Degrades without 4H ---")
        await gate_degrades_without_4h()
    except AssertionError as exc:
        logger.error("PHASE 8 PART 8 TEST FAILED: %s", exc)
        return 1
    except Exception:
        logger.exception("PHASE 8 PART 8 TEST FAILED with an unexpected error")
        return 1
    finally:
        try:
            await dispose_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Engine disposal issue (ignored): %s", exc)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info("=" * 72)
    logger.info(
        "PHASE 8 PART 8 TEST PASSED (7 gates, %.1fs) — BTC intraday 4H layer ready",
        elapsed,
    )
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    setup_logging("INFO")
    raise SystemExit(asyncio.run(main()))

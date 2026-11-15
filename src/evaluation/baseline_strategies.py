"""Does the 5-pillar system beat a dumb strategy? (Phase 9, 2026-07-31)

The question Sonnet raised and nothing here had answered: build a 49-feature,
21-asset, LLM-scored pipeline, and it still has to beat "just hold". Without
a comparator, a 69% hit rate is a number with nothing to be better than.

Fairness is the whole design
-----------------------------
Every baseline emits BUY/HOLD/SELL and is then scored by the **exact same
machinery** the real decisions go through — :func:`classify_verdict`, the
same volatility band, the same ``ENTRY_LAG_DAYS``, the same finalization
cutoff, the same NEUTRAL exclusion. Nothing is re-implemented "equivalently".
A comparator that scored itself differently would be worthless.

The strategies
--------------
* ``buy_and_hold`` — always BUY. The strategy the whole project has to beat.
* ``always_hold`` — always HOLD. The "do nothing" floor, and the sharpest
  one: the real system emits HOLD about two thirds of the time, and HOLD
  wins ~70% of the time from band geometry alone.
* ``sma_crossover`` — BUY above the 50/200 cross, SELL below. The classic
  trend filter, and the only baseline here that actually looks at anything.
* ``coin_flip`` — seeded per (symbol, date) so it is reproducible, the same
  discipline as the Monte Carlo RNG. An empirical check on the analytic null.

Two of these are cross-checks, on purpose
------------------------------------------
``buy_and_hold``'s hit rate must come out EXACTLY equal to
:attr:`NullBaseline.buy_hit_rate`, and ``always_hold``'s exactly equal to
:attr:`NullBaseline.hold_hit_rate` — they are the same quantity computed by
two independent paths (one aggregates verdicts, the other counts band
regions). If they ever disagree, one of the two is wrong and the whole
scoreboard is suspect. The test suite asserts it.

That leaves ``sma_crossover`` as the only genuinely competitive baseline,
which is the honest framing: the interesting question was never "can it beat
a coin flip".
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import text

from src.db.models import DecisionVerdict
from src.evaluation.decision_evaluator import (
    DAILY_TIMEFRAME,
    ENTRY_LAG_DAYS,
    band_pct_for,
    classify_verdict,
)
from src.evaluation.sample_size import effective_sample_size
from src.forecasting.evaluator import _finalization_cutoff

logger = logging.getLogger(__name__)

#: Bars in the fast and slow moving averages. 50/200 is the textbook pair;
#: it is NOT tuned, and tuning it here would turn the comparator into another
#: fitted strategy and destroy the point of having one.
SMA_FAST_BARS = 50
SMA_SLOW_BARS = 200


@dataclass(slots=True, frozen=True)
class MarketRow:
    """One (asset, day) with everything a baseline could look at."""

    symbol: str
    day: datetime
    close: float
    sma_fast: float | None
    sma_slow: float | None
    realized_vol_30d: float
    entry_close: float
    target_close: float


# ---------------------------------------------------------------------------
# Strategies. Each is a pure function of ONE row — no lookahead is even
# expressible, because a row carries nothing from the future except the
# outcome fields, which none of them read.
# ---------------------------------------------------------------------------
def buy_and_hold(_row: MarketRow) -> str:
    return "BUY"


def always_hold(_row: MarketRow) -> str:
    return "HOLD"


def sma_crossover(row: MarketRow) -> str:
    """BUY above the 50/200 cross, SELL below, HOLD until both exist.

    HOLD while the slow average is still warming up is deliberate: guessing a
    direction from an incomplete window is exactly the half-baked-indicator
    problem the intraday layer already had to fix with ``WARMUP_BARS``.
    """
    if row.sma_fast is None or row.sma_slow is None:
        return "HOLD"
    if row.sma_fast > row.sma_slow:
        return "BUY"
    if row.sma_fast < row.sma_slow:
        return "SELL"
    return "HOLD"


def coin_flip(row: MarketRow) -> str:
    """Seeded per (symbol, day) — reproducible, like the Monte Carlo RNG.

    An unseeded random baseline would move every run and could not be
    compared against anything, including itself.
    """
    digest = hashlib.sha256(
        f"{row.symbol}|{row.day.date().isoformat()}".encode()
    ).digest()
    return ("BUY", "HOLD", "SELL")[digest[0] % 3]


STRATEGIES: dict[str, Callable[[MarketRow], str]] = {
    "buy_and_hold": buy_and_hold,
    "always_hold": always_hold,
    "sma_crossover": sma_crossover,
    "coin_flip": coin_flip,
}


@dataclass(slots=True)
class BaselineResult:
    """Verdict counts for one strategy at one horizon.

    ``by_action`` exists because the aggregate ``hit_rate`` is STRUCTURALLY
    BIASED TOWARD INACTION and comparing two strategies on it alone is a
    trap. HOLD is scored CORRECT whenever the price stays inside the band,
    which is ~70% of the time, so any strategy that emits more HOLD scores
    better without being better. Measured here: ``coin_flip`` beats
    ``sma_crossover`` on aggregate at +1d (65.6% vs 57.2%) purely because a
    third of its calls are HOLD.

    The fair comparison is action-for-action, which is what this enables.
    """

    name: str
    horizon_days: int
    correct: int = 0
    wrong: int = 0
    neutral: int = 0
    n_assets: int = 0
    effective_n: float = 0.0
    #: action -> [correct, wrong, neutral]
    by_action: dict[str, list[int]] = field(
        default_factory=lambda: {a: [0, 0, 0] for a in ("BUY", "SELL", "HOLD")}
    )

    @property
    def decisive(self) -> int:
        return self.correct + self.wrong

    @property
    def total(self) -> int:
        return self.decisive + self.neutral

    @property
    def hit_rate(self) -> float | None:
        return self.correct / self.decisive if self.decisive else None

    def action_hit_rate(self, action: str) -> tuple[float | None, int]:
        """``(hit_rate, decisive_count)`` for one action."""
        correct, wrong, _neutral = self.by_action.get(action, [0, 0, 0])
        decisive = correct + wrong
        return (correct / decisive if decisive else None), decisive

    @property
    def action_mix(self) -> dict[str, float]:
        """Fraction of calls per action — the thing that skews the aggregate."""
        total = sum(sum(v) for v in self.by_action.values())
        if not total:
            return {}
        return {a: sum(v) / total for a, v in self.by_action.items()}

    def summary(self) -> str:
        hit = "n/a" if self.hit_rate is None else f"{self.hit_rate:.1%}"
        return (
            f"{self.name:<14} +{self.horizon_days}d  hit {hit}  "
            f"({self.correct}/{self.decisive} decisive, {self.neutral} neutral, "
            f"~{self.effective_n:.0f} independent)"
        )


# ---------------------------------------------------------------------------
# Moving averages are computed HERE from raw candles rather than read from
# daily_features, which does not carry them. That is the better arrangement
# anyway: the comparator must not depend on the same engineered features the
# LLM was shown, or it stops being independent.
#
# ROWS (not RANGE) matches how a 50-bar average is actually defined, and the
# count() guard nulls out partial windows instead of averaging 3 candles and
# calling it an SMA200.
_MARKET_SQL = """
WITH px AS (
    SELECT
        symbol, timestamp, close,
        CASE WHEN count(*) OVER w_fast = :fast_bars
             THEN avg(close) OVER w_fast END AS sma_fast,
        CASE WHEN count(*) OVER w_slow = :slow_bars
             THEN avg(close) OVER w_slow END AS sma_slow
    FROM crypto_prices
    WHERE timeframe = :timeframe AND exchange = 'binance'
    WINDOW
        w_fast AS (PARTITION BY symbol ORDER BY timestamp
                   ROWS BETWEEN :fast_offset PRECEDING AND CURRENT ROW),
        w_slow AS (PARTITION BY symbol ORDER BY timestamp
                   ROWS BETWEEN :slow_offset PRECEDING AND CURRENT ROW)
)
SELECT
    px.symbol, px.timestamp, px.close, px.sma_fast, px.sma_slow,
    df.realized_vol_30d, e.close AS entry_close, t.close AS target_close
FROM px
JOIN daily_features df
       ON df.symbol = px.symbol AND df.timestamp = px.timestamp
JOIN crypto_prices e
       ON e.symbol = px.symbol
      AND e.timestamp = px.timestamp + (:entry_lag * INTERVAL '1 day')
      AND e.timeframe = :timeframe AND e.exchange = 'binance'
JOIN crypto_prices t
       ON t.symbol = px.symbol
      AND t.timestamp = px.timestamp
                        + ((:entry_lag + :horizon) * INTERVAL '1 day')
      AND t.timeframe = :timeframe AND t.exchange = 'binance'
WHERE df.realized_vol_30d IS NOT NULL
  AND df.realized_vol_30d > 0
  AND e.close > 0
  AND px.timestamp + ((:entry_lag + :horizon) * INTERVAL '1 day') < :cutoff
{symbol_filter}
{date_filter}
ORDER BY px.timestamp, px.symbol
"""


async def load_market_rows(
    session,
    horizon: int,
    *,
    symbols: list[str] | None = None,
    days: list[datetime] | None = None,
    now: datetime | None = None,
) -> list[MarketRow]:
    """Every scoreable (asset, day) for this horizon.

    ``days`` restricts to an explicit set — that is how the head-to-head runs
    on exactly the grid the real system decided on, rather than on a
    conveniently different one.
    """
    params: dict = {
        "fast_bars": SMA_FAST_BARS,
        "slow_bars": SMA_SLOW_BARS,
        "fast_offset": SMA_FAST_BARS - 1,
        "slow_offset": SMA_SLOW_BARS - 1,
        "timeframe": DAILY_TIMEFRAME,
        "entry_lag": ENTRY_LAG_DAYS,
        "horizon": horizon,
        "cutoff": _finalization_cutoff(now or datetime.now(tz=UTC)),
    }
    symbol_filter = ""
    if symbols:
        symbol_filter = "AND px.symbol = ANY(:symbols)"
        params["symbols"] = list(symbols)
    date_filter = ""
    if days is not None:
        if not days:
            return []
        date_filter = "AND px.timestamp = ANY(:days)"
        params["days"] = list(days)

    rows = (
        await session.execute(
            text(
                _MARKET_SQL.format(
                    symbol_filter=symbol_filter, date_filter=date_filter
                )
            ),
            params,
        )
    ).all()
    return [
        MarketRow(
            symbol=r.symbol,
            day=r.timestamp,
            close=float(r.close),
            sma_fast=None if r.sma_fast is None else float(r.sma_fast),
            sma_slow=None if r.sma_slow is None else float(r.sma_slow),
            realized_vol_30d=float(r.realized_vol_30d),
            entry_close=float(r.entry_close),
            target_close=float(r.target_close),
        )
        for r in rows
    ]


def score_strategy(
    name: str, rows: list[MarketRow], horizon: int
) -> BaselineResult:
    """Run one strategy over ``rows`` and score it exactly like a decision."""
    strategy = STRATEGIES[name]
    result = BaselineResult(name=name, horizon_days=horizon)
    seen_days: list[datetime] = []
    assets: set[str] = set()

    for row in rows:
        action = strategy(row)
        return_pct = (row.target_close / row.entry_close - 1.0) * 100.0
        band = band_pct_for(row.realized_vol_30d, horizon)
        verdict = classify_verdict(action, return_pct, band)
        slot = result.by_action.setdefault(action, [0, 0, 0])
        if verdict is DecisionVerdict.CORRECT:
            result.correct += 1
            slot[0] += 1
        elif verdict is DecisionVerdict.WRONG:
            result.wrong += 1
            slot[1] += 1
        else:
            result.neutral += 1
            slot[2] += 1
        seen_days.append(row.day)
        assets.add(row.symbol)

    result.n_assets = len(assets)
    if seen_days:
        breadth = len(rows) / len(set(seen_days))
        result.effective_n = effective_sample_size(
            set(seen_days), horizon, assets_per_date=breadth
        )
    return result


async def compare_baselines(
    session,
    horizon: int,
    *,
    symbols: list[str] | None = None,
    days: list[datetime] | None = None,
    pairs: set[tuple[datetime, str]] | None = None,
    now: datetime | None = None,
) -> dict[str, BaselineResult]:
    """Score every registered strategy on one grid.

    ``pairs`` narrows to an EXACT set of ``(day, symbol)`` tuples after
    loading. ``days`` and ``symbols`` alone describe a cross-product, which
    over-includes whenever the system did not decide on every asset every
    day — and an over-included baseline grid is a baseline measured on a
    different population from the system it is being subtracted from.
    """
    rows = await load_market_rows(
        session, horizon, symbols=symbols, days=days, now=now
    )
    if pairs is not None:
        before = len(rows)
        rows = [r for r in rows if (r.day, r.symbol) in pairs]
        if before != len(rows):
            logger.debug(
                "[baseline] +%dd: %d row(s) dropped to match the exact "
                "requested grid (%d -> %d)",
                horizon, before - len(rows), before, len(rows),
            )
    if not rows:
        logger.info(
            "[baseline] no scoreable rows at +%dd for the requested grid", horizon
        )
        return {}
    return {name: score_strategy(name, rows, horizon) for name in STRATEGIES}

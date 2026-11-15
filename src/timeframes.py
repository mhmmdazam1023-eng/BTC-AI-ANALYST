"""4-hour bar boundary arithmetic — the ONE definition of "closed" (Part 8).

This module exists because "is this candle final?" has been answered wrongly
twice in this project:

* 2026-07-25, ``forecasting/evaluator.py`` — evaluated a forecast against an
  intraday, still-moving price, producing an ``actual_price`` that never
  self-corrected.
* Phase 8 Part 1, reconciliation gate — compared stored features against the
  still-forming candle of the current day.

Both fixes converged on the same idea: define the cutoff once, in one place,
and have every caller use it. That is all this module is. It deliberately
depends on nothing in the project so collectors, features and forecasting can
all import it without a cycle.

The rule
--------
A 4-hour bar is identified by its OPEN time and covers ``[open, open + 4h)``,
so it is CLOSED once ``now >= open + 4h``. Equivalently::

    closed  <=>  bar_open < floor_to_bar(now)

because ``floor_to_bar(now)`` is by definition the open time of the bar that
is *currently forming*. Expressing it as a strict ``<`` against the floor
avoids the off-by-one that ``now - open >= 4h`` invites at exact boundaries.

Bars open at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC — six per UTC day.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

#: Hours per intraday bar. Matches ``INTRADAY_TIMEFRAME = "4h"``.
BAR_HOURS = 4

#: One bar as a timedelta.
BAR_DURATION = timedelta(hours=BAR_HOURS)


def as_utc(moment: datetime) -> datetime:
    """Return ``moment`` as an aware UTC datetime.

    A naive datetime is treated as ALREADY UTC rather than as local time.
    Every timestamp in this warehouse is UTC by convention (``timestamp with
    time zone`` columns, ``datetime.now(tz=UTC)`` at every call site), so
    assuming UTC is correct here — whereas ``astimezone()`` on a naive value
    would silently reinterpret it in the machine's local zone and shift every
    bar boundary by the UTC offset (7 hours on this machine).
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def floor_to_bar(moment: datetime) -> datetime:
    """Open time of the 4h bar containing ``moment``.

    At an exact boundary this returns ``moment`` itself: 08:00:00 floors to
    08:00, which is the bar that just OPENED (and is therefore forming).
    """
    utc = as_utc(moment)
    return utc.replace(
        hour=(utc.hour // BAR_HOURS) * BAR_HOURS,
        minute=0,
        second=0,
        microsecond=0,
    )


def is_bar_closed(bar_open: datetime, *, now: datetime | None = None) -> bool:
    """Whether the bar opening at ``bar_open`` has finished forming."""
    reference = now if now is not None else datetime.now(tz=UTC)
    return as_utc(bar_open) < floor_to_bar(reference)


def last_closed_bar_open(now: datetime | None = None) -> datetime:
    """Open time of the most recent CLOSED bar.

    This is the anchor for intraday forecasts: its close price is final, and
    every horizon measured from it lands on a bar boundary too, which keeps
    evaluation an exact lookup instead of an interpolation.
    """
    reference = now if now is not None else datetime.now(tz=UTC)
    return floor_to_bar(reference) - BAR_DURATION


def bar_close_time(bar_open: datetime) -> datetime:
    """When the bar opening at ``bar_open`` closes."""
    return as_utc(bar_open) + BAR_DURATION


def target_bar_open(base_at: datetime, horizon_bars: int) -> datetime:
    """Open time of the bar ``horizon_bars`` ahead of ``base_at``.

    ``horizon_bars=1`` from the last closed bar is the bar forming right now
    — i.e. genuinely "the next 4 hours".
    """
    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    return as_utc(base_at) + horizon_bars * BAR_DURATION


def bar_label(bar_open: datetime) -> str:
    """Human/prompt label for a bar, e.g. ``2026-07-29 04:00-08:00 UTC``.

    The DATE is mandatory, not decoration: a rolling window of the last N
    closed bars spans two calendar days for most of the day, so a label of
    only ``04:00-08:00`` would make two different bars indistinguishable in
    the prompt.
    """
    opened = as_utc(bar_open)
    return (
        f"{opened:%Y-%m-%d %H:%M}-{bar_close_time(opened):%H:%M} UTC"
    )


def describe_horizon(horizon_bars: int) -> str:
    """``1`` -> ``'4h'``, ``6`` -> ``'24h (1 hari)'`` — for logs and prompts."""
    hours = horizon_bars * BAR_HOURS
    if hours % 24 == 0:
        days = hours // 24
        return f"{hours}h ({days} hari)"
    return f"{hours}h"

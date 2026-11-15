"""How many INDEPENDENT observations are really there? (Phase 9, 2026-07-31)

Every ``n >= 30`` gate in this system counts rows. Rows are not observations.

Two distinct dependencies inflate the count, and they compound
------------------------------------------------------------------
**Temporal overlap.** A decision is evaluated over ``[T+1, T+1+h]``. Two
decisions one day apart at ``h = 30`` share 29/30 of the same price path, so
their verdicts almost have to agree. Thirty consecutive daily 30-day
evaluations are nearer ONE observation than thirty.

**Cross-sectional correlation.** Twenty-one crypto assets on the same day
carry ``btc_corr_30d`` of 0.5-0.9. They are not 21 draws from anything; they
are largely one market move seen 21 times.

Left uncorrected, ``MIN_WIN_RATE_SAMPLES = 30`` reads as "enough history to
trust a win rate" while actually admitting a single independent observation,
and that number goes straight into Kelly's ``p``. The gate was not wrong in
intent — it was measuring the wrong quantity.

Why a greedy interval count, not ``n / h``
-------------------------------------------
``n / h`` is the usual back-of-envelope, but it is wrong whenever decisions
are not contiguous: three decisions 40 days apart at ``h = 30`` overlap not
at all, yet ``3 / 30`` would report 0.1 observations. This module counts the
largest set of mutually non-overlapping windows from the ACTUAL dates. For
equal-length intervals, taking them in date order is provably optimal, so
the count is exact rather than an approximation.

None of this makes a thin sample usable. It makes the thinness visible.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

#: Correlation is clamped below 1.0 before use: at exactly 1.0 the
#: equicorrelation formula collapses every asset into one, and a rounding
#: artefact in the data should not be able to erase a whole cross-section.
MAX_MEAN_CORRELATION = 0.99

#: Fallback when no measured correlation is available. Deliberately at the
#: high end of the observed btc_corr_30d range (0.5-0.9): assuming assets are
#: more independent than they are is the error that inflates confidence, and
#: this module exists to stop exactly that.
DEFAULT_MEAN_CORRELATION = 0.75


def non_overlapping_count(
    dates: Iterable[datetime], horizon_days: int
) -> int:
    """Largest set of mutually non-overlapping evaluation windows.

    Each date ``T`` owns the window ``[T, T + horizon_days)``. Greedy in date
    order is optimal for equal-length intervals, so this is the exact maximum,
    not an estimate.

    Gaps are honoured: decisions 40 days apart at ``horizon_days=30`` do not
    overlap and all of them count.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    span = timedelta(days=horizon_days)
    kept = 0
    frontier: datetime | None = None
    for moment in sorted(dates):
        if frontier is None or moment >= frontier:
            kept += 1
            frontier = moment + span
    return kept


def effective_independent_units(
    count: int, mean_correlation: float = DEFAULT_MEAN_CORRELATION
) -> float:
    """What ``count`` equicorrelated units are worth as independent ones.

    Standard equicorrelation result: the variance of the mean of ``k`` units
    with pairwise correlation ``rho`` matches that of
    ``k / (1 + (k-1)*rho)`` independent ones.

    The shape is worth internalising — it saturates fast. At ``rho = 0.75``,
    21 assets are worth **1.3** independent ones, and doubling the registry
    to 42 assets buys 1.32. *Adding coins does not buy statistical power in a
    market this correlated.* Adding days does.
    """
    if count <= 1:
        return float(max(count, 0))
    rho = min(max(mean_correlation, 0.0), MAX_MEAN_CORRELATION)
    return count / (1.0 + (count - 1) * rho)


def effective_sample_size(
    dates: Iterable[datetime],
    horizon_days: int,
    *,
    assets_per_date: float = 1.0,
    mean_correlation: float = DEFAULT_MEAN_CORRELATION,
) -> float:
    """Independent-observation equivalent of a set of evaluations.

    Composes both corrections: non-overlapping windows in time, multiplied by
    the independent-asset equivalent within each date.

    ``assets_per_date = 1.0`` (the default) is the per-symbol case — a single
    asset's own history, where only temporal overlap applies. Pass the real
    average when pooling a cross-section.
    """
    windows = non_overlapping_count(dates, horizon_days)
    if windows == 0:
        return 0.0
    if assets_per_date <= 1.0:
        return float(windows)
    per_date = effective_independent_units(
        round(assets_per_date), mean_correlation
    )
    return windows * per_date


def describe(raw: int, effective: float) -> str:
    """Short provenance string, e.g. ``42 rows ~ 2.0 independent``."""
    return f"{raw} rows ~ {effective:.1f} independent"

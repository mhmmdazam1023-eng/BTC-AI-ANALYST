"""SectorTracker: where is capital rotating? (Phase 8 Part 3)

Groups every active altcoin by its registered ``sector`` and averages each
group's BTC-relative strength. A positive sector average means that
narrative outperformed Bitcoin over the window — i.e. capital flowed in.

Bitcoin itself is EXCLUDED: it is the denominator. Its ``rs_7d`` is 0.0 by
construction, so including it would drag every sector average toward zero
and make the L1 bucket structurally weaker than the others.

Missing data (the requirement that decides whether this is trustworthy)
----------------------------------------------------------------------
Averages ignore NULLs (``pandas.mean`` skips NaN, like SQL ``AVG``). A coin
listed three weeks ago has an ``rs_7d`` but no ``rs_30d``, and must not
count toward the 30-day figure.

That leaves a subtler trap: the 7-day and 30-day averages can then rest on
DIFFERENT numbers of coins. ``coin_count`` and ``coin_count_30d`` are
therefore tracked and persisted separately — "AI +18%" reads identically
whether it came from three coins or one, and only the counts tell you which.

Status: a 2x2 over weekly vs monthly strength
---------------------------------------------
==================  ==========  ===========  ==============================
Status              ``rs_7d``   ``rs_30d``   Reading
==================  ==========  ===========  ==============================
``IN_FLOW``         > 0         > 0          leading and holding it
``ACCUMULATING``    > 0         < 0          weak month, bidding this week
                                             — bottoming / reversal
``PULLBACK``        < 0         > 0          strong month, correcting now
``OUT_OF_FLOW``     < 0         < 0          lagging outright
``NEUTRAL``         either leg exactly zero, or either leg missing
==================  ==========  ===========  ==============================

The quadrants are MUTUALLY EXCLUSIVE, so no ordering of the checks can
make a state unreachable. The earlier three-state version had
``ACCUMULATING`` defined as ``rs_7d < 0 AND rs_30d > 0`` — a strict subset
of ``OUT_OF_FLOW`` that had to be tested first or it could never fire. That
condition now correctly carries the name ``PULLBACK`` (strong month,
weekly correction), and ``ACCUMULATING`` was reassigned to its mirror
image, which is the genuine bottoming signal (user decision, 2026-07-28).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import Settings, get_settings
from src.db.engine import session_scope
from src.db.models import (
    BASE_SYMBOL,
    CoinRegistry,
    DailyFeature,
    SectorFlowStatus,
    SectorPerformance,
)

logger = logging.getLogger(__name__)

#: Sectors reported in the top/bottom shortlists.
SHORTLIST_SIZE = 3

#: Below this many contributing coins a sector average is one token's noise
#: wearing a sector's name. It is still computed and stored — suppressing it
#: would hide the thinness — but it is flagged in logs and in the prompt.
MIN_COINS_FOR_CONFIDENCE = 2

#: Risk-appetite heuristic: which buckets are the speculative end of the
#: market versus the defensive end. Used only to phrase the rotation
#: summary; it never changes a number.
_SPECULATIVE_SECTORS = frozenset({"Meme", "AI"})
_DEFENSIVE_SECTORS = frozenset({"L1", "RWA"})


class SectorTrackerError(RuntimeError):
    """Raised when sector performance cannot be computed at all."""


@dataclass(slots=True)
class SectorReading:
    """One sector's flow reading for a day."""

    sector: str
    avg_rs_7d: float | None
    avg_rs_30d: float | None
    status: SectorFlowStatus
    ranking: int
    coin_count: int
    coin_count_30d: int
    members: list[str] = field(default_factory=list)

    @property
    def is_thin(self) -> bool:
        """True when too few coins contributed to trust the average."""
        return self.coin_count < MIN_COINS_FOR_CONFIDENCE

    def as_dict(self) -> dict:
        return {
            "sector": self.sector,
            "avg_rs_7d": self.avg_rs_7d,
            "avg_rs_30d": self.avg_rs_30d,
            "status": self.status.value,
            "ranking": self.ranking,
            "coin_count": self.coin_count,
            "coin_count_30d": self.coin_count_30d,
        }

    def line(self) -> str:
        seven = "n/a" if self.avg_rs_7d is None else f"{self.avg_rs_7d:+.2f}%"
        thirty = "n/a" if self.avg_rs_30d is None else f"{self.avg_rs_30d:+.2f}%"
        thin = "  [THIN]" if self.is_thin else ""
        return (
            f"#{self.ranking} {self.sector:<14s} 7d {seven:>9s}  30d {thirty:>9s}  "
            f"{self.status.value:<12s} n={self.coin_count}"
            f"/{self.coin_count_30d}{thin}"
        )


@dataclass(slots=True)
class SectorSnapshot:
    """A full day's sector-rotation picture."""

    as_of: datetime
    rankings: list[SectorReading]
    #: Indonesian one-liner for logs and the terminal report.
    rotation_signal: str
    #: English equivalent injected into the Master Agent's prompt.
    rotation_signal_en: str = ""

    @property
    def top_sectors(self) -> list[SectorReading]:
        return self.rankings[:SHORTLIST_SIZE]

    @property
    def bottom_sectors(self) -> list[SectorReading]:
        return list(reversed(self.rankings[-SHORTLIST_SIZE:]))

    def to_prompt_context(self) -> str:
        """English one-paragraph brief for the Master Agent's prompt."""
        if not self.rankings:
            return "Sector Rotation: no sector data available today."

        def render(readings: list[SectorReading]) -> str:
            parts = []
            for reading in readings:
                if reading.avg_rs_7d is None:
                    continue
                note = f", only {reading.coin_count} coin" if reading.is_thin else ""
                parts.append(f"{reading.sector} ({reading.avg_rs_7d:+.1f}% vs BTC{note})")
            return ", ".join(parts) if parts else "none"

        def sectors_with(status: SectorFlowStatus) -> list[str]:
            return [r.sector for r in self.rankings if r.status is status]

        text = (
            f"Sector Rotation ({self.as_of.date()}, 7-day relative strength "
            f"vs BTC): Top sectors are {render(self.top_sectors)}. "
            f"Bottom sectors are {render(self.bottom_sectors)}. "
        )
        accumulating = sectors_with(SectorFlowStatus.ACCUMULATING)
        if accumulating:
            text += (
                f"Turning up after a weak month (possible bottoming): "
                f"{', '.join(accumulating)}. "
            )
        pullback = sectors_with(SectorFlowStatus.PULLBACK)
        if pullback:
            text += (
                f"Still ahead over 30 days but correcting this week: "
                f"{', '.join(pullback)}. "
            )
        text += self.rotation_signal_en
        return text


class SectorTracker:
    """Computes per-sector capital flow from ``daily_features``."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    # 1) LOAD
    # ------------------------------------------------------------------
    async def load_sector_data(self, as_of: datetime | None = None) -> pd.DataFrame:
        """Per-coin relative strength on one day, joined to its sector.

        Bitcoin is excluded here rather than in the aggregation, so it can
        never silently re-enter through a different code path.
        """
        async with session_scope() as session:
            if as_of is None:
                as_of = (
                    await session.execute(
                        select(func.max(DailyFeature.timestamp)).where(
                            DailyFeature.symbol == BASE_SYMBOL
                        )
                    )
                ).scalar_one_or_none()
                if as_of is None:
                    raise SectorTrackerError(
                        "daily_features is empty — run the collectors first"
                    )

            rows = (
                await session.execute(
                    select(
                        CoinRegistry.sector,
                        DailyFeature.symbol,
                        DailyFeature.rs_7d,
                        DailyFeature.rs_30d,
                    )
                    .join(CoinRegistry, CoinRegistry.symbol == DailyFeature.symbol)
                    .where(
                        DailyFeature.timestamp == as_of,
                        CoinRegistry.is_active.is_(True),
                        DailyFeature.symbol != BASE_SYMBOL,
                    )
                    .order_by(CoinRegistry.sector, DailyFeature.symbol)
                )
            ).all()

        frame = pd.DataFrame(rows, columns=["sector", "symbol", "rs_7d", "rs_30d"])
        if frame.empty:
            raise SectorTrackerError(
                f"no active altcoin features on {as_of.date()} — seed "
                "coins_registry (scripts/seed_sectors_phase8.py) and run the "
                "collectors"
            )
        frame["sector"] = frame["sector"].map(
            lambda value: value.value if hasattr(value, "value") else str(value)
        )
        for column in ("rs_7d", "rs_30d"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        frame.attrs["as_of"] = as_of
        logger.info(
            "[sector] loaded %d coin(s) across %d sector(s) for %s",
            len(frame), frame["sector"].nunique(), as_of.date(),
        )
        return frame

    # ------------------------------------------------------------------
    # 2) COMPUTE (pure — no I/O, unit-testable with synthetic frames)
    # ------------------------------------------------------------------
    @staticmethod
    def classify(
        avg_rs_7d: float | None, avg_rs_30d: float | None
    ) -> SectorFlowStatus:
        """Flow status for one sector: a 2x2 over weekly vs monthly strength.

        The four quadrants are mutually exclusive, so — unlike the earlier
        three-state version, where ACCUMULATING was a strict subset of
        OUT_OF_FLOW — no ordering of the checks can make a state
        unreachable. That whole class of bug is gone by construction.

        ``NEUTRAL`` covers both an exactly-flat leg and a missing one. Note
        the consequence: a sector whose members are too new to have an
        ``rs_30d`` reports NEUTRAL rather than being placed on the weekly
        leg alone. That is deliberate — a quadrant needs both axes — and it
        is why ``coin_count_30d`` is persisted alongside the status.
        """
        if avg_rs_7d is None or math.isnan(avg_rs_7d):
            return SectorFlowStatus.NEUTRAL
        if avg_rs_30d is None or math.isnan(avg_rs_30d):
            return SectorFlowStatus.NEUTRAL
        if avg_rs_7d > 0 and avg_rs_30d > 0:
            return SectorFlowStatus.IN_FLOW
        if avg_rs_7d > 0 and avg_rs_30d < 0:
            return SectorFlowStatus.ACCUMULATING
        if avg_rs_7d < 0 and avg_rs_30d > 0:
            return SectorFlowStatus.PULLBACK
        if avg_rs_7d < 0 and avg_rs_30d < 0:
            return SectorFlowStatus.OUT_OF_FLOW
        return SectorFlowStatus.NEUTRAL  # either leg exactly zero

    def compute(self, frame: pd.DataFrame) -> SectorSnapshot:
        """Aggregate per-coin relative strength into a sector picture."""
        as_of = frame.attrs.get("as_of") or datetime.now(tz=UTC)

        grouped = frame.groupby("sector", sort=False)
        # mean() skips NaN, count() counts only non-NaN — this IS the
        # missing-data requirement, and why the two counts can differ.
        aggregated = grouped.agg(
            avg_rs_7d=("rs_7d", "mean"),
            avg_rs_30d=("rs_30d", "mean"),
            coin_count=("rs_7d", "count"),
            coin_count_30d=("rs_30d", "count"),
        )
        members = grouped["symbol"].apply(list).to_dict()

        readings: list[SectorReading] = []
        for sector, row in aggregated.iterrows():
            if int(row["coin_count"]) == 0:
                logger.warning(
                    "[sector] %s has no coin with rs_7d on %s — excluded from "
                    "the ranking", sector, as_of.date(),
                )
                continue
            avg_7d = None if pd.isna(row["avg_rs_7d"]) else float(row["avg_rs_7d"])
            avg_30d = None if pd.isna(row["avg_rs_30d"]) else float(row["avg_rs_30d"])
            readings.append(
                SectorReading(
                    sector=str(sector),
                    avg_rs_7d=avg_7d,
                    avg_rs_30d=avg_30d,
                    status=self.classify(avg_7d, avg_30d),
                    ranking=0,  # assigned below
                    coin_count=int(row["coin_count"]),
                    coin_count_30d=int(row["coin_count_30d"]),
                    members=members.get(sector, []),
                )
            )

        if not readings:
            raise SectorTrackerError(
                f"no sector had a usable rs_7d on {as_of.date()}"
            )

        # Rank by 7-day strength, strongest first; unusable averages last.
        readings.sort(
            key=lambda reading: (
                reading.avg_rs_7d is None,
                -(reading.avg_rs_7d or 0.0),
            )
        )
        for position, reading in enumerate(readings, start=1):
            reading.ranking = position

        snapshot = SectorSnapshot(
            as_of=as_of,
            rankings=readings,
            rotation_signal=self._rotation_signal_id(readings),
        )
        snapshot.rotation_signal_en = self._rotation_signal_en(readings)

        logger.info("[sector] ranking on %s:", as_of.date())
        for reading in readings:
            logger.info("[sector]   %s", reading.line())
        thin = [r.sector for r in readings if r.is_thin]
        if thin:
            logger.warning(
                "[sector] thin sector(s) with <%d contributing coin(s): %s — "
                "treat their averages as a single token's noise",
                MIN_COINS_FOR_CONFIDENCE, ", ".join(thin),
            )
        logger.info("[sector] %s", snapshot.rotation_signal)
        return snapshot

    # ------------------------------------------------------------------
    @staticmethod
    def _leaders_laggards(
        readings: list[SectorReading],
    ) -> tuple[list[SectorReading], list[SectorReading]]:
        positive = [
            r for r in readings if r.avg_rs_7d is not None and r.avg_rs_7d > 0
        ]
        negative = [
            r for r in readings if r.avg_rs_7d is not None and r.avg_rs_7d < 0
        ]
        return positive[:2], list(reversed(negative))[:2]

    @classmethod
    def _risk_appetite(cls, readings: list[SectorReading]) -> str:
        """Speculative vs defensive buckets — phrasing only, never a number."""
        def mean_of(names: frozenset[str]) -> float | None:
            values = [
                r.avg_rs_7d
                for r in readings
                if r.sector in names and r.avg_rs_7d is not None
            ]
            return sum(values) / len(values) if values else None

        speculative = mean_of(_SPECULATIVE_SECTORS)
        defensive = mean_of(_DEFENSIVE_SECTORS)
        if speculative is None or defensive is None:
            return ""
        if speculative > defensive + 1.0:
            return "expanding"
        if defensive > speculative + 1.0:
            return "contracting"
        return "mixed"

    @classmethod
    def _rotation_signal_id(cls, readings: list[SectorReading]) -> str:
        """Indonesian one-liner (logs, terminal report)."""
        leaders, laggards = cls._leaders_laggards(readings)
        if not leaders and not laggards:
            return "Tidak ada rotasi sektor yang jelas — semua sektor datar vs BTC."
        if not leaders:
            return (
                "Tidak ada sektor yang mengungguli BTC — modal keluar dari "
                f"altcoin secara luas (terlemah: "
                f"{', '.join(r.sector for r in laggards)})."
            )
        if not laggards:
            return (
                "Seluruh sektor mengungguli BTC — risk-on luas (terkuat: "
                f"{', '.join(r.sector for r in leaders)})."
            )
        appetite = {
            "expanding": " Selera risiko menguat (narasi spekulatif memimpin).",
            "contracting": " Selera risiko melemah (aset mayor memimpin).",
            "mixed": "",
            "": "",
        }[cls._risk_appetite(readings)]
        return (
            f"Modal berotasi dari {' dan '.join(r.sector for r in laggards)} "
            f"ke {' dan '.join(r.sector for r in leaders)}.{appetite}"
        )

    @classmethod
    def _rotation_signal_en(cls, readings: list[SectorReading]) -> str:
        """English one-liner (Master Agent prompt)."""
        leaders, laggards = cls._leaders_laggards(readings)
        if not leaders and not laggards:
            return "No clear sector rotation; all sectors flat versus BTC."
        if not leaders:
            return "No sector is outperforming BTC — capital is leaving altcoins broadly."
        if not laggards:
            return "Every sector is outperforming BTC — broad risk-on."
        appetite = {
            "expanding": " Risk appetite is expanding: speculative narratives lead.",
            "contracting": " Risk appetite is contracting: majors lead.",
            "mixed": "",
            "": "",
        }[cls._risk_appetite(readings)]
        return (
            f"Capital is rotating out of "
            f"{' and '.join(r.sector for r in laggards)} into "
            f"{' and '.join(r.sector for r in leaders)}.{appetite}"
        )

    # ------------------------------------------------------------------
    # 3) SAVE
    # ------------------------------------------------------------------
    async def save(self, snapshot: SectorSnapshot) -> int:
        """Idempotent upsert into ``sector_performance`` (one row/sector/day)."""
        if not snapshot.rankings:
            logger.warning("[sector] nothing to save — empty ranking")
            return 0

        def to_decimal(value: float | None) -> Decimal | None:
            if value is None or math.isnan(value) or math.isinf(value):
                return None
            return Decimal(repr(round(value, 6)))

        payload = [
            {
                "timestamp": snapshot.as_of,
                "sector": reading.sector,
                "avg_rs_7d": to_decimal(reading.avg_rs_7d),
                "avg_rs_30d": to_decimal(reading.avg_rs_30d),
                "status": reading.status,
                "ranking": reading.ranking,
                "coin_count": reading.coin_count,
                "coin_count_30d": reading.coin_count_30d,
            }
            for reading in snapshot.rankings
        ]
        statement = pg_insert(SectorPerformance.__table__).values(payload)
        update_set = {
            column: statement.excluded[column]
            for column in (
                "avg_rs_7d", "avg_rs_30d", "status",
                "ranking", "coin_count", "coin_count_30d",
            )
        }
        update_set["ingested_at"] = func.now()
        statement = statement.on_conflict_do_update(
            index_elements=["timestamp", "sector"], set_=update_set
        )
        async with session_scope() as session:
            await session.execute(statement)
        logger.info(
            "[sector] upserted %d sector row(s) into 'sector_performance' for %s",
            len(payload), snapshot.as_of.date(),
        )
        return len(payload)

    # ------------------------------------------------------------------
    async def run(self, as_of: datetime | None = None) -> SectorSnapshot:
        """Load -> compute -> persist. Returns the snapshot."""
        frame = await self.load_sector_data(as_of)
        snapshot = self.compute(frame)
        await self.save(snapshot)
        return snapshot

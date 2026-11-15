"""Feature engineering layer (Phase 3).

Deterministic feature computation with pandas/numpy/`ta` — NEVER an LLM
(project rule 2026-07-08). Qwen AI is reserved for Phase 4 reasoning.
"""

from src.features.engine import FeatureEngine
from src.features.regime_detector import (
    RegimeDetectionError,
    RegimeDetector,
    RegimeSnapshot,
)
from src.features.sector_tracker import (
    SectorReading,
    SectorSnapshot,
    SectorTracker,
    SectorTrackerError,
)

__all__ = [
    "FeatureEngine",
    "RegimeDetector",
    "RegimeSnapshot",
    "RegimeDetectionError",
    "SectorTracker",
    "SectorSnapshot",
    "SectorReading",
    "SectorTrackerError",
]

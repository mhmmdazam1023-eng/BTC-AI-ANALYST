"""Forecasting & continuous-learning layer (Phase 6 + 7).

Deterministic quantitative forecasting (numpy Monte Carlo block-bootstrap),
realized-outcome evaluation, and a continuous-learning calibration loop that
feeds bias/volatility corrections back into the forecaster. No LLM
involvement — the statistical forecast becomes one INPUT to the Master Agent.
"""

from src.forecasting.engine import ForecastEngine, ForecastError
from src.forecasting.evaluator import ForecastEvaluator
from src.forecasting.kelly_criterion import (
    PositionSizer,
    PositionSizing,
    portfolio_exposure,
)
from src.forecasting.learning import LearningEvaluator
from src.forecasting.position_pipeline import PositionPipeline, PositionRun
from src.forecasting.scenario_engine import (
    ScenarioEngine,
    ScenarioEngineError,
    ScenarioSet,
)

__all__ = [
    "ForecastEngine",
    "ForecastError",
    "ForecastEvaluator",
    "LearningEvaluator",
    "ScenarioEngine",
    "ScenarioSet",
    "ScenarioEngineError",
    "PositionSizer",
    "PositionSizing",
    "portfolio_exposure",
    "PositionPipeline",
    "PositionRun",
]

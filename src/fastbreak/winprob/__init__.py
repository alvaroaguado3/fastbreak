"""Win-probability pipeline: estimate win likelihood and screen which stats are
genuinely predictive of winning (beyond the scoreboard).

Separate from the headline pipeline; shares only the project's design style.
"""
from .logistic import LogisticRegression
from .model import WinProbabilityModel
from .screening import PredictivityFilter, PredictiveStat
from .models import GameSnapshot, WinImpactInsight
from .pipeline import WinImpactPipeline

__all__ = [
    "LogisticRegression",
    "WinProbabilityModel",
    "PredictivityFilter",
    "PredictiveStat",
    "GameSnapshot",
    "WinImpactInsight",
    "WinImpactPipeline",
]

"""__init__.py — Public package exports for Hybrid V3 engines, preprocessing, search and diagnostic structures.

All executable definitions are retained from the research implementation;
only explanatory comments and internal docstrings are consolidated here.
"""

from .common import LaplaceDiagnostic, LocalScoreInfo, MixtureDiagnostic
from .engine import HybridV3ScoringEngine
from .preprocessing import QuantileDiscretizer
from .search import AtomicAdd2TabuGreedyHillClimbing, TabuGreedyHillClimbing

__all__ = [
    "HybridV3ScoringEngine",
    "QuantileDiscretizer",
    "AtomicAdd2TabuGreedyHillClimbing",
    "TabuGreedyHillClimbing",
    "LaplaceDiagnostic",
    "MixtureDiagnostic",
    "LocalScoreInfo",
]

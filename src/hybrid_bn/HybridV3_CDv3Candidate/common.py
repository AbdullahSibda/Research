"""common.py — Shared numerical constants and diagnostic record structures for local family scores and mixture/Laplace evaluations.

All executable definitions are retained from the research implementation;
only explanatory comments and internal docstrings are consolidated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

_EPS = 1e-12


@dataclass(frozen=True)
class LaplaceDiagnostic:
    child: str
    parents: Tuple[str, ...]
    discrete_configuration: object
    converged: bool
    iterations: int
    gradient_norm: float
    parameter_norm: float
    hessian_log_determinant: float


@dataclass(frozen=True)
class MixtureDiagnostic:
    child: str
    parents: Tuple[str, ...]
    discrete_configuration: object
    n_components: int
    method: str
    score: float
    selected: bool
    converged: bool
    iterations: int
    gradient_norm: float
    parameter_count: int
    min_effective_component_size: float
    hessian_log_determinant: float
    failure_reason: str = ""


@dataclass(frozen=True)
class LocalScoreInfo:
    child: str
    parents: Tuple[str, ...]
    quadrant: str
    family: str
    score: float
    parameter_count: int
    observed_configurations: int
    possible_configurations: int
    selected_components: int = 0

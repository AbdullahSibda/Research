"""engine.py — Central Hybrid V3 scorer dispatcher. Validates declared node types/cardinalities; preprocesses training data; routes local families to DD, CD, DC or CC; caches scores and exposes predictions/diagnostics.

All executable definitions are retained from the research implementation;
only explanatory comments and internal docstrings are consolidated here.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .cc import CCScorer
from .cd import CDScorer
from .common import LocalScoreInfo, _EPS
from .dc import DCScorer
from .dd import DDScorer


class HybridV3ScoringEngine:











    DISCRETE_CPT = "discrete_cpt"
    SOFTMAX_LAPLACE = "softmax_laplace_bma_v3"
    CONTINUOUS_V3 = "continuous_v3_corrected"
    CONDITIONAL_GAUSSIAN = "conditional_gaussian"
    CONDITIONAL_MIXTURE = "conditional_mixture_regression"

    def __init__(
        self,
        data: pd.DataFrame,
        discrete_nodes: Sequence[str],
        continuous_nodes: Sequence[str],
        cardinalities: Mapping[str, int],
        *,
        cc_scorer_file: str | Path | None = None,
        equivalent_sample_size: float = 2.0,
        continuous_coefficient_prior_variance: float = 1.0,
        continuous_intercept_prior_variance: float = 4.0,
        continuous_inverse_gamma_shape: float = 1.5,
        continuous_inverse_gamma_scale: float = 0.5,
        clg_coefficient_prior_variance: float = 9.0,
        clg_intercept_prior_variance: float = 40.0,
        clg_inverse_gamma_shape: float = 1.5,
        clg_inverse_gamma_scale: float = 0.5,
        softmax_coefficient_prior_variance: float = 2.0,
        softmax_intercept_prior_variance: float = 8.0,
        mixture_gate_coefficient_prior_variance: float = 0.5,
        mixture_gate_intercept_prior_variance: float = 0.5,
        mixture_max_components: int = 3,
        mixture_max_iter: int = 300,
        mixture_restarts: int = 3,
        mixture_hessian_step: float = 1e-4,
        mixture_min_effective_count: float = 5.0,
        logistic_max_iter: int = 300,
        cd_candidate_models: Sequence[str] = ("linear", "quadratic", "cubic", "spline5"),
    ) -> None:
        if data.empty:
            raise ValueError("HybridV3ScoringEngine requires non-empty training data.")
        if data.isna().any().any():
            raise ValueError("Training data contains missing values.")

        self.discrete_nodes = tuple(map(str, discrete_nodes))
        self.continuous_nodes = tuple(map(str, continuous_nodes))
        self._discrete_set = set(self.discrete_nodes)
        self._continuous_set = set(self.continuous_nodes)
        if self._discrete_set & self._continuous_set:
            raise ValueError("Discrete and continuous node sets must be disjoint.")
        if set(map(str, data.columns)) != self._discrete_set | self._continuous_set:
            raise ValueError("The declared node partition must exactly match the data columns.")

        self.cardinality = {str(k): int(v) for k, v in cardinalities.items()}
        if set(self.cardinality) != self._discrete_set:
            raise ValueError("Supply one cardinality for every discrete node only.")
        if any(value < 2 for value in self.cardinality.values()):
            raise ValueError("Every discrete node must have at least two states.")

        self.n = len(data)
        self.raw_data = data.copy().reset_index(drop=True)
        self.raw_data.columns = list(map(str, self.raw_data.columns))
        self.continuous_means: Dict[str, float] = {}
        self.continuous_stds: Dict[str, float] = {}
        self.data = self._fit_training_transform(self.raw_data)

        self.cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}
        self.score_info: Dict[Tuple[str, Tuple[str, ...]], LocalScoreInfo] = {}

        self.dd = DDScorer(
            self.data,
            self.cardinality,
            equivalent_sample_size=equivalent_sample_size,
        ) if self.discrete_nodes else None

        self.cd = CDScorer(
            self.data,
            self.cardinality,
            coefficient_prior_variance=softmax_coefficient_prior_variance,
            intercept_prior_variance=softmax_intercept_prior_variance,
            max_iter=logistic_max_iter,
            candidate_models=cd_candidate_models,
        ) if self.discrete_nodes and self.continuous_nodes else None

        self.dc = DCScorer(
            self.data,
            self.cardinality,
            clg_coefficient_prior_variance=clg_coefficient_prior_variance,
            clg_intercept_prior_variance=clg_intercept_prior_variance,
            clg_inverse_gamma_shape=clg_inverse_gamma_shape,
            clg_inverse_gamma_scale=clg_inverse_gamma_scale,
            continuous_coefficient_prior_variance=continuous_coefficient_prior_variance,
            continuous_intercept_prior_variance=continuous_intercept_prior_variance,
            continuous_inverse_gamma_shape=continuous_inverse_gamma_shape,
            continuous_inverse_gamma_scale=continuous_inverse_gamma_scale,
            mixture_gate_coefficient_prior_variance=mixture_gate_coefficient_prior_variance,
            mixture_gate_intercept_prior_variance=mixture_gate_intercept_prior_variance,
            mixture_max_components=mixture_max_components,
            mixture_max_iter=mixture_max_iter,
            mixture_restarts=mixture_restarts,
            mixture_hessian_step=mixture_hessian_step,
            mixture_min_effective_count=mixture_min_effective_count,
        ) if self.discrete_nodes and self.continuous_nodes else None

                                                                         
                                                                             
                                                                               
        self._cc = None
        self._cc_scorer_file = cc_scorer_file
        self._cc_raw_data = (
            self.raw_data.loc[:, list(self.continuous_nodes)].copy()
            if self.continuous_nodes else None
        )


    @property
    def cc(self) -> CCScorer:
        if not self.continuous_nodes or self._cc_raw_data is None:
            raise RuntimeError("CC scorer is unavailable because there are no continuous nodes.")
        if self._cc is None:
            self._cc = CCScorer(
                self._cc_raw_data,
                scorer_file=self._cc_scorer_file,
            )
        return self._cc

    def quadrant(self, child: str, parents: Sequence[str]) -> str:
        parents = self._validate_family(child, parents)
        c_parents = [p for p in parents if p in self._continuous_set]
        d_parents = [p for p in parents if p in self._discrete_set]
        if child in self._continuous_set:
            return "DC" if d_parents else "CC"
        return "CD" if c_parents else "DD"

    def score_local(self, child: str, parents: Sequence[str]) -> float:
        parents = self._validate_family(child, parents)
        key = (child, parents)
        if key in self.cache:
            return self.cache[key]

        c_parents = [p for p in parents if p in self._continuous_set]
        d_parents = [p for p in parents if p in self._discrete_set]

        if child in self._continuous_set and not d_parents:
            score = self.cc.score(child, c_parents)
            quadrant = "CC"
            family = self.CONTINUOUS_V3
            observed = 1
            q = 1
            parameter_count = -1                                                  
            selected_components = 0

        elif child in self._continuous_set:
            if self.dc is None:
                raise RuntimeError("DC scorer is unavailable.")
            score, family, observed, parameter_count, selected_components, q = self.dc.score(
                child, c_parents, d_parents, parents
            )
            quadrant = "DC"

        elif c_parents:
            if self.cd is None:
                raise RuntimeError("CD scorer is unavailable.")
            score, observed, parameter_count, q = self.cd.score(
                child, c_parents, d_parents, parents
            )
            quadrant = "CD"
            family = self.SOFTMAX_LAPLACE
            selected_components = 0

        else:
            if self.dd is None:
                raise RuntimeError("DD scorer is unavailable.")
            score, observed, parameter_count, q = self.dd.score(child, d_parents)
            quadrant = "DD"
            family = self.DISCRETE_CPT
            selected_components = 0

        if not np.isfinite(score):
            raise FloatingPointError(f"Non-finite score for {child!r} with parents {parents}: {score}")

        info = LocalScoreInfo(
            child=child,
            parents=parents,
            quadrant=quadrant,
            family=str(family),
            score=float(score),
            parameter_count=int(parameter_count),
            observed_configurations=int(observed),
            possible_configurations=int(q),
            selected_components=int(selected_components),
        )
        self.cache[key] = float(score)
        self.score_info[key] = info
        return float(score)

    def get_score_info(self, child: str, parents: Sequence[str]) -> LocalScoreInfo:
        parents = tuple(sorted(map(str, parents)))
        key = (str(child), parents)
        if key not in self.score_info:
            self.score_local(str(child), parents)
        return self.score_info[key]

    def score_info_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(info) for info in self.score_info.values()])

    def mixture_diagnostics_dataframe(self) -> pd.DataFrame:
        if self.dc is None:
            return pd.DataFrame()
        return pd.DataFrame([asdict(info) for info in self.dc.mixture_diagnostics])

    def laplace_diagnostics_dataframe(self) -> pd.DataFrame:
        if self.cd is None:
            return pd.DataFrame()
        return pd.DataFrame([asdict(info) for info in self.cd.diagnostics])

    def cd_model_diagnostics_dataframe(self) -> pd.DataFrame:
        if self.cd is None:
            return pd.DataFrame()
        return pd.DataFrame(list(getattr(self.cd, "model_diagnostics", [])))

    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if set(map(str, data.columns)) != self._discrete_set | self._continuous_set:
            raise ValueError("New data columns must exactly match the scorer's nodes.")
        transformed = data.copy().reset_index(drop=True)
        transformed.columns = list(map(str, transformed.columns))
        self._validate_discrete_data(transformed)
        for node in self.continuous_nodes:
            values = pd.to_numeric(transformed[node], errors="raise").to_numpy(float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Continuous column {node!r} contains invalid values.")
            transformed[node] = (values - self.continuous_means[node]) / self.continuous_stds[node]
        return transformed

    def preprocessing_metadata(self) -> Dict[str, object]:
        return {
            "continuous_means": dict(self.continuous_means),
            "continuous_stds": dict(self.continuous_stds),
            "cardinalities": dict(self.cardinality),
        }

    def _fit_training_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed = data.copy()
        self._validate_discrete_data(transformed)
        for node in self.continuous_nodes:
            values = pd.to_numeric(transformed[node], errors="raise").to_numpy(float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Continuous column {node!r} contains invalid values.")
            mean, std = float(values.mean()), float(values.std(ddof=0))
            if std < _EPS:
                std = 1.0
            self.continuous_means[node], self.continuous_stds[node] = mean, std
            transformed[node] = (values - mean) / std
        return transformed

    def _validate_discrete_data(self, data: pd.DataFrame) -> None:
        for node in self.discrete_nodes:
            values = pd.to_numeric(data[node], errors="raise").to_numpy(float)
            if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
                raise ValueError(f"Discrete column {node!r} must contain finite integer states.")
            values = values.astype(np.int64)
            if values.min() < 0 or values.max() >= self.cardinality[node]:
                raise ValueError(f"Discrete column {node!r} contains an out-of-range state.")
            data[node] = values

    def _validate_family(self, child: str, parents: Sequence[str]) -> Tuple[str, ...]:
        child = str(child)
        if child not in self.data.columns:
            raise KeyError(f"Unknown child node: {child!r}")
        parents = tuple(map(str, parents))
        if child in parents:
            raise ValueError("A node cannot be its own parent.")
        if len(set(parents)) != len(parents):
            raise ValueError("A parent list cannot contain duplicates.")
        parents = tuple(sorted(parents))
        unknown = [parent for parent in parents if parent not in self.data.columns]
        if unknown:
            raise KeyError(f"Unknown parent nodes: {unknown}")
        return parents

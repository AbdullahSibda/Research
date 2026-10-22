"""preprocessing.py — Quantile-based discretisation fit on training data and applied to evaluation data without refitting boundaries.

All executable definitions are retained from the research implementation;
only explanatory comments and internal docstrings are consolidated here.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd


class QuantileDiscretizer:


    def __init__(self, cardinalities: Mapping[str, int]) -> None:
        self.cardinalities = {str(k): int(v) for k, v in cardinalities.items()}
        if not self.cardinalities:
            raise ValueError("At least one discrete-node cardinality is required.")
        if any(value < 2 for value in self.cardinalities.values()):
            raise ValueError("Every discrete node must have at least two states.")
        self.bin_edges: Dict[str, np.ndarray] = {}
        self.is_fitted = False

    @property
    def categories(self) -> Dict[str, Tuple[int, ...]]:
        return {node: tuple(range(cardinality)) for node, cardinality in self.cardinalities.items()}

    def fit(self, data: pd.DataFrame) -> "QuantileDiscretizer":
        self._validate_columns(data)
        self.bin_edges = {}
        for node, cardinality in self.cardinalities.items():
            values = pd.to_numeric(data[node], errors="raise").to_numpy(float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Column {node!r} contains NaN or infinite values.")
            edges = np.quantile(values, np.linspace(0.0, 1.0, cardinality + 1))
            if np.unique(edges).size != cardinality + 1:
                raise ValueError(f"Column {node!r} cannot form {cardinality} distinct quantile bins.")
            edges = edges.astype(float, copy=True)
            edges[0], edges[-1] = -np.inf, np.inf
            self.bin_edges[node] = edges
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Fit the discretizer before calling transform().")
        self._validate_columns(data)
        transformed = data.copy()
        for node, cardinality in self.cardinalities.items():
            values = pd.to_numeric(transformed[node], errors="raise").to_numpy(float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Column {node!r} contains NaN or infinite values.")
            states = pd.cut(values, bins=self.bin_edges[node], labels=False, include_lowest=True, right=True)
            if np.any(pd.isna(states)):
                raise ValueError(f"Discretisation produced missing states for {node!r}.")
            states = np.asarray(states, dtype=np.int64)
            if states.min() < 0 or states.max() >= cardinality:
                raise ValueError(f"Discretisation produced an invalid state for {node!r}.")
            transformed[node] = states
        return transformed

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)

    def to_metadata(self) -> Dict[str, object]:
        if not self.is_fitted:
            raise RuntimeError("Fit the discretizer before exporting metadata.")
        return {
            "cardinalities": dict(self.cardinalities),
            "bin_edges": {node: edges.copy() for node, edges in self.bin_edges.items()},
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "QuantileDiscretizer":
        cardinalities = metadata.get("cardinalities")
        edges = metadata.get("bin_edges")
        if not isinstance(cardinalities, Mapping) or not isinstance(edges, Mapping):
            raise ValueError("Invalid discretizer metadata.")
        result = cls(cardinalities)
        if set(edges) != set(result.cardinalities):
            raise ValueError("Saved boundaries do not match saved cardinalities.")
        result.bin_edges = {str(node): np.asarray(value, dtype=float).copy() for node, value in edges.items()}
        for node, cardinality in result.cardinalities.items():
            values = result.bin_edges[node]
            if values.shape != (cardinality + 1,) or not np.all(np.diff(values) > 0):
                raise ValueError(f"Invalid saved boundaries for {node!r}.")
        result.is_fitted = True
        return result

    def _validate_columns(self, data: pd.DataFrame) -> None:
        missing = [node for node in self.cardinalities if node not in data.columns]
        if missing:
            raise ValueError(f"Data is missing required columns: {missing}")

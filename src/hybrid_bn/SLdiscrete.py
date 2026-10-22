"""Discrete Bayesian-network baseline: quantile preprocessing, BIC scoring and Tabu DAG search.

PURPOSE
-------
This module implements the discrete comparison model used in the research
experiments. It provides reusable components, not a standalone command-line
program. The caller is responsible for data preparation, selecting the
experimental random seeds and supplying the training/test partitions.

MAIN COMPONENTS
---------------
QuantileDiscretizer
    Learns per-variable quantile cut points from training observations and
    applies those same boundaries to new observations. Infinite outer bounds
    permit test values outside the original training range. Repeated quantile
    boundaries are rejected rather than silently reducing a variable's number
    of states. Existing integer-coded categorical features must be handled
    consistently by the calling pipeline; this class quantile-bins its inputs.

DiscreteScoringEngine
    Calculates and caches local multinomial Bayesian Information Criterion
    (BIC) scores for fully discrete data. For a child with r possible states
    and q parent-state configurations, the score is the maximised conditional
    multinomial log-likelihood minus 0.5 * (r - 1) * q * log(n).
    Parent order is canonicalised for caching. The scorer requires observed
    integer state labels in the range 0 through cardinality - 1, with no
    missing observations.

TabuGreedyHillClimbing
    Starts from an empty directed acyclic graph (DAG) and evaluates valid
    single-edge additions, removals and reversals. It rejects cycles and
    parent-count violations, updates only affected local scores, and uses a
    Tabu list to discourage immediate undoing of moves. An aspiration rule
    permits Tabu moves that surpass the best graph score observed so far.
    The best encountered DAG is returned, not necessarily the final visited
    graph. The search is heuristic and does not guarantee a global optimum.

HELPERS AND ENTRY POINTS
------------------------
default_cardinalities
    Assigns three states per node, except Dementia_Diagnosis, which is binary
    by default. This default is a convenience, not a universal data rule.

discretize_continuous_dataset
    Backward-compatible one-dataset discretisation helper. For leakage-safe
    cross-validation, fit QuantileDiscretizer on the training fold and reuse
    that fitted instance for the held-out fold.

IMPORTANT IMPLEMENTATION NOTES
------------------------------
- All exported names and callable signatures are preserved from the research
  implementation. This repository edition only consolidates documentation.
- Default search options are tabu_length=10, max_iter=300, max_parents=3 and
  improvement_tolerance=1e-10. Experiment runners may override these.
- No ELSA data, file-system locations or external data files are embedded.
- Dependencies: numpy, pandas and networkx.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


class QuantileDiscretizer:
    def __init__(self, cardinalities: Mapping[str, int]) -> None:
        self.cardinalities = {
            str(column): int(cardinality)
            for column, cardinality in cardinalities.items()
        }
        self.bin_edges: Dict[str, np.ndarray] = {}
        self.is_fitted = False

        if not self.cardinalities:
            raise ValueError("At least one column cardinality must be supplied.")

        invalid = {
            column: cardinality
            for column, cardinality in self.cardinalities.items()
            if cardinality < 2
        }
        if invalid:
            raise ValueError(
                "Every discrete variable must have at least two states. "
                f"Invalid entries: {invalid}"
            )

    def fit(self, data: pd.DataFrame) -> "QuantileDiscretizer":

        self._validate_columns(data)
        self.bin_edges = {}

        for column, cardinality in self.cardinalities.items():
            values = data[column].to_numpy(dtype=float)

            if not np.all(np.isfinite(values)):
                raise ValueError(f"Column {column!r} contains NaN or infinite values.")

            quantiles = np.linspace(0.0, 1.0, cardinality + 1)
            edges = np.quantile(values, quantiles)

            if np.unique(edges).size != cardinality + 1:
                raise ValueError(
                    f"Column {column!r} cannot be split into {cardinality} "
                    "distinct quantile bins because some boundaries coincide."
                )

            
            
            edges = edges.astype(float, copy=True)
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.bin_edges[column] = edges

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        if not self.is_fitted:
            raise RuntimeError("The discretizer must be fitted before transform().")

        self._validate_columns(data)
        transformed = data.copy()

        for column, cardinality in self.cardinalities.items():
            values = transformed[column].to_numpy(dtype=float)

            if not np.all(np.isfinite(values)):
                raise ValueError(f"Column {column!r} contains NaN or infinite values.")

            states = pd.cut(
                values,
                bins=self.bin_edges[column],
                labels=False,
                include_lowest=True,
                right=True,
            )

            if np.any(pd.isna(states)):
                raise ValueError(f"Discretisation produced missing states for {column!r}.")

            transformed[column] = np.asarray(states, dtype=np.int64)

            observed_states = transformed[column].to_numpy()
            if observed_states.min() < 0 or observed_states.max() >= cardinality:
                raise ValueError(f"Invalid state produced for column {column!r}.")

        return transformed

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:

        return self.fit(data).transform(data)

    def _validate_columns(self, data: pd.DataFrame) -> None:
        missing = [column for column in self.cardinalities if column not in data.columns]
        if missing:
            raise ValueError(f"Data is missing required columns: {missing}")


class DiscreteScoringEngine:










    def __init__(
        self,
        data: pd.DataFrame,
        cardinalities: Mapping[str, int] | None = None,
    ) -> None:
        if data.empty:
            raise ValueError("The scoring dataset cannot be empty.")
        if data.isna().any().any():
            raise ValueError("The scoring dataset contains missing values.")

        self.data = data.copy()
        self.n = len(self.data)
        self.cardinality = (
            {column: int(value) for column, value in cardinalities.items()}
            if cardinalities is not None
            else {column: int(self.data[column].nunique()) for column in self.data.columns}
        )
        self.cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}

        self._validate_discrete_data()

    def score_local(self, child: str, parents: Sequence[str]) -> float:

        parent_tuple = tuple(sorted(parents))
        cache_key = (child, parent_tuple)

        if cache_key in self.cache:
            return self.cache[cache_key]

        if child not in self.data.columns:
            raise KeyError(f"Unknown child node: {child!r}")
        if child in parent_tuple:
            raise ValueError("A node cannot be its own parent.")
        unknown_parents = [p for p in parent_tuple if p not in self.data.columns]
        if unknown_parents:
            raise KeyError(f"Unknown parent nodes: {unknown_parents}")

        child_cardinality = self.cardinality[child]

        if not parent_tuple:
            counts = np.bincount(
                self.data[child].to_numpy(dtype=np.int64),
                minlength=child_cardinality,
            )
            positive_counts = counts[counts > 0]
            log_likelihood = float(
                np.sum(positive_counts * np.log(positive_counts / self.n))
            )
            parent_configurations = 1
        else:
            grouped_counts = (
                self.data.groupby(list(parent_tuple) + [child], observed=True)
                .size()
                .rename("count")
                .reset_index()
            )
            parent_totals = grouped_counts.groupby(list(parent_tuple))["count"].transform("sum")
            probabilities = grouped_counts["count"] / parent_totals
            log_likelihood = float(
                np.sum(grouped_counts["count"] * np.log(probabilities))
            )
            parent_configurations = int(
                np.prod([self.cardinality[parent] for parent in parent_tuple])
            )

        parameter_count = (child_cardinality - 1) * parent_configurations
        penalty = 0.5 * parameter_count * np.log(self.n)
        score = float(log_likelihood - penalty)

        self.cache[cache_key] = score
        return score

    def _validate_discrete_data(self) -> None:
        missing_cardinalities = [
            column for column in self.data.columns if column not in self.cardinality
        ]
        if missing_cardinalities:
            raise ValueError(
                f"Missing cardinalities for columns: {missing_cardinalities}"
            )

        for column in self.data.columns:
            values = self.data[column].to_numpy()
            cardinality = self.cardinality[column]

            if cardinality < 2:
                raise ValueError(
                    f"Column {column!r} has invalid cardinality {cardinality}."
                )

            if not np.all(np.equal(values, np.floor(values))):
                raise ValueError(
                    f"Column {column!r} must contain integer state labels."
                )

            integer_values = values.astype(np.int64)
            if integer_values.min() < 0 or integer_values.max() >= cardinality:
                raise ValueError(
                    f"Column {column!r} contains a state outside "
                    f"0 to {cardinality - 1}."
                )

            self.data[column] = integer_values


class TabuGreedyHillClimbing:


    def __init__(
        self,
        nodes: Sequence[str],
        scorer: DiscreteScoringEngine,
        tabu_length: int = 10,
        max_iter: int = 300,
        max_parents: int = 3,
        improvement_tolerance: float = 1e-10,
    ) -> None:
        self.nodes = list(nodes)
        self.scorer = scorer
        self.tabu_length = int(tabu_length)
        self.max_iter = int(max_iter)
        self.max_parents = int(max_parents)
        self.improvement_tolerance = float(improvement_tolerance)

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        self.node_scores: Dict[str, float] = {}
        self.best_score_: float | None = None
        self.score_history_: List[float] = []

    def _initialize_scores(self) -> float:
        self.node_scores = {
            node: self.scorer.score_local(node, []) for node in self.nodes
        }
        return float(sum(self.node_scores.values()))

    @staticmethod
    def _is_tabu(
        action: Tuple[str, str, str],
        tabu_list: Mapping[Tuple[str, str, str], int],
        iteration: int,
    ) -> bool:
        return tabu_list.get(action, -1) > iteration

    def _admissible(
        self,
        action: Tuple[str, str, str],
        candidate_score: float,
        global_best_score: float,
        tabu_list: Mapping[Tuple[str, str, str], int],
        iteration: int,
    ) -> bool:
        if not self._is_tabu(action, tabu_list, iteration):
            return True
        return candidate_score > global_best_score + self.improvement_tolerance

    def search(self) -> List[Tuple[str, str]]:
        current_score = self._initialize_scores()
        best_graph = self.graph.copy()
        best_score = current_score
        tabu_list: Dict[Tuple[str, str, str], int] = {}
        self.score_history_ = [current_score]

        for iteration in range(self.max_iter):
            print(
                "Tabu Search (Discrete Baseline): "
                f"Iteration {iteration + 1}/{self.max_iter} | "
                f"Best Score: {best_score:.2f}",
                end="\r",
            )

            best_neighbor_score = -np.inf
            best_action: Tuple[str, str, str] | None = None
            best_node_score_updates: Dict[str, float] = {}

            for u in self.nodes:
                for v in self.nodes:
                    if u == v:
                        continue

                    
                    if not self.graph.has_edge(u, v) and not self.graph.has_edge(v, u):
                        parents_v = list(self.graph.predecessors(v))
                        if len(parents_v) >= self.max_parents:
                            continue

                        self.graph.add_edge(u, v)
                        valid = nx.is_directed_acyclic_graph(self.graph)
                        self.graph.remove_edge(u, v)
                        if not valid:
                            continue

                        new_v_score = self.scorer.score_local(v, parents_v + [u])
                        score = current_score + new_v_score - self.node_scores[v]
                        action = ("add", u, v)

                        if (
                            score > best_neighbor_score
                            and self._admissible(
                                action,
                                score,
                                best_score,
                                tabu_list,
                                iteration,
                            )
                        ):
                            best_neighbor_score = score
                            best_action = action
                            best_node_score_updates = {v: new_v_score}

                    
                    elif self.graph.has_edge(u, v):
                        parents_v = list(self.graph.predecessors(v))
                        new_parents_v = [parent for parent in parents_v if parent != u]
                        new_v_score = self.scorer.score_local(v, new_parents_v)

                        removal_score = current_score + new_v_score - self.node_scores[v]
                        removal_action = ("remove", u, v)
                        if (
                            removal_score > best_neighbor_score
                            and self._admissible(
                                removal_action,
                                removal_score,
                                best_score,
                                tabu_list,
                                iteration,
                            )
                        ):
                            best_neighbor_score = removal_score
                            best_action = removal_action
                            best_node_score_updates = {v: new_v_score}

                        parents_u = list(self.graph.predecessors(u))
                        if len(parents_u) >= self.max_parents:
                            continue

                        self.graph.remove_edge(u, v)
                        self.graph.add_edge(v, u)
                        valid = nx.is_directed_acyclic_graph(self.graph)
                        self.graph.remove_edge(v, u)
                        self.graph.add_edge(u, v)
                        if not valid:
                            continue

                        new_u_score = self.scorer.score_local(u, parents_u + [v])
                        reversal_score = (
                            current_score
                            + new_v_score
                            - self.node_scores[v]
                            + new_u_score
                            - self.node_scores[u]
                        )
                        reversal_action = ("reverse", u, v)

                        if (
                            reversal_score > best_neighbor_score
                            and self._admissible(
                                reversal_action,
                                reversal_score,
                                best_score,
                                tabu_list,
                                iteration,
                            )
                        ):
                            best_neighbor_score = reversal_score
                            best_action = reversal_action
                            best_node_score_updates = {
                                v: new_v_score,
                                u: new_u_score,
                            }

            if best_action is None:
                break

            action_type, u, v = best_action
            if action_type == "add":
                self.graph.add_edge(u, v)
                inverse_action = ("remove", u, v)
            elif action_type == "remove":
                self.graph.remove_edge(u, v)
                inverse_action = ("add", u, v)
            else:
                self.graph.remove_edge(u, v)
                self.graph.add_edge(v, u)
                inverse_action = ("reverse", v, u)

            self.node_scores.update(best_node_score_updates)
            current_score = float(best_neighbor_score)
            self.score_history_.append(current_score)

            if current_score > best_score + self.improvement_tolerance:
                best_score = current_score
                best_graph = self.graph.copy()

            tabu_list[inverse_action] = iteration + self.tabu_length + 1
            tabu_list = {
                action: expiry
                for action, expiry in tabu_list.items()
                if expiry > iteration + 1
            }

        print()
        self.graph = best_graph
        self.best_score_ = float(best_score)
        return list(self.graph.edges())


def default_cardinalities(
    nodes: Sequence[str],
    binary_node: str = "Dementia_Diagnosis",
) -> Dict[str, int]:

    return {node: 2 if node == binary_node else 3 for node in nodes}


def discretize_continuous_dataset(
    data: pd.DataFrame,
    payload: Mapping[str, object],
) -> pd.DataFrame:





    nodes = list(payload["nodes"])
    discretizer = QuantileDiscretizer(default_cardinalities(nodes))
    return discretizer.fit_transform(data)


__all__ = [
    "QuantileDiscretizer",
    "DiscreteScoringEngine",
    "TabuGreedyHillClimbing",
    "default_cardinalities",
    "discretize_continuous_dataset",
]
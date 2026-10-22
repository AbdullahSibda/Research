"""Continuous Bayesian-network baseline: linear-Gaussian BIC and Tabu DAG search.

Purpose
-------
Provides the fully continuous comparison architecture for synthetic and ELSA
experiments. All input columns are interpreted as continuous numeric variables;
this baseline does not preserve categorical likelihoods or model nonlinear or
multimodal conditional densities. The module defines reusable classes only and
has no command-line entry point.

ContinuousScoringEngine
-----------------------
For each child Y and candidate parent set P, fits a Gaussian linear-regression
conditional model Y = beta_0 + sum(beta_j * X_j) + epsilon, with independent
Gaussian residual error. Roots use an intercept-only Gaussian distribution.
Coefficients are estimated by NumPy least squares, and the residual variance
is the maximum-likelihood estimate RSS/n, bounded below by minimum_variance
(default 1e-12) for numerical stability. The local BIC score is

    -n/2 * [log(2*pi*sigma_squared) + 1] - (|P|+2)/2 * log(n).

The parameter count includes the intercept, one coefficient per parent, and
the residual variance. The scorer rejects empty datasets, datasets with fewer
than two rows, nonnumeric/nonfinite observations, unknown nodes, self-parenting,
and duplicate parents. Scores are cached by (child, sorted parent tuple), so
repeated local-family evaluations do not repeat regression fits.

TabuGreedyHillClimbing
---------------------
Begins with an empty directed acyclic graph (DAG) over the requested nodes and
scores each node as a root. At each iteration, considers valid single-edge
additions, deletions, and reversals; proposals that create cycles or exceed the
maximum parent count are disallowed. Because the BIC objective decomposes by
child family, candidate scores update only the families affected by a move.
The best admissible neighbour is applied, including potentially non-improving
moves; a finite Tabu list temporarily forbids inverse moves. The aspiration
criterion permits a Tabu move if it exceeds the best score seen so far.
The best graph encountered is restored after search, exposed in self.graph;
search() returns its directed edge list. self.best_score_ and
self.score_history_ record the best score and traversed objective values.

Default search settings: tabu_length=10, max_iter=300, max_parents=3 and
improvement_tolerance=1e-10. Individual benchmark runners may override these
settings (e.g., the synthetic experiments use 150 iterations). The module's
only third-party dependencies are NumPy, pandas and NetworkX.

Reproducibility
---------------
Input data must already be cleaned, completed, and transformed as appropriate
for the experiment. Cross-validation, data imputation, train/test separation,
random-seed control, result saving and evaluation are responsibilities of the
calling experiment scripts. Mathematical operations, class names, public
methods and default parameters are unchanged from the original baseline.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


class ContinuousScoringEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        minimum_variance: float = 1e-12,
    ) -> None:
        if data.empty:
            raise ValueError("The scoring dataset cannot be empty.")
        if len(data) < 2:
            raise ValueError("At least two observations are required.")
        if minimum_variance <= 0:
            raise ValueError("minimum_variance must be positive.")

        self.data = data.copy()
        self.n = len(self.data)
        self.minimum_variance = float(minimum_variance)
        self.cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}

        self._validate_continuous_data()

    def score_local(self, child: str, parents: Sequence[str]) -> float:

        parent_tuple = tuple(sorted(parents))
        cache_key = (child, parent_tuple)

        if cache_key in self.cache:
            return self.cache[cache_key]

        self._validate_family(child, parent_tuple)

        y = self.data[child].to_numpy(dtype=float)

        if not parent_tuple:
            mean = float(np.mean(y))
            residuals = y - mean
            parameter_count = 2
        else:
            parent_values = self.data.loc[:, list(parent_tuple)].to_numpy(dtype=float)
            design_matrix = np.column_stack(
                [np.ones(self.n, dtype=float), parent_values]
            )

            coefficients, _, _, _ = np.linalg.lstsq(
                design_matrix,
                y,
                rcond=None,
            )
            residuals = y - design_matrix @ coefficients
            parameter_count = len(parent_tuple) + 2

        residual_variance = float(np.dot(residuals, residuals) / self.n)
        residual_variance = max(residual_variance, self.minimum_variance)

        log_likelihood = -0.5 * self.n * (
            np.log(2.0 * np.pi * residual_variance) + 1.0
        )
        penalty = 0.5 * parameter_count * np.log(self.n)
        score = float(log_likelihood - penalty)

        self.cache[cache_key] = score
        return score

    def _validate_continuous_data(self) -> None:
        for column in self.data.columns:
            try:
                values = self.data[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Column {column!r} must contain numeric values."
                ) from exc

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Column {column!r} contains NaN or infinite values."
                )

            self.data[column] = values

    def _validate_family(
        self,
        child: str,
        parents: Tuple[str, ...],
    ) -> None:
        if child not in self.data.columns:
            raise KeyError(f"Unknown child node: {child!r}")
        if child in parents:
            raise ValueError("A node cannot be its own parent.")
        if len(set(parents)) != len(parents):
            raise ValueError("The parent list contains duplicates.")

        unknown_parents = [
            parent for parent in parents if parent not in self.data.columns
        ]
        if unknown_parents:
            raise KeyError(f"Unknown parent nodes: {unknown_parents}")


class TabuGreedyHillClimbing:


    def __init__(
        self,
        nodes: Sequence[str],
        scorer: ContinuousScoringEngine,
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
                "Tabu Search (Continuous Baseline): "
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


__all__ = [
    "ContinuousScoringEngine",
    "TabuGreedyHillClimbing",
]
"""search.py — Tabu-constrained greedy DAG search with edge additions, removals, reversals and optional atomic add2 moves; cycle checks, parent limits and aspiration handling.

All executable definitions are retained from the research implementation;
only explanatory comments and internal docstrings are consolidated here.
"""





from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np


class AtomicAdd2TabuGreedyHillClimbing:
    def __init__(
        self,
        nodes: Sequence[str],
        scorer,
        tabu_length: int = 10,
        max_iter: int = 150,
        max_parents: int = 3,
        improvement_tolerance: float = 1e-10,
        enable_add2: bool = True,
        progress_label: str = "Hybrid V3",
    ) -> None:
        self.nodes = list(map(str, nodes))
        self.scorer = scorer
        self.tabu_length = int(tabu_length)
        self.max_iter = int(max_iter)
        self.max_parents = int(max_parents)
        self.improvement_tolerance = float(improvement_tolerance)
        self.enable_add2 = bool(enable_add2)
        self.progress_label = str(progress_label)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        self.node_scores: Dict[str, float] = {}
        self.best_score_: float | None = None
        self.score_history_: List[float] = []
        self.action_history_: List[Tuple[str, ...]] = []
        self.iteration_diagnostics_: List[dict] = []

    @staticmethod
    def _is_tabu(action: Tuple[str, ...], tabu_list: Mapping[Tuple[str, ...], int], iteration: int) -> bool:
        if tabu_list.get(action, -1) > iteration:
            return True
        if action and action[0] == "add2":
            _, p1, p2, child = action
            return (
                tabu_list.get(("add", p1, child), -1) > iteration
                or tabu_list.get(("add", p2, child), -1) > iteration
            )
        return False

    def _admissible(self, action, candidate_score, global_best_score, tabu_list, iteration) -> bool:
        if not self._is_tabu(action, tabu_list, iteration):
            return True
        return candidate_score > global_best_score + self.improvement_tolerance

    def _initialize_scores(self) -> float:
        self.node_scores = {node: self.scorer.score_local(node, []) for node in self.nodes}
        return float(sum(self.node_scores[node] for node in self.nodes))

    def _ordinary_candidates(self, current_score: float):

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
                    new_v = float(self.scorer.score_local(v, parents_v + [u]))
                    yield ("add", u, v), current_score + new_v - self.node_scores[v], {v: new_v}

                elif self.graph.has_edge(u, v):
                    parents_v = list(self.graph.predecessors(v))
                    new_parents_v = [p for p in parents_v if p != u]
                    new_v = float(self.scorer.score_local(v, new_parents_v))
                    yield ("remove", u, v), current_score + new_v - self.node_scores[v], {v: new_v}

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
                    new_u = float(self.scorer.score_local(u, parents_u + [v]))
                    score = current_score + new_v - self.node_scores[v] + new_u - self.node_scores[u]
                    yield ("reverse", u, v), score, {v: new_v, u: new_u}

    def _add2_candidates(self, current_score: float):
        if not self.enable_add2:
            return
        for child in self.nodes:
            current_parents = list(self.graph.predecessors(child))
            if self.max_parents - len(current_parents) < 2:
                continue
            proposal = [
                node for node in self.nodes
                if node != child
                and node not in current_parents
                and not self.graph.has_edge(child, node)
                and not self.graph.has_edge(node, child)
            ]
            for p1, p2 in itertools.combinations(proposal, 2):
                self.graph.add_edge(p1, child)
                self.graph.add_edge(p2, child)
                valid = nx.is_directed_acyclic_graph(self.graph)
                self.graph.remove_edge(p2, child)
                self.graph.remove_edge(p1, child)
                if not valid:
                    continue
                new_child = float(self.scorer.score_local(child, current_parents + [p1, p2]))
                score = current_score + new_child - self.node_scores[child]
                yield ("add2", p1, p2, child), score, {child: new_child}

    def search(self) -> List[Tuple[str, str]]:
        current_score = self._initialize_scores()
        best_graph = self.graph.copy()
        best_score = current_score
        tabu_list: Dict[Tuple[str, ...], int] = {}
        self.score_history_ = [current_score]
        self.action_history_ = []
        self.iteration_diagnostics_ = []

        for iteration in range(self.max_iter):
            best_neighbor_score = -np.inf
            best_action = None
            best_updates: Dict[str, float] = {}
            ordinary_count = 0
            add2_count = 0

                                                                             
                                                                      
            for action, score, updates in self._ordinary_candidates(current_score):
                ordinary_count += 1
                if score > best_neighbor_score and self._admissible(action, score, best_score, tabu_list, iteration):
                    best_neighbor_score = float(score)
                    best_action = action
                    best_updates = updates

            for action, score, updates in self._add2_candidates(current_score) or ():
                add2_count += 1
                if score > best_neighbor_score and self._admissible(action, score, best_score, tabu_list, iteration):
                    best_neighbor_score = float(score)
                    best_action = action
                    best_updates = updates

            if best_action is None:
                break

            kind = best_action[0]
            inverse_actions: list[Tuple[str, ...]] = []
            if kind == "add":
                _, u, v = best_action
                self.graph.add_edge(u, v)
                inverse_actions = [("remove", u, v)]
            elif kind == "remove":
                _, u, v = best_action
                self.graph.remove_edge(u, v)
                inverse_actions = [("add", u, v)]
            elif kind == "reverse":
                _, u, v = best_action
                self.graph.remove_edge(u, v)
                self.graph.add_edge(v, u)
                inverse_actions = [("reverse", v, u)]
            elif kind == "add2":
                _, p1, p2, child = best_action
                self.graph.add_edge(p1, child)
                self.graph.add_edge(p2, child)
                inverse_actions = [("remove", p1, child), ("remove", p2, child)]
            else:
                raise ValueError(best_action)

            if not nx.is_directed_acyclic_graph(self.graph):
                raise RuntimeError(f"Applied action produced a cycle: {best_action}")
            if any(self.graph.in_degree(node) > self.max_parents for node in self.nodes):
                raise RuntimeError(f"Applied action exceeded max_parents: {best_action}")

            self.node_scores.update(best_updates)
            current_score = float(best_neighbor_score)
            self.score_history_.append(current_score)
            self.action_history_.append(best_action)

            if current_score > best_score + self.improvement_tolerance:
                best_score = current_score
                best_graph = self.graph.copy()

            expiry = iteration + self.tabu_length + 1
            for inverse in inverse_actions:
                tabu_list[inverse] = expiry
            tabu_list = {a: e for a, e in tabu_list.items() if e > iteration + 1}

            self.iteration_diagnostics_.append({
                "iteration": iteration + 1,
                "ordinary_candidate_count": ordinary_count,
                "add2_candidate_count": add2_count,
                "selected_action": list(best_action),
                "current_score": current_score,
                "best_score": best_score,
                "edge_count": self.graph.number_of_edges(),
            })
            print(
                f"{self.progress_label}: iter={iteration + 1} | chose={best_action} | "
                f"best={best_score:.3f} | edges={self.graph.number_of_edges()} | "
                f"ordinary={ordinary_count} | add2={add2_count}",
                flush=True,
            )

        self.graph = best_graph
        self.best_score_ = float(best_score)
        return list(self.graph.edges())


                                                                      
TabuGreedyHillClimbing = AtomicAdd2TabuGreedyHillClimbing

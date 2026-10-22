"""Continuous-Generator Benchmark: Multimodal Ground Truth.

Purpose
-------
Generate reproducible continuous networks containing multimodal roots and
nonlinear conditional relationships. Selected roots use Gaussian mixtures;
selected non-root nodes employ threshold-gated Gaussian regressions, alongside
ordinary Gaussian families. Gate thresholds are fitted from an independent
calibration sample and stored as fixed generating parameters before the
benchmark observations are sampled.

Core workflow
-------------
1. Generate an 18-node directed acyclic graph using a seeded node ordering,
   an edge-inclusion probability, and a maximum parent count.
2. Draw and retain the parameters of the ground-truth conditional models.
3. Forward-sample observations in topological order with reproducible random
   streams, maintaining nested samples for benchmarking across sample sizes.
4. Expose the sample as a DataFrame and as a payload containing the DAG,
   node-type metadata, exact generating parameters, seed and sample size.
5. Optionally serialize the payload, visualize the network and inspect
   generated continuous distributions.

Representation
--------------
All 18 generating variables are continuous. The continuous, multiclass and
'discrete' node groups are metadata for later learner-side transformations;
they do not imply that categorical data are simulated in this module.

Usage
-----
Instantiate the generator with the desired seed and sample count; call
generate_dag(), then generate_data(), and use get_payload(), get_dataframe(),
or save() as required. The main() entry point runs a standalone example.
This module does not perform Bayesian-network structure learning.
"""

from __future__ import annotations

import math
import pickle
import random
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


CONTINUOUS_NODES = [
    "Age",
    "Total_Word_Recall",
    "Verbal_Fluency",
    "Walking_Speed",
    "Grip_Strength",
    "Chair_Stand_Time",
    "BMI",
]

MULTICLASS_NODES = [
    "Education_Level",
    "Self_Memory",
    "Hearing",
    "Eyesight",
    "Depression",
    "Loneliness",
    "Social_Demands",
    "Managing_Meds",
    "Grocery_Shopping",
    "Vigorous_Activity",
]

BINARY_NODES = ["Dementia_Diagnosis"]

REGIME_LABELS = {
    "sharp_multimodal": "Sharp multimodal",
    "subtle_multimodal": "Subtle multimodal",
    "unimodal": "Unimodal",
}


class StochasticContinuousMultimodalGenerator:


    def __init__(
        self,
        num_samples: int = 5000,
        seed: int = 42,
        edge_prob: float = 0.15,
        max_parents: int = 3,
        gate_calibration_size: int = 10000,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if not 0.0 <= edge_prob <= 1.0:
            raise ValueError("edge_prob must lie between 0 and 1.")
        if max_parents < 0:
            raise ValueError("max_parents cannot be negative.")
        if gate_calibration_size <= 0:
            raise ValueError("gate_calibration_size must be positive.")

        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.edge_prob = float(edge_prob)
        self.max_parents = int(max_parents)
        self.gate_calibration_size = int(gate_calibration_size)

        self.continuous_nodes = CONTINUOUS_NODES.copy()
        self.multiclass_nodes = MULTICLASS_NODES.copy()
        self.binary_nodes = BINARY_NODES.copy()
        self.discrete_nodes = self.multiclass_nodes + self.binary_nodes
        self.nodes = self.continuous_nodes + self.discrete_nodes

        self.graph: nx.DiGraph | None = None
        self.data: dict[str, np.ndarray] = {}
        self.node_regimes: dict[str, str] = {}
        self.true_parameters: dict[str, dict[str, Any]] = {}

    def generate_dag(self) -> nx.DiGraph:

        graph_rng = random.Random(self.seed)

        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)

        shuffled_nodes = self.nodes.copy()
        graph_rng.shuffle(shuffled_nodes)

        for index, parent in enumerate(shuffled_nodes):
            for child in shuffled_nodes[index + 1 :]:
                if (
                    graph_rng.random() < self.edge_prob
                    and graph.in_degree(child) < self.max_parents
                ):
                    graph.add_edge(parent, child)

        self.graph = graph
        return graph

    def _parameter_rng(self) -> np.random.Generator:

        return np.random.default_rng(np.random.SeedSequence([self.seed, 1]))

    def _node_rng(
        self,
        node: str,
        *,
        stage: int,
        purpose: int,
    ) -> np.random.Generator:


        node_index = self.nodes.index(node)
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, stage, node_index, purpose])
        )

    @staticmethod
    def _draw_mixture_parameters(
        parameter_rng: np.random.Generator,
        *,
        sharp: bool,
    ) -> dict[str, Any]:

        num_components = int(parameter_rng.integers(2, 4))

        if sharp:
            base_means = np.array([-3.5, 0.0, 3.5], dtype=float)
            stds = parameter_rng.uniform(0.25, 0.5, size=num_components)
        else:
            base_means = np.array([-2.0, 0.0, 2.0], dtype=float)
            stds = parameter_rng.uniform(0.6, 0.9, size=num_components)

        means = parameter_rng.choice(
            base_means,
            size=num_components,
            replace=False,
        )
        weights = parameter_rng.dirichlet(np.ones(num_components) * 4.0)

        return {
            "num_components": num_components,
            "means": means.astype(float).tolist(),
            "stds": stds.astype(float).tolist(),
            "weights": weights.astype(float).tolist(),
        }

    @staticmethod
    def _sample_gaussian_mixture(
        num_samples: int,
        parameters: dict[str, Any],
        component_rng: np.random.Generator,
        normal_rng: np.random.Generator,
    ) -> np.ndarray:

        means = np.asarray(parameters["means"], dtype=float)
        stds = np.asarray(parameters["stds"], dtype=float)
        weights = np.asarray(parameters["weights"], dtype=float)

        component_uniforms = component_rng.random(num_samples)
        component_indices = np.searchsorted(
            np.cumsum(weights),
            component_uniforms,
            side="right",
        )
        component_indices = np.minimum(component_indices, len(weights) - 1)
        standard_normals = normal_rng.standard_normal(num_samples)

        return (
            means[component_indices]
            + stds[component_indices] * standard_normals
        )

    def _build_parameters_and_calibration(self) -> None:


        if self.graph is None:
            raise RuntimeError("Call generate_dag() before generate_data().")

        parameter_rng = self._parameter_rng()
        calibration_data: dict[str, np.ndarray] = {}
        topological_order = list(nx.topological_sort(self.graph))

        regime_names = np.array(
            ["sharp_multimodal", "subtle_multimodal", "unimodal"],
            dtype=object,
        )

        self.node_regimes = {
            node: str(
                parameter_rng.choice(
                    regime_names,
                    p=[0.55, 0.30, 0.15],
                )
            )
            for node in topological_order
        }
        self.true_parameters = {}

        for node in topological_order:
            parents = list(self.graph.predecessors(node))
            regime = self.node_regimes[node]

            if not parents:
                if regime in {"sharp_multimodal", "subtle_multimodal"}:
                    mixture_parameters = self._draw_mixture_parameters(
                        parameter_rng,
                        sharp=(regime == "sharp_multimodal"),
                    )
                    parameters = {
                        "family": "gaussian_mixture_root",
                        "regime": regime,
                        **mixture_parameters,
                    }
                    calibration_data[node] = self._sample_gaussian_mixture(
                        self.gate_calibration_size,
                        parameters,
                        self._node_rng(node, stage=2, purpose=0),
                        self._node_rng(node, stage=2, purpose=1),
                    )
                else:
                    mean = float(parameter_rng.uniform(-2.0, 2.0))
                    std = float(parameter_rng.uniform(0.6, 1.2))
                    parameters = {
                        "family": "gaussian_root",
                        "regime": regime,
                        "mean": mean,
                        "std": std,
                    }
                    calibration_data[node] = self._node_rng(
                        node,
                        stage=2,
                        purpose=1,
                    ).normal(
                        loc=mean,
                        scale=std,
                        size=self.gate_calibration_size,
                    )

                self.true_parameters[node] = parameters
                continue

            intercept = float(parameter_rng.uniform(-1.0, 1.0))

            if regime in {"sharp_multimodal", "subtle_multimodal"}:
                conditional_mean = np.full(
                    self.gate_calibration_size,
                    intercept,
                    dtype=float,
                )
                gates: dict[str, dict[str, float]] = {}

                shift_range = (
                    (2.2, 3.2)
                    if regime == "sharp_multimodal"
                    else (0.9, 1.6)
                )

                for parent in parents:
                    threshold = float(np.median(calibration_data[parent]))
                    shift = float(parameter_rng.uniform(*shift_range))
                    gates[parent] = {
                        "threshold": threshold,
                        "shift": shift,
                    }
                    conditional_mean += np.where(
                        calibration_data[parent] > threshold,
                        shift,
                        -shift,
                    )

                noise_std = 0.35 if regime == "sharp_multimodal" else 0.50
                residuals = self._node_rng(
                    node,
                    stage=2,
                    purpose=1,
                ).normal(
                    loc=0.0,
                    scale=noise_std,
                    size=self.gate_calibration_size,
                )
                calibration_data[node] = conditional_mean + residuals
                self.true_parameters[node] = {
                    "family": "gated_gaussian",
                    "regime": regime,
                    "parents": parents.copy(),
                    "intercept": intercept,
                    "gates": gates,
                    "noise_std": float(noise_std),
                }
                continue

            coefficients: dict[str, float] = {}
            conditional_mean = np.full(
                self.gate_calibration_size,
                intercept,
                dtype=float,
            )

            for parent in parents:
                direction = float(parameter_rng.choice([-1.0, 1.0]))
                coefficient = direction * float(parameter_rng.uniform(0.4, 0.8))
                coefficients[parent] = coefficient
                conditional_mean += coefficient * calibration_data[parent]

            noise_std = 0.70
            residuals = self._node_rng(
                node,
                stage=2,
                purpose=1,
            ).normal(
                loc=0.0,
                scale=noise_std,
                size=self.gate_calibration_size,
            )
            calibration_data[node] = conditional_mean + residuals
            self.true_parameters[node] = {
                "family": "linear_gaussian",
                "regime": regime,
                "parents": parents.copy(),
                "intercept": intercept,
                "coefficients": coefficients,
                "noise_std": float(noise_std),
            }

    def generate_data(self) -> pd.DataFrame:

        if self.graph is None:
            raise RuntimeError("Call generate_dag() before generate_data().")

        self.data = {}
        self._build_parameters_and_calibration()
        topological_order = list(nx.topological_sort(self.graph))

        for node in topological_order:
            parameters = self.true_parameters[node]
            family = parameters["family"]

            if family == "gaussian_mixture_root":
                self.data[node] = self._sample_gaussian_mixture(
                    self.num_samples,
                    parameters,
                    self._node_rng(node, stage=3, purpose=0),
                    self._node_rng(node, stage=3, purpose=1),
                )
                continue

            if family == "gaussian_root":
                self.data[node] = self._node_rng(
                    node,
                    stage=3,
                    purpose=1,
                ).normal(
                    loc=parameters["mean"],
                    scale=parameters["std"],
                    size=self.num_samples,
                )
                continue

            conditional_mean = np.full(
                self.num_samples,
                parameters["intercept"],
                dtype=float,
            )

            if family == "gated_gaussian":
                for parent, gate in parameters["gates"].items():
                    conditional_mean += np.where(
                        self.data[parent] > gate["threshold"],
                        gate["shift"],
                        -gate["shift"],
                    )
            elif family == "linear_gaussian":
                for parent, coefficient in parameters["coefficients"].items():
                    conditional_mean += coefficient * self.data[parent]
            else:
                raise ValueError(f"Unknown ground-truth family: {family}")

            residuals = self._node_rng(
                node,
                stage=3,
                purpose=1,
            ).normal(
                loc=0.0,
                scale=parameters["noise_std"],
                size=self.num_samples,
            )
            self.data[node] = conditional_mean + residuals

        return self.get_dataframe()

    def get_dataframe(self, num_rows: int | None = None) -> pd.DataFrame:

        if not self.data:
            raise RuntimeError("Call generate_data() before get_dataframe().")

        dataframe = pd.DataFrame(self.data, columns=self.nodes)

        if num_rows is None:
            return dataframe
        if not 0 < num_rows <= self.num_samples:
            raise ValueError(
                f"num_rows must be between 1 and {self.num_samples}."
            )
        return dataframe.iloc[:num_rows].copy()

    def get_payload(self) -> dict[str, Any]:

        if self.graph is None or not self.data:
            raise RuntimeError("Generate the DAG and data before building the payload.")

        return {
            "dataframe": self.get_dataframe(),
            "graph": self.graph.copy(),
            "nodes": self.nodes.copy(),
            "discrete_nodes": self.discrete_nodes.copy(),
            "continuous_nodes": self.continuous_nodes.copy(),
            "multiclass_nodes": self.multiclass_nodes.copy(),
            "binary_nodes": self.binary_nodes.copy(),
            "node_regimes": self.node_regimes.copy(),
            "true_parameters": self.true_parameters.copy(),
            "generator": "multimodal",
            "seed": self.seed,
            "num_samples": self.num_samples,
            "gate_calibration_size": self.gate_calibration_size,
        }

    def save(self, filepath: str | Path = "Data/GTmultimodal.pkl") -> Path:

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self.get_payload(), file)
        return output_path.resolve()


def visualize_ground_truth(
    generator: StochasticContinuousMultimodalGenerator,
) -> None:

    if generator.graph is None:
        raise RuntimeError("Generate the DAG before visualizing it.")

    graph = generator.graph
    layer_by_node: dict[str, int] = {}

    for node in nx.topological_sort(graph):
        parents = list(graph.predecessors(node))
        layer_by_node[node] = (
            0 if not parents else max(layer_by_node[parent] for parent in parents) + 1
        )

    nodes_by_layer: dict[int, list[str]] = {}
    for node, layer in layer_by_node.items():
        nodes_by_layer.setdefault(layer, []).append(node)

    positions: dict[str, tuple[float, float]] = {}
    for layer, layer_nodes in nodes_by_layer.items():
        width = len(layer_nodes)
        for index, node in enumerate(layer_nodes):
            positions[node] = ((index - width / 2) + 0.5, -layer)

    node_colors = [
        "#87CEFA" if node in generator.discrete_nodes else "#F08080"
        for node in graph.nodes
    ]

    plt.figure(figsize=(16, 12))
    plt.title(
        "Purely Continuous Ground Truth Network (Multimodal)",
        fontsize=18,
        fontweight="bold",
    )
    nx.draw(
        graph,
        positions,
        with_labels=True,
        node_color=node_colors,
        node_size=3500,
        font_size=8,
        font_weight="bold",
        edge_color="dimgray",
        arrows=True,
        arrowsize=20,
        width=1.5,
    )

    discrete_patch = mpatches.Patch(
        color="#87CEFA", label="Target discrete in hybrid"
    )
    continuous_patch = mpatches.Patch(
        color="#F08080", label="Target continuous in hybrid"
    )
    plt.legend(
        handles=[discrete_patch, continuous_patch],
        loc="upper right",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def verify_continuous_distributions(
    dataframe: pd.DataFrame,
    nodes: list[str],
    generator: StochasticContinuousMultimodalGenerator,
    title_prefix: str = "Multimodal ground truth",
) -> None:

    num_columns = 3
    num_rows = math.ceil(len(nodes) / num_columns)
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(16, 4 * num_rows),
    )
    axes = np.asarray(axes).reshape(-1)

    for index, node in enumerate(nodes):
        axis = axes[index]
        sns.histplot(
            dataframe[node],
            kde=True,
            ax=axis,
            bins=40,
            stat="density",
        )
        regime_label = REGIME_LABELS.get(
            generator.node_regimes.get(node, ""),
            "Unknown",
        )
        axis.set_title(
            f"{node} [{regime_label}]",
            fontweight="bold",
            fontsize=11,
        )
        axis.set_xlabel("Value")
        axis.set_ylabel("Density")
        axis.grid(axis="y", alpha=0.3)

    for index in range(len(nodes), len(axes)):
        axes[index].set_visible(False)

    figure.suptitle(
        f"{title_prefix} — continuous marginal distributions",
        fontsize=18,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


def main() -> None:
    generator = StochasticContinuousMultimodalGenerator(
        num_samples=5000,
        seed=42,
    )
    graph = generator.generate_dag()
    dataframe = generator.generate_data()
    output_path = generator.save("Data/GTmultimodal.pkl")

    print("Generated purely continuous multimodal ground truth.")
    print(f"Data shape: {dataframe.shape}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Saved to: {output_path}")

    visualize_ground_truth(generator)
    verify_continuous_distributions(
        dataframe,
        generator.nodes,
        generator,
    )


if __name__ == "__main__":
    main()

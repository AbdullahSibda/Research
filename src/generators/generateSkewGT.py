"""Continuous-Generator Benchmark: Skew-Normal Ground Truth.

Purpose
-------
Generate reproducible continuous networks with asymmetric distributions.
Root variables are skew-normal, and non-root variables combine linear
conditional locations with skew-normal residuals. Skew-normal draws use
paired standard-normal variates so a fixed seed supports nested samples.

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


class StochasticContinuousSkewedGenerator:


    def __init__(
        self,
        num_samples: int = 5000,
        seed: int = 42,
        edge_prob: float = 0.15,
        max_parents: int = 3,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if not 0.0 <= edge_prob <= 1.0:
            raise ValueError("edge_prob must lie between 0 and 1.")
        if max_parents < 0:
            raise ValueError("max_parents cannot be negative.")

        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.edge_prob = float(edge_prob)
        self.max_parents = int(max_parents)

        self.continuous_nodes = CONTINUOUS_NODES.copy()
        self.multiclass_nodes = MULTICLASS_NODES.copy()
        self.binary_nodes = BINARY_NODES.copy()
        self.discrete_nodes = self.multiclass_nodes + self.binary_nodes
        self.nodes = self.continuous_nodes + self.discrete_nodes

        self.graph: nx.DiGraph | None = None
        self.data: dict[str, np.ndarray] = {}
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

    def _observation_rng(self, node: str) -> np.random.Generator:


        node_index = self.nodes.index(node)
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, 2, node_index])
        )

    @staticmethod
    def _draw_skew_parameters(
        parameter_rng: np.random.Generator,
        *,
        is_root: bool,
    ) -> dict[str, float]:

        direction = float(parameter_rng.choice([-1.0, 1.0]))
        shape = direction * float(parameter_rng.uniform(8.0, 25.0))
        loc = float(parameter_rng.uniform(-2.0, 2.0))
        scale_range = (0.8, 2.2) if is_root else (0.4, 1.2)
        scale = float(parameter_rng.uniform(*scale_range))

        return {
            "shape": shape,
            "loc": loc,
            "scale": scale,
        }

    @staticmethod
    def _sample_skew_normal(
        num_samples: int,
        parameters: dict[str, float],
        observation_rng: np.random.Generator,
    ) -> np.ndarray:

        shape = parameters["shape"]
        delta = shape / math.sqrt(1.0 + shape**2)


        normal_pairs = observation_rng.standard_normal((num_samples, 2))
        standard_skew = (
            delta * np.abs(normal_pairs[:, 0])
            + math.sqrt(1.0 - delta**2) * normal_pairs[:, 1]
        )
        return parameters["loc"] + parameters["scale"] * standard_skew

    def generate_data(self) -> pd.DataFrame:

        if self.graph is None:
            raise RuntimeError("Call generate_dag() before generate_data().")

        self.data = {}
        self.true_parameters = {}

        parameter_rng = self._parameter_rng()
        topological_order = list(nx.topological_sort(self.graph))

        for node in topological_order:
            parents = list(self.graph.predecessors(node))
            observation_rng = self._observation_rng(node)

            if not parents:
                root_parameters = self._draw_skew_parameters(
                    parameter_rng,
                    is_root=True,
                )
                self.data[node] = self._sample_skew_normal(
                    self.num_samples,
                    root_parameters,
                    observation_rng,
                )
                self.true_parameters[node] = {
                    "family": "skew_normal_root",
                    **root_parameters,
                }
                continue

            intercept = float(parameter_rng.uniform(-1.0, 1.0))
            coefficients: dict[str, float] = {}
            conditional_location = np.full(
                self.num_samples,
                intercept,
                dtype=float,
            )

            for parent in parents:
                direction = float(parameter_rng.choice([-1.0, 1.0]))
                coefficient = direction * float(parameter_rng.uniform(0.3, 0.8))
                coefficients[parent] = coefficient
                conditional_location += coefficient * self.data[parent]

            residual_parameters = self._draw_skew_parameters(
                parameter_rng,
                is_root=False,
            )
            residuals = self._sample_skew_normal(
                self.num_samples,
                residual_parameters,
                observation_rng,
            )
            self.data[node] = conditional_location + residuals
            self.true_parameters[node] = {
                "family": "skew_normal_regression",
                "parents": parents.copy(),
                "intercept": intercept,
                "coefficients": coefficients,
                "residual_shape": residual_parameters["shape"],
                "residual_loc": residual_parameters["loc"],
                "residual_scale": residual_parameters["scale"],
            }

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
            "true_parameters": self.true_parameters.copy(),
            "generator": "skew",
            "seed": self.seed,
            "num_samples": self.num_samples,
        }

    def save(self, filepath: str | Path = "Data/GTskew.pkl") -> Path:

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self.get_payload(), file)
        return output_path.resolve()


def visualize_ground_truth(generator: StochasticContinuousSkewedGenerator) -> None:

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
        "Purely Continuous Ground Truth Network (Skewed Unimodal)",
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
    title_prefix: str = "Skewed ground truth",
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
        axis.set_title(
            f"{node} (skew={dataframe[node].skew():.2f})",
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
    generator = StochasticContinuousSkewedGenerator(num_samples=5000, seed=42)
    graph = generator.generate_dag()
    dataframe = generator.generate_data()
    output_path = generator.save("Data/GTskew.pkl")

    print("Generated purely continuous skewed ground truth.")
    print(f"Data shape: {dataframe.shape}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Saved to: {output_path}")

    visualize_ground_truth(generator)
    verify_continuous_distributions(dataframe, generator.nodes)


if __name__ == "__main__":
    main()

"""Native mixed-variable ground-truth Bayesian network generator.

Purpose
-------
Defines a reproducible 18-node Bayesian network with seven continuous variables,
ten three-category variables, and one binary dementia indicator. A seeded random
DAG is generated under configurable edge-probability and maximum-parent limits.
Conditional distributions are parameterized in topological order, and samples
are drawn using node-specific deterministic random streams.

Distribution Families
---------------------
Continuous children use Gaussian or skew-normal conditional regressions with
continuous-parent slopes and discrete-parent state offsets. The multimodal
settings add parent-dependent threshold gates; root nodes may use two- or
three-component Gaussian or skew-normal mixtures. Discrete children are drawn
from categorical softmax probabilities using categorical-parent effects and
bounded continuous-parent features. In multimodal modes, additional quadratic
and threshold effects create nonlinear continuous-to-discrete dependencies.

Public Functionality
--------------------
NativeMixedGenerator.generate_dag() constructs the DAG and samples its local
parameters. generate_data() and sample() return reproducible synthetic samples.
get_payload() exports the true graph, node types, cardinalities, and parameters.
local_log_prob() and joint_log_prob() evaluate exact native ground-truth log
probabilities or densities. discrete_probabilities(),
continuous_interval_probability(), and sample_continuous_truncated() support
common-resolution particle-based evaluation without changing the native types.

The four generateMixed*GT.py wrappers select the distribution regime through
the MODE class attribute. No external participant data is used.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import norm, skewnorm


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
DISCRETE_NODES = MULTICLASS_NODES + BINARY_NODES
ALL_NODES = CONTINUOUS_NODES + DISCRETE_NODES

CARDINALITIES = {
    **{node: 3 for node in MULTICLASS_NODES},
    **{node: 2 for node in BINARY_NODES},
}

_EPS = 1e-12


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _safe_log(x: np.ndarray | float) -> np.ndarray | float:
    return np.log(np.maximum(x, _EPS))


def _stable_code(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


@dataclass(frozen=True)
class ContinuousParams:
    kind: str
    parents: tuple[str, ...]
    intercept: float
    c_coefficients: Mapping[str, float]
    d_offsets: Mapping[str, tuple[float, ...]]
    scale: float
    skew_shape: float
    gate_parent: str | None
    gate_threshold: float
    gate_shift: float
    root_weights: tuple[float, ...]
    root_locations: tuple[float, ...]
    root_scales: tuple[float, ...]
    root_shapes: tuple[float, ...]


@dataclass(frozen=True)
class DiscreteParams:
    parents: tuple[str, ...]
    cardinality: int
    intercept_logits: tuple[float, ...]
    c_linear: Mapping[str, tuple[float, ...]]
    c_quadratic: Mapping[str, tuple[float, ...]]
    c_threshold: Mapping[str, tuple[float, ...]]
    d_effects: Mapping[str, tuple[tuple[float, ...], ...]]
    thresholds: Mapping[str, float]


class NativeMixedGenerator:
    MODE: str = "unimodal"

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
            raise ValueError("edge_prob must lie in [0,1].")
        if max_parents < 0:
            raise ValueError("max_parents cannot be negative.")

        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.edge_prob = float(edge_prob)
        self.max_parents = int(max_parents)

        self.continuous_nodes = CONTINUOUS_NODES.copy()
        self.multiclass_nodes = MULTICLASS_NODES.copy()
        self.binary_nodes = BINARY_NODES.copy()
        self.discrete_nodes = DISCRETE_NODES.copy()
        self.nodes = ALL_NODES.copy()
        self.cardinalities = dict(CARDINALITIES)

        self.graph: nx.DiGraph | None = None
        self.true_parameters: dict[str, ContinuousParams | DiscreteParams] = {}
        self.data: dict[str, np.ndarray] = {}





    def generate_dag(self) -> nx.DiGraph:
        rng = random.Random(self.seed)
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        order = self.nodes.copy()
        rng.shuffle(order)

        for i, parent in enumerate(order):
            for child in order[i + 1 :]:
                if (
                    rng.random() < self.edge_prob
                    and graph.in_degree(child) < self.max_parents
                ):
                    graph.add_edge(parent, child)

        self.graph = graph
        self._draw_parameters()
        return graph

    def _parameter_rng(self) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, 71001]))

    def _node_rng(
        self, node: str, *, sample_seed: int, purpose: int
    ) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, int(sample_seed), _stable_code(node), int(purpose)]
            )
        )

    def _draw_parameters(self) -> None:
        if self.graph is None:
            raise RuntimeError("Call generate_dag() first.")

        rng = self._parameter_rng()
        self.true_parameters = {}

        for child in nx.topological_sort(self.graph):
            parents = tuple(str(p) for p in self.graph.predecessors(child))
            if child in self.discrete_nodes:
                self.true_parameters[child] = self._draw_discrete_params(
                    child, parents, rng
                )
            else:
                self.true_parameters[child] = self._draw_continuous_params(
                    child, parents, rng
                )

    def _draw_discrete_params(
        self,
        child: str,
        parents: tuple[str, ...],
        rng: np.random.Generator,
    ) -> DiscreteParams:
        r = self.cardinalities[child]

        intercept = rng.normal(0.0, 0.45, size=r)
        intercept -= intercept.mean()

        c_linear: dict[str, tuple[float, ...]] = {}
        c_quad: dict[str, tuple[float, ...]] = {}
        c_thresh: dict[str, tuple[float, ...]] = {}
        thresholds: dict[str, float] = {}
        d_effects: dict[str, tuple[tuple[float, ...], ...]] = {}

        for parent in parents:
            if parent in self.continuous_nodes:
                w = rng.normal(0.0, 0.85, size=r)
                w -= w.mean()
                c_linear[parent] = tuple(map(float, w))

                if self.MODE in {"multimodal", "skew_multi"}:
                    q = rng.normal(0.0, 0.45, size=r)
                    q -= q.mean()
                    t = rng.normal(0.0, 0.70, size=r)
                    t -= t.mean()
                    c_quad[parent] = tuple(map(float, q))
                    c_thresh[parent] = tuple(map(float, t))
                    thresholds[parent] = float(rng.normal(0.0, 0.55))
                else:
                    c_quad[parent] = tuple([0.0] * r)
                    c_thresh[parent] = tuple([0.0] * r)
                    thresholds[parent] = 0.0
            else:
                k = self.cardinalities[parent]
                table = rng.normal(0.0, 0.65, size=(k, r))
                table -= table.mean(axis=1, keepdims=True)
                d_effects[parent] = tuple(
                    tuple(map(float, row)) for row in table
                )

        return DiscreteParams(
            parents=parents,
            cardinality=r,
            intercept_logits=tuple(map(float, intercept)),
            c_linear=c_linear,
            c_quadratic=c_quad,
            c_threshold=c_thresh,
            d_effects=d_effects,
            thresholds=thresholds,
        )

    def _draw_continuous_params(
        self,
        child: str,
        parents: tuple[str, ...],
        rng: np.random.Generator,
    ) -> ContinuousParams:
        c_parents = [p for p in parents if p in self.continuous_nodes]
        d_parents = [p for p in parents if p in self.discrete_nodes]

        intercept = float(rng.normal(0.0, 0.8))
        c_coefficients = {
            p: float(rng.uniform(0.20, 0.65) * rng.choice([-1.0, 1.0]))
            for p in c_parents
        }
        d_offsets = {}
        for p in d_parents:
            k = self.cardinalities[p]
            offsets = rng.normal(0.0, 0.8, size=k)
            offsets -= offsets.mean()
            d_offsets[p] = tuple(map(float, offsets))

        scale = float(rng.uniform(0.55, 1.20))
        skew_shape = 0.0
        if self.MODE in {"skew", "skew_multi"}:
            skew_shape = float(rng.uniform(4.0, 10.0) * rng.choice([-1.0, 1.0]))

        gate_parent = None
        gate_threshold = 0.0
        gate_shift = 0.0

        if self.MODE in {"multimodal", "skew_multi"} and parents:
            gate_parent = str(rng.choice(list(parents)))
            gate_threshold = float(rng.normal(0.0, 0.5))
            gate_shift = float(rng.uniform(1.25, 2.75) * rng.choice([-1.0, 1.0]))



        if self.MODE in {"multimodal", "skew_multi"} and not parents:
            k = int(rng.choice([2, 3]))
            weights = rng.dirichlet(np.ones(k) * 2.0)
            base = np.sort(rng.normal(0.0, 1.0, size=k))
            gaps = np.linspace(-1.6, 1.6, k) * rng.uniform(0.8, 1.4)
            locations = base * 0.25 + gaps
            scales = rng.uniform(0.45, 0.85, size=k)
            if self.MODE == "skew_multi":
                shapes = rng.uniform(3.0, 8.0, size=k) * rng.choice(
                    [-1.0, 1.0], size=k
                )
            else:
                shapes = np.zeros(k)
            root_weights = tuple(map(float, weights))
            root_locations = tuple(map(float, locations))
            root_scales = tuple(map(float, scales))
            root_shapes = tuple(map(float, shapes))
            kind = "root_mixture"
        else:
            root_weights = (1.0,)
            root_locations = (0.0,)
            root_scales = (scale,)
            root_shapes = (skew_shape,)
            kind = "conditional"

        return ContinuousParams(
            kind=kind,
            parents=parents,
            intercept=intercept,
            c_coefficients=c_coefficients,
            d_offsets=d_offsets,
            scale=scale,
            skew_shape=skew_shape,
            gate_parent=gate_parent,
            gate_threshold=gate_threshold,
            gate_shift=gate_shift,
            root_weights=root_weights,
            root_locations=root_locations,
            root_scales=root_scales,
            root_shapes=root_shapes,
        )





    def _continuous_mean(
        self,
        child: str,
        parent_values: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        params = self.true_parameters[child]
        if not isinstance(params, ContinuousParams):
            raise TypeError(child)

        if parent_values:
            first = next(iter(parent_values.values()))
            n = len(np.asarray(first))
        else:
            n = 1
        mu = np.full(n, params.intercept, dtype=float)

        for p, beta in params.c_coefficients.items():
            mu += float(beta) * np.asarray(parent_values[p], dtype=float)

        for p, offsets in params.d_offsets.items():
            states = np.asarray(parent_values[p], dtype=np.int64)
            mu += np.asarray(offsets, dtype=float)[states]

        if params.gate_parent is not None:
            g = params.gate_parent
            values = np.asarray(parent_values[g], dtype=float)
            if g in self.discrete_nodes:
                gate = values >= max(1, self.cardinalities[g] // 2)
            else:

                gate = np.tanh(values / 2.0) > params.gate_threshold
            mu += params.gate_shift * gate.astype(float)

        return mu

    def discrete_probabilities(
        self,
        child: str,
        parent_values: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        params = self.true_parameters[child]
        if not isinstance(params, DiscreteParams):
            raise TypeError(child)

        if parent_values:
            n = len(np.asarray(next(iter(parent_values.values()))))
        else:
            n = 1

        logits = np.tile(
            np.asarray(params.intercept_logits, dtype=float),
            (n, 1),
        )

        for p in params.parents:
            if p in self.continuous_nodes:
                raw = np.asarray(parent_values[p], dtype=float)
                z = np.tanh(raw / 2.0)
                logits += z[:, None] * np.asarray(
                    params.c_linear[p], dtype=float
                )[None, :]

                if self.MODE in {"multimodal", "skew_multi"}:
                    logits += (z * z)[:, None] * np.asarray(
                        params.c_quadratic[p], dtype=float
                    )[None, :]
                    threshold = params.thresholds[p]
                    step = (z > threshold).astype(float)
                    logits += step[:, None] * np.asarray(
                        params.c_threshold[p], dtype=float
                    )[None, :]
            else:
                states = np.asarray(parent_values[p], dtype=np.int64)
                table = np.asarray(params.d_effects[p], dtype=float)
                logits += table[states]

        return _softmax(logits)

    def _parent_map(
        self,
        child: str,
        dataframe: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        params = self.true_parameters[child]
        parents = params.parents
        return {
            p: dataframe[p].to_numpy()
            for p in parents
        }





    def sample(
        self,
        num_samples: int,
        sample_seed: int = 0,
    ) -> pd.DataFrame:
        if self.graph is None:
            self.generate_dag()
        assert self.graph is not None

        n = int(num_samples)
        if n <= 0:
            raise ValueError("num_samples must be positive.")

        values: dict[str, np.ndarray] = {}
        for child in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(child))
            pmap = {p: values[p] for p in parents}

            if child in self.discrete_nodes:
                probs = self.discrete_probabilities(child, pmap)
                rng = self._node_rng(
                    child, sample_seed=sample_seed, purpose=1
                )
                u = rng.random(n)
                cdf = np.cumsum(probs, axis=1)
                state = np.sum(u[:, None] > cdf, axis=1)
                values[child] = state.astype(np.int64)
                continue

            params = self.true_parameters[child]
            assert isinstance(params, ContinuousParams)
            rng = self._node_rng(
                child, sample_seed=sample_seed, purpose=2
            )

            if params.kind == "root_mixture":
                comp_rng = self._node_rng(
                    child, sample_seed=sample_seed, purpose=3
                )
                components = comp_rng.choice(
                    len(params.root_weights),
                    size=n,
                    p=np.asarray(params.root_weights),
                )
                x = np.empty(n, dtype=float)
                for k in range(len(params.root_weights)):
                    mask = components == k
                    if not np.any(mask):
                        continue
                    loc = params.intercept + params.root_locations[k]
                    scale = params.root_scales[k]
                    shape = params.root_shapes[k]
                    if abs(shape) < 1e-12:
                        x[mask] = rng.normal(loc, scale, int(mask.sum()))
                    else:
                        x[mask] = skewnorm.rvs(
                            shape,
                            loc=loc,
                            scale=scale,
                            size=int(mask.sum()),
                            random_state=rng,
                        )
                values[child] = x
            else:
                mu = self._continuous_mean(child, pmap)
                if abs(params.skew_shape) < 1e-12:
                    values[child] = rng.normal(mu, params.scale, n)
                else:
                    values[child] = skewnorm.rvs(
                        params.skew_shape,
                        loc=mu,
                        scale=params.scale,
                        size=n,
                        random_state=rng,
                    )

        return pd.DataFrame(values, columns=self.nodes)

    def generate_data(self) -> dict[str, np.ndarray]:
        frame = self.sample(self.num_samples, sample_seed=100)
        self.data = {
            node: frame[node].to_numpy(copy=True)
            for node in self.nodes
        }
        return self.data





    def local_log_prob(
        self,
        child: str,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        params = self.true_parameters[child]
        pmap = self._parent_map(child, dataframe)

        if isinstance(params, DiscreteParams):
            probs = self.discrete_probabilities(child, pmap)
            y = dataframe[child].to_numpy(np.int64)
            if probs.shape[0] == 1 and len(dataframe) != 1:
                probs = np.repeat(probs, len(dataframe), axis=0)
            return _safe_log(probs[np.arange(len(dataframe)), y])

        y = dataframe[child].to_numpy(float)

        if params.kind == "root_mixture":
            terms = []
            for w, loc0, scale, shape in zip(
                params.root_weights,
                params.root_locations,
                params.root_scales,
                params.root_shapes,
            ):
                loc = params.intercept + loc0
                if abs(shape) < 1e-12:
                    lp = norm.logpdf(y, loc=loc, scale=scale)
                else:
                    lp = skewnorm.logpdf(
                        y, shape, loc=loc, scale=scale
                    )
                terms.append(math.log(w) + lp)
            return logsumexp(np.vstack(terms), axis=0)

        mu = self._continuous_mean(child, pmap)
        if abs(params.skew_shape) < 1e-12:
            return norm.logpdf(y, loc=mu, scale=params.scale)
        return skewnorm.logpdf(
            y, params.skew_shape, loc=mu, scale=params.scale
        )

    def joint_log_prob(
        self,
        dataframe: pd.DataFrame,
        *,
        return_nodewise: bool = False,
    ):
        if self.graph is None:
            raise RuntimeError("Ground-truth DAG is not initialised.")
        total = np.zeros(len(dataframe), dtype=float)
        nodewise = {}
        for child in nx.topological_sort(self.graph):
            lp = np.asarray(self.local_log_prob(child, dataframe), float)
            nodewise[child] = lp
            total += lp
        if return_nodewise:
            return total, nodewise
        return total





    def continuous_interval_probability(
        self,
        child: str,
        low: np.ndarray,
        high: np.ndarray,
        parent_values: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        params = self.true_parameters[child]
        if not isinstance(params, ContinuousParams):
            raise TypeError(child)

        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)

        if params.kind == "root_mixture":
            out = np.zeros_like(low, dtype=float)
            for w, loc0, scale, shape in zip(
                params.root_weights,
                params.root_locations,
                params.root_scales,
                params.root_shapes,
            ):
                loc = params.intercept + loc0
                if abs(shape) < 1e-12:
                    mass = norm.cdf(
                        (high - loc) / scale
                    ) - norm.cdf((low - loc) / scale)
                else:
                    mass = skewnorm.cdf(
                        high, shape, loc=loc, scale=scale
                    ) - skewnorm.cdf(
                        low, shape, loc=loc, scale=scale
                    )
                out += w * mass
            return np.maximum(out, _EPS)

        mu = self._continuous_mean(child, parent_values)
        if abs(params.skew_shape) < 1e-12:
            mass = norm.cdf(
                (high - mu) / params.scale
            ) - norm.cdf((low - mu) / params.scale)
        else:
            mass = skewnorm.cdf(
                high,
                params.skew_shape,
                loc=mu,
                scale=params.scale,
            ) - skewnorm.cdf(
                low,
                params.skew_shape,
                loc=mu,
                scale=params.scale,
            )
        return np.maximum(mass, _EPS)

    def sample_continuous_truncated(
        self,
        child: str,
        low: np.ndarray,
        high: np.ndarray,
        parent_values: Mapping[str, np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray:
        params = self.true_parameters[child]
        if not isinstance(params, ContinuousParams):
            raise TypeError(child)

        low = np.asarray(low, float)
        high = np.asarray(high, float)
        n = len(low)

        if params.kind == "root_mixture":
            masses = np.empty((n, len(params.root_weights)), dtype=float)
            for k, (w, loc0, scale, shape) in enumerate(zip(
                params.root_weights,
                params.root_locations,
                params.root_scales,
                params.root_shapes,
            )):
                loc = params.intercept + loc0
                if abs(shape) < 1e-12:
                    m = norm.cdf((high-loc)/scale) - norm.cdf((low-loc)/scale)
                else:
                    m = (
                        skewnorm.cdf(high, shape, loc=loc, scale=scale)
                        - skewnorm.cdf(low, shape, loc=loc, scale=scale)
                    )
                masses[:, k] = w * np.maximum(m, _EPS)

            probs = masses / masses.sum(axis=1, keepdims=True)
            u = rng.random(n)
            comp = np.sum(u[:, None] > np.cumsum(probs, axis=1), axis=1)
            out = np.empty(n, float)

            for k, (loc0, scale, shape) in enumerate(zip(
                params.root_locations,
                params.root_scales,
                params.root_shapes,
            )):
                mask = comp == k
                if not np.any(mask):
                    continue
                loc = params.intercept + loc0
                a = low[mask]
                b = high[mask]
                u2 = rng.random(int(mask.sum()))
                if abs(shape) < 1e-12:
                    fa = norm.cdf((a-loc)/scale)
                    fb = norm.cdf((b-loc)/scale)
                    p = np.clip(fa + u2 * (fb-fa), 1e-12, 1-1e-12)
                    out[mask] = loc + scale * norm.ppf(p)
                else:
                    fa = skewnorm.cdf(a, shape, loc=loc, scale=scale)
                    fb = skewnorm.cdf(b, shape, loc=loc, scale=scale)
                    p = np.clip(fa + u2 * (fb-fa), 1e-12, 1-1e-12)
                    out[mask] = skewnorm.ppf(
                        p, shape, loc=loc, scale=scale
                    )
            return out

        mu = self._continuous_mean(child, parent_values)
        u = rng.random(n)

        if abs(params.skew_shape) < 1e-12:
            fa = norm.cdf((low-mu)/params.scale)
            fb = norm.cdf((high-mu)/params.scale)
            p = np.clip(fa + u * (fb-fa), 1e-12, 1-1e-12)
            return mu + params.scale * norm.ppf(p)

        fa = skewnorm.cdf(
            low, params.skew_shape, loc=mu, scale=params.scale
        )
        fb = skewnorm.cdf(
            high, params.skew_shape, loc=mu, scale=params.scale
        )
        p = np.clip(fa + u * (fb-fa), 1e-12, 1-1e-12)
        return skewnorm.ppf(
            p, params.skew_shape, loc=mu, scale=params.scale
        )





    def get_payload(self) -> dict[str, Any]:
        if self.graph is None:
            raise RuntimeError("Call generate_dag() first.")

        serial_params = {}
        for node, params in self.true_parameters.items():
            if isinstance(params, ContinuousParams):
                serial_params[node] = {
                    "type": "continuous",
                    **params.__dict__,
                }
            else:
                serial_params[node] = {
                    "type": "discrete",
                    **params.__dict__,
                }

        return {
            "graph": self.graph.copy(),
            "nodes": self.nodes.copy(),
            "continuous_nodes": self.continuous_nodes.copy(),
            "discrete_nodes": self.discrete_nodes.copy(),
            "multiclass_nodes": self.multiclass_nodes.copy(),
            "binary_nodes": self.binary_nodes.copy(),
            "cardinalities": dict(self.cardinalities),
            "true_parameters": serial_params,
            "generator_mode": self.MODE,
            "native_mixed_ground_truth": True,
        }

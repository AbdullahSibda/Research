"""RQ1 native mixed-variable benchmark.

This single file contains the benchmark runner and its supporting statistical
evaluators as readable, namespaced Python functions and classes. It uses
the separate engines in src/hybrid_bn and generators in src/generators.
Statistical routines originate from the frozen research implementations.
The experiment retains its original checkpoints, seeds and evaluation metrics.

The internal _rq1_* namespaces avoid collisions between formerly separate
evaluation modules. No compressed or encoded source is embedded.
"""
from __future__ import annotations
import sys as _rq1_sys
import types as _rq1_types
from pathlib import Path as _rq1_Path
import importlib.util as _rq1_importutil

_rq1_inline_modules = {}
def _rq1_find_root():
    here = _rq1_Path(__file__).resolve().parent
    for root in [here, *here.parents]:
        if (root / "src" / "hybrid_bn" / "SLhybrid.py").is_file() and (root / "src" / "generators").is_dir():
            return root
    raise FileNotFoundError("Repository root with src/hybrid_bn and src/generators not found")

def _rq1_inline_load(path, name=None):
    path = _rq1_Path(path)
    if path.name in _rq1_inline_modules:
        return _rq1_inline_modules[path.name]
    full = path.resolve()
    if str(full.parent) not in _rq1_sys.path:
        _rq1_sys.path.insert(0, str(full.parent))
    spec = _rq1_importutil.spec_from_file_location(name or full.stem, str(full))
    if spec is None or spec.loader is None:
        raise ImportError(full)
    mod = _rq1_importutil.module_from_spec(spec)
    _rq1_sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# Evaluation implementation: 07_v2_hybrid_v3_representation_correct_kl.py
# ======================================================================
import argparse as _rq1_s0_argparse
import gc as _rq1_s0_gc
import hashlib as _rq1_s0_hashlib
import importlib.util as _rq1_s0_importlib
import json as _rq1_s0_json
import math as _rq1_s0_math
import os as _rq1_s0_os
import sys as _rq1_s0_sys
import time as _rq1_s0_time
import traceback as _rq1_s0_traceback
import warnings as _rq1_s0_warnings
from dataclasses import asdict as _rq1_s0_asdict, dataclass as _rq1_s0_dataclass
from datetime import datetime as _rq1_s0_datetime, timezone as _rq1_s0_timezone
from pathlib import Path as _rq1_s0_Path
from typing import Any as _rq1_s0_Any, Iterable as _rq1_s0_Iterable, Mapping as _rq1_s0_Mapping, Sequence as _rq1_s0_Sequence
_rq1_s0_sys.dont_write_bytecode = True
import networkx as _rq1_s0_nx
import numpy as _rq1_s0_np
import pandas as _rq1_s0_pd
from scipy.linalg import helmert as _rq1_s0_helmert
from scipy.optimize import minimize as _rq1_s0_minimize
from scipy.special import gammaln as _rq1_s0_gammaln, logsumexp as _rq1_s0_logsumexp, owens_t as _rq1_s0_owens_t
from scipy.stats import norm as _rq1_s0_norm, skewnorm as _rq1_s0_skewnorm
from sklearn.exceptions import ConvergenceWarning as _rq1_s0_ConvergenceWarning
from sklearn.linear_model import LogisticRegression as _rq1_s0_LogisticRegression
_rq1_s0__MIN_PROBABILITY = 1e-300
_rq1_s0__MIN_STANDARD_DEVIATION = 1e-06
_rq1_s0_SUPPORTED_TRUE_FAMILIES = {'gaussian_root', 'linear_gaussian', 'gaussian_mixture_root', 'gated_gaussian', 'skew_normal_root', 'skew_normal_regression', 'skew_mixture_root', 'gated_skew_mixture'}
_rq1_s0_PARADIGMS = ('discrete', 'continuous', 'hybrid')
_rq1_s0__V2_Q_MIXTURE_MAX_ITER = 1000
_rq1_s0__V2_Q_MIXTURE_RESTARTS = 3
_rq1_s0__V2_Q_MIXTURE_MIN_EFFECTIVE_COUNT = 5.0
_rq1_s0_KL_SCHEMA_VERSION = 'hybrid-v3-representation-correct-kl-v2'
_rq1_s0_GENERATOR_SPECS = {'unimodal': ('generateUnimodalGT.py', 'StochasticContinuousGenerator'), 'skew': ('generateSkewGT.py', 'StochasticContinuousSkewedGenerator'), 'multimodal': ('generateMultimodalGT.py', 'StochasticContinuousMultimodalGenerator'), 'skew_multi': ('generateSkewMultiGT.py', 'StochasticContinuousSkewedMultimodalGenerator')}

def _rq1_s0_json_safe(value: Any) -> Any:
    if isinstance(value, _rq1_s0_Mapping):
        return {str(k): _rq1_s0_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_rq1_s0_json_safe(v) for v in value]
    if isinstance(value, _rq1_s0_np.ndarray):
        return _rq1_s0_json_safe(value.tolist())
    if isinstance(value, _rq1_s0_np.generic):
        return _rq1_s0_json_safe(value.item())
    if isinstance(value, _rq1_s0_Path):
        return str(value)
    if isinstance(value, float):
        if _rq1_s0_np.isnan(value):
            return 'NaN'
        if _rq1_s0_np.isposinf(value):
            return 'Infinity'
        if _rq1_s0_np.isneginf(value):
            return '-Infinity'
    return value

def _rq1_s0_decode_float(value: Any) -> float:
    if value == 'Infinity':
        return _rq1_s0_np.inf
    if value == '-Infinity':
        return -_rq1_s0_np.inf
    if value == 'NaN':
        return _rq1_s0_np.nan
    return float(value)

def _rq1_s0_atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(_rq1_s0_json.dumps(_rq1_s0_json_safe(payload), indent=2), encoding='utf-8')
    _rq1_s0_os.replace(tmp, path)

def _rq1_s0_discover_root(here: _rq1_s0_Path, explicit: _rq1_s0_Path | None) -> _rq1_s0_Path:
    return _rq1_find_root()

def _rq1_s0_import_generator(root: Path, generator: str):
    filename, class_name = _rq1_s0_GENERATOR_SPECS[generator]
    module = _rq1_inline_load(root / 'src' / 'generators' / filename, f'_hv3_kl_generator_{generator}_{_rq1_s0_os.getpid()}')
    if not hasattr(module, class_name):
        raise AttributeError(f'{filename} does not expose {class_name}.')
    return getattr(module, class_name)

def _rq1_s0_structure_dataframe_sha256(frame: _rq1_s0_pd.DataFrame) -> str:
    h = _rq1_s0_hashlib.sha256()
    h.update('|'.join(map(str, frame.columns)).encode('utf-8'))
    hashed = _rq1_s0_pd.util.hash_pandas_object(frame, index=True).to_numpy(_rq1_s0_np.uint64)
    h.update(_rq1_s0_np.ascontiguousarray(hashed).tobytes())
    return h.hexdigest()

def _rq1_s0_deterministic_mc_seed(generator: str, seed: int) -> int:
    message = f'fresh-supervisor-kl-v1|{generator}|{seed}'.encode('utf-8')
    digest = _rq1_s0_hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], 'little') % (2 ** 32 - 1)

def _rq1_s0_build_graph(nodes: Sequence[str], edges: Iterable[Sequence[str]]) -> _rq1_s0_nx.DiGraph:
    g = _rq1_s0_nx.DiGraph()
    g.add_nodes_from(map(str, nodes))
    g.add_edges_from(((str(e[0]), str(e[1])) for e in edges))
    if not _rq1_s0_nx.is_directed_acyclic_graph(g):
        raise ValueError('Graph is not a DAG.')
    return g

def _rq1_s0_default_cardinalities(payload: Mapping[str, Any]) -> dict[str, int]:
    discrete = list(map(str, payload['discrete_nodes']))
    binary = set(map(str, payload.get('binary_nodes', [])))
    return {node: 2 if node in binary else 3 for node in discrete}

def _rq1_s0__configuration_storage_key(value: Any) -> str:
    if value is None:
        return 'none'
    if isinstance(value, (tuple, list, _rq1_s0_np.ndarray)):
        values = list(value)
    else:
        values = [value]
    return '|'.join((str(int(v)) for v in values))

class _rq1_s0_TrueContinuousDAG:
    """Sample and evaluate the eight families used by the four generators."""

    def __init__(self, graph: _rq1_s0_nx.DiGraph, parameters: Mapping[str, Mapping[str, Any]], node_order: Sequence[str]) -> None:
        if not _rq1_s0_nx.is_directed_acyclic_graph(graph):
            raise ValueError('The true graph is not a DAG.')
        self.graph = graph.copy()
        self.parameters = {str(node): dict(values) for node, values in parameters.items()}
        self.node_order = [str(node) for node in node_order]
        self.topological_order = list(_rq1_s0_nx.topological_sort(self.graph))
        missing = set(self.graph.nodes) - set(self.parameters)
        if missing:
            raise ValueError(f'Missing true parameters for: {sorted(missing)}')
        families = {str(values.get('family')) for values in self.parameters.values()}
        unsupported = families - _rq1_s0_SUPPORTED_TRUE_FAMILIES
        if unsupported:
            raise ValueError(f'Unsupported true parameter families: {sorted(unsupported)}')

    @staticmethod
    def _linear_location(parameters: Mapping[str, Any], dataframe: pd.DataFrame) -> _rq1_s0_np.ndarray:
        location = _rq1_s0_np.full(len(dataframe), float(parameters['intercept']), dtype=float)
        for parent, coefficient in parameters['coefficients'].items():
            location += float(coefficient) * dataframe[parent].to_numpy(dtype=float)
        return location

    @staticmethod
    def _gated_location(parameters: Mapping[str, Any], dataframe: pd.DataFrame) -> _rq1_s0_np.ndarray:
        location = _rq1_s0_np.full(len(dataframe), float(parameters['intercept']), dtype=float)
        for parent, gate in parameters['gates'].items():
            parent_values = dataframe[parent].to_numpy(dtype=float)
            location += _rq1_s0_np.where(parent_values > float(gate['threshold']), float(gate['shift']), -float(gate['shift']))
        return location

    @staticmethod
    def _mixture_logpdf(weights: Sequence[float], component_logpdfs: Sequence[_rq1_s0_np.ndarray]) -> _rq1_s0_np.ndarray:
        weights_array = _rq1_s0_np.asarray(weights, dtype=float)
        log_weights = _rq1_s0_np.log(_rq1_s0_np.clip(weights_array, _rq1_s0__MIN_PROBABILITY, None))
        return _rq1_s0_logsumexp(_rq1_s0_np.vstack([log_weight + component for log_weight, component in zip(log_weights, component_logpdfs)]), axis=0)

    @staticmethod
    def _sample_component_indices(weights: Sequence[float], size: int, rng: _rq1_s0_np.random.Generator) -> _rq1_s0_np.ndarray:
        probabilities = _rq1_s0_np.asarray(weights, dtype=float)
        probabilities = probabilities / probabilities.sum()
        return rng.choice(len(probabilities), size=size, p=probabilities)

    def node_logpdf(self, node: str, dataframe: pd.DataFrame) -> np.ndarray:
        parameters = self.parameters[node]
        family = str(parameters['family'])
        values = dataframe[node].to_numpy(dtype=float)
        if family == 'gaussian_root':
            return _rq1_s0_norm.logpdf(values, loc=float(parameters['mean']), scale=float(parameters['std']))
        if family == 'linear_gaussian':
            location = self._linear_location(parameters, dataframe)
            return _rq1_s0_norm.logpdf(values, loc=location, scale=float(parameters['noise_std']))
        if family == 'gaussian_mixture_root':
            component_logpdfs = [_rq1_s0_norm.logpdf(values, loc=float(mean), scale=float(std)) for mean, std in zip(parameters['means'], parameters['stds'])]
            return self._mixture_logpdf(parameters['weights'], component_logpdfs)
        if family == 'gated_gaussian':
            location = self._gated_location(parameters, dataframe)
            return _rq1_s0_norm.logpdf(values, loc=location, scale=float(parameters['noise_std']))
        if family == 'skew_normal_root':
            return _rq1_s0_skewnorm.logpdf(values, float(parameters['shape']), loc=float(parameters['loc']), scale=float(parameters['scale']))
        if family == 'skew_normal_regression':
            location = self._linear_location(parameters, dataframe)
            factor = float(parameters.get('residual_scale_factor', 1.0))
            residual = values - location
            return _rq1_s0_skewnorm.logpdf(residual, float(parameters['residual_shape']), loc=factor * float(parameters['residual_loc']), scale=factor * float(parameters['residual_scale']))
        if family == 'skew_mixture_root':
            component_logpdfs = [_rq1_s0_skewnorm.logpdf(values, float(shape), loc=float(mean), scale=float(scale)) for mean, scale, shape in zip(parameters['means'], parameters['scales'], parameters['shapes'])]
            return self._mixture_logpdf(parameters['weights'], component_logpdfs)
        if family == 'gated_skew_mixture':
            location = self._gated_location(parameters, dataframe)
            residual = values - location
            mixture = parameters['residual_mixture']
            factor = float(parameters['residual_scale_factor'])
            component_logpdfs = [_rq1_s0_skewnorm.logpdf(residual, float(shape), loc=factor * float(mean), scale=factor * float(scale)) for mean, scale, shape in zip(mixture['means'], mixture['scales'], mixture['shapes'])]
            return self._mixture_logpdf(mixture['weights'], component_logpdfs)
        raise ValueError(f'Unknown true family {family!r} for node {node!r}.')

    @staticmethod
    def _stable_interval_probability(cdf_lower: _rq1_s0_np.ndarray, cdf_upper: _rq1_s0_np.ndarray, sf_lower: _rq1_s0_np.ndarray, sf_upper: _rq1_s0_np.ndarray) -> _rq1_s0_np.ndarray:
        """Return a stable probability for lower < X <= upper."""
        cdf_difference = _rq1_s0_np.asarray(cdf_upper) - _rq1_s0_np.asarray(cdf_lower)
        survival_difference = _rq1_s0_np.asarray(sf_lower) - _rq1_s0_np.asarray(sf_upper)
        probability = _rq1_s0_np.maximum(cdf_difference, survival_difference)
        return _rq1_s0_np.clip(probability, _rq1_s0__MIN_PROBABILITY, 1.0)

    @classmethod
    def _normal_interval_probability(cls, lower: np.ndarray, upper: np.ndarray, location: np.ndarray | float, scale: float) -> np.ndarray:
        return cls._stable_interval_probability(_rq1_s0_norm.cdf(lower, loc=location, scale=scale), _rq1_s0_norm.cdf(upper, loc=location, scale=scale), _rq1_s0_norm.sf(lower, loc=location, scale=scale), _rq1_s0_norm.sf(upper, loc=location, scale=scale))

    @classmethod
    def _skew_interval_probability(cls, lower: _rq1_s0_np.ndarray, upper: _rq1_s0_np.ndarray, shape: float, location: _rq1_s0_np.ndarray | float, scale: float) -> _rq1_s0_np.ndarray:
        lower = _rq1_s0_np.asarray(lower, dtype=float)
        upper = _rq1_s0_np.asarray(upper, dtype=float)
        location = _rq1_s0_np.asarray(location, dtype=float)
        z_lower = (lower - location) / float(scale)
        z_upper = (upper - location) / float(scale)
        t_lower = _rq1_s0_owens_t(z_lower, float(shape))
        t_upper = _rq1_s0_owens_t(z_upper, float(shape))
        cdf_lower = _rq1_s0_np.clip(_rq1_s0_norm.cdf(z_lower) - 2.0 * t_lower, 0.0, 1.0)
        cdf_upper = _rq1_s0_np.clip(_rq1_s0_norm.cdf(z_upper) - 2.0 * t_upper, 0.0, 1.0)
        sf_lower = _rq1_s0_np.clip(_rq1_s0_norm.sf(z_lower) + 2.0 * t_lower, 0.0, 1.0)
        sf_upper = _rq1_s0_np.clip(_rq1_s0_norm.sf(z_upper) + 2.0 * t_upper, 0.0, 1.0)
        return cls._stable_interval_probability(cdf_lower, cdf_upper, sf_lower, sf_upper)

    def node_log_interval_probability(self, node: str, dataframe: pd.DataFrame, lower: _rq1_s0_np.ndarray, upper: _rq1_s0_np.ndarray) -> _rq1_s0_np.ndarray:
        """Log true conditional probability of the observed node interval."""
        parameters = self.parameters[node]
        family = str(parameters['family'])
        if family == 'gaussian_root':
            probability = self._normal_interval_probability(lower, upper, float(parameters['mean']), float(parameters['std']))
        elif family == 'linear_gaussian':
            location = self._linear_location(parameters, dataframe)
            probability = self._normal_interval_probability(lower, upper, location, float(parameters['noise_std']))
        elif family == 'gaussian_mixture_root':
            component_probabilities = [self._normal_interval_probability(lower, upper, float(mean), float(std)) for mean, std in zip(parameters['means'], parameters['stds'])]
            probability = _rq1_s0_np.sum(_rq1_s0_np.asarray(parameters['weights'], dtype=float)[:, None] * _rq1_s0_np.vstack(component_probabilities), axis=0)
        elif family == 'gated_gaussian':
            location = self._gated_location(parameters, dataframe)
            probability = self._normal_interval_probability(lower, upper, location, float(parameters['noise_std']))
        elif family == 'skew_normal_root':
            probability = self._skew_interval_probability(lower, upper, float(parameters['shape']), float(parameters['loc']), float(parameters['scale']))
        elif family == 'skew_normal_regression':
            location = self._linear_location(parameters, dataframe)
            factor = float(parameters.get('residual_scale_factor', 1.0))
            probability = self._skew_interval_probability(lower, upper, float(parameters['residual_shape']), location + factor * float(parameters['residual_loc']), factor * float(parameters['residual_scale']))
        elif family == 'skew_mixture_root':
            component_probabilities = [self._skew_interval_probability(lower, upper, float(shape), float(mean), float(scale)) for mean, scale, shape in zip(parameters['means'], parameters['scales'], parameters['shapes'])]
            probability = _rq1_s0_np.sum(_rq1_s0_np.asarray(parameters['weights'], dtype=float)[:, None] * _rq1_s0_np.vstack(component_probabilities), axis=0)
        elif family == 'gated_skew_mixture':
            location = self._gated_location(parameters, dataframe)
            mixture = parameters['residual_mixture']
            factor = float(parameters['residual_scale_factor'])
            component_probabilities = [self._skew_interval_probability(lower, upper, float(shape), location + factor * float(mean), factor * float(scale)) for mean, scale, shape in zip(mixture['means'], mixture['scales'], mixture['shapes'])]
            probability = _rq1_s0_np.sum(_rq1_s0_np.asarray(mixture['weights'], dtype=float)[:, None] * _rq1_s0_np.vstack(component_probabilities), axis=0)
        else:
            raise ValueError(f'Unknown true family {family!r} for node {node!r}.')
        probability = _rq1_s0_np.clip(probability, _rq1_s0__MIN_PROBABILITY, 1.0)
        return _rq1_s0_np.log(probability)

    def joint_logscore_for_representation(self, dataframe: pd.DataFrame, *, discretised_nodes: Sequence[str], transformer: Any | None, return_nodewise: bool=False) -> _rq1_s0_np.ndarray | tuple[_rq1_s0_np.ndarray, dict[str, _rq1_s0_np.ndarray]]:
        """Evaluate true P in the same raw/discrete representation as Q."""
        dataframe = dataframe[self.node_order].reset_index(drop=True)
        discrete_set = {str(node) for node in discretised_nodes}
        if discrete_set:
            if transformer is None:
                raise ValueError('A saved quantile transform is required for discretised nodes.')
            if transformer.nodes != discrete_set:
                raise ValueError('The true-score discretised nodes do not match the saved transform.')
            transformed = transformer.transform(dataframe)
        else:
            if transformer is not None:
                raise ValueError('Continuous true scoring must not receive a discretiser.')
            transformed = dataframe
        total = _rq1_s0_np.zeros(len(dataframe), dtype=float)
        nodewise: dict[str, _rq1_s0_np.ndarray] = {}
        for node in self.topological_order:
            if node in discrete_set:
                states = transformed[node].to_numpy(dtype=int)
                edges = transformer.bin_edges[node]
                lower = edges[states]
                upper = edges[states + 1]
                values = self.node_log_interval_probability(node, dataframe, lower, upper)
            else:
                values = self.node_logpdf(node, dataframe)
            if values.shape != (len(dataframe),) or not _rq1_s0_np.all(_rq1_s0_np.isfinite(values)):
                raise FloatingPointError(f'Invalid true representation score for {node!r}.')
            nodewise[node] = values
            total += values
        if return_nodewise:
            return (total, nodewise)
        return total

    def joint_logpdf(self, dataframe: pd.DataFrame, *, return_nodewise: bool=False) -> _rq1_s0_np.ndarray | tuple[_rq1_s0_np.ndarray, dict[str, _rq1_s0_np.ndarray]]:
        dataframe = dataframe[self.node_order].reset_index(drop=True)
        total = _rq1_s0_np.zeros(len(dataframe), dtype=float)
        nodewise: dict[str, _rq1_s0_np.ndarray] = {}
        for node in self.topological_order:
            values = self.node_logpdf(node, dataframe)
            if values.shape != (len(dataframe),) or not _rq1_s0_np.all(_rq1_s0_np.isfinite(values)):
                raise FloatingPointError(f'Invalid true log-density for {node!r}.')
            nodewise[node] = values
            total += values
        if return_nodewise:
            return (total, nodewise)
        return total

    def sample(self, size: int, random_state: int) -> _rq1_s0_pd.DataFrame:
        if size <= 0:
            raise ValueError('Monte Carlo sample size must be positive.')
        rng = _rq1_s0_np.random.default_rng(random_state)
        data: dict[str, _rq1_s0_np.ndarray] = {}
        for node in self.topological_order:
            parameters = self.parameters[node]
            family = str(parameters['family'])
            parent_frame = _rq1_s0_pd.DataFrame(data)
            if family == 'gaussian_root':
                values = rng.normal(loc=float(parameters['mean']), scale=float(parameters['std']), size=size)
            elif family == 'linear_gaussian':
                location = self._linear_location(parameters, parent_frame)
                values = location + rng.normal(loc=0.0, scale=float(parameters['noise_std']), size=size)
            elif family == 'gaussian_mixture_root':
                component = self._sample_component_indices(parameters['weights'], size, rng)
                means = _rq1_s0_np.asarray(parameters['means'], dtype=float)
                stds = _rq1_s0_np.asarray(parameters['stds'], dtype=float)
                values = rng.normal(loc=means[component], scale=stds[component])
            elif family == 'gated_gaussian':
                location = self._gated_location(parameters, parent_frame)
                values = location + rng.normal(loc=0.0, scale=float(parameters['noise_std']), size=size)
            elif family == 'skew_normal_root':
                values = _rq1_s0_skewnorm.rvs(float(parameters['shape']), loc=float(parameters['loc']), scale=float(parameters['scale']), size=size, random_state=rng)
            elif family == 'skew_normal_regression':
                location = self._linear_location(parameters, parent_frame)
                residual = _rq1_s0_skewnorm.rvs(float(parameters['residual_shape']), loc=float(parameters['residual_loc']), scale=float(parameters['residual_scale']), size=size, random_state=rng)
                factor = float(parameters.get('residual_scale_factor', 1.0))
                values = location + factor * residual
            elif family == 'skew_mixture_root':
                component = self._sample_component_indices(parameters['weights'], size, rng)
                means = _rq1_s0_np.asarray(parameters['means'], dtype=float)
                scales = _rq1_s0_np.asarray(parameters['scales'], dtype=float)
                shapes = _rq1_s0_np.asarray(parameters['shapes'], dtype=float)
                values = _rq1_s0_np.empty(size, dtype=float)
                for index in range(len(parameters['weights'])):
                    mask = component == index
                    if _rq1_s0_np.any(mask):
                        values[mask] = _rq1_s0_skewnorm.rvs(float(shapes[index]), loc=float(means[index]), scale=float(scales[index]), size=int(mask.sum()), random_state=rng)
            elif family == 'gated_skew_mixture':
                location = self._gated_location(parameters, parent_frame)
                mixture = parameters['residual_mixture']
                component = self._sample_component_indices(mixture['weights'], size, rng)
                means = _rq1_s0_np.asarray(mixture['means'], dtype=float)
                scales = _rq1_s0_np.asarray(mixture['scales'], dtype=float)
                shapes = _rq1_s0_np.asarray(mixture['shapes'], dtype=float)
                residual = _rq1_s0_np.empty(size, dtype=float)
                for index in range(len(mixture['weights'])):
                    mask = component == index
                    if _rq1_s0_np.any(mask):
                        residual[mask] = _rq1_s0_skewnorm.rvs(float(shapes[index]), loc=float(means[index]), scale=float(scales[index]), size=int(mask.sum()), random_state=rng)
                values = location + float(parameters['residual_scale_factor']) * residual
            else:
                raise ValueError(f'Unknown true family {family!r} for node {node!r}.')
            data[node] = _rq1_s0_np.asarray(values, dtype=float)
        return _rq1_s0_pd.DataFrame(data, columns=self.node_order)

class _rq1_s0_TransformedTruthParticleLikelihood:
    """SMC likelihood for p_T(x_C, d_D) with latent within-bin values."""

    def __init__(self, true_model: _rq1_s0_TrueContinuousDAG, transformer: Any, discrete_nodes: Sequence[str], *, particles: int=256, replicates: int=2, batch_size: int=64) -> None:
        if particles < 8:
            raise ValueError('particles must be >= 8')
        if replicates < 1:
            raise ValueError('replicates must be >= 1')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.true_model = true_model
        self.transformer = transformer
        self.discrete_nodes = tuple(map(str, discrete_nodes))
        self.discrete_set = set(self.discrete_nodes)
        self.particles = int(particles)
        self.replicates = int(replicates)
        self.batch_size = int(batch_size)

    @staticmethod
    def _systematic_indices(log_weights: _rq1_s0_np.ndarray, rng: _rq1_s0_np.random.Generator) -> _rq1_s0_np.ndarray:
        """Row-wise systematic resampling indices for log weights [B,P]."""
        b, p = log_weights.shape
        normalized = _rq1_s0_np.exp(log_weights - _rq1_s0_logsumexp(log_weights, axis=1, keepdims=True))
        cdf = _rq1_s0_np.cumsum(normalized, axis=1)
        cdf[:, -1] = 1.0
        starts = rng.random(b) / p
        positions = starts[:, None] + _rq1_s0_np.arange(p, dtype=float)[None, :] / p
        out = _rq1_s0_np.empty((b, p), dtype=_rq1_s0_np.int64)
        for i in range(b):
            out[i] = _rq1_s0_np.searchsorted(cdf[i], positions[i], side='right')
        return _rq1_s0_np.minimum(out, p - 1)

    @staticmethod
    def _categorical_rows(probabilities: _rq1_s0_np.ndarray, rng: _rq1_s0_np.random.Generator) -> _rq1_s0_np.ndarray:
        cdf = _rq1_s0_np.cumsum(probabilities, axis=1)
        cdf[:, -1] = 1.0
        u = rng.random(len(probabilities))
        return _rq1_s0_np.sum(u[:, None] > cdf, axis=1).astype(int)

    @staticmethod
    def _normal_prob(lower, upper, loc, scale):
        return _rq1_s0_TrueContinuousDAG._normal_interval_probability(lower, upper, loc, scale)

    @staticmethod
    def _skew_prob(lower, upper, shape, loc, scale):
        return _rq1_s0_TrueContinuousDAG._skew_interval_probability(lower, upper, shape, loc, scale)

    @staticmethod
    def _truncated_normal(lower, upper, loc, scale, rng):
        lower = _rq1_s0_np.asarray(lower, dtype=float)
        upper = _rq1_s0_np.asarray(upper, dtype=float)
        loc = _rq1_s0_np.asarray(loc, dtype=float)
        fl = _rq1_s0_norm.cdf(lower, loc=loc, scale=scale)
        fu = _rq1_s0_norm.cdf(upper, loc=loc, scale=scale)
        sl = _rq1_s0_norm.sf(lower, loc=loc, scale=scale)
        su = _rq1_s0_norm.sf(upper, loc=loc, scale=scale)
        u = rng.random(_rq1_s0_np.broadcast(lower, loc).shape)
        use_sf = fu > 0.9
        q = fl + u * _rq1_s0_np.maximum(fu - fl, 0.0)
        s = su + u * _rq1_s0_np.maximum(sl - su, 0.0)
        tiny = _rq1_s0_np.finfo(float).tiny
        q = _rq1_s0_np.clip(q, tiny, 1.0 - _rq1_s0_np.finfo(float).eps)
        s = _rq1_s0_np.clip(s, tiny, 1.0 - _rq1_s0_np.finfo(float).eps)
        x_cdf = _rq1_s0_norm.ppf(q, loc=loc, scale=scale)
        x_sf = _rq1_s0_norm.isf(s, loc=loc, scale=scale)
        return _rq1_s0_np.where(use_sf, x_sf, x_cdf)

    @staticmethod
    def _truncated_skew(lower, upper, shape, loc, scale, rng):
        """Sample a skew-normal conditional truncated to ``(lower, upper]``.

        Numerically identical algorithm to the original implementation, but
        avoid eagerly evaluating BOTH expensive skewnorm.ppf and skewnorm.isf
        branches for every particle.  ``np.where`` evaluates its two inputs
        before selecting, which roughly doubled inverse-CDF work on skew and
        skew-multimodal KL runs.

        The CDF/SF construction and RNG draw are intentionally unchanged so
        this is a performance-only patch, not an estimator/model change.
        """
        lower = _rq1_s0_np.asarray(lower, dtype=float)
        upper = _rq1_s0_np.asarray(upper, dtype=float)
        loc = _rq1_s0_np.asarray(loc, dtype=float)
        z_lower = (lower - loc) / float(scale)
        z_upper = (upper - loc) / float(scale)
        t_lower = _rq1_s0_owens_t(z_lower, float(shape))
        t_upper = _rq1_s0_owens_t(z_upper, float(shape))
        fl = _rq1_s0_np.clip(_rq1_s0_norm.cdf(z_lower) - 2.0 * t_lower, 0.0, 1.0)
        fu = _rq1_s0_np.clip(_rq1_s0_norm.cdf(z_upper) - 2.0 * t_upper, 0.0, 1.0)
        sl = _rq1_s0_np.clip(_rq1_s0_norm.sf(z_lower) + 2.0 * t_lower, 0.0, 1.0)
        su = _rq1_s0_np.clip(_rq1_s0_norm.sf(z_upper) + 2.0 * t_upper, 0.0, 1.0)
        u = rng.random(_rq1_s0_np.broadcast(lower, loc).shape)
        use_sf = fu > 0.9
        q = fl + u * _rq1_s0_np.maximum(fu - fl, 0.0)
        s = su + u * _rq1_s0_np.maximum(sl - su, 0.0)
        tiny = _rq1_s0_np.finfo(float).tiny
        q = _rq1_s0_np.clip(q, tiny, 1.0 - _rq1_s0_np.finfo(float).eps)
        s = _rq1_s0_np.clip(s, tiny, 1.0 - _rq1_s0_np.finfo(float).eps)
        samples = _rq1_s0_np.empty(_rq1_s0_np.broadcast(lower, loc).shape, dtype=float)
        cdf_mask = ~use_sf
        if _rq1_s0_np.any(cdf_mask):
            samples[cdf_mask] = _rq1_s0_skewnorm.ppf(q[cdf_mask], shape, loc=loc[cdf_mask], scale=scale)
        if _rq1_s0_np.any(use_sf):
            samples[use_sf] = _rq1_s0_skewnorm.isf(s[use_sf], shape, loc=loc[use_sf], scale=scale)
        return samples

    def _component_spec(self, node: str, parent_frame: pd.DataFrame):
        """Return list[(prior_weight, loc_vector, scale, shape|None)]."""
        p = self.true_model.parameters[node]
        family = str(p['family'])
        n = len(parent_frame)
        if family == 'gaussian_root':
            return [(1.0, _rq1_s0_np.full(n, float(p['mean'])), float(p['std']), None)]
        if family == 'linear_gaussian':
            loc = self.true_model._linear_location(p, parent_frame)
            return [(1.0, loc, float(p['noise_std']), None)]
        if family == 'gaussian_mixture_root':
            return [(float(w), _rq1_s0_np.full(n, float(m)), float(sd), None) for w, m, sd in zip(p['weights'], p['means'], p['stds'])]
        if family == 'gated_gaussian':
            loc = self.true_model._gated_location(p, parent_frame)
            return [(1.0, loc, float(p['noise_std']), None)]
        if family == 'skew_normal_root':
            return [(1.0, _rq1_s0_np.full(n, float(p['loc'])), float(p['scale']), float(p['shape']))]
        if family == 'skew_normal_regression':
            base = self.true_model._linear_location(p, parent_frame)
            factor = float(p.get('residual_scale_factor', 1.0))
            loc = base + factor * float(p['residual_loc'])
            scale = factor * float(p['residual_scale'])
            return [(1.0, loc, scale, float(p['residual_shape']))]
        if family == 'skew_mixture_root':
            return [(float(w), _rq1_s0_np.full(n, float(m)), float(sc), float(sh)) for w, m, sc, sh in zip(p['weights'], p['means'], p['scales'], p['shapes'])]
        if family == 'gated_skew_mixture':
            base = self.true_model._gated_location(p, parent_frame)
            mix = p['residual_mixture']
            factor = float(p['residual_scale_factor'])
            return [(float(w), base + factor * float(m), factor * float(sc), float(sh)) for w, m, sc, sh in zip(mix['weights'], mix['means'], mix['scales'], mix['shapes'])]
        raise ValueError(f'Unsupported family {family!r}')

    def _sample_truncated_conditional(self, node: str, parent_frame: pd.DataFrame, lower: _rq1_s0_np.ndarray, upper: _rq1_s0_np.ndarray, rng: _rq1_s0_np.random.Generator) -> _rq1_s0_np.ndarray:
        specs = self._component_spec(node, parent_frame)
        component_probs = []
        for weight, loc, scale, shape in specs:
            if shape is None:
                prob = self._normal_prob(lower, upper, loc, scale)
            else:
                prob = self._skew_prob(lower, upper, shape, loc, scale)
            component_probs.append(float(weight) * _rq1_s0_np.asarray(prob, dtype=float))
        weighted = _rq1_s0_np.column_stack(component_probs)
        total = _rq1_s0_np.clip(weighted.sum(axis=1), _rq1_s0__MIN_PROBABILITY, None)
        posterior = weighted / total[:, None]
        component = self._categorical_rows(posterior, rng)
        samples = _rq1_s0_np.empty(len(parent_frame), dtype=float)
        for k, (_, loc, scale, shape) in enumerate(specs):
            mask = component == k
            if not _rq1_s0_np.any(mask):
                continue
            if shape is None:
                samples[mask] = self._truncated_normal(lower[mask], upper[mask], _rq1_s0_np.asarray(loc)[mask], scale, rng)
            else:
                samples[mask] = self._truncated_skew(lower[mask], upper[mask], shape, _rq1_s0_np.asarray(loc)[mask], scale, rng)
        if not _rq1_s0_np.all(_rq1_s0_np.isfinite(samples)):
            raise FloatingPointError(f'Non-finite truncated samples for {node}')
        return samples

    def _parent_frame(self, node: str, raw_batch: _rq1_s0_pd.DataFrame, latent: Mapping[str, _rq1_s0_np.ndarray]) -> _rq1_s0_pd.DataFrame:
        b = len(raw_batch)
        p = self.particles
        data = {}
        for parent in self.true_model.graph.predecessors(node):
            parent = str(parent)
            if parent in self.discrete_set:
                data[parent] = latent[parent].reshape(-1)
            else:
                data[parent] = _rq1_s0_np.repeat(raw_batch[parent].to_numpy(dtype=float), p)
        return _rq1_s0_pd.DataFrame(data, index=_rq1_s0_np.arange(b * p))

    def _one_replicate(self, raw: pd.DataFrame, transformed: pd.DataFrame, seed: int) -> _rq1_s0_np.ndarray:
        rng = _rq1_s0_np.random.default_rng(seed)
        out = _rq1_s0_np.empty(len(raw), dtype=float)
        pcount = self.particles
        for start in range(0, len(raw), self.batch_size):
            stop = min(start + self.batch_size, len(raw))
            rb = raw.iloc[start:stop].reset_index(drop=True)
            tb = transformed.iloc[start:stop].reset_index(drop=True)
            b = len(rb)
            loglik = _rq1_s0_np.zeros(b, dtype=float)
            latent: dict[str, _rq1_s0_np.ndarray] = {}
            for node in self.true_model.topological_order:
                node = str(node)
                parent_frame = self._parent_frame(node, rb, latent)
                if node in self.discrete_set:
                    states = tb[node].to_numpy(dtype=int)
                    edges = self.transformer.bin_edges[node]
                    lower_b = edges[states]
                    upper_b = edges[states + 1]
                    lower = _rq1_s0_np.repeat(lower_b, pcount)
                    upper = _rq1_s0_np.repeat(upper_b, pcount)
                    log_inc_flat = self.true_model.node_log_interval_probability(node, parent_frame, lower, upper)
                    sampled = self._sample_truncated_conditional(node, parent_frame, lower, upper, rng)
                    latent[node] = sampled.reshape(b, pcount)
                else:
                    frame = parent_frame.copy()
                    frame[node] = _rq1_s0_np.repeat(rb[node].to_numpy(dtype=float), pcount)
                    log_inc_flat = self.true_model.node_logpdf(node, frame)
                log_inc = _rq1_s0_np.asarray(log_inc_flat, dtype=float).reshape(b, pcount)
                if not _rq1_s0_np.all(_rq1_s0_np.isfinite(log_inc)):
                    raise FloatingPointError(f'Non-finite particle weights at {node}')
                loglik += _rq1_s0_logsumexp(log_inc, axis=1) - _rq1_s0_math.log(pcount)
                indices = self._systematic_indices(log_inc, rng)
                if latent:
                    row = _rq1_s0_np.arange(b)[:, None]
                    for key in latent:
                        latent[key] = latent[key][row, indices]
            out[start:stop] = loglik
        return out

    def loglikelihood(self, raw_evaluation: pd.DataFrame, base_seed: int) -> tuple[_rq1_s0_np.ndarray, dict[str, Any]]:
        raw = raw_evaluation[self.true_model.node_order].reset_index(drop=True)
        transformed = self.transformer.transform(raw)
        replicate_logs = []
        for r in range(self.replicates):
            seed = int(_rq1_s0_np.random.SeedSequence([base_seed, 77123, r]).generate_state(1)[0])
            replicate_logs.append(self._one_replicate(raw, transformed, seed))
        matrix = _rq1_s0_np.vstack(replicate_logs)
        combined = _rq1_s0_logsumexp(matrix, axis=0) - _rq1_s0_math.log(self.replicates)
        if self.replicates > 1:
            row_sd = _rq1_s0_np.std(matrix, axis=0, ddof=1)
            replicate_mean_logs = _rq1_s0_np.mean(matrix, axis=1)
            diagnostics = {'replicate_mean_log_p': replicate_mean_logs.tolist(), 'replicate_mean_log_p_std': float(_rq1_s0_np.std(replicate_mean_logs, ddof=1)), 'median_row_log_p_replicate_sd': float(_rq1_s0_np.median(row_sd)), 'p95_row_log_p_replicate_sd': float(_rq1_s0_np.quantile(row_sd, 0.95)), 'max_row_log_p_replicate_sd': float(_rq1_s0_np.max(row_sd))}
        else:
            diagnostics = {'replicate_mean_log_p': [float(_rq1_s0_np.mean(matrix[0]))], 'replicate_mean_log_p_std': _rq1_s0_np.nan, 'median_row_log_p_replicate_sd': _rq1_s0_np.nan, 'p95_row_log_p_replicate_sd': _rq1_s0_np.nan, 'max_row_log_p_replicate_sd': _rq1_s0_np.nan}
        diagnostics.update({'particles': self.particles, 'replicates': self.replicates, 'batch_size': self.batch_size, 'combination': 'log(mean of independent particle likelihood estimates)'})
        return (_rq1_s0_np.asarray(combined, dtype=float), diagnostics)

    def validate_continuous_identity(self, raw: pd.DataFrame) -> float:
        """No-discretisation special case must equal exact joint logpdf."""
        exact = self.true_model.joint_logpdf(raw)
        total = _rq1_s0_np.zeros(len(raw), dtype=float)
        for node in self.true_model.topological_order:
            total += self.true_model.node_logpdf(str(node), raw)
        return float(_rq1_s0_np.max(_rq1_s0_np.abs(_rq1_s0_np.asarray(exact) - total)))

class _rq1_s0_SavedQuantileTransform:
    """Apply the exact bin boundaries saved during structure learning."""

    def __init__(self, cardinalities: Mapping[str, int], bin_edges: Mapping[str, Sequence[Any]]) -> None:
        self.cardinalities = {str(node): int(cardinality) for node, cardinality in cardinalities.items()}
        self.bin_edges = {str(node): _rq1_s0_np.asarray([_rq1_s0_decode_float(value) for value in edges], dtype=float) for node, edges in bin_edges.items()}
        if set(self.cardinalities) != set(self.bin_edges):
            raise ValueError('Saved cardinalities and bin edges refer to different node sets.')
        for node, cardinality in self.cardinalities.items():
            edges = self.bin_edges[node]
            if edges.shape != (cardinality + 1,) or not _rq1_s0_np.all(_rq1_s0_np.diff(edges) > 0):
                raise ValueError(f'Invalid saved boundaries for {node!r}.')

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> '_rq1_s0_SavedQuantileTransform':
        return cls(metadata['cardinalities'], metadata['bin_edges'])

    @property
    def nodes(self) -> set[str]:
        return set(self.cardinalities)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        transformed = dataframe.copy()
        for node, cardinality in self.cardinalities.items():
            values = transformed[node].to_numpy(dtype=float)
            internal_edges = self.bin_edges[node][1:-1]
            states = _rq1_s0_np.searchsorted(internal_edges, values, side='left').astype(_rq1_s0_np.int64)
            if states.min(initial=0) < 0 or states.max(initial=0) >= cardinality:
                raise ValueError(f'Saved discretisation produced an invalid state for {node!r}.')
            transformed[node] = states
        return transformed

@_rq1_s0_dataclass
class _rq1_s0_GaussianConditional:
    child: str
    parents: tuple[str, ...]
    coefficients: np.ndarray
    standard_deviation: float

@_rq1_s0_dataclass
class _rq1_s0_CPTConditional:
    child: str
    parents: tuple[str, ...]
    cardinality: int
    marginal_probabilities: np.ndarray
    tables: dict[tuple[int, ...], np.ndarray]

@_rq1_s0_dataclass
class _rq1_s0_LogisticConditional:
    kind: str
    parents: tuple[str, ...]
    cardinality: int
    prior_probabilities: np.ndarray
    classifier: Any = None
    feature_means: np.ndarray | None = None
    feature_stds: np.ndarray | None = None
    classes: np.ndarray | None = None
    converged: bool = True

@_rq1_s0_dataclass
class _rq1_s0_MixtureConditional:
    """Evaluation-time fitted V2 mixture-of-linear-experts conditional."""
    child: str
    parents: tuple[str, ...]
    n_components: int
    feature_means: np.ndarray
    feature_stds: np.ndarray
    child_mean: float
    child_std: float
    betas: np.ndarray
    variances: np.ndarray
    gate_coefficients: np.ndarray
    converged: bool
    iterations: int
    gradient_norm: float
    min_effective_component_size: float

class _rq1_s0_LearnedDAGDistribution:
    """Fit Q for the continuous, discrete, or hybrid learned DAG."""

    def __init__(self, graph: nx.DiGraph, nodes: Sequence[str], paradigm: str, *, discrete_nodes: Sequence[str]=(), continuous_nodes: Sequence[str]=(), transform: _rq1_s0_SavedQuantileTransform | None=None, cpt_alpha: float=1.0, min_regime_size: int=5, logistic_c: float=1000.0, hybrid_selected_k_by_configuration: Mapping[str, Mapping[str, int]] | None=None, hybrid_continuous_scaling: Mapping[str, Any] | None=None, hybrid_prior_hyperparameters: Mapping[str, float] | None=None) -> None:
        if paradigm not in _rq1_s0_PARADIGMS:
            raise ValueError(f'Unknown paradigm: {paradigm!r}')
        if cpt_alpha <= 0:
            raise ValueError('cpt_alpha must be positive.')
        if min_regime_size < 1:
            raise ValueError('min_regime_size must be positive.')
        if logistic_c <= 0:
            raise ValueError('logistic_c must be positive.')
        self.graph = graph.copy()
        self.nodes = [str(node) for node in nodes]
        self.paradigm = paradigm
        self.discrete_nodes = set((str(node) for node in discrete_nodes))
        self.continuous_nodes = set((str(node) for node in continuous_nodes))
        self.transformer = transform
        self.cpt_alpha = float(cpt_alpha)
        self.min_regime_size = int(min_regime_size)
        self.logistic_c = float(logistic_c)
        self.hybrid_selected_k_by_configuration = {str(child): {str(key): int(value) for key, value in choices.items()} for child, choices in (hybrid_selected_k_by_configuration or {}).items()}
        self.hybrid_continuous_scaling = dict(hybrid_continuous_scaling or {})
        self.hybrid_prior_hyperparameters = {str(key): float(value) for key, value in (hybrid_prior_hyperparameters or {}).items()}
        self.models: dict[str, Any] = {}
        self.diagnostics = {'gaussian_models': 0, 'mixture_models_attempted': 0, 'mixture_models_fitted': 0, 'mixture_models_nonconverged': 0, 'mixture_k2_models': 0, 'mixture_k3_models': 0, 'cpt_models': 0, 'logistic_models_attempted': 0, 'logistic_models_fitted': 0, 'logistic_models_nonconverged': 0, 'prior_fallbacks': 0, 'regime_fallbacks': 0}
        if set(self.graph.nodes) != set(self.nodes):
            raise ValueError('Learned graph nodes do not match the saved node list.')
        if self.paradigm == 'continuous':
            if self.transformer is not None:
                raise ValueError('The continuous model must not have a discretiser.')
        elif self.paradigm == 'discrete':
            if self.transformer is None or self.transformer.nodes != set(self.nodes):
                raise ValueError('The discrete model requires saved boundaries for every node.')
        else:
            if self.discrete_nodes | self.continuous_nodes != set(self.nodes):
                raise ValueError('Hybrid node sets must cover all nodes.')
            if self.discrete_nodes & self.continuous_nodes:
                raise ValueError('Hybrid node sets overlap.')
            if self.transformer is None or self.transformer.nodes != self.discrete_nodes:
                raise ValueError('Hybrid saved boundaries must match the designated discrete nodes.')
            if set(self.hybrid_selected_k_by_configuration) != self.continuous_nodes:
                raise ValueError('Hybrid V2 evaluation requires saved final K choices for every designated continuous child.')
            means = self.hybrid_continuous_scaling.get('continuous_means', {})
            stds = self.hybrid_continuous_scaling.get('continuous_stds', {})
            if set(means) != self.continuous_nodes or set(stds) != self.continuous_nodes:
                raise ValueError('Hybrid V2 evaluation requires the saved structure-learning continuous means/stds for every continuous node.')
            if any((float(stds[node]) <= 0.0 for node in self.continuous_nodes)):
                raise ValueError('Saved Hybrid V2 continuous standard deviations must be positive.')
            required_priors = {'continuous_coefficient_prior_variance', 'continuous_intercept_prior_variance', 'continuous_inverse_gamma_shape', 'continuous_inverse_gamma_scale', 'clg_coefficient_prior_variance', 'clg_intercept_prior_variance', 'clg_inverse_gamma_shape', 'clg_inverse_gamma_scale', 'mixture_gate_coefficient_prior_variance', 'mixture_gate_intercept_prior_variance'}
            missing_priors = required_priors - set(self.hybrid_prior_hyperparameters)
            if missing_priors:
                raise ValueError('Hybrid V2 evaluation is missing saved frozen priors: ' + ', '.join(sorted(missing_priors)))
            if any((self.hybrid_prior_hyperparameters[name] <= 0.0 for name in required_priors)):
                raise ValueError('Saved Hybrid V2 prior hyperparameters must be positive.')

    @staticmethod
    def _configuration_key(values: Sequence[Any]) -> tuple[int, ...]:
        return tuple((int(value) for value in values))

    @staticmethod
    def _design_matrix(dataframe: pd.DataFrame, parents: Sequence[str]) -> _rq1_s0_np.ndarray:
        if not parents:
            return _rq1_s0_np.ones((len(dataframe), 1), dtype=float)
        return _rq1_s0_np.column_stack([_rq1_s0_np.ones(len(dataframe), dtype=float), dataframe[list(parents)].to_numpy(dtype=float)])

    def _fit_gaussian(self, dataframe: pd.DataFrame, child: str, parents: Sequence[str]) -> _rq1_s0_GaussianConditional:
        parent_tuple = tuple(parents)
        y = dataframe[child].to_numpy(dtype=float)
        design = self._design_matrix(dataframe, parent_tuple)
        coefficients, _, _, _ = _rq1_s0_np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coefficients
        standard_deviation = max(float(_rq1_s0_np.sqrt(_rq1_s0_np.mean(residuals ** 2))), _rq1_s0__MIN_STANDARD_DEVIATION)
        self.diagnostics['gaussian_models'] += 1
        return _rq1_s0_GaussianConditional(child=child, parents=parent_tuple, coefficients=coefficients, standard_deviation=standard_deviation)

    @staticmethod
    def _eval_gaussian(model: _rq1_s0_GaussianConditional, dataframe: pd.DataFrame) -> _rq1_s0_np.ndarray:
        if model.parents:
            design = _rq1_s0_np.column_stack([_rq1_s0_np.ones(len(dataframe), dtype=float), dataframe[list(model.parents)].to_numpy(dtype=float)])
        else:
            design = _rq1_s0_np.ones((len(dataframe), 1), dtype=float)
        location = design @ model.coefficients
        return _rq1_s0_norm.logpdf(dataframe[model.child].to_numpy(dtype=float), loc=location, scale=model.standard_deviation)

    def _smoothed_probabilities(self, values: pd.Series, cardinality: int) -> np.ndarray:
        counts = values.astype(int).value_counts().reindex(range(cardinality), fill_value=0).to_numpy(dtype=float)
        return (counts + self.cpt_alpha) / (counts.sum() + self.cpt_alpha * cardinality)

    def _fit_cpt(self, dataframe: pd.DataFrame, child: str, parents: Sequence[str], cardinality: int) -> _rq1_s0_CPTConditional:
        parent_tuple = tuple(parents)
        marginal = self._smoothed_probabilities(dataframe[child], cardinality)
        tables: dict[tuple[int, ...], np.ndarray] = {}
        if parent_tuple:
            group_key: Any = parent_tuple[0] if len(parent_tuple) == 1 else list(parent_tuple)
            for key, group in dataframe.groupby(group_key, sort=False, observed=True):
                normalized = key if isinstance(key, tuple) else (key,)
                tables[self._configuration_key(normalized)] = self._smoothed_probabilities(group[child], cardinality)
        self.diagnostics['cpt_models'] += 1
        return _rq1_s0_CPTConditional(child=child, parents=parent_tuple, cardinality=cardinality, marginal_probabilities=marginal, tables=tables)

    @staticmethod
    def _eval_cpt(model: _rq1_s0_CPTConditional, dataframe: pd.DataFrame) -> _rq1_s0_np.ndarray:
        child_values = dataframe[model.child].to_numpy(dtype=int)
        if not model.parents:
            probabilities = model.marginal_probabilities[child_values]
            return _rq1_s0_np.log(_rq1_s0_np.clip(probabilities, _rq1_s0__MIN_PROBABILITY, None))
        parent_values = dataframe[list(model.parents)].to_numpy(dtype=int)
        result = _rq1_s0_np.empty(len(dataframe), dtype=float)
        for index, (configuration, child_value) in enumerate(zip(parent_values, child_values)):
            probabilities = model.tables.get(tuple((int(value) for value in configuration)), model.marginal_probabilities)
            result[index] = _rq1_s0_np.log(max(float(probabilities[child_value]), _rq1_s0__MIN_PROBABILITY))
        return result

    def _fit_logistic(self, dataframe: pd.DataFrame, child: str, parents: Sequence[str], cardinality: int, fallback_prior: _rq1_s0_np.ndarray) -> _rq1_s0_LogisticConditional:
        parent_tuple = tuple(parents)
        y = dataframe[child].to_numpy(dtype=int)
        local_prior = self._smoothed_probabilities(dataframe[child], cardinality) if len(dataframe) else fallback_prior
        minimum_size = max(self.min_regime_size, cardinality * 3, len(parent_tuple) + 2)
        if len(dataframe) < minimum_size or _rq1_s0_np.unique(y).size < 2:
            self.diagnostics['prior_fallbacks'] += 1
            return _rq1_s0_LogisticConditional(kind='prior', parents=parent_tuple, cardinality=cardinality, prior_probabilities=local_prior)
        self.diagnostics['logistic_models_attempted'] += 1
        features = dataframe[list(parent_tuple)].to_numpy(dtype=float)
        feature_means = features.mean(axis=0)
        feature_stds = features.std(axis=0, ddof=0)
        feature_stds = _rq1_s0_np.where(feature_stds < _rq1_s0__MIN_STANDARD_DEVIATION, 1.0, feature_stds)
        standardized = (features - feature_means) / feature_stds
        classifier = _rq1_s0_LogisticRegression(solver='lbfgs', C=self.logistic_c, max_iter=1000)
        converged = True
        try:
            with _rq1_s0_warnings.catch_warnings(record=True) as captured:
                _rq1_s0_warnings.simplefilter('always', _rq1_s0_ConvergenceWarning)
                classifier.fit(standardized, y)
                converged = not any((issubclass(warning.category, _rq1_s0_ConvergenceWarning) for warning in captured))
        except Exception:
            self.diagnostics['prior_fallbacks'] += 1
            return _rq1_s0_LogisticConditional(kind='prior', parents=parent_tuple, cardinality=cardinality, prior_probabilities=local_prior)
        self.diagnostics['logistic_models_fitted'] += 1
        if not converged:
            self.diagnostics['logistic_models_nonconverged'] += 1
        return _rq1_s0_LogisticConditional(kind='logistic', parents=parent_tuple, cardinality=cardinality, prior_probabilities=local_prior, classifier=classifier, feature_means=feature_means, feature_stds=feature_stds, classes=classifier.classes_.astype(int), converged=converged)

    @staticmethod
    def _eval_logistic(model: _rq1_s0_LogisticConditional, dataframe: pd.DataFrame, indices: _rq1_s0_np.ndarray, child_values: _rq1_s0_np.ndarray) -> _rq1_s0_np.ndarray:
        if model.kind == 'prior':
            return _rq1_s0_np.log(_rq1_s0_np.clip(model.prior_probabilities[child_values], _rq1_s0__MIN_PROBABILITY, None))
        features = dataframe.iloc[indices][list(model.parents)].to_numpy(dtype=float)
        standardized = (features - model.feature_means) / model.feature_stds
        predicted = model.classifier.predict_proba(standardized)
        full_probabilities = _rq1_s0_np.full((len(indices), model.cardinality), _rq1_s0__MIN_PROBABILITY, dtype=float)
        full_probabilities[:, model.classes] = predicted
        full_probabilities /= full_probabilities.sum(axis=1, keepdims=True)
        return _rq1_s0_np.log(_rq1_s0_np.clip(full_probabilities[_rq1_s0_np.arange(len(indices)), child_values], _rq1_s0__MIN_PROBABILITY, None))

    @staticmethod
    def _configuration_storage_key(values: Sequence[Any]) -> str:
        if len(values) == 0:
            return 'none'
        return '|'.join((str(int(value)) for value in values))

    def _selected_components(self, child: str, configuration: Sequence[Any]) -> int:
        child_choices = self.hybrid_selected_k_by_configuration.get(str(child))
        if child_choices is None:
            raise ValueError(f'Missing saved Hybrid V2 K choices for {child!r}.')
        key = self._configuration_storage_key(configuration)
        if key not in child_choices:
            raise ValueError(f'Missing saved Hybrid V2 K choice for {child!r}, configuration {key!r}.')
        value = int(child_choices[key])
        if value not in (1, 2, 3):
            raise ValueError(f'Invalid saved Hybrid V2 component count K={value} for {child!r}.')
        return value

    def _saved_hybrid_continuous_scaling(self, child: str, parents: Sequence[str]) -> tuple[_rq1_s0_np.ndarray, _rq1_s0_np.ndarray, float, float]:
        means = self.hybrid_continuous_scaling['continuous_means']
        stds = self.hybrid_continuous_scaling['continuous_stds']
        parent_tuple = tuple(parents)
        feature_means = _rq1_s0_np.asarray([float(means[parent]) for parent in parent_tuple], dtype=float)
        feature_stds = _rq1_s0_np.asarray([float(stds[parent]) for parent in parent_tuple], dtype=float)
        child_mean = float(means[child])
        child_std = float(stds[child])
        return (feature_means, feature_stds, child_mean, child_std)

    def _hybrid_mixture_prior_values(self, *, prior_family: str, p: int) -> tuple[_rq1_s0_np.ndarray, _rq1_s0_np.ndarray, float, float]:
        if prior_family not in {'continuous', 'clg'}:
            raise ValueError(f'Unknown Hybrid V2 mixture prior family: {prior_family!r}.')
        priors = self.hybrid_prior_hyperparameters
        expert_variances = _rq1_s0_np.full(p, priors[f'{prior_family}_coefficient_prior_variance'], dtype=float)
        expert_variances[0] = priors[f'{prior_family}_intercept_prior_variance']
        a0 = priors[f'{prior_family}_inverse_gamma_shape']
        b0 = priors[f'{prior_family}_inverse_gamma_scale']
        return (expert_variances, 1.0 / expert_variances, a0, b0)

    @staticmethod
    def _mixture_initializations(X: _rq1_s0_np.ndarray, y: _rq1_s0_np.ndarray, n_components: int) -> list[_rq1_s0_np.ndarray]:
        n, p = X.shape
        beta_global, *_ = _rq1_s0_np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ beta_global
        residual_variance = max(float(_rq1_s0_np.var(residual)), 0.05)
        response_scale = max(float(_rq1_s0_np.std(y, ddof=0)), 0.25)

        def pack(betas: _rq1_s0_np.ndarray, log_variances: _rq1_s0_np.ndarray, gamma: _rq1_s0_np.ndarray) -> _rq1_s0_np.ndarray:
            return _rq1_s0_np.concatenate([betas.ravel(), log_variances.ravel(), gamma.ravel()])
        starts: list[_rq1_s0_np.ndarray] = []
        offsets = _rq1_s0_np.linspace(-0.75, 0.75, n_components) * response_scale
        betas = _rq1_s0_np.tile(beta_global, (n_components, 1))
        betas[:, 0] += offsets
        log_variances = _rq1_s0_np.full(n_components, _rq1_s0_np.log(residual_variance))
        starts.append(pack(betas, log_variances, _rq1_s0_np.zeros((n_components - 1, p))))
        if n >= max(10, n_components * (p + 2)):
            boundaries = _rq1_s0_np.quantile(residual, _rq1_s0_np.linspace(0.0, 1.0, n_components + 1)[1:-1])
            labels = _rq1_s0_np.digitize(residual, boundaries, right=True)
            q_betas = _rq1_s0_np.tile(beta_global, (n_components, 1))
            q_log_variances = _rq1_s0_np.full(n_components, _rq1_s0_np.log(residual_variance))
            for component in range(n_components):
                mask = labels == component
                if int(mask.sum()) >= p + 2:
                    beta_component, *_ = _rq1_s0_np.linalg.lstsq(X[mask], y[mask], rcond=None)
                    q_betas[component] = beta_component
                    component_residual = y[mask] - X[mask] @ beta_component
                    q_log_variances[component] = _rq1_s0_np.log(max(float(_rq1_s0_np.var(component_residual)), 0.03))
            starts.append(pack(q_betas, q_log_variances, _rq1_s0_np.zeros((n_components - 1, p))))
        base = starts[0]
        while len(starts) < _rq1_s0__V2_Q_MIXTURE_RESTARTS:
            index = len(starts)
            perturbation = 0.04 * _rq1_s0_np.sin(_rq1_s0_np.arange(base.size, dtype=float) + 1.7 * index)
            starts.append(base + perturbation)
        return starts[:_rq1_s0__V2_Q_MIXTURE_RESTARTS]

    def _fit_mixture(self, dataframe: pd.DataFrame, child: str, parents: Sequence[str], n_components: int, *, prior_family: str) -> _rq1_s0_MixtureConditional:
        if n_components not in (2, 3):
            raise ValueError('Mixture fitting requires K=2 or K=3.')
        parent_tuple = tuple(parents)
        self.diagnostics['mixture_models_attempted'] += 1
        self.diagnostics[f'mixture_k{n_components}_models'] += 1
        feature_means, feature_stds, child_mean, child_std = self._saved_hybrid_continuous_scaling(child, parent_tuple)
        n = len(dataframe)
        if parent_tuple:
            raw_features = dataframe[list(parent_tuple)].to_numpy(dtype=float)
            features = (raw_features - feature_means) / feature_stds
            X = _rq1_s0_np.column_stack([_rq1_s0_np.ones(n), features])
        else:
            X = _rq1_s0_np.ones((n, 1), dtype=float)
        raw_y = dataframe[child].to_numpy(dtype=float)
        y = (raw_y - child_mean) / child_std
        _, p = X.shape
        H = _rq1_s0_helmert(n_components, full=False).T
        free_components = n_components - 1
        beta_size = n_components * p
        variance_size = n_components
        expert_variances, expert_precision, a0, b0 = self._hybrid_mixture_prior_values(prior_family=prior_family, p=p)
        expert_logdet = float(_rq1_s0_np.sum(_rq1_s0_np.log(expert_variances)))
        gate_variances = _rq1_s0_np.full((free_components, p), self.hybrid_prior_hyperparameters['mixture_gate_coefficient_prior_variance'], dtype=float)
        gate_variances[:, 0] = self.hybrid_prior_hyperparameters['mixture_gate_intercept_prior_variance']

        def unpack(theta: _rq1_s0_np.ndarray):
            theta = _rq1_s0_np.asarray(theta, dtype=float)
            betas = theta[:beta_size].reshape(n_components, p)
            start = beta_size
            log_variances = theta[start:start + variance_size]
            gamma = theta[start + variance_size:].reshape(free_components, p)
            return (betas, log_variances, gamma)
        cached_theta: _rq1_s0_np.ndarray | None = None
        cached_state: Any = None

        def state(theta: _rq1_s0_np.ndarray):
            nonlocal cached_theta, cached_state
            theta = _rq1_s0_np.asarray(theta, dtype=float)
            if cached_theta is not None and _rq1_s0_np.array_equal(theta, cached_theta):
                return cached_state
            betas, log_variances, gamma = unpack(theta)
            if _rq1_s0_np.any(_rq1_s0_np.abs(log_variances) > 30.0):
                raise FloatingPointError('V2 Q mixture log-variance left the stable range.')
            variances = _rq1_s0_np.exp(log_variances)
            gate_coefficients = H @ gamma
            logits = X @ gate_coefficients.T
            log_gate = logits - _rq1_s0_logsumexp(logits, axis=1, keepdims=True)
            means = X @ betas.T
            residuals = y[:, None] - means
            log_normal = -0.5 * (_rq1_s0_np.log(2.0 * _rq1_s0_np.pi) + log_variances[None, :] + residuals ** 2 / variances[None, :])
            component_log_joint = log_gate + log_normal
            log_density = _rq1_s0_logsumexp(component_log_joint, axis=1)
            responsibilities = _rq1_s0_np.exp(component_log_joint - log_density[:, None])
            gate_probabilities = _rq1_s0_np.exp(log_gate)
            result = (betas, log_variances, gamma, variances, gate_coefficients, residuals, log_density, responsibilities, gate_probabilities)
            cached_theta = theta.copy()
            cached_state = result
            return result

        def log_joint(theta: _rq1_s0_np.ndarray) -> float:
            betas, log_variances, gamma, variances, _, _, log_density, _, _ = state(theta)
            value = float(_rq1_s0_np.sum(log_density))
            beta_quadratic = _rq1_s0_np.sum(betas ** 2 * expert_precision[None, :], axis=1)
            value += float(_rq1_s0_np.sum(-0.5 * (p * _rq1_s0_np.log(2.0 * _rq1_s0_np.pi) + expert_logdet + p * log_variances + beta_quadratic / variances)))
            value += float(_rq1_s0_np.sum(a0 * _rq1_s0_np.log(b0) - _rq1_s0_gammaln(a0) - (a0 + 1.0) * log_variances - b0 / variances + log_variances))
            value += float(_rq1_s0_np.sum(-0.5 * (gamma ** 2 / gate_variances + _rq1_s0_np.log(2.0 * _rq1_s0_np.pi * gate_variances))))
            return value

        def objective(theta: _rq1_s0_np.ndarray) -> float:
            try:
                value = -log_joint(theta)
            except (FloatingPointError, OverflowError, ValueError):
                return 1e+100
            return float(value) if _rq1_s0_np.isfinite(value) else 1e+100

        def gradient(theta: _rq1_s0_np.ndarray) -> _rq1_s0_np.ndarray:
            betas, log_variances, gamma, variances, _, residuals, _, responsibilities, gate_probabilities = state(theta)
            del log_variances, gamma
            beta_gradient = _rq1_s0_np.empty_like(betas)
            log_variance_gradient = _rq1_s0_np.empty(n_components, dtype=float)
            for component in range(n_components):
                weights = responsibilities[:, component]
                beta_gradient[component] = -X.T @ (weights * residuals[:, component]) / variances[component] + betas[component] * expert_precision / variances[component]
                beta_quadratic = float(_rq1_s0_np.sum(betas[component] ** 2 * expert_precision))
                log_variance_gradient[component] = float(_rq1_s0_np.sum(weights * (0.5 - 0.5 * residuals[:, component] ** 2 / variances[component]))) + 0.5 * p + a0 - 0.5 * beta_quadratic / variances[component] - b0 / variances[component]
            gate_gradient = H.T @ ((gate_probabilities - responsibilities).T @ X)
            gate_gradient += unpack(theta)[2] / gate_variances
            result = _rq1_s0_np.concatenate([beta_gradient.ravel(), log_variance_gradient.ravel(), gate_gradient.ravel()])
            if not _rq1_s0_np.all(_rq1_s0_np.isfinite(result)):
                raise FloatingPointError('Non-finite V2 Q mixture MAP gradient.')
            return result
        starts = self._mixture_initializations(X, y, n_components)
        bounds = [(None, None)] * beta_size + [(-8.0, 4.0)] * variance_size + [(None, None)] * ((n_components - 1) * p)
        candidates: list[tuple[Any, float, float]] = []
        for initial in starts:
            result = _rq1_s0_minimize(objective, initial, method='L-BFGS-B', jac=gradient, bounds=bounds, options={'maxiter': _rq1_s0__V2_Q_MIXTURE_MAX_ITER, 'ftol': 1e-10, 'gtol': 1e-07, 'maxls': 50})
            if not _rq1_s0_np.isfinite(result.fun):
                continue
            theta = _rq1_s0_np.asarray(result.x, dtype=float)
            try:
                effective_counts = state(theta)[7].sum(axis=0)
                min_effective_count = float(_rq1_s0_np.min(effective_counts))
                grad_norm = float(_rq1_s0_np.linalg.norm(gradient(theta), ord=_rq1_s0_np.inf))
            except (FloatingPointError, OverflowError, ValueError):
                continue
            if _rq1_s0_np.isfinite(min_effective_count) and min_effective_count >= _rq1_s0__V2_Q_MIXTURE_MIN_EFFECTIVE_COUNT and _rq1_s0_np.isfinite(grad_norm):
                candidates.append((result, min_effective_count, grad_norm))
        if not candidates:
            raise FloatingPointError(f'All evaluation-time V2 MAP mixture fits failed the finite/interior checks for {child!r} | {parent_tuple}, K={n_components}.')
        best_result, min_effective_count, grad_norm = min(candidates, key=lambda item: float(item[0].fun))
        theta = _rq1_s0_np.asarray(best_result.x, dtype=float)
        betas, _, _, variances, gate_coefficients, _, _, _, _ = state(theta)
        converged = bool(best_result.success or grad_norm <= 0.001)
        self.diagnostics['mixture_models_fitted'] += 1
        if not converged:
            self.diagnostics['mixture_models_nonconverged'] += 1
        return _rq1_s0_MixtureConditional(child=child, parents=parent_tuple, n_components=int(n_components), feature_means=feature_means, feature_stds=feature_stds, child_mean=child_mean, child_std=child_std, betas=_rq1_s0_np.asarray(betas, dtype=float), variances=_rq1_s0_np.asarray(variances, dtype=float), gate_coefficients=_rq1_s0_np.asarray(gate_coefficients, dtype=float), converged=converged, iterations=int(getattr(best_result, 'nit', 0)), gradient_norm=grad_norm, min_effective_component_size=min_effective_count)

    @staticmethod
    def _eval_mixture(model: _rq1_s0_MixtureConditional, dataframe: pd.DataFrame) -> _rq1_s0_np.ndarray:
        n = len(dataframe)
        if model.parents:
            raw_features = dataframe[list(model.parents)].to_numpy(dtype=float)
            features = (raw_features - model.feature_means) / model.feature_stds
            X = _rq1_s0_np.column_stack([_rq1_s0_np.ones(n), features])
        else:
            X = _rq1_s0_np.ones((n, 1), dtype=float)
        raw_y = dataframe[model.child].to_numpy(dtype=float)
        y = (raw_y - model.child_mean) / model.child_std
        logits = X @ model.gate_coefficients.T
        log_gate = logits - _rq1_s0_logsumexp(logits, axis=1, keepdims=True)
        means = X @ model.betas.T
        log_components = -0.5 * (_rq1_s0_np.log(2.0 * _rq1_s0_np.pi) + _rq1_s0_np.log(model.variances)[None, :] + (y[:, None] - means) ** 2 / model.variances[None, :])
        return _rq1_s0_logsumexp(log_gate + log_components, axis=1) - _rq1_s0_np.log(model.child_std)

    def _eval_regime_v2_continuous(self, model: Mapping[str, Any], raw_data: pd.DataFrame, model_data: pd.DataFrame) -> _rq1_s0_np.ndarray:
        result = _rq1_s0_np.empty(len(raw_data), dtype=float)
        discrete_parents = list(model['discrete_parents'])
        key_matrix = model_data[discrete_parents].to_numpy(dtype=int)
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, row in enumerate(key_matrix):
            groups.setdefault(tuple((int(value) for value in row)), []).append(index)
        for key, indices_list in groups.items():
            indices = _rq1_s0_np.asarray(indices_list, dtype=int)
            local_model = model['regimes'].get(key, model['fallback'])
            local_data = raw_data.iloc[indices]
            if isinstance(local_model, _rq1_s0_GaussianConditional):
                values = self._eval_gaussian(local_model, local_data)
            elif isinstance(local_model, _rq1_s0_MixtureConditional):
                values = self._eval_mixture(local_model, local_data)
            else:
                raise ValueError(f"Unknown Hybrid V2 continuous Q model for {model['child']!r}.")
            result[indices] = values
        return result

    def fit(self, raw_training_data: pd.DataFrame) -> '_rq1_s0_LearnedDAGDistribution':
        raw_training_data = raw_training_data[self.nodes].copy().reset_index(drop=True)
        if not _rq1_s0_np.all(_rq1_s0_np.isfinite(raw_training_data.to_numpy(dtype=float))):
            raise ValueError('Training data contain non-finite values.')
        model_training_data = self.transformer.transform(raw_training_data) if self.transformer else raw_training_data.copy()
        self.models.clear()
        for child in self.nodes:
            parents = sorted((str(parent) for parent in self.graph.predecessors(child)))
            if self.paradigm == 'continuous':
                self.models[child] = self._fit_gaussian(raw_training_data, child, parents)
                continue
            if self.paradigm == 'discrete':
                cardinality = self.transformer.cardinalities[child]
                self.models[child] = self._fit_cpt(model_training_data, child, parents, cardinality)
                continue
            continuous_parents = [parent for parent in parents if parent in self.continuous_nodes]
            discrete_parents = [parent for parent in parents if parent in self.discrete_nodes]
            if child in self.continuous_nodes:
                if not discrete_parents:
                    selected_k = self._selected_components(child, ())
                    if selected_k == 1:
                        self.models[child] = self._fit_gaussian(raw_training_data, child, continuous_parents)
                    else:
                        self.models[child] = self._fit_mixture(raw_training_data, child, continuous_parents, selected_k, prior_family='continuous')
                    continue
                fallback = self._fit_gaussian(raw_training_data, child, continuous_parents)
                regimes: dict[tuple[int, ...], _rq1_s0_GaussianConditional | _rq1_s0_MixtureConditional] = {}
                group_key: Any = discrete_parents[0] if len(discrete_parents) == 1 else discrete_parents
                for key, group in model_training_data.groupby(group_key, sort=False, observed=True):
                    normalized_values = key if isinstance(key, tuple) else (key,)
                    normalized = self._configuration_key(normalized_values)
                    minimum_size = max(self.min_regime_size, len(continuous_parents) + 2)
                    if len(group) < minimum_size:
                        self.diagnostics['regime_fallbacks'] += 1
                        continue
                    selected_k = self._selected_components(child, normalized)
                    raw_group = raw_training_data.loc[group.index]
                    if selected_k == 1:
                        local_model: _rq1_s0_GaussianConditional | _rq1_s0_MixtureConditional = self._fit_gaussian(raw_group, child, continuous_parents)
                    else:
                        local_model = self._fit_mixture(raw_group, child, continuous_parents, selected_k, prior_family='clg')
                    regimes[normalized] = local_model
                self.models[child] = {'kind': 'regime_v2_continuous', 'child': child, 'discrete_parents': tuple(discrete_parents), 'regimes': regimes, 'fallback': fallback}
                continue
            cardinality = self.transformer.cardinalities[child]
            if not continuous_parents:
                self.models[child] = self._fit_cpt(model_training_data, child, discrete_parents, cardinality)
                continue
            global_prior = self._smoothed_probabilities(model_training_data[child], cardinality)
            fallback = self._fit_logistic(model_training_data, child, continuous_parents, cardinality, global_prior)
            regimes: dict[tuple[int, ...], _rq1_s0_LogisticConditional] = {}
            if discrete_parents:
                group_key = discrete_parents[0] if len(discrete_parents) == 1 else discrete_parents
                grouped = model_training_data.groupby(group_key, sort=False, observed=True)
            else:
                grouped = [((), model_training_data)]
            for key, group in grouped:
                if discrete_parents:
                    normalized_values = key if isinstance(key, tuple) else (key,)
                    normalized = self._configuration_key(normalized_values)
                else:
                    normalized = ()
                regimes[normalized] = self._fit_logistic(group, child, continuous_parents, cardinality, global_prior)
            self.models[child] = {'kind': 'regime_logistic', 'child': child, 'discrete_parents': tuple(discrete_parents), 'cardinality': cardinality, 'regimes': regimes, 'fallback': fallback}
        return self

    def _eval_regime_gaussian(self, model: Mapping[str, Any], raw_data: pd.DataFrame, model_data: pd.DataFrame) -> _rq1_s0_np.ndarray:
        result = _rq1_s0_np.empty(len(raw_data), dtype=float)
        discrete_parents = list(model['discrete_parents'])
        key_matrix = model_data[discrete_parents].to_numpy(dtype=int)
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, row in enumerate(key_matrix):
            groups.setdefault(tuple((int(value) for value in row)), []).append(index)
        for key, indices_list in groups.items():
            indices = _rq1_s0_np.asarray(indices_list, dtype=int)
            local_model = model['regimes'].get(key, model['fallback'])
            result[indices] = self._eval_gaussian(local_model, raw_data.iloc[indices])
        return result

    def _eval_regime_logistic(self, model: Mapping[str, Any], model_data: pd.DataFrame) -> _rq1_s0_np.ndarray:
        result = _rq1_s0_np.empty(len(model_data), dtype=float)
        child_values = model_data[model['child']].to_numpy(dtype=int)
        discrete_parents = list(model['discrete_parents'])
        if discrete_parents:
            key_matrix = model_data[discrete_parents].to_numpy(dtype=int)
        else:
            key_matrix = _rq1_s0_np.empty((len(model_data), 0), dtype=int)
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, row in enumerate(key_matrix):
            groups.setdefault(tuple((int(value) for value in row)), []).append(index)
        for key, indices_list in groups.items():
            indices = _rq1_s0_np.asarray(indices_list, dtype=int)
            local_model = model['regimes'].get(key, model['fallback'])
            result[indices] = self._eval_logistic(local_model, model_data, indices, child_values[indices])
        return result

    def joint_logscore(self, raw_evaluation_data: pd.DataFrame, *, return_nodewise: bool=False) -> _rq1_s0_np.ndarray | tuple[_rq1_s0_np.ndarray, dict[str, _rq1_s0_np.ndarray]]:
        if not self.models:
            raise RuntimeError('Fit Q before evaluating it.')
        raw_data = raw_evaluation_data[self.nodes].copy().reset_index(drop=True)
        model_data = self.transformer.transform(raw_data) if self.transformer else raw_data.copy()
        total = _rq1_s0_np.zeros(len(raw_data), dtype=float)
        nodewise: dict[str, _rq1_s0_np.ndarray] = {}
        for child in self.nodes:
            model = self.models[child]
            if isinstance(model, _rq1_s0_GaussianConditional):
                values = self._eval_gaussian(model, raw_data)
            elif isinstance(model, _rq1_s0_MixtureConditional):
                values = self._eval_mixture(model, raw_data)
            elif isinstance(model, _rq1_s0_CPTConditional):
                values = self._eval_cpt(model, model_data)
            elif isinstance(model, _rq1_s0_Mapping) and model['kind'] == 'regime_gaussian':
                values = self._eval_regime_gaussian(model, raw_data, model_data)
            elif isinstance(model, _rq1_s0_Mapping) and model['kind'] == 'regime_v2_continuous':
                values = self._eval_regime_v2_continuous(model, raw_data, model_data)
            elif isinstance(model, _rq1_s0_Mapping) and model['kind'] == 'regime_logistic':
                values = self._eval_regime_logistic(model, model_data)
            else:
                raise ValueError(f'Unknown fitted Q model for {child!r}.')
            if values.shape != (len(raw_data),) or not _rq1_s0_np.all(_rq1_s0_np.isfinite(values)):
                raise FloatingPointError(f'Invalid learned Q score for {child!r}.')
            nodewise[child] = values
            total += values
        if return_nodewise:
            return (total, nodewise)
        return total

class _rq1_s0_HybridV3LearnedDAGDistribution(_rq1_s0_LearnedDAGDistribution):
    """Established Hybrid V2 Q evaluator with corrected V3 predictive CC."""

    def __init__(self, *, corrected_cc_module, graph: nx.DiGraph, nodes: Sequence[str], discrete_nodes: Sequence[str], continuous_nodes: Sequence[str], transform: _rq1_s0_SavedQuantileTransform, dc_selected_k_by_configuration: Mapping[str, Mapping[str, int]], hybrid_continuous_scaling: Mapping[str, Any], hybrid_prior_hyperparameters: Mapping[str, float], cpt_alpha: float, min_regime_size: int, logistic_c: float) -> None:
        self.corrected_cc_module = corrected_cc_module
        self.cc_engine = None
        self.cc_children: set[str] = set()
        selected = {str(child): {str(k): int(v) for k, v in choices.items()} for child, choices in dc_selected_k_by_configuration.items()}
        for child in map(str, continuous_nodes):
            selected.setdefault(child, {'none': 1})
        super().__init__(graph=graph, nodes=nodes, paradigm='hybrid', discrete_nodes=discrete_nodes, continuous_nodes=continuous_nodes, transform=transform, cpt_alpha=cpt_alpha, min_regime_size=min_regime_size, logistic_c=logistic_c, hybrid_selected_k_by_configuration=selected, hybrid_continuous_scaling=hybrid_continuous_scaling, hybrid_prior_hyperparameters=hybrid_prior_hyperparameters)
        self.diagnostics['cc_v3_models'] = 0

    def fit(self, raw_training_data: pd.DataFrame) -> '_rq1_s0_HybridV3LearnedDAGDistribution':
        raw_training_data = raw_training_data[self.nodes].copy().reset_index(drop=True)
        if not _rq1_s0_np.all(_rq1_s0_np.isfinite(raw_training_data.to_numpy(dtype=float))):
            raise ValueError('Training data contain non-finite values.')
        model_training_data = self.transformer.transform(raw_training_data)
        self.models.clear()
        self.cc_children.clear()
        continuous_order = [n for n in self.nodes if n in self.continuous_nodes]
        needs_cc = any((child in self.continuous_nodes and (not any((p in self.discrete_nodes for p in self.graph.predecessors(child)))) for child in self.nodes))
        if needs_cc:
            if self.corrected_cc_module is None:
                raise RuntimeError('A corrected Final V3 module is required for final CC families.')
            self.cc_engine = self.corrected_cc_module.V3CScoringEngine(raw_training_data[continuous_order].copy())
        else:
            self.cc_engine = None
        for child in self.nodes:
            parents = sorted((str(p) for p in self.graph.predecessors(child)))
            continuous_parents = [p for p in parents if p in self.continuous_nodes]
            discrete_parents = [p for p in parents if p in self.discrete_nodes]
            if child in self.continuous_nodes and (not discrete_parents):
                self.cc_engine.score_local(child, continuous_parents)
                self.models[child] = {'kind': 'corrected_v3_cc', 'child': child, 'parents': tuple(continuous_parents)}
                self.cc_children.add(child)
                self.diagnostics['cc_v3_models'] += 1
                continue
            if child in self.continuous_nodes:
                fallback = self._fit_gaussian(raw_training_data, child, continuous_parents)
                regimes = {}
                group_key: Any = discrete_parents[0] if len(discrete_parents) == 1 else discrete_parents
                for key, group in model_training_data.groupby(group_key, sort=False, observed=True):
                    normalized_values = key if isinstance(key, tuple) else (key,)
                    normalized = self._configuration_key(normalized_values)
                    minimum_size = max(self.min_regime_size, len(continuous_parents) + 2)
                    if len(group) < minimum_size:
                        self.diagnostics['regime_fallbacks'] += 1
                        continue
                    selected_k = self._selected_components(child, normalized)
                    raw_group = raw_training_data.loc[group.index]
                    if selected_k == 1:
                        local_model = self._fit_gaussian(raw_group, child, continuous_parents)
                    else:
                        local_model = self._fit_mixture(raw_group, child, continuous_parents, selected_k, prior_family='clg')
                    regimes[normalized] = local_model
                self.models[child] = {'kind': 'regime_v2_continuous', 'child': child, 'discrete_parents': tuple(discrete_parents), 'regimes': regimes, 'fallback': fallback}
                continue
            cardinality = self.transformer.cardinalities[child]
            if not continuous_parents:
                self.models[child] = self._fit_cpt(model_training_data, child, discrete_parents, cardinality)
                continue
            global_prior = self._smoothed_probabilities(model_training_data[child], cardinality)
            fallback = self._fit_logistic(model_training_data, child, continuous_parents, cardinality, global_prior)
            regimes = {}
            if discrete_parents:
                group_key = discrete_parents[0] if len(discrete_parents) == 1 else discrete_parents
                grouped = model_training_data.groupby(group_key, sort=False, observed=True)
            else:
                grouped = [((), model_training_data)]
            for key, group in grouped:
                if discrete_parents:
                    normalized_values = key if isinstance(key, tuple) else (key,)
                    normalized = self._configuration_key(normalized_values)
                else:
                    normalized = ()
                regimes[normalized] = self._fit_logistic(group, child, continuous_parents, cardinality, global_prior)
            self.models[child] = {'kind': 'regime_logistic', 'child': child, 'discrete_parents': tuple(discrete_parents), 'cardinality': cardinality, 'regimes': regimes, 'fallback': fallback}
        return self

    def joint_logscore(self, raw_evaluation_data: pd.DataFrame, *, return_nodewise: bool=False):
        if not self.models:
            raise RuntimeError('Fit Q before evaluating it.')
        raw_data = raw_evaluation_data[self.nodes].copy().reset_index(drop=True)
        model_data = self.transformer.transform(raw_data)
        total = _rq1_s0_np.zeros(len(raw_data), dtype=float)
        nodewise: dict[str, _rq1_s0_np.ndarray] = {}
        continuous_order = [n for n in self.nodes if n in self.continuous_nodes]
        cc_eval = raw_data[continuous_order]
        for child in _rq1_s0_nx.topological_sort(self.graph):
            child = str(child)
            model = self.models[child]
            if isinstance(model, _rq1_s0_GaussianConditional):
                values = self._eval_gaussian(model, raw_data)
            elif isinstance(model, _rq1_s0_MixtureConditional):
                values = self._eval_mixture(model, raw_data)
            elif isinstance(model, _rq1_s0_CPTConditional):
                values = self._eval_cpt(model, model_data)
            elif isinstance(model, _rq1_s0_Mapping) and model['kind'] == 'corrected_v3_cc':
                values = self.cc_engine.local_predictive_logpdf(child, model['parents'], cc_eval)
            elif isinstance(model, _rq1_s0_Mapping) and model['kind'] == 'regime_v2_continuous':
                values = self._eval_regime_v2_continuous(model, raw_data, model_data)
            elif isinstance(model, _rq1_s0_Mapping) and model['kind'] == 'regime_logistic':
                values = self._eval_regime_logistic(model, model_data)
            else:
                raise ValueError(f'Unknown fitted Q model for {child!r}.')
            values = _rq1_s0_np.asarray(values, dtype=float)
            if values.shape != (len(raw_data),) or not _rq1_s0_np.all(_rq1_s0_np.isfinite(values)):
                raise FloatingPointError(f'Invalid learned Q score for {child!r}.')
            nodewise[child] = values
            total += values
        return (total, nodewise) if return_nodewise else total

def _rq1_s0__frozen_prior_hyperparameters_from_scorer(scorer) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if getattr(scorer, 'dd', None) is not None:
        out['equivalent_sample_size'] = float(scorer.dd.equivalent_sample_size)
    if getattr(scorer, 'cd', None) is not None:
        out.update({'softmax_coefficient_prior_variance': float(scorer.cd.coefficient_prior_variance), 'softmax_intercept_prior_variance': float(scorer.cd.intercept_prior_variance), 'logistic_max_iter': int(scorer.cd.max_iter)})
    if getattr(scorer, 'dc', None) is not None:
        dc = scorer.dc
        for name in ('clg_coefficient_prior_variance', 'clg_intercept_prior_variance', 'clg_inverse_gamma_shape', 'clg_inverse_gamma_scale', 'continuous_coefficient_prior_variance', 'continuous_intercept_prior_variance', 'continuous_inverse_gamma_shape', 'continuous_inverse_gamma_scale', 'mixture_gate_coefficient_prior_variance', 'mixture_gate_intercept_prior_variance'):
            out[name] = float(getattr(dc, name))
        for name in ('mixture_max_components', 'mixture_max_iter', 'mixture_restarts'):
            out[name] = int(getattr(dc, name))
        out['mixture_hessian_step'] = float(dc.mixture_hessian_step)
        out['mixture_min_effective_count'] = float(dc.mixture_min_effective_count)
    return out

def _rq1_s0_reconstruct_final_evaluation_metadata(*, model_module, cc_scorer_file: Path, raw_training: pd.DataFrame, transformer: _rq1_s0_SavedQuantileTransform, learned_graph: nx.DiGraph, nodes: Sequence[str], discrete_nodes: Sequence[str], continuous_nodes: Sequence[str], cardinalities: Mapping[str, int], saved_final_score: float | None, score_tolerance: float) -> dict[str, Any]:
    """Rebuild metadata for pre-KL-ready 02 cells and parity-gate it."""
    hybrid_training = transformer.transform(raw_training)
    scorer = model_module.HybridScoringEngine(hybrid_training.copy(), discrete_nodes, continuous_nodes, cardinalities, cc_scorer_file=cc_scorer_file)
    total = 0.0
    local_rows = []
    for child in nodes:
        parents = tuple(sorted(map(str, learned_graph.predecessors(child))))
        total += float(scorer.score_local(str(child), parents))
        local_rows.append(_rq1_s0_asdict(scorer.get_score_info(str(child), parents)))
    if saved_final_score is not None:
        diff = abs(float(total) - float(saved_final_score))
        if diff > score_tolerance:
            raise AssertionError(f'Reconstructed training/scorer does not reproduce the saved final DAG score: reconstructed={total:.12f}, saved={saved_final_score:.12f}, abs_diff={diff:.3e} > {score_tolerance:.3e}. Do not compute KL from a non-identical reconstruction.')
    else:
        diff = None
    dc_diag = scorer.mixture_diagnostics_dataframe().to_dict(orient='records') if getattr(scorer, 'dc', None) is not None else []
    cd_diag = scorer.laplace_diagnostics_dataframe().to_dict(orient='records') if getattr(scorer, 'cd', None) is not None else []
    selected_k: dict[str, dict[str, int]] = {}
    for row in dc_diag:
        if bool(row.get('selected', False)):
            selected_k.setdefault(str(row['child']), {})[_rq1_s0__configuration_storage_key(row.get('discrete_configuration'))] = int(row['n_components'])
    return {'schema_version': 'hybrid-v3-predictive-q-metadata-v1-reconstructed', 'metadata_source': 'reconstructed_from_pre_kl_ready_structure_cell', 'reconstructed_final_dag_score': float(total), 'saved_final_dag_score': saved_final_score, 'reconstruction_score_abs_diff': diff, 'hybrid_continuous_scaling': scorer.preprocessing_metadata(), 'frozen_prior_hyperparameters': _rq1_s0__frozen_prior_hyperparameters_from_scorer(scorer), 'final_local_score_info': local_rows, 'dc_selected_k_by_configuration': selected_k, 'dc_mixture_diagnostics': dc_diag, 'cd_laplace_diagnostics': cd_diag}

def _rq1_s0__load_manifest_base_sample_size(structure_json: Path) -> int | None:
    for parent in structure_json.parents:
        manifest = parent / 'manifest.json'
        if not manifest.exists():
            continue
        try:
            payload = _rq1_s0_json.loads(manifest.read_text(encoding='utf-8'))
        except Exception:
            continue
        sizes = payload.get('experiment_scope', {}).get('sample_sizes')
        if sizes:
            return int(max(map(int, sizes)))
    return None

def _rq1_s0_reconstruct_generator_payload_and_training(*, root: Path, cell: Mapping[str, Any], structure_json: Path, explicit_base_sample_size: int | None) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    definition = cell.get('experiment_definition', {})
    generator_name = str(definition['generator'])
    seed = int(definition['seed'])
    n = int(definition['sample_size'])
    base_n = explicit_base_sample_size or definition.get('training_base_sample_size') or cell.get('settings', {}).get('training_base_sample_size') or _rq1_s0__load_manifest_base_sample_size(structure_json) or n
    base_n = int(base_n)
    if base_n < n:
        raise ValueError(f'Reconstruction base sample size {base_n} < cell n={n}.')
    Generator = _rq1_s0_import_generator(root, generator_name)
    generator = Generator(num_samples=base_n, seed=seed)
    generator.generate_dag()
    base_data = generator.generate_data().reset_index(drop=True)
    payload = generator.get_payload()
    nodes = list(map(str, payload['nodes']))
    raw_training = base_data[nodes].iloc[:n].copy().reset_index(drop=True)
    expected_hash = cell.get('dataset_sha256')
    if expected_hash:
        actual_hash = _rq1_s0_structure_dataframe_sha256(raw_training)
        if actual_hash != expected_hash:
            raise ValueError(f'Deterministic training-data reconstruction failed the structure-cell hash check (base_n={base_n}). For an older multi-size run, pass --reconstruction-base-sample-size equal to the maximum n used by that 02 invocation, or rerun the cell with the updated KL-ready 02.')
    return (raw_training, payload)

def _rq1_s0_load_cell_inputs(*, root: Path, cell: _rq1_s0_Mapping[str, Any], structure_json: Path, explicit_base_sample_size: int | None) -> tuple[_rq1_s0_pd.DataFrame, _rq1_s0_Mapping[str, Any], str]:
    raw_meta = cell.get('raw_training_data', {})
    records = raw_meta.get('records') if isinstance(raw_meta, _rq1_s0_Mapping) else None
    ground_truth = cell.get('ground_truth')
    if records is not None and ground_truth and ground_truth.get('true_parameters'):
        columns = list(map(str, raw_meta['column_order']))
        raw = _rq1_s0_pd.DataFrame.from_records(records, columns=columns).reset_index(drop=True)
        expected_hash = raw_meta.get('sha256') or cell.get('dataset_sha256')
        if expected_hash and _rq1_s0_structure_dataframe_sha256(raw) != expected_hash:
            raise ValueError('Embedded raw training records failed their saved hash check.')
        payload = {'nodes': list(map(str, ground_truth['nodes'])), 'discrete_nodes': list(map(str, ground_truth['discrete_nodes'])), 'continuous_nodes': list(map(str, ground_truth['continuous_nodes'])), 'binary_nodes': list(map(str, ground_truth.get('binary_nodes', []))), 'multiclass_nodes': list(map(str, ground_truth.get('multiclass_nodes', []))), 'node_regimes': ground_truth.get('node_regimes', {}), 'true_parameters': ground_truth['true_parameters'], 'graph': _rq1_s0_build_graph(ground_truth['nodes'], ground_truth['true_edges'])}
        return (raw, payload, 'embedded_kl_ready_cell')
    raw, payload = _rq1_s0_reconstruct_generator_payload_and_training(root=root, cell=cell, structure_json=structure_json, explicit_base_sample_size=explicit_base_sample_size)
    return (raw, payload, 'deterministic_generator_reconstruction')

def _rq1_s0_discover_structure_files(structure_dir: Path, *, generators: set[str] | None, seeds: set[int] | None, sample_sizes: set[int] | None) -> list[Path]:
    files = sorted((structure_dir / 'runs').rglob('hybrid_v3_structure.json'))
    selected = []
    for path in files:
        try:
            cell = _rq1_s0_json.loads(path.read_text(encoding='utf-8'))
            d = cell['experiment_definition']
            g, s, n = (str(d['generator']), int(d['seed']), int(d['sample_size']))
        except Exception:
            continue
        if generators is not None and g not in generators:
            continue
        if seeds is not None and s not in seeds:
            continue
        if sample_sizes is not None and n not in sample_sizes:
            continue
        selected.append(path)
    return selected

def _rq1_s0_output_path(output_dir: Path, generator: str, seed: int, n: int) -> Path:
    return output_dir / 'runs' / generator / f'seed_{seed:02d}' / f'n_{n:04d}' / 'hybrid_v3_kl.json'

def _rq1_s0_evaluate_cell(*, root: Path, structure_json: Path, cell: _rq1_s0_Mapping[str, Any], mc_samples: int, cpt_alpha: float, min_regime_size: int, logistic_c: float, reconstruction_base_sample_size: int | None, metadata_score_tolerance: float, particles: int=256, particle_replicates: int=2, particle_batch_size: int=64) -> dict[str, Any]:
    definition = cell['experiment_definition']
    generator_name = str(definition['generator'])
    seed = int(definition['seed'])
    n = int(definition['sample_size'])
    raw_training, payload, training_source = _rq1_s0_load_cell_inputs(root=root, cell=cell, structure_json=structure_json, explicit_base_sample_size=reconstruction_base_sample_size)
    nodes = list(map(str, payload['nodes']))
    discrete_nodes = list(map(str, payload['discrete_nodes']))
    continuous_nodes = list(map(str, payload['continuous_nodes']))
    cardinalities = _rq1_s0_default_cardinalities(payload)
    true_graph = payload['graph'].copy()
    learned_graph = _rq1_s0_build_graph(nodes, cell['learned_edges'])
    transformer = _rq1_s0_SavedQuantileTransform.from_metadata(cell['discretizer_metadata'])
    model_file = (root / 'src' / 'hybrid_bn' / 'SLhybrid.py').resolve()
    cc_file = (root / 'src' / 'hybrid_bn' / 'SLhybrid.py').resolve()
    model_module = _rq1_inline_load(model_file, f'_hv3_kl_model_{seed}_{n}_{_rq1_s0_os.getpid()}')
    evaluation_metadata = cell.get('evaluation_metadata')
    metadata_source = 'saved_kl_ready_structure_cell'
    if not isinstance(evaluation_metadata, _rq1_s0_Mapping):
        saved_score = cell.get('learned_dag_score_recomputed_main')
        evaluation_metadata = _rq1_s0_reconstruct_final_evaluation_metadata(model_module=model_module, cc_scorer_file=cc_file, raw_training=raw_training, transformer=transformer, learned_graph=learned_graph, nodes=nodes, discrete_nodes=discrete_nodes, continuous_nodes=continuous_nodes, cardinalities=cardinalities, saved_final_score=float(saved_score) if saved_score is not None else None, score_tolerance=metadata_score_tolerance)
        metadata_source = 'reconstructed_and_final_score_parity_checked'
    selected_k = evaluation_metadata.get('dc_selected_k_by_configuration', {})
    scaling = evaluation_metadata.get('hybrid_continuous_scaling', {})
    priors = evaluation_metadata.get('frozen_prior_hyperparameters', {})
    discrete_set = set(discrete_nodes)
    final_quadrants = {}
    for child in nodes:
        parents = list(map(str, learned_graph.predecessors(child)))
        d_parents = [p for p in parents if p in discrete_set]
        c_parents = [p for p in parents if p not in discrete_set]
        if child in discrete_set:
            quadrant = 'CD' if c_parents else 'DD'
        else:
            quadrant = 'DC' if d_parents else 'CC'
        final_quadrants[child] = quadrant
        if quadrant == 'DC' and child not in selected_k:
            raise ValueError(f'Missing DC selected-K metadata for final child {child!r}.')
    cc_module = None
    if any((q == 'CC' for q in final_quadrants.values())):
        cc_module = _rq1_inline_load(cc_file, f'_hv3_kl_cc_{seed}_{n}_{_rq1_s0_os.getpid()}')
    true_model = _rq1_s0_TrueContinuousDAG(true_graph, payload['true_parameters'], nodes)
    mc_seed = _rq1_s0_deterministic_mc_seed(generator_name, seed)
    evaluation_sample = true_model.sample(mc_samples, mc_seed)
    truth_estimator = _rq1_s0_TransformedTruthParticleLikelihood(true_model, transformer, discrete_nodes, particles=particles, replicates=particle_replicates, batch_size=particle_batch_size)
    identity_check = truth_estimator.validate_continuous_identity(evaluation_sample.iloc[:min(128, len(evaluation_sample))].copy())
    if identity_check > 1e-10:
        raise AssertionError(f'Continuous special-case identity check failed: {identity_check:.3e}')
    log_p, particle_diagnostics = truth_estimator.loglikelihood(evaluation_sample, mc_seed)
    q = _rq1_s0_HybridV3LearnedDAGDistribution(corrected_cc_module=cc_module, graph=learned_graph, nodes=nodes, discrete_nodes=discrete_nodes, continuous_nodes=continuous_nodes, transform=transformer, dc_selected_k_by_configuration=selected_k, hybrid_continuous_scaling=scaling, hybrid_prior_hyperparameters=priors, cpt_alpha=cpt_alpha, min_regime_size=min_regime_size, logistic_c=logistic_c).fit(raw_training)
    log_q, q_nodewise = q.joint_logscore(evaluation_sample, return_nodewise=True)
    ratio = log_p - log_q
    estimate = float(_rq1_s0_np.mean(ratio))
    se = float(_rq1_s0_np.std(ratio, ddof=1) / _rq1_s0_np.sqrt(len(ratio)))
    node_details = [{'node': node, 'final_quadrant': final_quadrants[node], 'kl_contribution': None, 'mean_log_p': None, 'mean_log_q': float(_rq1_s0_np.mean(q_nodewise[node])), 'note': 'undefined: transformed P_T need not factorise over original DAG'} for node in nodes]
    structure = cell.get('structure_metrics', {})
    return {'schema_version': _rq1_s0_KL_SCHEMA_VERSION, 'status': 'complete', 'source_structure_json': str(structure_json), 'experiment_definition': {'generator': generator_name, 'seed': seed, 'sample_size': n}, 'input_reconstruction': {'training_data_source': training_source, 'evaluation_metadata_source': metadata_source}, 'structure_metrics': structure, 'kl': {'estimate': estimate, 'standard_error': se, 'lower_95': float(estimate - 1.96 * se), 'upper_95': float(estimate + 1.96 * se), 'mean_log_p': float(_rq1_s0_np.mean(log_p)), 'mean_log_q': float(_rq1_s0_np.mean(log_q)), 'mc_samples': int(mc_samples), 'mc_seed': int(mc_seed), 'materially_negative': bool(estimate < -3.0 * se), 'definition': 'mean(log P_T(T(x)) - log Q(T(x))) for fresh x~P', 'true_transformed_density': 'joint p_T(x_C,d_D) estimated by SMC marginalisation of latent raw values inside saved training-quantile intervals', 'dequantisation': False, 'particle_likelihood': particle_diagnostics, 'continuous_identity_max_abs_diff': float(identity_check), 'nodewise_kl_decomposition_available': False}, 'node_contributions': node_details, 'q_fit_diagnostics': dict(q.diagnostics), 'final_family_quadrants': final_quadrants, 'evaluation_settings': {'cpt_alpha': float(cpt_alpha), 'min_regime_size': int(min_regime_size), 'logistic_c': float(logistic_c), 'cc_q': 'corrected Final V3 posterior predictive', 'dd_q': 'additive-smoothed CPT', 'cd_q': 'near-MLE multinomial logistic, V2 evaluator convention', 'dc_q': 'saved-K Gaussian/MAP MoE, V2 evaluator convention', 'p_side': 'representation-correct SMC marginal transformed truth', 'particles': int(particles), 'particle_replicates': int(particle_replicates), 'particle_batch_size': int(particle_batch_size)}}

def _rq1_s0_rebuild_csvs(output_dir: Path) -> tuple[_rq1_s0_pd.DataFrame, _rq1_s0_pd.DataFrame, _rq1_s0_pd.DataFrame]:
    rows = []
    node_rows = []
    for path in sorted((output_dir / 'runs').rglob('hybrid_v3_kl.json')):
        try:
            r = _rq1_s0_json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if r.get('schema_version') != _rq1_s0_KL_SCHEMA_VERSION or r.get('status') != 'complete':
            continue
        d = r['experiment_definition']
        k = r['kl']
        s = r.get('structure_metrics', {})
        rows.append({'Generator': d['generator'], 'Seed': int(d['seed']), 'Sample_Size': int(d['sample_size']), 'SHD': s.get('shd'), 'Directed_Precision': s.get('directed_precision'), 'Directed_Recall': s.get('directed_recall'), 'Directed_F1': s.get('directed_f1'), 'Learned_Edges': s.get('learned_edge_count'), 'KL_Estimate': k['estimate'], 'KL_Standard_Error': k['standard_error'], 'KL_CI_Lower': k['lower_95'], 'KL_CI_Upper': k['upper_95'], 'Mean_Log_P': k['mean_log_p'], 'Mean_Log_Q': k['mean_log_q'], 'KL_MC_Samples': k['mc_samples'], 'KL_MC_Seed': k['mc_seed'], 'KL_Materially_Negative': k['materially_negative'], 'Training_Data_Source': r['input_reconstruction']['training_data_source'], 'Evaluation_Metadata_Source': r['input_reconstruction']['evaluation_metadata_source'], 'Result_JSON': str(path)})
        for nr in r.get('node_contributions', []):
            node_rows.append({'Generator': d['generator'], 'Seed': int(d['seed']), 'Sample_Size': int(d['sample_size']), 'Node': nr['node'], 'Final_Quadrant': nr['final_quadrant'], 'KL_Contribution': nr.get('kl_contribution'), 'Mean_Log_P': nr.get('mean_log_p'), 'Mean_Log_Q': nr.get('mean_log_q'), 'Note': nr.get('note', '')})
    runs = _rq1_s0_pd.DataFrame(rows)
    nodes = _rq1_s0_pd.DataFrame(node_rows)
    if not runs.empty:
        runs = runs.sort_values(['Generator', 'Seed', 'Sample_Size']).reset_index(drop=True)
    if not nodes.empty:
        nodes = nodes.sort_values(['Generator', 'Seed', 'Sample_Size', 'Node']).reset_index(drop=True)
    summary_rows = []
    if not runs.empty:
        for (g, n), group in runs.groupby(['Generator', 'Sample_Size'], sort=True):
            vals = group['KL_Estimate'].astype(float)
            summary_rows.append({'Generator': g, 'Sample_Size': int(n), 'Runs': int(len(group)), 'KL_Mean': float(vals.mean()), 'KL_Std': float(vals.std(ddof=1)) if len(vals) > 1 else _rq1_s0_np.nan, 'SHD_Mean': float(_rq1_s0_pd.to_numeric(group['SHD'], errors='coerce').mean()), 'Directed_F1_Mean': float(_rq1_s0_pd.to_numeric(group['Directed_F1'], errors='coerce').mean())})
    summary = _rq1_s0_pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / 'hybrid_v3_kl_results.csv', index=False)
    nodes.to_csv(output_dir / 'hybrid_v3_kl_node_contributions.csv', index=False)
    nodes.to_csv(output_dir / 'hybrid_v3_q_nodewise_diagnostics.csv', index=False)
    summary.to_csv(output_dir / 'hybrid_v3_kl_summary.csv', index=False)
    return (runs, nodes, summary)

def _rq1_s0_parse_args() -> _rq1_s0_argparse.Namespace:
    p = _rq1_s0_argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=_rq1_s0_Path, default=None)
    p.add_argument('--structure-results-dir', type=_rq1_s0_Path, default=None)
    p.add_argument('--output-dir', type=_rq1_s0_Path, default=None)
    p.add_argument('--generators', nargs='+', choices=tuple(_rq1_s0_GENERATOR_SPECS), default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    p.add_argument('--sample-sizes', nargs='+', type=int, default=None)
    p.add_argument('--mc-samples', type=int, default=5000)
    p.add_argument('--cpt-alpha', type=float, default=1.0)
    p.add_argument('--min-regime-size', type=int, default=5)
    p.add_argument('--logistic-c', type=float, default=1000.0)
    p.add_argument('--metadata-score-tolerance', type=float, default=1e-06)
    p.add_argument('--particles', type=int, default=256, help='Particles per transformed-truth likelihood replicate.')
    p.add_argument('--particle-replicates', type=int, default=2, help='Independent particle likelihood replicates.')
    p.add_argument('--particle-batch-size', type=int, default=64, help='Fresh rows processed together by the particle estimator.')
    p.add_argument('--reconstruction-base-sample-size', type=int, default=None, help='Only needed for older 02 cells when the original invocation used multiple sample sizes.')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--stop-on-error', action='store_true')
    args = p.parse_args()
    if args.mc_samples < 2:
        p.error('--mc-samples must be >=2')
    if args.cpt_alpha <= 0 or args.min_regime_size < 1 or args.logistic_c <= 0:
        p.error('Invalid Q-fit settings.')
    if args.metadata_score_tolerance <= 0:
        p.error('--metadata-score-tolerance must be positive.')
    if args.particles < 8 or args.particle_replicates < 1 or args.particle_batch_size < 1:
        p.error('Invalid particle settings.')
    return args

def _rq1_s0_main() -> int:
    args = _rq1_s0_parse_args()
    here = _rq1_s0_Path(__file__).resolve().parent
    root = _rq1_s0_discover_root(here, args.project_root)
    if args.structure_results_dir is None:
        preferred = root / 'Outputs' / 'HybridV3_StructureLearning_NoAdd2'
        fallback = root / 'Outputs' / 'HybridV3_StructureLearning'
        structure_dir = preferred if preferred.exists() else fallback
    else:
        structure_dir = args.structure_results_dir.expanduser().resolve() if args.structure_results_dir.is_absolute() else (root / args.structure_results_dir).resolve()
    if not structure_dir.exists():
        raise FileNotFoundError(structure_dir)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None and args.output_dir.is_absolute() else (root / args.output_dir).resolve() if args.output_dir is not None else root / 'Outputs' / 'HybridV3_KL_RepresentationCorrect'
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _rq1_s0_discover_structure_files(structure_dir, generators=set(args.generators) if args.generators else None, seeds=set(args.seeds) if args.seeds else None, sample_sizes=set(args.sample_sizes) if args.sample_sizes else None)
    if not files:
        raise RuntimeError(f'No matching hybrid_v3_structure.json files under {structure_dir}')
    print('=' * 124)
    print('HYBRID V3 — REPRESENTATION-CORRECT FRESH-P KL')
    print('=' * 124)
    print(f'Project root:      {root}')
    print(f'Structure results: {structure_dir}')
    print(f'KL output:         {output_dir}')
    print(f'Cells:             {len(files)}')
    print(f'MC samples:        {args.mc_samples}')
    print('P: joint transformed truth via latent-bin particle marginalisation')
    print(f'Particles:          {args.particles} x {args.particle_replicates} replicates | batch={args.particle_batch_size}')
    print('Q: DD CPT | CD logistic | DC saved-K V2 predictive | CC corrected V3 predictive')
    print('=' * 124, flush=True)
    completed = skipped = failed = 0
    started = _rq1_s0_time.perf_counter()
    failure_path = output_dir / 'failures.jsonl'
    for index, path in enumerate(files, 1):
        cell = _rq1_s0_json.loads(path.read_text(encoding='utf-8'))
        d = cell['experiment_definition']
        g, seed, n = (str(d['generator']), int(d['seed']), int(d['sample_size']))
        out = _rq1_s0_output_path(output_dir, g, seed, n)
        print(f'\n[{index}/{len(files)}] {g} seed={seed} n={n}', flush=True)
        if out.exists() and (not args.overwrite):
            try:
                old = _rq1_s0_json.loads(out.read_text(encoding='utf-8'))
                if old.get('schema_version') == _rq1_s0_KL_SCHEMA_VERSION and old.get('status') == 'complete' and (int(old.get('kl', {}).get('mc_samples', -1)) == args.mc_samples) and (int(old.get('evaluation_settings', {}).get('particles', -1)) == args.particles) and (int(old.get('evaluation_settings', {}).get('particle_replicates', -1)) == args.particle_replicates) and (int(old.get('evaluation_settings', {}).get('particle_batch_size', -1)) == args.particle_batch_size):
                    skipped += 1
                    print('    matching KL result exists; skipping.', flush=True)
                    continue
            except Exception:
                pass
        try:
            cell_started = _rq1_s0_time.perf_counter()
            result = _rq1_s0_evaluate_cell(root=root, structure_json=path, cell=cell, mc_samples=int(args.mc_samples), cpt_alpha=float(args.cpt_alpha), min_regime_size=int(args.min_regime_size), logistic_c=float(args.logistic_c), reconstruction_base_sample_size=args.reconstruction_base_sample_size, metadata_score_tolerance=float(args.metadata_score_tolerance), particles=int(args.particles), particle_replicates=int(args.particle_replicates), particle_batch_size=int(args.particle_batch_size))
            result['runtime_seconds'] = float(_rq1_s0_time.perf_counter() - cell_started)
            _rq1_s0_atomic_write_json(out, result)
            completed += 1
            k = result['kl']
            s = result.get('structure_metrics', {})
            print(f"    KL={k['estimate']:.6f} ± {1.96 * k['standard_error']:.6f} | SHD={s.get('shd', 'NA')} | F1={s.get('directed_f1', 'NA')} | metadata={result['input_reconstruction']['evaluation_metadata_source']} | runtime={result['runtime_seconds']:.1f}s", flush=True)
        except Exception as exc:
            failed += 1
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with failure_path.open('a', encoding='utf-8') as f:
                f.write(_rq1_s0_json.dumps(_rq1_s0_json_safe({'source': str(path), 'generator': g, 'seed': seed, 'sample_size': n, 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': _rq1_s0_traceback.format_exc()})) + '\n')
            print(f'    FAILED: {type(exc).__name__}: {exc}', flush=True)
            if args.stop_on_error:
                _rq1_s0_rebuild_csvs(output_dir)
                raise
        finally:
            _rq1_s0_gc.collect()
        _rq1_s0_rebuild_csvs(output_dir)
    runs, _, _ = _rq1_s0_rebuild_csvs(output_dir)
    print('\n' + '=' * 124)
    print('HYBRID V3 KL COMPLETE')
    print('=' * 124)
    print(f'Completed now: {completed}')
    print(f'Skipped existing: {skipped}')
    print(f'Failed now: {failed}')
    print(f'Rows available: {len(runs)}')
    if not runs.empty:
        print(f"Mean KL={runs['KL_Estimate'].mean():.6f}")
    print(f"Results: {output_dir / 'hybrid_v3_kl_results.csv'}")
    print(f"Summary: {output_dir / 'hybrid_v3_kl_summary.csv'}")
    print(f"Node contributions: {output_dir / 'hybrid_v3_kl_node_contributions.csv'}")
    print(f'Elapsed: {_rq1_s0_time.perf_counter() - started:.1f}s')
    print('=' * 124)
    return 0 if failed == 0 else 1


# ======================================================================
# Evaluation implementation: 15_discrete_baseline_representation_correct_kl.py
# ======================================================================
import argparse as _rq1_s1_argparse
import importlib.util as _rq1_s1_importlib
import json as _rq1_s1_json
import math as _rq1_s1_math
import os as _rq1_s1_os
import sys as _rq1_s1_sys
import time as _rq1_s1_time
import traceback as _rq1_s1_traceback
from pathlib import Path as _rq1_s1_Path
from typing import Any as _rq1_s1_Any, Iterable as _rq1_s1_Iterable, Mapping as _rq1_s1_Mapping, Sequence as _rq1_s1_Sequence
_rq1_s1_sys.dont_write_bytecode = True
import networkx as _rq1_s1_nx
import numpy as _rq1_s1_np
import pandas as _rq1_s1_pd
_rq1_s1_GENERATOR_SPECS = {'unimodal': ('generateUnimodalGT.py', 'StochasticContinuousGenerator'), 'skew': ('generateSkewGT.py', 'StochasticContinuousSkewedGenerator'), 'multimodal': ('generateMultimodalGT.py', 'StochasticContinuousMultimodalGenerator'), 'skew_multi': ('generateSkewMultiGT.py', 'StochasticContinuousSkewedMultimodalGenerator')}
_rq1_s1_SCHEMA_VERSION = 'discrete-baseline-representation-correct-kl-v1'
_rq1_s1__MIN_PROB = 1e-300

def _rq1_s1_json_safe(value: Any) -> Any:
    if isinstance(value, _rq1_s1_Mapping):
        return {str(k): _rq1_s1_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_rq1_s1_json_safe(v) for v in value]
    if isinstance(value, _rq1_s1_np.ndarray):
        return _rq1_s1_json_safe(value.tolist())
    if isinstance(value, _rq1_s1_np.generic):
        return _rq1_s1_json_safe(value.item())
    if isinstance(value, _rq1_s1_Path):
        return str(value)
    if isinstance(value, float):
        if _rq1_s1_math.isnan(value):
            return 'NaN'
        if _rq1_s1_math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
    return value

def _rq1_s1_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(_rq1_s1_json.dumps(_rq1_s1_json_safe(payload), indent=2), encoding='utf-8')
    _rq1_s1_os.replace(tmp, path)

def _rq1_s1_discover_root(here: _rq1_s1_Path, explicit: _rq1_s1_Path | None) -> _rq1_s1_Path:
    return _rq1_find_root()

def _rq1_s1_find_corrected_kl_script(root: Path, here: Path, explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    candidates.extend([here / '07_v2_hybrid_v3_representation_correct_kl.py', root / 'experiments' / 'rq1' / '07_v2_hybrid_v3_representation_correct_kl.py'])
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError('Could not find 07_v2_hybrid_v3_representation_correct_kl.py. Place this script beside 07_v2 or pass --corrected-kl-script.')

def _rq1_s1_import_generator(root: Path, generator: str):
    filename, class_name = _rq1_s1_GENERATOR_SPECS[generator]
    module = _rq1_inline_load(root / 'src' / 'generators' / filename, f'_disc_rc_generator_{generator}_{_rq1_s1_os.getpid()}')
    return getattr(module, class_name)

def _rq1_s1_graph_from_edges(nodes: Sequence[str], edges: Iterable[Sequence[str]]) -> _rq1_s1_nx.DiGraph:
    g = _rq1_s1_nx.DiGraph()
    g.add_nodes_from(map(str, nodes))
    g.add_edges_from(((str(u), str(v)) for u, v in edges))
    if not _rq1_s1_nx.is_directed_acyclic_graph(g):
        raise ValueError('Learned graph is not a DAG.')
    return g

def _rq1_s1_structure_metrics(true_edges: Iterable[tuple[str, str]], learned_edges: Iterable[tuple[str, str]]) -> dict[str, Any]:
    true_d = {(str(u), str(v)) for u, v in true_edges}
    learned_d = {(str(u), str(v)) for u, v in learned_edges}
    true_s = {frozenset((u, v)) for u, v in true_d}
    learned_s = {frozenset((u, v)) for u, v in learned_d}
    stp = len(true_s & learned_s)
    sfp = len(learned_s - true_s)
    sfn = len(true_s - learned_s)
    sp = stp / (stp + sfp) if stp + sfp else 0.0
    sr = stp / (stp + sfn) if stp + sfn else 0.0
    sf1 = 2 * sp * sr / (sp + sr) if sp + sr else 0.0
    dtp = len(true_d & learned_d)
    dfp = len(learned_d - true_d)
    dfn = len(true_d - learned_d)
    dp = dtp / (dtp + dfp) if dtp + dfp else 0.0
    dr = dtp / (dtp + dfn) if dtp + dfn else 0.0
    df1 = 2 * dp * dr / (dp + dr) if dp + dr else 0.0
    reversals = sum(((v, u) in true_d and (u, v) not in true_d for u, v in learned_d))
    extra = len(learned_s - true_s)
    missing = len(true_s - learned_s)
    return {'true_edge_count': len(true_d), 'learned_edge_count': len(learned_d), 'skeleton_true_positives': stp, 'skeleton_false_positives': sfp, 'skeleton_false_negatives': sfn, 'skeleton_precision': float(sp), 'skeleton_recall': float(sr), 'skeleton_f1': float(sf1), 'directed_true_positives': dtp, 'directed_false_positives': dfp, 'directed_false_negatives': dfn, 'directed_precision': float(dp), 'directed_recall': float(dr), 'directed_f1': float(df1), 'reversal_count': int(reversals), 'extra_adjacencies': int(extra), 'missing_adjacencies': int(missing), 'shd': int(reversals + extra + missing)}

class _rq1_s1_SmoothedDiscreteDAG:
    """Additive-smoothed CPT distribution over a learned discrete DAG."""

    def __init__(self, graph: nx.DiGraph, nodes: Sequence[str], cardinalities: Mapping[str, int], *, alpha: float=1.0) -> None:
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        self.graph = graph.copy()
        self.nodes = list(map(str, nodes))
        self.cardinalities = {str(k): int(v) for k, v in cardinalities.items()}
        self.alpha = float(alpha)
        self.tables: dict[str, dict[tuple[int, ...], np.ndarray]] = {}
        self.parent_orders: dict[str, tuple[str, ...]] = {}

    def fit(self, discrete_training: pd.DataFrame) -> '_rq1_s1_SmoothedDiscreteDAG':
        self.tables = {}
        self.parent_orders = {}
        for child in self.nodes:
            parents = tuple(sorted(map(str, self.graph.predecessors(child))))
            self.parent_orders[child] = parents
            r = self.cardinalities[child]
            counts: dict[tuple[int, ...], _rq1_s1_np.ndarray] = {}
            child_values = discrete_training[child].to_numpy(dtype=_rq1_s1_np.int64)
            if not parents:
                arr = _rq1_s1_np.bincount(child_values, minlength=r).astype(float)
                counts[()] = arr
            else:
                pmat = discrete_training[list(parents)].to_numpy(dtype=_rq1_s1_np.int64)
                for i in range(len(discrete_training)):
                    config = tuple((int(x) for x in pmat[i]))
                    arr = counts.get(config)
                    if arr is None:
                        arr = _rq1_s1_np.zeros(r, dtype=float)
                        counts[config] = arr
                    arr[int(child_values[i])] += 1.0
            tables: dict[tuple[int, ...], _rq1_s1_np.ndarray] = {}
            for config, arr in counts.items():
                posterior = arr + self.alpha
                posterior /= posterior.sum()
                tables[config] = posterior
            self.tables[child] = tables
        return self

    def joint_logscore(self, raw_evaluation: pd.DataFrame, transformer) -> tuple[_rq1_s1_np.ndarray, dict[str, _rq1_s1_np.ndarray]]:
        d = transformer.transform(raw_evaluation)
        n = len(d)
        total = _rq1_s1_np.zeros(n, dtype=float)
        nodewise: dict[str, _rq1_s1_np.ndarray] = {}
        for child in self.nodes:
            parents = self.parent_orders[child]
            r = self.cardinalities[child]
            fallback = _rq1_s1_np.full(r, 1.0 / r, dtype=float)
            table = self.tables[child]
            child_values = d[child].to_numpy(dtype=_rq1_s1_np.int64)
            logs = _rq1_s1_np.empty(n, dtype=float)
            if not parents:
                probs = table.get((), fallback)
                logs[:] = _rq1_s1_np.log(_rq1_s1_np.clip(probs[child_values], _rq1_s1__MIN_PROB, 1.0))
            else:
                pmat = d[list(parents)].to_numpy(dtype=_rq1_s1_np.int64)
                for i in range(n):
                    config = tuple((int(x) for x in pmat[i]))
                    probs = table.get(config, fallback)
                    logs[i] = _rq1_s1_math.log(max(float(probs[int(child_values[i])]), _rq1_s1__MIN_PROB))
            nodewise[child] = logs
            total += logs
        return (total, nodewise)

def _rq1_s1_rebuild_csvs(output_dir: Path) -> tuple[_rq1_s1_pd.DataFrame, _rq1_s1_pd.DataFrame]:
    rows = []
    for path in sorted(output_dir.glob('runs/*/seed_*/n_*/discrete_kl.json')):
        try:
            payload = _rq1_s1_json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if payload.get('status') != 'complete':
            continue
        exp = payload['experiment_definition']
        kl = payload['kl']
        sm = payload['structure_metrics']
        rows.append({'Generator': exp['generator'], 'Seed': int(exp['seed']), 'Sample_Size': int(exp['sample_size']), 'KL': float(kl['estimate']), 'KL_SE': float(kl['standard_error']), 'KL_Lower95': float(kl['lower_95']), 'KL_Upper95': float(kl['upper_95']), 'Mean_Log_P': float(kl['mean_log_p']), 'Mean_Log_Q': float(kl['mean_log_q']), 'SHD': int(sm['shd']), 'Precision': float(sm['directed_precision']), 'Recall': float(sm['directed_recall']), 'F1': float(sm['directed_f1']), 'Learned_Edges': int(sm['learned_edge_count']), 'Runtime_S': float(payload['runtime_seconds'])})
    df = _rq1_s1_pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(['Generator', 'Seed', 'Sample_Size']).reset_index(drop=True)
    df.to_csv(output_dir / 'discrete_representation_correct_results.csv', index=False)
    if len(df):
        summary = df.groupby(['Generator', 'Sample_Size'], as_index=False).agg(Runs=('KL', 'size'), Mean_KL=('KL', 'mean'), SD_KL=('KL', 'std'), Mean_SHD=('SHD', 'mean'), Mean_F1=('F1', 'mean'), Mean_Runtime_S=('Runtime_S', 'mean'))
    else:
        summary = _rq1_s1_pd.DataFrame()
    summary.to_csv(output_dir / 'discrete_representation_correct_summary.csv', index=False)
    return (df, summary)

def _rq1_s1_run(args: argparse.Namespace) -> dict[str, Any]:
    here = _rq1_s1_Path(__file__).resolve().parent
    root = _rq1_s1_discover_root(here, args.project_root)
    corrected_kl_path = _rq1_s1_find_corrected_kl_script(root, here, args.corrected_kl_script)
    kl07v2 = _rq1_inline_load(corrected_kl_path, f'_disc_rc_kl07v2_{_rq1_s1_os.getpid()}')
    discrete_module = _rq1_inline_load(root / 'src' / 'hybrid_bn' / 'SLdiscrete.py', f'_disc_rc_model_{_rq1_s1_os.getpid()}')
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None and args.output_dir.is_absolute() else (root / args.output_dir).resolve() if args.output_dir is not None else root / 'Outputs' / 'DiscreteBaseline_RepresentationCorrect'
    run_dir = output_dir / 'runs' / args.generator / f'seed_{args.seed:02d}' / f'n_{args.n:04d}'
    structure_path = run_dir / 'discrete_structure.json'
    kl_path = run_dir / 'discrete_kl.json'
    if kl_path.exists() and (not args.overwrite):
        print(f'Result already exists: {kl_path}')
        print('Use --overwrite to rerun.')
        _rq1_s1_rebuild_csvs(output_dir)
        return _rq1_s1_json.loads(kl_path.read_text(encoding='utf-8'))
    print('=' * 124)
    print('DISCRETE BASELINE — REPRESENTATION-CORRECT KL')
    print('=' * 124)
    print(f'Generator:      {args.generator}')
    print(f'Seed / n:       {args.seed} / {args.n}')
    print('Representation: all 18 variables quantile-discretised')
    print(f'Structure:      BIC + tabu GHC | max_parents={args.max_parents} | tabu={args.tabu_length} | max_iter={args.max_iter}')
    print(f'KL:             fresh={args.mc_samples} | particles={args.particles} x {args.particle_replicates} | batch={args.particle_batch_size}')
    print('=' * 124, flush=True)
    started = _rq1_s1_time.perf_counter()
    Generator = _rq1_s1_import_generator(root, args.generator)
    generator = Generator(num_samples=args.n, seed=args.seed)
    generator.generate_dag()
    raw_training = generator.generate_data()
    payload = generator.get_payload()
    nodes = list(map(str, payload['nodes']))
    raw_training = raw_training[nodes].copy().reset_index(drop=True)
    true_graph = payload['graph'].copy()
    cardinalities = discrete_module.default_cardinalities(nodes)
    discretizer = discrete_module.QuantileDiscretizer(cardinalities)
    discrete_training = discretizer.fit_transform(raw_training.copy())
    scorer = discrete_module.DiscreteScoringEngine(discrete_training, cardinalities)
    search = discrete_module.TabuGreedyHillClimbing(nodes, scorer, tabu_length=args.tabu_length, max_iter=args.max_iter, max_parents=args.max_parents, improvement_tolerance=1e-10)
    structure_started = _rq1_s1_time.perf_counter()
    learned_edges = search.search()
    structure_runtime = _rq1_s1_time.perf_counter() - structure_started
    learned_graph = _rq1_s1_graph_from_edges(nodes, learned_edges)
    metrics = _rq1_s1_structure_metrics(true_graph.edges(), learned_graph.edges())
    learned_score = float(sum((scorer.score_local(child, list(learned_graph.predecessors(child))) for child in nodes)))
    score_parity = abs(learned_score - float(search.best_score_)) if search.best_score_ is not None else float('nan')
    if _rq1_s1_np.isfinite(score_parity) and score_parity > 1e-08:
        raise AssertionError(f'Returned graph score parity failed: {score_parity:.3e}')
    discretizer_metadata = {'cardinalities': dict(cardinalities), 'bin_edges': {node: _rq1_s1_np.asarray(discretizer.bin_edges[node], dtype=float).tolist() for node in nodes}}
    structure_payload = {'schema_version': _rq1_s1_SCHEMA_VERSION, 'status': 'complete', 'experiment_definition': {'generator': args.generator, 'seed': int(args.seed), 'sample_size': int(args.n), 'representation': 'fully_discrete_quantile'}, 'settings': {'max_parents': int(args.max_parents), 'tabu_length': int(args.tabu_length), 'max_iter': int(args.max_iter), 'improvement_tolerance': 1e-10, 'score': 'multinomial_BIC'}, 'true_edges': sorted([[str(u), str(v)] for u, v in true_graph.edges()]), 'learned_edges': sorted([[str(u), str(v)] for u, v in learned_graph.edges()]), 'structure_metrics': metrics, 'best_structure_score': learned_score, 'returned_score_parity_abs_diff': score_parity, 'score_history': list(map(float, search.score_history_)), 'structure_runtime_seconds': float(structure_runtime), 'discretizer_metadata': discretizer_metadata}
    _rq1_s1_atomic_json(structure_path, structure_payload)
    print(f"\nStructure complete: SHD={metrics['shd']} | P={metrics['directed_precision']:.3f} | R={metrics['directed_recall']:.3f} | F1={metrics['directed_f1']:.3f} | edges={metrics['learned_edge_count']} | score={learned_score:.3f}", flush=True)
    transformer = kl07v2.SavedQuantileTransform(cardinalities, discretizer.bin_edges)
    true_model = kl07v2.TrueContinuousDAG(true_graph, payload['true_parameters'], nodes)
    mc_seed = kl07v2.deterministic_mc_seed(args.generator, args.seed)
    evaluation_sample = true_model.sample(args.mc_samples, mc_seed)
    truth_estimator = kl07v2.TransformedTruthParticleLikelihood(true_model, transformer, nodes, particles=args.particles, replicates=args.particle_replicates, batch_size=args.particle_batch_size)
    kl_started = _rq1_s1_time.perf_counter()
    log_p, particle_diagnostics = truth_estimator.loglikelihood(evaluation_sample, mc_seed)
    q = _rq1_s1_SmoothedDiscreteDAG(learned_graph, nodes, cardinalities, alpha=args.cpt_alpha).fit(discrete_training)
    log_q, q_nodewise = q.joint_logscore(evaluation_sample, transformer)
    ratio = log_p - log_q
    estimate = float(_rq1_s1_np.mean(ratio))
    se = float(_rq1_s1_np.std(ratio, ddof=1) / _rq1_s1_np.sqrt(len(ratio)))
    kl_runtime = _rq1_s1_time.perf_counter() - kl_started
    node_diagnostics = _rq1_s1_pd.DataFrame([{'Node': node, 'Parents': '|'.join(sorted(map(str, learned_graph.predecessors(node)))), 'Parent_Count': int(learned_graph.in_degree(node)), 'Mean_Log_Q': float(_rq1_s1_np.mean(q_nodewise[node])), 'KL_Contribution': _rq1_s1_np.nan, 'Note': 'nodewise KL undefined: transformed P_T need not factorise over the original/learned DAG'} for node in nodes])
    run_dir.mkdir(parents=True, exist_ok=True)
    node_diagnostics.to_csv(run_dir / 'q_node_diagnostics.csv', index=False)
    total_runtime = _rq1_s1_time.perf_counter() - started
    result = {'schema_version': _rq1_s1_SCHEMA_VERSION, 'status': 'complete', 'experiment_definition': {'generator': args.generator, 'seed': int(args.seed), 'sample_size': int(args.n), 'representation': 'fully_discrete_quantile'}, 'source_structure_json': str(structure_path), 'structure_metrics': metrics, 'structure': {'best_score': learned_score, 'runtime_seconds': float(structure_runtime), 'learned_edges': sorted([[str(u), str(v)] for u, v in learned_graph.edges()])}, 'kl': {'estimate': estimate, 'standard_error': se, 'lower_95': float(estimate - 1.96 * se), 'upper_95': float(estimate + 1.96 * se), 'mean_log_p': float(_rq1_s1_np.mean(log_p)), 'mean_log_q': float(_rq1_s1_np.mean(log_q)), 'mc_samples': int(args.mc_samples), 'mc_seed': int(mc_seed), 'definition': 'mean(log P_TD(T_D(x)) - log Q(T_D(x))) for fresh x~P', 'representation': 'all variables discretised with training-fitted quantile bins', 'true_transformed_probability': 'joint fully-discrete mass estimated by SMC marginalisation of every latent raw value inside its observed saved bin', 'q': f'learned discrete DAG with additive-smoothed CPTs alpha={args.cpt_alpha}', 'nodewise_kl_decomposition_available': False, 'particle_likelihood': particle_diagnostics}, 'evaluation_settings': {'cpt_alpha': float(args.cpt_alpha), 'particles': int(args.particles), 'particle_replicates': int(args.particle_replicates), 'particle_batch_size': int(args.particle_batch_size), 'fresh_mc_samples': int(args.mc_samples), 'same_mc_seed_mapping_as_hybrid_07_v2': True}, 'runtime_seconds': float(total_runtime), 'structure_runtime_seconds': float(structure_runtime), 'kl_runtime_seconds': float(kl_runtime)}
    _rq1_s1_atomic_json(kl_path, result)
    _, _ = _rq1_s1_rebuild_csvs(output_dir)
    print('\n' + '=' * 124)
    print('DISCRETE BASELINE COMPLETE')
    print('=' * 124)
    print(f'KL={estimate:.6f} ± {1.96 * se:.6f} (95% half-width) | SE={se:.6f}')
    print(f"SHD={metrics['shd']} | P={metrics['directed_precision']:.3f} | R={metrics['directed_recall']:.3f} | F1={metrics['directed_f1']:.3f} | edges={metrics['learned_edge_count']}")
    print(f'Structure score={learned_score:.3f}')
    print(f'Structure runtime={structure_runtime:.1f}s')
    print(f'KL runtime={kl_runtime:.1f}s')
    print(f'Total runtime={total_runtime:.1f}s')
    print(f'Structure: {structure_path}')
    print(f'KL result: {kl_path}')
    print(f"Results CSV: {output_dir / 'discrete_representation_correct_results.csv'}")
    print(f"Summary CSV: {output_dir / 'discrete_representation_correct_summary.csv'}")
    print('=' * 124)
    return result

def _rq1_s1_parse_args() -> _rq1_s1_argparse.Namespace:
    p = _rq1_s1_argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=_rq1_s1_Path, default=None)
    p.add_argument('--corrected-kl-script', type=_rq1_s1_Path, default=None)
    p.add_argument('--output-dir', type=_rq1_s1_Path, default=None)
    p.add_argument('--generator', choices=tuple(_rq1_s1_GENERATOR_SPECS), default='multimodal')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n', type=int, default=5000)
    p.add_argument('--max-parents', type=int, default=3)
    p.add_argument('--tabu-length', type=int, default=10)
    p.add_argument('--max-iter', type=int, default=300)
    p.add_argument('--mc-samples', type=int, default=5000)
    p.add_argument('--particles', type=int, default=256)
    p.add_argument('--particle-replicates', type=int, default=2)
    p.add_argument('--particle-batch-size', type=int, default=64)
    p.add_argument('--cpt-alpha', type=float, default=1.0)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.n < 1 or args.mc_samples < 2:
        p.error('n must be positive and mc-samples must be >=2')
    if args.max_parents != 3:
        p.error('Current benchmark contract requires --max-parents 3')
    if args.tabu_length < 0 or args.max_iter < 1:
        p.error('invalid search settings')
    if args.particles < 2 or args.particle_replicates < 1:
        p.error('invalid particle settings')
    if args.particle_batch_size < 1 or args.cpt_alpha <= 0:
        p.error('invalid batch size / CPT alpha')
    return args

def _rq1_s1_main() -> int:
    args = _rq1_s1_parse_args()
    try:
        _rq1_s1_run(args)
        return 0
    except Exception as exc:
        print(f'\nFAILED: {type(exc).__name__}: {exc}\n{_rq1_s1_traceback.format_exc()}', flush=True)
        return 1


# ======================================================================
# Evaluation implementation: 16_hybrid_v3_cd_v3_structure_learning.py
# ======================================================================
import argparse as _rq1_s2_argparse
import gc as _rq1_s2_gc
import hashlib as _rq1_s2_hashlib
import importlib.util as _rq1_s2_importlib
import itertools as _rq1_s2_itertools
import json as _rq1_s2_json
import math as _rq1_s2_math
import multiprocessing as _rq1_s2_mp
import os as _rq1_s2_os
import platform as _rq1_s2_platform
import sys as _rq1_s2_sys
import time as _rq1_s2_time
import traceback as _rq1_s2_traceback
from collections import Counter as _rq1_s2_Counter
from concurrent.futures import ProcessPoolExecutor as _rq1_s2_ProcessPoolExecutor, as_completed as _rq1_s2_as_completed
from dataclasses import dataclass as _rq1_s2_dataclass
from pathlib import Path as _rq1_s2_Path
from typing import Any as _rq1_s2_Any, Dict as _rq1_s2_Dict, Iterable as _rq1_s2_Iterable, Mapping as _rq1_s2_Mapping, Sequence as _rq1_s2_Sequence, Tuple as _rq1_s2_Tuple
_rq1_s2_sys.dont_write_bytecode = True
import networkx as _rq1_s2_nx
import numpy as _rq1_s2_np
import pandas as _rq1_s2_pd
_rq1_s2_SCHEMA_VERSION = 'hybrid-v3-cdv3-candidate-standard-ghc-noadd2-v1'
_rq1_s2_GENERATOR_SPECS = {'unimodal': ('generateUnimodalGT.py', 'StochasticContinuousGenerator'), 'skew': ('generateSkewGT.py', 'StochasticContinuousSkewedGenerator'), 'multimodal': ('generateMultimodalGT.py', 'StochasticContinuousMultimodalGenerator'), 'skew_multi': ('generateSkewMultiGT.py', 'StochasticContinuousSkewedMultimodalGenerator')}
_rq1_s2_DEFAULT_GENERATORS = ('multimodal',)
_rq1_s2_DEFAULT_SEEDS = (0,)
_rq1_s2_DEFAULT_SAMPLE_SIZES = (5000,)
_rq1_s2__WORKER_SCORER = None
_rq1_s2__WORKER_TASK_COUNT = 0

def _rq1_s2_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _rq1_s2_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_rq1_s2_json_safe(v) for v in value]
    if isinstance(value, _rq1_s2_np.ndarray):
        return value.tolist()
    if isinstance(value, _rq1_s2_np.integer):
        return int(value)
    if isinstance(value, _rq1_s2_np.floating):
        value = float(value)
    if isinstance(value, float):
        if _rq1_s2_math.isnan(value):
            return 'NaN'
        if _rq1_s2_math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
    return value

def _rq1_s2_atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(_rq1_s2_json.dumps(_rq1_s2_json_safe(dict(payload)), indent=2, sort_keys=True), encoding='utf-8')
    _rq1_s2_os.replace(temp, path)

def _rq1_s2_file_sha256(path: Path) -> str:
    h = _rq1_s2_hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def _rq1_s2_dataframe_sha256(frame: _rq1_s2_pd.DataFrame) -> str:
    h = _rq1_s2_hashlib.sha256()
    h.update('|'.join(map(str, frame.columns)).encode('utf-8'))
    hashed = _rq1_s2_pd.util.hash_pandas_object(frame, index=True).to_numpy(_rq1_s2_np.uint64)
    h.update(_rq1_s2_np.ascontiguousarray(hashed).tobytes())
    return h.hexdigest()

def _rq1_s2_package_sha256(package_dir: Path) -> str:
    h = _rq1_s2_hashlib.sha256()
    for path in sorted(package_dir.rglob('*.py')):
        h.update(str(path.relative_to(package_dir)).encode('utf-8'))
        h.update(path.read_bytes())
    return h.hexdigest()

def _rq1_s2_discover_root(here: _rq1_s2_Path, explicit: _rq1_s2_Path | None) -> _rq1_s2_Path:
    return _rq1_find_root()

def _rq1_s2_import_generator(root: Path, generator: str):
    filename, class_name = _rq1_s2_GENERATOR_SPECS[generator]
    module = _rq1_inline_load(root / 'src' / 'generators' / filename, f'_hv3_structure_generator_{generator}_{_rq1_s2_os.getpid()}')
    if not hasattr(module, class_name):
        raise AttributeError(f'{filename} does not expose {class_name}.')
    return getattr(module, class_name)

def _rq1_s2_default_cardinalities(payload: Mapping[str, Any]) -> _rq1_s2_Dict[str, int]:
    discrete = list(map(str, payload['discrete_nodes']))
    binary = set(map(str, payload.get('binary_nodes', [])))
    return {node: 2 if node in binary else 3 for node in discrete}

def _rq1_s2_family_key(child: str, parents: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    return (str(child), tuple(sorted(map(str, parents))))

def _rq1_s2_graph_from_edges(nodes: Sequence[str], edges: Iterable[_rq1_s2_Tuple[str, str]]) -> _rq1_s2_nx.DiGraph:
    g = _rq1_s2_nx.DiGraph()
    g.add_nodes_from(map(str, nodes))
    g.add_edges_from(((str(u), str(v)) for u, v in edges))
    if not _rq1_s2_nx.is_directed_acyclic_graph(g):
        raise ValueError('Learned graph is not a DAG.')
    return g

def _rq1_s2_edge_type(parent: str, child: str, discrete_set: set[str]) -> str:
    p = 'D' if parent in discrete_set else 'C'
    c = 'D' if child in discrete_set else 'C'
    return f'{p}->{c}'

def _rq1_s2_final_family_quadrant(child: str, parents: Sequence[str], discrete_set: set[str]) -> str:
    child_is_d = child in discrete_set
    has_d = any((p in discrete_set for p in parents))
    has_c = any((p not in discrete_set for p in parents))
    if child_is_d:
        return 'CD' if has_c else 'DD'
    return 'DC' if has_d else 'CC'

def _rq1_s2_calculate_structure_metrics(true_edges: Iterable[_rq1_s2_Tuple[str, str]], learned_edges: Iterable[_rq1_s2_Tuple[str, str]]) -> _rq1_s2_Dict[str, Any]:
    true_directed = set(((str(u), str(v)) for u, v in true_edges))
    learned_directed = set(((str(u), str(v)) for u, v in learned_edges))
    true_skeleton = {frozenset((u, v)) for u, v in true_directed}
    learned_skeleton = {frozenset((u, v)) for u, v in learned_directed}
    skeleton_tp = len(true_skeleton & learned_skeleton)
    skeleton_fp = len(learned_skeleton - true_skeleton)
    skeleton_fn = len(true_skeleton - learned_skeleton)
    skeleton_precision = skeleton_tp / (skeleton_tp + skeleton_fp) if skeleton_tp + skeleton_fp else 0.0
    skeleton_recall = skeleton_tp / (skeleton_tp + skeleton_fn) if skeleton_tp + skeleton_fn else 0.0
    skeleton_f1 = 2 * skeleton_precision * skeleton_recall / (skeleton_precision + skeleton_recall) if skeleton_precision + skeleton_recall else 0.0
    directed_tp = len(true_directed & learned_directed)
    directed_fp = len(learned_directed - true_directed)
    directed_fn = len(true_directed - learned_directed)
    directed_precision = directed_tp / (directed_tp + directed_fp) if directed_tp + directed_fp else 0.0
    directed_recall = directed_tp / (directed_tp + directed_fn) if directed_tp + directed_fn else 0.0
    directed_f1 = 2 * directed_precision * directed_recall / (directed_precision + directed_recall) if directed_precision + directed_recall else 0.0
    reversal_count = sum(((child, parent) in true_directed and (parent, child) not in true_directed for parent, child in learned_directed))
    extra_adjacencies = len(learned_skeleton - true_skeleton)
    missing_adjacencies = len(true_skeleton - learned_skeleton)
    shd = reversal_count + extra_adjacencies + missing_adjacencies
    return {'true_edge_count': len(true_directed), 'learned_edge_count': len(learned_directed), 'skeleton_true_positives': skeleton_tp, 'skeleton_false_positives': skeleton_fp, 'skeleton_false_negatives': skeleton_fn, 'skeleton_precision': float(skeleton_precision), 'skeleton_recall': float(skeleton_recall), 'skeleton_f1': float(skeleton_f1), 'directed_true_positives': directed_tp, 'directed_false_positives': directed_fp, 'directed_false_negatives': directed_fn, 'directed_precision': float(directed_precision), 'directed_recall': float(directed_recall), 'directed_f1': float(directed_f1), 'reversal_count': int(reversal_count), 'extra_adjacencies': int(extra_adjacencies), 'missing_adjacencies': int(missing_adjacencies), 'shd': int(shd)}

def _rq1_s2__worker_initializer(model_file: str, cc_scorer_file: str, training_frame: pd.DataFrame, discrete_nodes: tuple[str, ...], continuous_nodes: tuple[str, ...], cardinalities: _rq1_s2_Dict[str, int]) -> None:
    global _rq1_s2__WORKER_SCORER, _rq1_s2__WORKER_TASK_COUNT
    model_path = _rq1_s2_Path(model_file).resolve()
    module = _rq1_inline_load(model_path, f'_hv3_parallel_worker_model_{_rq1_s2_os.getpid()}')
    if not hasattr(module, 'HybridScoringEngine'):
        raise AttributeError(f'{model_path} does not expose HybridScoringEngine.')
    _rq1_s2__WORKER_SCORER = module.HybridScoringEngine(training_frame.copy(), discrete_nodes, continuous_nodes, cardinalities, cc_scorer_file=_rq1_s2_Path(cc_scorer_file).resolve())
    _rq1_s2__WORKER_TASK_COUNT = 0

def _rq1_s2__evict_worker_family(key: tuple[str, tuple[str, ...]]) -> None:
    scorer = _rq1_s2__WORKER_SCORER
    if scorer is None:
        return
    for attr in ('cache', 'score_info'):
        mapping = getattr(scorer, attr, None)
        if isinstance(mapping, dict):
            mapping.pop(key, None)
    cc = getattr(scorer, '_cc', None)
    if cc is not None:
        engine = getattr(cc, 'engine', None)
        if engine is not None:
            for attr in ('cache', 'results', 'diagnostics', 'combined_results'):
                mapping = getattr(engine, attr, None)
                if isinstance(mapping, dict):
                    mapping.pop(key, None)
            base = getattr(engine, '_base', None)
            if base is not None:
                for attr in ('cache', 'results', 'diagnostics', 'combined_results'):
                    mapping = getattr(base, attr, None)
                    if isinstance(mapping, dict):
                        mapping.pop(key, None)
                mean = getattr(base, 'mean_engine', None)
                if mean is not None:
                    for attr in ('cache', 'local_families', 'diagnostics', 'posterior_state_logweights', 'contextual_refinement_diagnostics'):
                        mapping = getattr(mean, attr, None)
                        if isinstance(mapping, dict):
                            mapping.pop(key, None)
    cd = getattr(scorer, 'cd', None)
    if cd is not None and isinstance(getattr(cd, 'diagnostics', None), list):
        cd.diagnostics.clear()
    if cd is not None and isinstance(getattr(cd, 'model_diagnostics', None), list):
        cd.model_diagnostics.clear()
    dc = getattr(scorer, 'dc', None)
    if dc is not None and isinstance(getattr(dc, 'mixture_diagnostics', None), list):
        dc.mixture_diagnostics.clear()

def _rq1_s2__worker_score_family(key: tuple[str, tuple[str, ...]]) -> tuple[tuple[str, tuple[str, ...]], float, str, float, int]:
    global _rq1_s2__WORKER_TASK_COUNT
    if _rq1_s2__WORKER_SCORER is None:
        raise RuntimeError('Worker scorer is not initialized.')
    child, parents = key
    started = _rq1_s2_time.perf_counter()
    try:
        value = float(_rq1_s2__WORKER_SCORER.score_local(child, parents))
        quadrant = str(_rq1_s2__WORKER_SCORER.quadrant(child, parents))
        elapsed = float(_rq1_s2_time.perf_counter() - started)
        return (key, value, quadrant, elapsed, _rq1_s2_os.getpid())
    finally:
        _rq1_s2__evict_worker_family(key)
        _rq1_s2__WORKER_TASK_COUNT += 1
        if _rq1_s2__WORKER_TASK_COUNT % 16 == 0:
            _rq1_s2_gc.collect()

@_rq1_s2_dataclass(frozen=True)
class _rq1_s2_Candidate:
    action: tuple[str, ...]
    new_v_key: tuple[str, tuple[str, ...]]
    new_u_key: tuple[str, tuple[str, ...]] | None = None

class _rq1_s2_ParallelAtomicAdd2TabuGreedyHillClimbing:
    """Exact decision semantics; parallel missing-family evaluation only."""

    def __init__(self, *, nodes: Sequence[str], model_file: _rq1_s2_Path, cc_scorer_file: _rq1_s2_Path, training_frame: pd.DataFrame, discrete_nodes: Sequence[str], continuous_nodes: Sequence[str], cardinalities: Mapping[str, int], workers: int, tabu_length: int=10, max_iter: int=150, max_parents: int=3, improvement_tolerance: float=1e-10, enable_add2: bool=True, progress_label: str='Hybrid V3 parallel') -> None:
        self.nodes = list(map(str, nodes))
        self.model_file = _rq1_s2_Path(model_file).resolve()
        self.cc_scorer_file = _rq1_s2_Path(cc_scorer_file).resolve()
        self.training_frame = training_frame.copy()
        self.discrete_nodes = tuple(map(str, discrete_nodes))
        self.continuous_nodes = tuple(map(str, continuous_nodes))
        self.cardinalities = {str(k): int(v) for k, v in cardinalities.items()}
        self.workers = int(workers)
        self.tabu_length = int(tabu_length)
        self.max_iter = int(max_iter)
        self.max_parents = int(max_parents)
        self.improvement_tolerance = float(improvement_tolerance)
        self.enable_add2 = bool(enable_add2)
        self.progress_label = str(progress_label)
        if self.workers < 1:
            raise ValueError('workers must be >= 1.')
        self.graph = _rq1_s2_nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        self.family_scores: _rq1_s2_Dict[tuple[str, tuple[str, ...]], float] = {}
        self.family_quadrants: _rq1_s2_Dict[tuple[str, tuple[str, ...]], str] = {}
        self.family_runtimes: _rq1_s2_Dict[tuple[str, tuple[str, ...]], float] = {}
        self.family_worker_pid: _rq1_s2_Dict[tuple[str, tuple[str, ...]], int] = {}
        self.node_scores: _rq1_s2_Dict[str, float] = {}
        self.best_score_: float | None = None
        self.score_history_: list[float] = []
        self.action_history_: list[tuple[str, ...]] = []
        self.iteration_diagnostics_: list[dict[str, Any]] = []

    @staticmethod
    def _is_tabu(action: tuple[str, ...], tabu_list: Mapping[tuple[str, ...], int], iteration: int) -> bool:
        if tabu_list.get(action, -1) > iteration:
            return True
        if action and action[0] == 'add2':
            _, p1, p2, child = action
            return tabu_list.get(('add', p1, child), -1) > iteration or tabu_list.get(('add', p2, child), -1) > iteration
        return False

    def _admissible(self, action: tuple[str, ...], candidate_score: float, global_best_score: float, tabu_list: Mapping[tuple[str, ...], int], iteration: int) -> bool:
        if not self._is_tabu(action, tabu_list, iteration):
            return True
        return candidate_score > global_best_score + self.improvement_tolerance

    def _score_missing(self, executor: _rq1_s2_ProcessPoolExecutor, keys: Sequence[tuple[str, tuple[str, ...]]], *, phase: str) -> float:
        missing = [key for key in keys if key not in self.family_scores]
        missing = list(dict.fromkeys(missing))
        if not missing:
            return 0.0
        started = _rq1_s2_time.perf_counter()
        futures = {executor.submit(_rq1_s2__worker_score_family, key): key for key in missing}
        completed = 0
        next_report = started + 20.0
        for future in _rq1_s2_as_completed(futures):
            key = futures[future]
            try:
                returned_key, value, quadrant, family_s, pid = future.result()
            except Exception as exc:
                for other in futures:
                    other.cancel()
                raise RuntimeError(f'Parallel family score failed for {key[0]}|{list(key[1])}: {exc}') from exc
            if returned_key != key:
                raise RuntimeError(f'Worker returned wrong key: expected={key}, got={returned_key}')
            self.family_scores[key] = float(value)
            self.family_quadrants[key] = str(quadrant)
            self.family_runtimes[key] = float(family_s)
            self.family_worker_pid[key] = int(pid)
            completed += 1
            now = _rq1_s2_time.perf_counter()
            if completed == len(missing) or now >= next_report:
                print(f'    {phase}: {completed}/{len(missing)} new families | wall={now - started:.1f}s | master_cache={len(self.family_scores)}', flush=True)
                next_report = now + 20.0
        return float(_rq1_s2_time.perf_counter() - started)

    def _enumerate_ordinary(self) -> tuple[list[_rq1_s2_Candidate], list[tuple[str, tuple[str, ...]]]]:
        candidates: list[_rq1_s2_Candidate] = []
        needed: list[tuple[str, tuple[str, ...]]] = []
        for u in self.nodes:
            for v in self.nodes:
                if u == v:
                    continue
                if not self.graph.has_edge(u, v) and (not self.graph.has_edge(v, u)):
                    parents_v = list(self.graph.predecessors(v))
                    if len(parents_v) >= self.max_parents:
                        continue
                    self.graph.add_edge(u, v)
                    valid = _rq1_s2_nx.is_directed_acyclic_graph(self.graph)
                    self.graph.remove_edge(u, v)
                    if not valid:
                        continue
                    key_v = _rq1_s2_family_key(v, parents_v + [u])
                    candidates.append(_rq1_s2_Candidate(('add', u, v), key_v))
                    needed.append(key_v)
                elif self.graph.has_edge(u, v):
                    parents_v = list(self.graph.predecessors(v))
                    new_parents_v = [p for p in parents_v if p != u]
                    key_v = _rq1_s2_family_key(v, new_parents_v)
                    candidates.append(_rq1_s2_Candidate(('remove', u, v), key_v))
                    needed.append(key_v)
                    parents_u = list(self.graph.predecessors(u))
                    if len(parents_u) >= self.max_parents:
                        continue
                    self.graph.remove_edge(u, v)
                    self.graph.add_edge(v, u)
                    valid = _rq1_s2_nx.is_directed_acyclic_graph(self.graph)
                    self.graph.remove_edge(v, u)
                    self.graph.add_edge(u, v)
                    if not valid:
                        continue
                    key_u = _rq1_s2_family_key(u, parents_u + [v])
                    candidates.append(_rq1_s2_Candidate(('reverse', u, v), key_v, key_u))
                    needed.append(key_u)
        return (candidates, needed)

    def _enumerate_add2(self) -> tuple[list[_rq1_s2_Candidate], list[tuple[str, tuple[str, ...]]]]:
        candidates: list[_rq1_s2_Candidate] = []
        needed: list[tuple[str, tuple[str, ...]]] = []
        if not self.enable_add2:
            return (candidates, needed)
        for child in self.nodes:
            current_parents = list(self.graph.predecessors(child))
            if self.max_parents - len(current_parents) < 2:
                continue
            proposal = [node for node in self.nodes if node != child and node not in current_parents and (not self.graph.has_edge(child, node)) and (not self.graph.has_edge(node, child))]
            for p1, p2 in _rq1_s2_itertools.combinations(proposal, 2):
                self.graph.add_edge(p1, child)
                self.graph.add_edge(p2, child)
                valid = _rq1_s2_nx.is_directed_acyclic_graph(self.graph)
                self.graph.remove_edge(p2, child)
                self.graph.remove_edge(p1, child)
                if not valid:
                    continue
                key_child = _rq1_s2_family_key(child, current_parents + [p1, p2])
                candidates.append(_rq1_s2_Candidate(('add2', p1, p2, child), key_child))
                needed.append(key_child)
        return (candidates, needed)

    def _candidate_score(self, candidate: _rq1_s2_Candidate, current_score: float) -> tuple[float, _rq1_s2_Dict[str, float]]:
        action = candidate.action
        kind = action[0]
        if kind in {'add', 'remove'}:
            _, _u, v = action
            new_v = float(self.family_scores[candidate.new_v_key])
            return (float(current_score + new_v - self.node_scores[v]), {v: new_v})
        if kind == 'reverse':
            _, u, v = action
            if candidate.new_u_key is None:
                raise RuntimeError('Reverse candidate missing new_u_key.')
            new_v = float(self.family_scores[candidate.new_v_key])
            new_u = float(self.family_scores[candidate.new_u_key])
            score = current_score + new_v - self.node_scores[v] + new_u - self.node_scores[u]
            return (float(score), {v: new_v, u: new_u})
        if kind == 'add2':
            _, _p1, _p2, child = action
            new_child = float(self.family_scores[candidate.new_v_key])
            score = current_score + new_child - self.node_scores[child]
            return (float(score), {child: new_child})
        raise ValueError(action)

    def search(self) -> list[tuple[str, str]]:
        mp_context = _rq1_s2_mp.get_context('spawn')
        with _rq1_s2_ProcessPoolExecutor(max_workers=self.workers, mp_context=mp_context, initializer=_rq1_s2__worker_initializer, initargs=(str(self.model_file), str(self.cc_scorer_file), self.training_frame, self.discrete_nodes, self.continuous_nodes, self.cardinalities)) as executor:
            root_keys = [_rq1_s2_family_key(node, ()) for node in self.nodes]
            root_wall = self._score_missing(executor, root_keys, phase='Root initialization')
            self.node_scores = {node: self.family_scores[_rq1_s2_family_key(node, ())] for node in self.nodes}
            current_score = float(sum((self.node_scores[n] for n in self.nodes)))
            best_graph = self.graph.copy()
            best_score = current_score
            tabu_list: _rq1_s2_Dict[tuple[str, ...], int] = {}
            self.best_score_ = float(best_score)
            self.score_history_ = [float(current_score)]
            self.action_history_ = []
            self.iteration_diagnostics_ = []
            print(f'    Roots complete: score={current_score:.3f} | wall={root_wall:.1f}s', flush=True)
            for iteration in range(self.max_iter):
                iteration_started = _rq1_s2_time.perf_counter()
                print(f'  Iteration {iteration + 1}/{self.max_iter} | best={best_score:.3f} | edges={self.graph.number_of_edges()} | families={len(self.family_scores)}', flush=True)
                ordinary, ordinary_needed = self._enumerate_ordinary()
                add2, add2_needed = self._enumerate_add2()
                candidates = [*ordinary, *add2]
                if not candidates:
                    print('    No legal candidates; search stops.', flush=True)
                    break
                ordinary_unique = list(dict.fromkeys(ordinary_needed))
                add2_unique = list(dict.fromkeys(add2_needed))
                all_unique = list(dict.fromkeys([*ordinary_unique, *add2_unique]))
                ordinary_new = sum((k not in self.family_scores for k in ordinary_unique))
                add2_new = sum((k not in self.family_scores for k in add2_unique))
                total_new = sum((k not in self.family_scores for k in all_unique))
                batch_wall = self._score_missing(executor, all_unique, phase=f'Iteration {iteration + 1} family scoring')
                required_quadrants = _rq1_s2_Counter((self.family_quadrants[k] for k in all_unique if k in self.family_quadrants))
                best_neighbor_score = -_rq1_s2_np.inf
                best_action: tuple[str, ...] | None = None
                best_updates: _rq1_s2_Dict[str, float] = {}
                for candidate in candidates:
                    score, updates = self._candidate_score(candidate, current_score)
                    action = candidate.action
                    if score > best_neighbor_score and self._admissible(action, score, best_score, tabu_list, iteration):
                        best_neighbor_score = float(score)
                        best_action = action
                        best_updates = updates
                if best_action is None:
                    print('    No admissible candidate; search stops.', flush=True)
                    break
                kind = best_action[0]
                inverse_actions: list[tuple[str, ...]] = []
                if kind == 'add':
                    _, u, v = best_action
                    self.graph.add_edge(u, v)
                    inverse_actions = [('remove', u, v)]
                elif kind == 'remove':
                    _, u, v = best_action
                    self.graph.remove_edge(u, v)
                    inverse_actions = [('add', u, v)]
                elif kind == 'reverse':
                    _, u, v = best_action
                    self.graph.remove_edge(u, v)
                    self.graph.add_edge(v, u)
                    inverse_actions = [('reverse', v, u)]
                elif kind == 'add2':
                    _, p1, p2, child = best_action
                    self.graph.add_edge(p1, child)
                    self.graph.add_edge(p2, child)
                    inverse_actions = [('remove', p1, child), ('remove', p2, child)]
                else:
                    raise ValueError(best_action)
                if not _rq1_s2_nx.is_directed_acyclic_graph(self.graph):
                    raise RuntimeError(f'Applied action produced cycle: {best_action}')
                if any((self.graph.in_degree(node) > self.max_parents for node in self.nodes)):
                    raise RuntimeError(f'Applied action exceeded max_parents: {best_action}')
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
                tabu_list = {action: exp for action, exp in tabu_list.items() if exp > iteration + 1}
                iteration_wall = float(_rq1_s2_time.perf_counter() - iteration_started)
                selected_child = best_action[3] if kind == 'add2' else best_action[2]
                selected_parents = list(self.graph.predecessors(selected_child))
                selected_family_key = _rq1_s2_family_key(selected_child, selected_parents)
                selected_quadrant = self.family_quadrants.get(selected_family_key, _rq1_s2_final_family_quadrant(selected_child, selected_parents, set(self.discrete_nodes)))
                diag = {'iteration': int(iteration + 1), 'ordinary_candidate_count': int(len(ordinary)), 'add2_candidate_count': int(len(add2)), 'candidate_count': int(len(candidates)), 'ordinary_unique_family_count': int(len(ordinary_unique)), 'add2_unique_family_count': int(len(add2_unique)), 'unique_required_family_count': int(len(all_unique)), 'ordinary_new_family_count': int(ordinary_new), 'add2_new_family_count': int(add2_new), 'new_family_count': int(total_new), 'required_family_quadrant_counts': dict(sorted(required_quadrants.items())), 'family_batch_wall_s': float(batch_wall), 'iteration_wall_s': iteration_wall, 'selected_action': list(best_action), 'selected_action_is_add2': bool(kind == 'add2'), 'selected_family_quadrant': str(selected_quadrant), 'selected_neighbor_score': float(current_score), 'global_best_score': float(best_score), 'edge_count_after_action': int(self.graph.number_of_edges()), 'master_family_cache_size': int(len(self.family_scores))}
                self.iteration_diagnostics_.append(diag)
                print(f'    chose={best_action} [{selected_quadrant}] | current={current_score:.3f} | best={best_score:.3f} | ordinary={len(ordinary)} | add2={len(add2)} | new={total_new} | batch={batch_wall:.1f}s | iter={iteration_wall:.1f}s', flush=True)
            self.graph = best_graph
            self.best_score_ = float(best_score)
        return list(self.graph.edges())

def _rq1_s2_build_main_scorer(*, model_module, hybrid_training: pd.DataFrame, discrete_nodes: Sequence[str], continuous_nodes: Sequence[str], cardinalities: Mapping[str, int], cc_scorer_file: Path):
    return model_module.HybridScoringEngine(hybrid_training.copy(), discrete_nodes, continuous_nodes, cardinalities, cc_scorer_file=cc_scorer_file)

def _rq1_s2_reconcile_final_families(*, scorer, search: _rq1_s2_ParallelAtomicAdd2TabuGreedyHillClimbing, graph: nx.DiGraph) -> tuple[_rq1_s2_pd.DataFrame, float]:
    rows = []
    max_diff = 0.0
    for child in search.nodes:
        parents = tuple(sorted(map(str, graph.predecessors(child))))
        key = _rq1_s2_family_key(child, parents)
        parallel = float(search.family_scores[key])
        main = float(scorer.score_local(child, parents))
        diff = abs(parallel - main)
        max_diff = max(max_diff, diff)
        rows.append({'Child': child, 'Parents': '|'.join(parents), 'Parent_Count': len(parents), 'Quadrant': str(scorer.quadrant(child, parents)), 'Parallel_Score': parallel, 'Main_Score': main, 'Abs_Diff': diff})
    return (_rq1_s2_pd.DataFrame(rows), float(max_diff))

def _rq1_s2_run_one_cell(*, model_module, model_file: Path, cc_scorer_file: Path, raw_training: _rq1_s2_pd.DataFrame, nodes: Sequence[str], discrete_nodes: Sequence[str], continuous_nodes: Sequence[str], cardinalities: Mapping[str, int], true_graph: nx.DiGraph, workers: int, tabu_length: int, max_iter: int, max_parents: int, enable_add2: bool, parity_tolerance: float) -> tuple[_rq1_s2_Dict[str, Any], _rq1_s2_pd.DataFrame]:
    started = _rq1_s2_time.perf_counter()
    discretizer = model_module.QuantileDiscretizer(cardinalities)
    hybrid_training = discretizer.fit_transform(raw_training.copy())
    search = _rq1_s2_ParallelAtomicAdd2TabuGreedyHillClimbing(nodes=nodes, model_file=model_file, cc_scorer_file=cc_scorer_file, training_frame=hybrid_training, discrete_nodes=discrete_nodes, continuous_nodes=continuous_nodes, cardinalities=cardinalities, workers=workers, tabu_length=tabu_length, max_iter=max_iter, max_parents=max_parents, enable_add2=enable_add2)
    search_started = _rq1_s2_time.perf_counter()
    learned_edges = search.search()
    search_runtime = float(_rq1_s2_time.perf_counter() - search_started)
    learned_graph = _rq1_s2_graph_from_edges(nodes, learned_edges)
    structure = _rq1_s2_calculate_structure_metrics(true_graph.edges(), learned_graph.edges())
    main_started = _rq1_s2_time.perf_counter()
    main_scorer = _rq1_s2_build_main_scorer(model_module=model_module, hybrid_training=hybrid_training, discrete_nodes=discrete_nodes, continuous_nodes=continuous_nodes, cardinalities=cardinalities, cc_scorer_file=cc_scorer_file)
    main_scorer_build_s = float(_rq1_s2_time.perf_counter() - main_started)
    parity_df, max_diff = _rq1_s2_reconcile_final_families(scorer=main_scorer, search=search, graph=learned_graph)
    if max_diff > parity_tolerance:
        raise AssertionError(f'Worker/main final-family score mismatch {max_diff} > tolerance {parity_tolerance}.')
    recomputed_score = float(sum((main_scorer.score_local(child, list(learned_graph.predecessors(child))) for child in nodes)))
    discrete_set = set(map(str, discrete_nodes))
    learned_edge_types = _rq1_s2_Counter((_rq1_s2_edge_type(str(u), str(v), discrete_set) for u, v in learned_graph.edges()))
    true_edge_types = _rq1_s2_Counter((_rq1_s2_edge_type(str(u), str(v), discrete_set) for u, v in true_graph.edges()))
    final_family_quadrants = _rq1_s2_Counter((str(main_scorer.quadrant(child, list(learned_graph.predecessors(child)))) for child in nodes))
    selected_action_quadrants = _rq1_s2_Counter((str(d.get('selected_family_quadrant', 'UNKNOWN')) for d in search.iteration_diagnostics_))
    selected_add2_count = sum((bool(d.get('selected_action_is_add2', False)) for d in search.iteration_diagnostics_))
    worker_counts = _rq1_s2_Counter(search.family_worker_pid.values())
    scored_quadrants = _rq1_s2_Counter(search.family_quadrants.values())
    cd_v3_model_df = main_scorer.cd_model_diagnostics_dataframe() if hasattr(main_scorer, 'cd_model_diagnostics_dataframe') else _rq1_s2_pd.DataFrame()
    if not cd_v3_model_df.empty and 'candidate' in cd_v3_model_df.columns:
        cd_v3_weight_summary = cd_v3_model_df.groupby('candidate', as_index=False)['posterior_model_weight'].agg(['mean', 'sum']).reset_index().to_dict(orient='records')
    else:
        cd_v3_weight_summary = []
    diagnostics = {'parallel_search_unique_family_count': int(len(search.family_scores)), 'parallel_search_family_quadrant_counts': dict(sorted(scored_quadrants.items())), 'parallel_search_worker_count_requested': int(workers), 'parallel_search_worker_pids_used': int(len(worker_counts)), 'parallel_search_families_by_worker': {str(pid): int(count) for pid, count in sorted(worker_counts.items())}, 'parallel_search_sum_family_runtime_s': float(sum(search.family_runtimes.values())), 'parallel_search_wall_s': search_runtime, 'parallel_family_runtime_to_wall_ratio': float(sum(search.family_runtimes.values()) / max(search_runtime, 1e-12)), 'worker_main_final_family_max_abs_diff': float(max_diff), 'worker_main_parity_tolerance': float(parity_tolerance), 'true_edge_type_counts': dict(sorted(true_edge_types.items())), 'learned_edge_type_counts': dict(sorted(learned_edge_types.items())), 'learned_final_family_quadrant_counts': dict(sorted(final_family_quadrants.items())), 'selected_action_quadrant_counts': dict(sorted(selected_action_quadrants.items())), 'selected_add2_action_count': int(selected_add2_count), 'search_iteration_count': int(len(search.iteration_diagnostics_)), 'predictive_kl_evaluated': False, 'predictive_kl_reason': 'Evaluated separately by representation-correct CD V3 KL script.', 'cd_v3_candidate_weight_summary_final_families': cd_v3_weight_summary}
    result = {'status': 'complete', 'learned_edges': sorted([[str(u), str(v)] for u, v in learned_graph.edges()]), 'structure_metrics': structure, 'best_structure_score_parallel': float(search.best_score_), 'learned_dag_score_recomputed_main': float(recomputed_score), 'parallel_best_minus_main_learned_score': float(float(search.best_score_) - recomputed_score), 'runtime_seconds': search_runtime, 'main_scorer_build_seconds': main_scorer_build_s, 'total_runtime_seconds': float(_rq1_s2_time.perf_counter() - started), 'score_history': [float(v) for v in search.score_history_], 'action_history': [list(a) for a in search.action_history_], 'iteration_diagnostics': search.iteration_diagnostics_, 'diagnostics': diagnostics, 'cd_v3_model_diagnostics': cd_v3_model_df.to_dict(orient='records'), 'discretizer_metadata': {'cardinalities': dict(cardinalities), 'bin_edges': {node: _rq1_s2_np.asarray(edges, dtype=float).tolist() for node, edges in discretizer.bin_edges.items()}}}
    return (result, parity_df)

def _rq1_s2_result_path(output_dir: Path, generator: str, seed: int, sample_size: int) -> Path:
    return output_dir / 'runs' / generator / f'seed_{seed:02d}' / f'n_{sample_size:04d}' / 'hybrid_v3_structure.json'

def _rq1_s2_parity_path(output_dir: Path, generator: str, seed: int, sample_size: int) -> Path:
    return output_dir / 'parity' / f'{generator}__seed{seed}__n{sample_size}.csv'

def _rq1_s2_cell_matches(path: Path, *, settings: Mapping[str, Any], provenance: Mapping[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        payload = _rq1_s2_json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return False
    return payload.get('schema_version') == _rq1_s2_SCHEMA_VERSION and payload.get('status') == 'complete' and (payload.get('settings') == dict(settings)) and (payload.get('provenance') == dict(provenance))

def _rq1_s2_build_summary(runs: _rq1_s2_pd.DataFrame) -> _rq1_s2_pd.DataFrame:
    if runs.empty:
        return _rq1_s2_pd.DataFrame()
    metrics = ['Learned_Edges', 'Skeleton_Precision', 'Skeleton_Recall', 'Skeleton_F1', 'Directed_Precision', 'Directed_Recall', 'Directed_F1', 'Reversals', 'Extra_Adjacencies', 'Missing_Adjacencies', 'SHD', 'Structure_Runtime_s', 'Search_Unique_Local_Families', 'Selected_Add2_Actions', 'First_Iteration_New_Families', 'First_Iteration_Batch_s']
    rows = []
    for (generator, n), group in runs.groupby(['Generator', 'Sample_Size'], sort=True):
        row = {'Generator': generator, 'Sample_Size': int(n), 'Cells': int(len(group))}
        for metric in metrics:
            vals = _rq1_s2_pd.to_numeric(group[metric], errors='coerce')
            row[f'{metric}_Mean'] = float(vals.mean())
            row[f'{metric}_SD'] = float(vals.std(ddof=1)) if len(vals) > 1 else float('nan')
        rows.append(row)
    return _rq1_s2_pd.DataFrame(rows)

def _rq1_s2_write_report(path: Path, runs: pd.DataFrame, *, target_count: int, model_file: Path, workers: int, enable_add2: bool) -> None:
    lines = ['HYBRID V3 + CD V3 CANDIDATE — STANDARD NO-ADD2 GHC', '=' * 116, f'Completed cells: {len(runs)}/{target_count}', f'Workers: {workers}', f'Model: {model_file}', f"Atomic ADD2: {('enabled, exhaustive' if enable_add2 else 'disabled')}", 'Predictive KL: NOT YET EVALUATED in this structure-only runner', '']
    if runs.empty:
        lines.append('No completed cells.')
    else:
        lines.extend(['OVERALL', '-' * 116, f"Mean SHD: {runs['SHD'].mean():.4f}", f"Mean directed precision: {runs['Directed_Precision'].mean():.4f}", f"Mean directed recall: {runs['Directed_Recall'].mean():.4f}", f"Mean directed F1: {runs['Directed_F1'].mean():.4f}", f"Mean search runtime: {runs['Structure_Runtime_s'].mean():.3f}s", f"Max worker/main final-family score diff: {runs['Worker_Main_Max_Abs_Diff'].max():.3e}", f"Mean selected ADD2 actions: {runs['Selected_Add2_Actions'].mean():.3f}", '', 'BY GENERATOR x N', '-' * 116])
        table = runs.groupby(['Generator', 'Sample_Size'], as_index=False).agg(Cells=('Seed', 'count'), SHD=('SHD', 'mean'), Directed_Precision=('Directed_Precision', 'mean'), Directed_Recall=('Directed_Recall', 'mean'), Directed_F1=('Directed_F1', 'mean'), Structure_Runtime_s=('Structure_Runtime_s', 'mean'), Learned_Edges=('Learned_Edges', 'mean'), Selected_Add2_Actions=('Selected_Add2_Actions', 'mean'), Search_Unique_Local_Families=('Search_Unique_Local_Families', 'mean'))
        lines.append(table.to_string(index=False))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def _rq1_s2_parse_args() -> _rq1_s2_argparse.Namespace:
    p = _rq1_s2_argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=_rq1_s2_Path, default=None)
    p.add_argument('--generators', nargs='+', choices=tuple(_rq1_s2_GENERATOR_SPECS), default=list(_rq1_s2_DEFAULT_GENERATORS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(_rq1_s2_DEFAULT_SEEDS))
    p.add_argument('--sample-sizes', nargs='+', type=int, default=list(_rq1_s2_DEFAULT_SAMPLE_SIZES))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--tabu-length', type=int, default=10)
    p.add_argument('--max-iter', type=int, default=150)
    p.add_argument('--max-parents', type=int, default=3)
    p.add_argument('--no-add2', action='store_true')
    p.add_argument('--parity-tolerance', type=float, default=1e-06)
    p.add_argument('--output-dir', type=_rq1_s2_Path, default=None)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def _rq1_s2_main() -> int:
    args = _rq1_s2_parse_args()
    here = _rq1_s2_Path(__file__).resolve().parent
    root = _rq1_s2_discover_root(here, args.project_root)
    model_file = (root / 'src' / 'hybrid_bn' / 'SLhybrid.py').resolve()
    package_dir = (root / 'src' / 'hybrid_bn' / 'HybridV3_CDv3Candidate').resolve()
    cc_scorer_file = (root / 'src' / 'hybrid_bn' / 'SLhybrid.py').resolve()
    cc_production_file = (root / 'src' / 'hybrid_bn' / 'SLhybrid.py').resolve()
    model_module = _rq1_inline_load(model_file, '_hv3_parallel_structure_main_model')
    generators = tuple(dict.fromkeys(map(str, args.generators)))
    seeds = tuple(dict.fromkeys(map(int, args.seeds)))
    sample_sizes = tuple(sorted(set(map(int, args.sample_sizes))))
    if not sample_sizes or min(sample_sizes) < 30:
        raise ValueError('All sample sizes must be >= 30.')
    if args.workers < 1:
        raise ValueError('--workers must be >= 1.')
    if args.max_parents != 3:
        raise ValueError('Current Hybrid V3 benchmark contract requires max_parents=3.')
    if args.parity_tolerance <= 0:
        raise ValueError('--parity-tolerance must be positive.')
    enable_add2 = False
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else root / 'Outputs' / 'HybridV3_CDv3Candidate_StructureLearning_NoAdd2'
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {'model_file_sha256': _rq1_s2_file_sha256(model_file), 'hybrid_package_sha256': _rq1_s2_package_sha256(package_dir), 'cc_corrected_sha256': _rq1_s2_file_sha256(cc_scorer_file), 'cc_production_sha256': _rq1_s2_file_sha256(cc_production_file)}
    settings = {'workers': int(args.workers), 'tabu_length': int(args.tabu_length), 'max_iter': int(args.max_iter), 'max_parents': int(args.max_parents), 'enable_add2': bool(enable_add2), 'add2_screening': False, 'parallelized_component': 'missing local-family scalar scores only', 'serial_candidate_decision_order': True, 'parity_tolerance': float(args.parity_tolerance), 'predictive_kl_evaluated': False, 'search_family': 'standard_tabu_ghc_noadd2', 'cd_model': 'softmax_laplace_bma_v3_linear_quadratic_cubic_spline5'}
    target_count = len(generators) * len(seeds) * len(sample_sizes)
    failure_log = output_dir / 'failures.jsonl'
    print('=' * 132)
    print('HYBRID V3 + CD V3 CANDIDATE — STANDARD NO-ADD2 GHC')
    print('=' * 132)
    print(f'Project root: {root}')
    print(f'Model:        {model_file}')
    print(f'CC scorer:    {cc_scorer_file}')
    print(f'Generators:   {generators}')
    print(f'Seeds:        {seeds}')
    print(f'Sample sizes: {sample_sizes}')
    print(f'Workers:      {args.workers}')
    print(f'Search:       tabu={args.tabu_length}, max_iter={args.max_iter}, max_parents={args.max_parents}, add2=OFF')
    print('KL:           deferred until mixed predictive logpdf is implemented')
    print('=' * 132, flush=True)
    started_all = _rq1_s2_time.perf_counter()
    completed_now = 0
    skipped = 0
    failed = 0
    for generator_name in generators:
        Generator = _rq1_s2_import_generator(root, generator_name)
        for seed in seeds:
            print(f'\n=== {generator_name} | seed={seed} | generating n={max(sample_sizes)} base dataset ===', flush=True)
            generator = Generator(num_samples=max(sample_sizes), seed=int(seed))
            generator.generate_dag()
            base_data = generator.generate_data()
            payload = generator.get_payload()
            nodes = list(map(str, payload['nodes']))
            discrete_nodes = list(map(str, payload['discrete_nodes']))
            continuous_nodes = list(map(str, payload['continuous_nodes']))
            cardinalities = _rq1_s2_default_cardinalities(payload)
            true_graph = payload['graph'].copy()
            base_data = base_data[nodes].copy().reset_index(drop=True)
            print(f'GT: nodes={len(nodes)} | true_edges={true_graph.number_of_edges()} | D={len(discrete_nodes)} | C={len(continuous_nodes)}', flush=True)
            for n in sample_sizes:
                path = _rq1_s2_result_path(output_dir, generator_name, int(seed), int(n))
                cell_settings = dict(settings)
                cell_settings.update({'generator': generator_name, 'seed': int(seed), 'sample_size': int(n)})
                print(f'\n[RUN] {generator_name} seed={seed} n={n}', flush=True)
                if not args.overwrite and _rq1_s2_cell_matches(path, settings=cell_settings, provenance=provenance):
                    skipped += 1
                    print('    completed matching result exists; skipping.', flush=True)
                    continue
                raw_training = base_data.iloc[:n].copy().reset_index(drop=True)
                data_hash = _rq1_s2_dataframe_sha256(raw_training)
                try:
                    result, parity_df = _rq1_s2_run_one_cell(model_module=model_module, model_file=model_file, cc_scorer_file=cc_scorer_file, raw_training=raw_training, nodes=nodes, discrete_nodes=discrete_nodes, continuous_nodes=continuous_nodes, cardinalities=cardinalities, true_graph=true_graph, workers=int(args.workers), tabu_length=int(args.tabu_length), max_iter=int(args.max_iter), max_parents=int(args.max_parents), enable_add2=enable_add2, parity_tolerance=float(args.parity_tolerance))
                    cell = {'schema_version': _rq1_s2_SCHEMA_VERSION, 'status': 'complete', 'experiment_definition': {'generator': generator_name, 'seed': int(seed), 'sample_size': int(n), 'all_ground_truth_variables_continuous': True, 'hybrid_discrete_nodes': discrete_nodes, 'hybrid_continuous_nodes': continuous_nodes}, 'settings': cell_settings, 'provenance': provenance, 'dataset_sha256': data_hash, 'true_edges': sorted([[str(u), str(v)] for u, v in true_graph.edges()]), **result}
                    _rq1_s2_atomic_write_json(path, cell)
                    ppath = _rq1_s2_parity_path(output_dir, generator_name, int(seed), int(n))
                    ppath.parent.mkdir(parents=True, exist_ok=True)
                    parity_df.to_csv(ppath, index=False)
                    completed_now += 1
                    m = result['structure_metrics']
                    d = result['diagnostics']
                    print(f"    OK | SHD={m['shd']} | edges={m['learned_edge_count']} | P={m['directed_precision']:.3f} | R={m['directed_recall']:.3f} | F1={m['directed_f1']:.3f} | runtime={result['runtime_seconds']:.1f}s | families={d['parallel_search_unique_family_count']} | add2_selected={d['selected_add2_action_count']} | parity={d['worker_main_final_family_max_abs_diff']:.2e}", flush=True)
                except Exception as error:
                    failed += 1
                    record = {'generator': generator_name, 'seed': int(seed), 'sample_size': int(n), 'error_type': type(error).__name__, 'error_message': str(error), 'traceback': _rq1_s2_traceback.format_exc()}
                    with failure_log.open('a', encoding='utf-8') as f:
                        f.write(_rq1_s2_json.dumps(_rq1_s2_json_safe(record)) + '\n')
                    print(f'    FAILED: {type(error).__name__}: {error}', flush=True)
                    print(record['traceback'], flush=True)
    rows = []
    for path in sorted((output_dir / 'runs').rglob('hybrid_v3_structure.json')):
        try:
            p = _rq1_s2_json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if p.get('schema_version') != _rq1_s2_SCHEMA_VERSION or p.get('status') != 'complete':
            continue
        if p.get('provenance') != provenance:
            continue
        s = p.get('settings', {})
        base_ok = all((s.get(k) == v for k, v in settings.items()))
        if not base_ok:
            continue
        d = p['experiment_definition']
        if str(d.get('generator')) not in set(generators) or int(d.get('seed')) not in set(seeds) or int(d.get('sample_size')) not in set(sample_sizes):
            continue
        m = p['structure_metrics']
        diag = p['diagnostics']
        iters = p.get('iteration_diagnostics', [])
        first = iters[0] if iters else {}
        rows.append({'Model': 'hybrid_v3_modular', 'Generator': d['generator'], 'Seed': d['seed'], 'Sample_Size': d['sample_size'], 'Workers': settings['workers'], 'True_Edges': m['true_edge_count'], 'Learned_Edges': m['learned_edge_count'], 'Skeleton_Precision': m['skeleton_precision'], 'Skeleton_Recall': m['skeleton_recall'], 'Skeleton_F1': m['skeleton_f1'], 'Directed_Precision': m['directed_precision'], 'Directed_Recall': m['directed_recall'], 'Directed_F1': m['directed_f1'], 'Reversals': m['reversal_count'], 'Extra_Adjacencies': m['extra_adjacencies'], 'Missing_Adjacencies': m['missing_adjacencies'], 'SHD': m['shd'], 'Structure_Runtime_s': p['runtime_seconds'], 'Total_Runtime_s': p['total_runtime_seconds'], 'Best_Structure_Score': p['best_structure_score_parallel'], 'Learned_DAG_Score_Main': p['learned_dag_score_recomputed_main'], 'Best_Minus_Main': p['parallel_best_minus_main_learned_score'], 'Search_Unique_Local_Families': diag['parallel_search_unique_family_count'], 'Worker_Main_Max_Abs_Diff': diag['worker_main_final_family_max_abs_diff'], 'Search_Iterations': diag['search_iteration_count'], 'Selected_Add2_Actions': diag['selected_add2_action_count'], 'First_Iteration_New_Families': first.get('new_family_count', float('nan')), 'First_Iteration_Batch_s': first.get('family_batch_wall_s', float('nan')), 'Learned_Edge_Type_Counts': _rq1_s2_json.dumps(diag['learned_edge_type_counts'], sort_keys=True), 'Final_Family_Quadrant_Counts': _rq1_s2_json.dumps(diag['learned_final_family_quadrant_counts'], sort_keys=True), 'Selected_Action_Quadrant_Counts': _rq1_s2_json.dumps(diag['selected_action_quadrant_counts'], sort_keys=True)})
    runs = _rq1_s2_pd.DataFrame(rows)
    runs_path = output_dir / 'hybrid_v3_structure_results.csv'
    summary_path = output_dir / 'hybrid_v3_structure_summary.csv'
    report_path = output_dir / 'report.txt'
    runs.to_csv(runs_path, index=False)
    summary = _rq1_s2_build_summary(runs)
    summary.to_csv(summary_path, index=False)
    _rq1_s2_write_report(report_path, runs, target_count=target_count, model_file=model_file, workers=int(args.workers), enable_add2=enable_add2)
    manifest = {'schema_version': _rq1_s2_SCHEMA_VERSION, 'status': 'COMPLETE' if len(runs) >= target_count and failed == 0 else 'INCOMPLETE', 'experiment_scope': {'generators': list(generators), 'seeds': list(seeds), 'sample_sizes': list(sample_sizes), 'target_cells': int(target_count)}, 'settings': settings, 'provenance': provenance, 'counts': {'completed_now': int(completed_now), 'skipped_existing': int(skipped), 'failed_now': int(failed), 'matching_completed_rows': int(len(runs))}, 'runtime_seconds': float(_rq1_s2_time.perf_counter() - started_all), 'environment': {'python': _rq1_s2_sys.version, 'platform': _rq1_s2_platform.platform(), 'numpy': _rq1_s2_np.__version__, 'pandas': _rq1_s2_pd.__version__, 'cpu_count': _rq1_s2_os.cpu_count()}, 'outputs': {'runs_csv': str(runs_path), 'summary_csv': str(summary_path), 'report': str(report_path), 'failure_log': str(failure_log) if failure_log.exists() else None, 'parity_dir': str(output_dir / 'parity')}, 'scientific_note': 'This runner evaluates structure only. Predictive KL is deferred until coherent mixed predictive logpdf APIs exist for DD/CD/DC.'}
    _rq1_s2_atomic_write_json(output_dir / 'manifest.json', manifest)
    print('\n' + '=' * 132)
    print('HYBRID V3 STRUCTURE RUN')
    print('=' * 132)
    print(f"STATUS: {manifest['status']}")
    print(f'Completed now: {completed_now}')
    print(f'Skipped existing: {skipped}')
    print(f'Failed now: {failed}')
    print(f'Rows available: {len(runs)}/{target_count}')
    if not runs.empty:
        print(f"Mean SHD={runs['SHD'].mean():.3f} | P={runs['Directed_Precision'].mean():.3f} | R={runs['Directed_Recall'].mean():.3f} | F1={runs['Directed_F1'].mean():.3f} | runtime={runs['Structure_Runtime_s'].mean():.1f}s | max parity={runs['Worker_Main_Max_Abs_Diff'].max():.2e}", flush=True)
    print(f'Results: {runs_path}')
    print(f'Summary: {summary_path}')
    print(f'Report:  {report_path}')
    print('=' * 132)
    return 0 if failed == 0 else 1


# ======================================================================
# Evaluation implementation: 17_hybrid_v3_cd_v3_representation_correct_kl.py
# ======================================================================
import argparse as _rq1_s3_argparse
import gc as _rq1_s3_gc
import hashlib as _rq1_s3_hashlib
import importlib.util as _rq1_s3_importlib
import json as _rq1_s3_json
import math as _rq1_s3_math
import os as _rq1_s3_os
import sys as _rq1_s3_sys
import time as _rq1_s3_time
import traceback as _rq1_s3_traceback
import warnings as _rq1_s3_warnings
from dataclasses import asdict as _rq1_s3_asdict, dataclass as _rq1_s3_dataclass
from datetime import datetime as _rq1_s3_datetime, timezone as _rq1_s3_timezone
from pathlib import Path as _rq1_s3_Path
from typing import Any as _rq1_s3_Any, Iterable as _rq1_s3_Iterable, Mapping as _rq1_s3_Mapping, Sequence as _rq1_s3_Sequence
_rq1_s3_sys.dont_write_bytecode = True
import networkx as _rq1_s3_nx
import numpy as _rq1_s3_np
import pandas as _rq1_s3_pd
from scipy.linalg import helmert as _rq1_s3_helmert
from scipy.optimize import minimize as _rq1_s3_minimize
from scipy.special import gammaln as _rq1_s3_gammaln, logsumexp as _rq1_s3_logsumexp, owens_t as _rq1_s3_owens_t
from scipy.stats import norm as _rq1_s3_norm, skewnorm as _rq1_s3_skewnorm
from sklearn.exceptions import ConvergenceWarning as _rq1_s3_ConvergenceWarning
from sklearn.linear_model import LogisticRegression as _rq1_s3_LogisticRegression
_rq1_s3__MIN_PROBABILITY = 1e-300
_rq1_s3__MIN_STANDARD_DEVIATION = 1e-06
_rq1_s3_SUPPORTED_TRUE_FAMILIES = {'gaussian_root', 'linear_gaussian', 'gaussian_mixture_root', 'gated_gaussian', 'skew_normal_root', 'skew_normal_regression', 'skew_mixture_root', 'gated_skew_mixture'}
_rq1_s3_PARADIGMS = ('discrete', 'continuous', 'hybrid')
_rq1_s3__V2_Q_MIXTURE_MAX_ITER = 1000
_rq1_s3__V2_Q_MIXTURE_RESTARTS = 3
_rq1_s3__V2_Q_MIXTURE_MIN_EFFECTIVE_COUNT = 5.0
_rq1_s3_KL_SCHEMA_VERSION = 'hybrid-v3-representation-correct-kl-v2'
_rq1_s3_GENERATOR_SPECS = {'unimodal': ('generateUnimodalGT.py', 'StochasticContinuousGenerator'), 'skew': ('generateSkewGT.py', 'StochasticContinuousSkewedGenerator'), 'multimodal': ('generateMultimodalGT.py', 'StochasticContinuousMultimodalGenerator'), 'skew_multi': ('generateSkewMultiGT.py', 'StochasticContinuousSkewedMultimodalGenerator')}

def _rq1_s3_json_safe(value: Any) -> Any:
    if isinstance(value, _rq1_s3_Mapping):
        return {str(k): _rq1_s3_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_rq1_s3_json_safe(v) for v in value]
    if isinstance(value, _rq1_s3_np.ndarray):
        return _rq1_s3_json_safe(value.tolist())
    if isinstance(value, _rq1_s3_np.generic):
        return _rq1_s3_json_safe(value.item())
    if isinstance(value, _rq1_s3_Path):
        return str(value)
    if isinstance(value, float):
        if _rq1_s3_np.isnan(value):
            return 'NaN'
        if _rq1_s3_np.isposinf(value):
            return 'Infinity'
        if _rq1_s3_np.isneginf(value):
            return '-Infinity'
    return value

def _rq1_s3_decode_float(value: Any) -> float:
    if value == 'Infinity':
        return _rq1_s3_np.inf
    if value == '-Infinity':
        return -_rq1_s3_np.inf
    if value == 'NaN':
        return _rq1_s3_np.nan
    return float(value)

def _rq1_s3_atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(_rq1_s3_json.dumps(_rq1_s3_json_safe(payload), indent=2), encoding='utf-8')
    _rq1_s3_os.replace(tmp, path)

def _rq1_s3_discover_root(here: _rq1_s3_Path, explicit: _rq1_s3_Path | None) -> _rq1_s3_Path:
    return _rq1_find_root()

def _rq1_s3_import_generator(root: Path, generator: str):
    filename, class_name = _rq1_s3_GENERATOR_SPECS[generator]
    module = _rq1_inline_load(root / 'src' / 'generators' / filename, f'_hv3_kl_generator_{generator}_{_rq1_s3_os.getpid()}')
    if not hasattr(module, class_name):
        raise AttributeError(f'{filename} does not expose {class_name}.')
    return getattr(module, class_name)

def _rq1_s3_structure_dataframe_sha256(frame: _rq1_s3_pd.DataFrame) -> str:
    h = _rq1_s3_hashlib.sha256()
    h.update('|'.join(map(str, frame.columns)).encode('utf-8'))
    hashed = _rq1_s3_pd.util.hash_pandas_object(frame, index=True).to_numpy(_rq1_s3_np.uint64)
    h.update(_rq1_s3_np.ascontiguousarray(hashed).tobytes())
    return h.hexdigest()

def _rq1_s3_deterministic_mc_seed(generator: str, seed: int) -> int:
    message = f'fresh-supervisor-kl-v1|{generator}|{seed}'.encode('utf-8')
    digest = _rq1_s3_hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], 'little') % (2 ** 32 - 1)

def _rq1_s3_build_graph(nodes: Sequence[str], edges: Iterable[Sequence[str]]) -> _rq1_s3_nx.DiGraph:
    g = _rq1_s3_nx.DiGraph()
    g.add_nodes_from(map(str, nodes))
    g.add_edges_from(((str(e[0]), str(e[1])) for e in edges))
    if not _rq1_s3_nx.is_directed_acyclic_graph(g):
        raise ValueError('Graph is not a DAG.')
    return g

def _rq1_s3_default_cardinalities(payload: Mapping[str, Any]) -> dict[str, int]:
    discrete = list(map(str, payload['discrete_nodes']))
    binary = set(map(str, payload.get('binary_nodes', [])))
    return {node: 2 if node in binary else 3 for node in discrete}

def _rq1_s3__configuration_storage_key(value: Any) -> str:
    if value is None:
        return 'none'
    if isinstance(value, (tuple, list, _rq1_s3_np.ndarray)):
        values = list(value)
    else:
        values = [value]
    return '|'.join((str(int(v)) for v in values))

class _rq1_s3_TrueContinuousDAG:
    """Sample and evaluate the eight families used by the four generators."""

    def __init__(self, graph: _rq1_s3_nx.DiGraph, parameters: Mapping[str, Mapping[str, Any]], node_order: Sequence[str]) -> None:
        if not _rq1_s3_nx.is_directed_acyclic_graph(graph):
            raise ValueError('The true graph is not a DAG.')
        self.graph = graph.copy()
        self.parameters = {str(node): dict(values) for node, values in parameters.items()}
        self.node_order = [str(node) for node in node_order]
        self.topological_order = list(_rq1_s3_nx.topological_sort(self.graph))
        missing = set(self.graph.nodes) - set(self.parameters)
        if missing:
            raise ValueError(f'Missing true parameters for: {sorted(missing)}')
        families = {str(values.get('family')) for values in self.parameters.values()}
        unsupported = families - _rq1_s3_SUPPORTED_TRUE_FAMILIES
        if unsupported:
            raise ValueError(f'Unsupported true parameter families: {sorted(unsupported)}')

    @staticmethod
    def _linear_location(parameters: Mapping[str, Any], dataframe: pd.DataFrame) -> _rq1_s3_np.ndarray:
        location = _rq1_s3_np.full(len(dataframe), float(parameters['intercept']), dtype=float)
        for parent, coefficient in parameters['coefficients'].items():
            location += float(coefficient) * dataframe[parent].to_numpy(dtype=float)
        return location

    @staticmethod
    def _gated_location(parameters: Mapping[str, Any], dataframe: pd.DataFrame) -> _rq1_s3_np.ndarray:
        location = _rq1_s3_np.full(len(dataframe), float(parameters['intercept']), dtype=float)
        for parent, gate in parameters['gates'].items():
            parent_values = dataframe[parent].to_numpy(dtype=float)
            location += _rq1_s3_np.where(parent_values > float(gate['threshold']), float(gate['shift']), -float(gate['shift']))
        return location

    @staticmethod
    def _mixture_logpdf(weights: Sequence[float], component_logpdfs: Sequence[_rq1_s3_np.ndarray]) -> _rq1_s3_np.ndarray:
        weights_array = _rq1_s3_np.asarray(weights, dtype=float)
        log_weights = _rq1_s3_np.log(_rq1_s3_np.clip(weights_array, _rq1_s3__MIN_PROBABILITY, None))
        return _rq1_s3_logsumexp(_rq1_s3_np.vstack([log_weight + component for log_weight, component in zip(log_weights, component_logpdfs)]), axis=0)

    @staticmethod
    def _sample_component_indices(weights: Sequence[float], size: int, rng: _rq1_s3_np.random.Generator) -> _rq1_s3_np.ndarray:
        probabilities = _rq1_s3_np.asarray(weights, dtype=float)
        probabilities = probabilities / probabilities.sum()
        return rng.choice(len(probabilities), size=size, p=probabilities)

    def node_logpdf(self, node: str, dataframe: pd.DataFrame) -> np.ndarray:
        parameters = self.parameters[node]
        family = str(parameters['family'])
        values = dataframe[node].to_numpy(dtype=float)
        if family == 'gaussian_root':
            return _rq1_s3_norm.logpdf(values, loc=float(parameters['mean']), scale=float(parameters['std']))
        if family == 'linear_gaussian':
            location = self._linear_location(parameters, dataframe)
            return _rq1_s3_norm.logpdf(values, loc=location, scale=float(parameters['noise_std']))
        if family == 'gaussian_mixture_root':
            component_logpdfs = [_rq1_s3_norm.logpdf(values, loc=float(mean), scale=float(std)) for mean, std in zip(parameters['means'], parameters['stds'])]
            return self._mixture_logpdf(parameters['weights'], component_logpdfs)
        if family == 'gated_gaussian':
            location = self._gated_location(parameters, dataframe)
            return _rq1_s3_norm.logpdf(values, loc=location, scale=float(parameters['noise_std']))
        if family == 'skew_normal_root':
            return _rq1_s3_skewnorm.logpdf(values, float(parameters['shape']), loc=float(parameters['loc']), scale=float(parameters['scale']))
        if family == 'skew_normal_regression':
            location = self._linear_location(parameters, dataframe)
            factor = float(parameters.get('residual_scale_factor', 1.0))
            residual = values - location
            return _rq1_s3_skewnorm.logpdf(residual, float(parameters['residual_shape']), loc=factor * float(parameters['residual_loc']), scale=factor * float(parameters['residual_scale']))
        if family == 'skew_mixture_root':
            component_logpdfs = [_rq1_s3_skewnorm.logpdf(values, float(shape), loc=float(mean), scale=float(scale)) for mean, scale, shape in zip(parameters['means'], parameters['scales'], parameters['shapes'])]
            return self._mixture_logpdf(parameters['weights'], component_logpdfs)
        if family == 'gated_skew_mixture':
            location = self._gated_location(parameters, dataframe)
            residual = values - location
            mixture = parameters['residual_mixture']
            factor = float(parameters['residual_scale_factor'])
            component_logpdfs = [_rq1_s3_skewnorm.logpdf(residual, float(shape), loc=factor * float(mean), scale=factor * float(scale)) for mean, scale, shape in zip(mixture['means'], mixture['scales'], mixture['shapes'])]
            return self._mixture_logpdf(mixture['weights'], component_logpdfs)
        raise ValueError(f'Unknown true family {family!r} for node {node!r}.')

    @staticmethod
    def _stable_interval_probability(cdf_lower: _rq1_s3_np.ndarray, cdf_upper: _rq1_s3_np.ndarray, sf_lower: _rq1_s3_np.ndarray, sf_upper: _rq1_s3_np.ndarray) -> _rq1_s3_np.ndarray:
        """Return a stable probability for lower < X <= upper."""
        cdf_difference = _rq1_s3_np.asarray(cdf_upper) - _rq1_s3_np.asarray(cdf_lower)
        survival_difference = _rq1_s3_np.asarray(sf_lower) - _rq1_s3_np.asarray(sf_upper)
        probability = _rq1_s3_np.maximum(cdf_difference, survival_difference)
        return _rq1_s3_np.clip(probability, _rq1_s3__MIN_PROBABILITY, 1.0)

    @classmethod
    def _normal_interval_probability(cls, lower: np.ndarray, upper: np.ndarray, location: np.ndarray | float, scale: float) -> np.ndarray:
        return cls._stable_interval_probability(_rq1_s3_norm.cdf(lower, loc=location, scale=scale), _rq1_s3_norm.cdf(upper, loc=location, scale=scale), _rq1_s3_norm.sf(lower, loc=location, scale=scale), _rq1_s3_norm.sf(upper, loc=location, scale=scale))

    @classmethod
    def _skew_interval_probability(cls, lower: _rq1_s3_np.ndarray, upper: _rq1_s3_np.ndarray, shape: float, location: _rq1_s3_np.ndarray | float, scale: float) -> _rq1_s3_np.ndarray:
        lower = _rq1_s3_np.asarray(lower, dtype=float)
        upper = _rq1_s3_np.asarray(upper, dtype=float)
        location = _rq1_s3_np.asarray(location, dtype=float)
        z_lower = (lower - location) / float(scale)
        z_upper = (upper - location) / float(scale)
        t_lower = _rq1_s3_owens_t(z_lower, float(shape))
        t_upper = _rq1_s3_owens_t(z_upper, float(shape))
        cdf_lower = _rq1_s3_np.clip(_rq1_s3_norm.cdf(z_lower) - 2.0 * t_lower, 0.0, 1.0)
        cdf_upper = _rq1_s3_np.clip(_rq1_s3_norm.cdf(z_upper) - 2.0 * t_upper, 0.0, 1.0)
        sf_lower = _rq1_s3_np.clip(_rq1_s3_norm.sf(z_lower) + 2.0 * t_lower, 0.0, 1.0)
        sf_upper = _rq1_s3_np.clip(_rq1_s3_norm.sf(z_upper) + 2.0 * t_upper, 0.0, 1.0)
        return cls._stable_interval_probability(cdf_lower, cdf_upper, sf_lower, sf_upper)

    def node_log_interval_probability(self, node: str, dataframe: pd.DataFrame, lower: _rq1_s3_np.ndarray, upper: _rq1_s3_np.ndarray) -> _rq1_s3_np.ndarray:
        """Log true conditional probability of the observed node interval."""
        parameters = self.parameters[node]
        family = str(parameters['family'])
        if family == 'gaussian_root':
            probability = self._normal_interval_probability(lower, upper, float(parameters['mean']), float(parameters['std']))
        elif family == 'linear_gaussian':
            location = self._linear_location(parameters, dataframe)
            probability = self._normal_interval_probability(lower, upper, location, float(parameters['noise_std']))
        elif family == 'gaussian_mixture_root':
            component_probabilities = [self._normal_interval_probability(lower, upper, float(mean), float(std)) for mean, std in zip(parameters['means'], parameters['stds'])]
            probability = _rq1_s3_np.sum(_rq1_s3_np.asarray(parameters['weights'], dtype=float)[:, None] * _rq1_s3_np.vstack(component_probabilities), axis=0)
        elif family == 'gated_gaussian':
            location = self._gated_location(parameters, dataframe)
            probability = self._normal_interval_probability(lower, upper, location, float(parameters['noise_std']))
        elif family == 'skew_normal_root':
            probability = self._skew_interval_probability(lower, upper, float(parameters['shape']), float(parameters['loc']), float(parameters['scale']))
        elif family == 'skew_normal_regression':
            location = self._linear_location(parameters, dataframe)
            factor = float(parameters.get('residual_scale_factor', 1.0))
            probability = self._skew_interval_probability(lower, upper, float(parameters['residual_shape']), location + factor * float(parameters['residual_loc']), factor * float(parameters['residual_scale']))
        elif family == 'skew_mixture_root':
            component_probabilities = [self._skew_interval_probability(lower, upper, float(shape), float(mean), float(scale)) for mean, scale, shape in zip(parameters['means'], parameters['scales'], parameters['shapes'])]
            probability = _rq1_s3_np.sum(_rq1_s3_np.asarray(parameters['weights'], dtype=float)[:, None] * _rq1_s3_np.vstack(component_probabilities), axis=0)
        elif family == 'gated_skew_mixture':
            location = self._gated_location(parameters, dataframe)
            mixture = parameters['residual_mixture']
            factor = float(parameters['residual_scale_factor'])
            component_probabilities = [self._skew_interval_probability(lower, upper, float(shape), location + factor * float(mean), factor * float(scale)) for mean, scale, shape in zip(mixture['means'], mixture['scales'], mixture['shapes'])]
            probability = _rq1_s3_np.sum(_rq1_s3_np.asarray(mixture['weights'], dtype=float)[:, None] * _rq1_s3_np.vstack(component_probabilities), axis=0)
        else:
            raise ValueError(f'Unknown true family {family!r} for node {node!r}.')
        probability = _rq1_s3_np.clip(probability, _rq1_s3__MIN_PROBABILITY, 1.0)
        return _rq1_s3_np.log(probability)

    def joint_logscore_for_representation(self, dataframe: pd.DataFrame, *, discretised_nodes: Sequence[str], transformer: Any | None, return_nodewise: bool=False) -> _rq1_s3_np.ndarray | tuple[_rq1_s3_np.ndarray, dict[str, _rq1_s3_np.ndarray]]:
        """Evaluate true P in the same raw/discrete representation as Q."""
        dataframe = dataframe[self.node_order].reset_index(drop=True)
        discrete_set = {str(node) for node in discretised_nodes}
        if discrete_set:
            if transformer is None:
                raise ValueError('A saved quantile transform is required for discretised nodes.')
            if transformer.nodes != discrete_set:
                raise ValueError('The true-score discretised nodes do not match the saved transform.')
            transformed = transformer.transform(dataframe)
        else:
            if transformer is not None:
                raise ValueError('Continuous true scoring must not receive a discretiser.')
            transformed = dataframe
        total = _rq1_s3_np.zeros(len(dataframe), dtype=float)
        nodewise: dict[str, _rq1_s3_np.ndarray] = {}
        for node in self.topological_order:
            if node in discrete_set:
                states = transformed[node].to_numpy(dtype=int)
                edges = transformer.bin_edges[node]
                lower = edges[states]
                upper = edges[states + 1]
                values = self.node_log_interval_probability(node, dataframe, lower, upper)
            else:
                values = self.node_logpdf(node, dataframe)
            if values.shape != (len(dataframe),) or not _rq1_s3_np.all(_rq1_s3_np.isfinite(values)):
                raise FloatingPointError(f'Invalid true representation score for {node!r}.')
            nodewise[node] = values
            total += values
        if return_nodewise:
            return (total, nodewise)
        return total

    def joint_logpdf(self, dataframe: pd.DataFrame, *, return_nodewise: bool=False) -> _rq1_s3_np.ndarray | tuple[_rq1_s3_np.ndarray, dict[str, _rq1_s3_np.ndarray]]:
        dataframe = dataframe[self.node_order].reset_index(drop=True)
        total = _rq1_s3_np.zeros(len(dataframe), dtype=float)
        nodewise: dict[str, _rq1_s3_np.ndarray] = {}
        for node in self.topological_order:
            values = self.node_logpdf(node, dataframe)
            if values.shape != (len(dataframe),) or not _rq1_s3_np.all(_rq1_s3_np.isfinite(values)):
                raise FloatingPointError(f'Invalid true log-density for {node!r}.')
            nodewise[node] = values
            total += values
        if return_nodewise:
            return (total, nodewise)
        return total

    def sample(self, size: int, random_state: int) -> _rq1_s3_pd.DataFrame:
        if size <= 0:
            raise ValueError('Monte Carlo sample size must be positive.')
        rng = _rq1_s3_np.random.default_rng(random_state)
        data: dict[str, _rq1_s3_np.ndarray] = {}
        for node in self.topological_order:
            parameters = self.parameters[node]
            family = str(parameters['family'])
            parent_frame = _rq1_s3_pd.DataFrame(data)
            if family == 'gaussian_root':
                values = rng.normal(loc=float(parameters['mean']), scale=float(parameters['std']), size=size)
            elif family == 'linear_gaussian':
                location = self._linear_location(parameters, parent_frame)
                values = location + rng.normal(loc=0.0, scale=float(parameters['noise_std']), size=size)
            elif family == 'gaussian_mixture_root':
                component = self._sample_component_indices(parameters['weights'], size, rng)
                means = _rq1_s3_np.asarray(parameters['means'], dtype=float)
                stds = _rq1_s3_np.asarray(parameters['stds'], dtype=float)
                values = rng.normal(loc=means[component], scale=stds[component])
            elif family == 'gated_gaussian':
                location = self._gated_location(parameters, parent_frame)
                values = location + rng.normal(loc=0.0, scale=float(parameters['noise_std']), size=size)
            elif family == 'skew_normal_root':
                values = _rq1_s3_skewnorm.rvs(float(parameters['shape']), loc=float(parameters['loc']), scale=float(parameters['scale']), size=size, random_state=rng)
            elif family == 'skew_normal_regression':
                location = self._linear_location(parameters, parent_frame)
                residual = _rq1_s3_skewnorm.rvs(float(parameters['residual_shape']), loc=float(parameters['residual_loc']), scale=float(parameters['residual_scale']), size=size, random_state=rng)
                factor = float(parameters.get('residual_scale_factor', 1.0))
                values = location + factor * residual
            elif family == 'skew_mixture_root':
                component = self._sample_component_indices(parameters['weights'], size, rng)
                means = _rq1_s3_np.asarray(parameters['means'], dtype=float)
                scales = _rq1_s3_np.asarray(parameters['scales'], dtype=float)
                shapes = _rq1_s3_np.asarray(parameters['shapes'], dtype=float)
                values = _rq1_s3_np.empty(size, dtype=float)
                for index in range(len(parameters['weights'])):
                    mask = component == index
                    if _rq1_s3_np.any(mask):
                        values[mask] = _rq1_s3_skewnorm.rvs(float(shapes[index]), loc=float(means[index]), scale=float(scales[index]), size=int(mask.sum()), random_state=rng)
            elif family == 'gated_skew_mixture':
                location = self._gated_location(parameters, parent_frame)
                mixture = parameters['residual_mixture']
                component = self._sample_component_indices(mixture['weights'], size, rng)
                means = _rq1_s3_np.asarray(mixture['means'], dtype=float)
                scales = _rq1_s3_np.asarray(mixture['scales'], dtype=float)
                shapes = _rq1_s3_np.asarray(mixture['shapes'], dtype=float)
                residual = _rq1_s3_np.empty(size, dtype=float)
                for index in range(len(mixture['weights'])):
                    mask = component == index
                    if _rq1_s3_np.any(mask):
                        residual[mask] = _rq1_s3_skewnorm.rvs(float(shapes[index]), loc=float(means[index]), scale=float(scales[index]), size=int(mask.sum()), random_state=rng)
                values = location + float(parameters['residual_scale_factor']) * residual
            else:
                raise ValueError(f'Unknown true family {family!r} for node {node!r}.')
            data[node] = _rq1_s3_np.asarray(values, dtype=float)
        return _rq1_s3_pd.DataFrame(data, columns=self.node_order)

class _rq1_s3_TransformedTruthParticleLikelihood:
    """SMC likelihood for p_T(x_C, d_D) with latent within-bin values."""

    def __init__(self, true_model: _rq1_s3_TrueContinuousDAG, transformer: Any, discrete_nodes: Sequence[str], *, particles: int=256, replicates: int=2, batch_size: int=64) -> None:
        if particles < 8:
            raise ValueError('particles must be >= 8')
        if replicates < 1:
            raise ValueError('replicates must be >= 1')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.true_model = true_model
        self.transformer = transformer
        self.discrete_nodes = tuple(map(str, discrete_nodes))
        self.discrete_set = set(self.discrete_nodes)
        self.particles = int(particles)
        self.replicates = int(replicates)
        self.batch_size = int(batch_size)

    @staticmethod
    def _systematic_indices(log_weights: _rq1_s3_np.ndarray, rng: _rq1_s3_np.random.Generator) -> _rq1_s3_np.ndarray:
        """Row-wise systematic resampling indices for log weights [B,P]."""
        b, p = log_weights.shape
        normalized = _rq1_s3_np.exp(log_weights - _rq1_s3_logsumexp(log_weights, axis=1, keepdims=True))
        cdf = _rq1_s3_np.cumsum(normalized, axis=1)
        cdf[:, -1] = 1.0
        starts = rng.random(b) / p
        positions = starts[:, None] + _rq1_s3_np.arange(p, dtype=float)[None, :] / p
        out = _rq1_s3_np.empty((b, p), dtype=_rq1_s3_np.int64)
        for i in range(b):
            out[i] = _rq1_s3_np.searchsorted(cdf[i], positions[i], side='right')
        return _rq1_s3_np.minimum(out, p - 1)

    @staticmethod
    def _categorical_rows(probabilities: _rq1_s3_np.ndarray, rng: _rq1_s3_np.random.Generator) -> _rq1_s3_np.ndarray:
        cdf = _rq1_s3_np.cumsum(probabilities, axis=1)
        cdf[:, -1] = 1.0
        u = rng.random(len(probabilities))
        return _rq1_s3_np.sum(u[:, None] > cdf, axis=1).astype(int)

    @staticmethod
    def _normal_prob(lower, upper, loc, scale):
        return _rq1_s3_TrueContinuousDAG._normal_interval_probability(lower, upper, loc, scale)

    @staticmethod
    def _skew_prob(lower, upper, shape, loc, scale):
        return _rq1_s3_TrueContinuousDAG._skew_interval_probability(lower, upper, shape, loc, scale)

    @staticmethod
    def _truncated_normal(lower, upper, loc, scale, rng):
        lower = _rq1_s3_np.asarray(lower, dtype=float)
        upper = _rq1_s3_np.asarray(upper, dtype=float)
        loc = _rq1_s3_np.asarray(loc, dtype=float)
        fl = _rq1_s3_norm.cdf(lower, loc=loc, scale=scale)
        fu = _rq1_s3_norm.cdf(upper, loc=loc, scale=scale)
        sl = _rq1_s3_norm.sf(lower, loc=loc, scale=scale)
        su = _rq1_s3_norm.sf(upper, loc=loc, scale=scale)
        u = rng.random(_rq1_s3_np.broadcast(lower, loc).shape)
        use_sf = fu > 0.9
        q = fl + u * _rq1_s3_np.maximum(fu - fl, 0.0)
        s = su + u * _rq1_s3_np.maximum(sl - su, 0.0)
        tiny = _rq1_s3_np.finfo(float).tiny
        q = _rq1_s3_np.clip(q, tiny, 1.0 - _rq1_s3_np.finfo(float).eps)
        s = _rq1_s3_np.clip(s, tiny, 1.0 - _rq1_s3_np.finfo(float).eps)
        x_cdf = _rq1_s3_norm.ppf(q, loc=loc, scale=scale)
        x_sf = _rq1_s3_norm.isf(s, loc=loc, scale=scale)
        return _rq1_s3_np.where(use_sf, x_sf, x_cdf)

    @staticmethod
    def _truncated_skew(lower, upper, shape, loc, scale, rng):
        lower = _rq1_s3_np.asarray(lower, dtype=float)
        upper = _rq1_s3_np.asarray(upper, dtype=float)
        loc = _rq1_s3_np.asarray(loc, dtype=float)
        z_lower = (lower - loc) / float(scale)
        z_upper = (upper - loc) / float(scale)
        t_lower = _rq1_s3_owens_t(z_lower, float(shape))
        t_upper = _rq1_s3_owens_t(z_upper, float(shape))
        fl = _rq1_s3_np.clip(_rq1_s3_norm.cdf(z_lower) - 2.0 * t_lower, 0.0, 1.0)
        fu = _rq1_s3_np.clip(_rq1_s3_norm.cdf(z_upper) - 2.0 * t_upper, 0.0, 1.0)
        sl = _rq1_s3_np.clip(_rq1_s3_norm.sf(z_lower) + 2.0 * t_lower, 0.0, 1.0)
        su = _rq1_s3_np.clip(_rq1_s3_norm.sf(z_upper) + 2.0 * t_upper, 0.0, 1.0)
        u = rng.random(_rq1_s3_np.broadcast(lower, loc).shape)
        use_sf = fu > 0.9
        q = fl + u * _rq1_s3_np.maximum(fu - fl, 0.0)
        s = su + u * _rq1_s3_np.maximum(sl - su, 0.0)
        tiny = _rq1_s3_np.finfo(float).tiny
        q = _rq1_s3_np.clip(q, tiny, 1.0 - _rq1_s3_np.finfo(float).eps)
        s = _rq1_s3_np.clip(s, tiny, 1.0 - _rq1_s3_np.finfo(float).eps)
        x_cdf = _rq1_s3_skewnorm.ppf(q, shape, loc=loc, scale=scale)
        x_sf = _rq1_s3_skewnorm.isf(s, shape, loc=loc, scale=scale)
        return _rq1_s3_np.where(use_sf, x_sf, x_cdf)

    def _component_spec(self, node: str, parent_frame: pd.DataFrame):
        """Return list[(prior_weight, loc_vector, scale, shape|None)]."""
        p = self.true_model.parameters[node]
        family = str(p['family'])
        n = len(parent_frame)
        if family == 'gaussian_root':
            return [(1.0, _rq1_s3_np.full(n, float(p['mean'])), float(p['std']), None)]
        if family == 'linear_gaussian':
            loc = self.true_model._linear_location(p, parent_frame)
            return [(1.0, loc, float(p['noise_std']), None)]
        if family == 'gaussian_mixture_root':
            return [(float(w), _rq1_s3_np.full(n, float(m)), float(sd), None) for w, m, sd in zip(p['weights'], p['means'], p['stds'])]
        if family == 'gated_gaussian':
            loc = self.true_model._gated_location(p, parent_frame)
            return [(1.0, loc, float(p['noise_std']), None)]
        if family == 'skew_normal_root':
            return [(1.0, _rq1_s3_np.full(n, float(p['loc'])), float(p['scale']), float(p['shape']))]
        if family == 'skew_normal_regression':
            base = self.true_model._linear_location(p, parent_frame)
            factor = float(p.get('residual_scale_factor', 1.0))
            loc = base + factor * float(p['residual_loc'])
            scale = factor * float(p['residual_scale'])
            return [(1.0, loc, scale, float(p['residual_shape']))]
        if family == 'skew_mixture_root':
            return [(float(w), _rq1_s3_np.full(n, float(m)), float(sc), float(sh)) for w, m, sc, sh in zip(p['weights'], p['means'], p['scales'], p['shapes'])]
        if family == 'gated_skew_mixture':
            base = self.true_model._gated_location(p, parent_frame)
            mix = p['residual_mixture']
            factor = float(p['residual_scale_factor'])
            return [(float(w), base + factor * float(m), factor * float(sc), float(sh)) for w, m, sc, sh in zip(mix['weights'], mix['means'], mix['scales'], mix['shapes'])]
        raise ValueError(f'Unsupported family {family!r}')

    def _sample_truncated_conditional(self, node: str, parent_frame: pd.DataFrame, lower: _rq1_s3_np.ndarray, upper: _rq1_s3_np.ndarray, rng: _rq1_s3_np.random.Generator) -> _rq1_s3_np.ndarray:
        specs = self._component_spec(node, parent_frame)
        component_probs = []
        for weight, loc, scale, shape in specs:
            if shape is None:
                prob = self._normal_prob(lower, upper, loc, scale)
            else:
                prob = self._skew_prob(lower, upper, shape, loc, scale)
            component_probs.append(float(weight) * _rq1_s3_np.asarray(prob, dtype=float))
        weighted = _rq1_s3_np.column_stack(component_probs)
        total = _rq1_s3_np.clip(weighted.sum(axis=1), _rq1_s3__MIN_PROBABILITY, None)
        posterior = weighted / total[:, None]
        component = self._categorical_rows(posterior, rng)
        samples = _rq1_s3_np.empty(len(parent_frame), dtype=float)
        for k, (_, loc, scale, shape) in enumerate(specs):
            mask = component == k
            if not _rq1_s3_np.any(mask):
                continue
            if shape is None:
                samples[mask] = self._truncated_normal(lower[mask], upper[mask], _rq1_s3_np.asarray(loc)[mask], scale, rng)
            else:
                samples[mask] = self._truncated_skew(lower[mask], upper[mask], shape, _rq1_s3_np.asarray(loc)[mask], scale, rng)
        if not _rq1_s3_np.all(_rq1_s3_np.isfinite(samples)):
            raise FloatingPointError(f'Non-finite truncated samples for {node}')
        return samples

    def _parent_frame(self, node: str, raw_batch: _rq1_s3_pd.DataFrame, latent: Mapping[str, _rq1_s3_np.ndarray]) -> _rq1_s3_pd.DataFrame:
        b = len(raw_batch)
        p = self.particles
        data = {}
        for parent in self.true_model.graph.predecessors(node):
            parent = str(parent)
            if parent in self.discrete_set:
                data[parent] = latent[parent].reshape(-1)
            else:
                data[parent] = _rq1_s3_np.repeat(raw_batch[parent].to_numpy(dtype=float), p)
        return _rq1_s3_pd.DataFrame(data, index=_rq1_s3_np.arange(b * p))

    def _one_replicate(self, raw: pd.DataFrame, transformed: pd.DataFrame, seed: int) -> _rq1_s3_np.ndarray:
        rng = _rq1_s3_np.random.default_rng(seed)
        out = _rq1_s3_np.empty(len(raw), dtype=float)
        pcount = self.particles
        for start in range(0, len(raw), self.batch_size):
            stop = min(start + self.batch_size, len(raw))
            rb = raw.iloc[start:stop].reset_index(drop=True)
            tb = transformed.iloc[start:stop].reset_index(drop=True)
            b = len(rb)
            loglik = _rq1_s3_np.zeros(b, dtype=float)
            latent: dict[str, _rq1_s3_np.ndarray] = {}
            for node in self.true_model.topological_order:
                node = str(node)
                parent_frame = self._parent_frame(node, rb, latent)
                if node in self.discrete_set:
                    states = tb[node].to_numpy(dtype=int)
                    edges = self.transformer.bin_edges[node]
                    lower_b = edges[states]
                    upper_b = edges[states + 1]
                    lower = _rq1_s3_np.repeat(lower_b, pcount)
                    upper = _rq1_s3_np.repeat(upper_b, pcount)
                    log_inc_flat = self.true_model.node_log_interval_probability(node, parent_frame, lower, upper)
                    sampled = self._sample_truncated_conditional(node, parent_frame, lower, upper, rng)
                    latent[node] = sampled.reshape(b, pcount)
                else:
                    frame = parent_frame.copy()
                    frame[node] = _rq1_s3_np.repeat(rb[node].to_numpy(dtype=float), pcount)
                    log_inc_flat = self.true_model.node_logpdf(node, frame)
                log_inc = _rq1_s3_np.asarray(log_inc_flat, dtype=float).reshape(b, pcount)
                if not _rq1_s3_np.all(_rq1_s3_np.isfinite(log_inc)):
                    raise FloatingPointError(f'Non-finite particle weights at {node}')
                loglik += _rq1_s3_logsumexp(log_inc, axis=1) - _rq1_s3_math.log(pcount)
                indices = self._systematic_indices(log_inc, rng)
                if latent:
                    row = _rq1_s3_np.arange(b)[:, None]
                    for key in latent:
                        latent[key] = latent[key][row, indices]
            out[start:stop] = loglik
        return out

    def loglikelihood(self, raw_evaluation: pd.DataFrame, base_seed: int) -> tuple[_rq1_s3_np.ndarray, dict[str, Any]]:
        raw = raw_evaluation[self.true_model.node_order].reset_index(drop=True)
        transformed = self.transformer.transform(raw)
        replicate_logs = []
        for r in range(self.replicates):
            seed = int(_rq1_s3_np.random.SeedSequence([base_seed, 77123, r]).generate_state(1)[0])
            replicate_logs.append(self._one_replicate(raw, transformed, seed))
        matrix = _rq1_s3_np.vstack(replicate_logs)
        combined = _rq1_s3_logsumexp(matrix, axis=0) - _rq1_s3_math.log(self.replicates)
        if self.replicates > 1:
            row_sd = _rq1_s3_np.std(matrix, axis=0, ddof=1)
            replicate_mean_logs = _rq1_s3_np.mean(matrix, axis=1)
            diagnostics = {'replicate_mean_log_p': replicate_mean_logs.tolist(), 'replicate_mean_log_p_std': float(_rq1_s3_np.std(replicate_mean_logs, ddof=1)), 'median_row_log_p_replicate_sd': float(_rq1_s3_np.median(row_sd)), 'p95_row_log_p_replicate_sd': float(_rq1_s3_np.quantile(row_sd, 0.95)), 'max_row_log_p_replicate_sd': float(_rq1_s3_np.max(row_sd))}
        else:
            diagnostics = {'replicate_mean_log_p': [float(_rq1_s3_np.mean(matrix[0]))], 'replicate_mean_log_p_std': _rq1_s3_np.nan, 'median_row_log_p_replicate_sd': _rq1_s3_np.nan, 'p95_row_log_p_replicate_sd': _rq1_s3_np.nan, 'max_row_log_p_replicate_sd': _rq1_s3_np.nan}
        diagnostics.update({'particles': self.particles, 'replicates': self.replicates, 'batch_size': self.batch_size, 'combination': 'log(mean of independent particle likelihood estimates)'})
        return (_rq1_s3_np.asarray(combined, dtype=float), diagnostics)

    def validate_continuous_identity(self, raw: pd.DataFrame) -> float:
        """No-discretisation special case must equal exact joint logpdf."""
        exact = self.true_model.joint_logpdf(raw)
        total = _rq1_s3_np.zeros(len(raw), dtype=float)
        for node in self.true_model.topological_order:
            total += self.true_model.node_logpdf(str(node), raw)
        return float(_rq1_s3_np.max(_rq1_s3_np.abs(_rq1_s3_np.asarray(exact) - total)))

class _rq1_s3_SavedQuantileTransform:
    """Apply the exact bin boundaries saved during structure learning."""

    def __init__(self, cardinalities: Mapping[str, int], bin_edges: Mapping[str, Sequence[Any]]) -> None:
        self.cardinalities = {str(node): int(cardinality) for node, cardinality in cardinalities.items()}
        self.bin_edges = {str(node): _rq1_s3_np.asarray([_rq1_s3_decode_float(value) for value in edges], dtype=float) for node, edges in bin_edges.items()}
        if set(self.cardinalities) != set(self.bin_edges):
            raise ValueError('Saved cardinalities and bin edges refer to different node sets.')
        for node, cardinality in self.cardinalities.items():
            edges = self.bin_edges[node]
            if edges.shape != (cardinality + 1,) or not _rq1_s3_np.all(_rq1_s3_np.diff(edges) > 0):
                raise ValueError(f'Invalid saved boundaries for {node!r}.')

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> '_rq1_s3_SavedQuantileTransform':
        return cls(metadata['cardinalities'], metadata['bin_edges'])

    @property
    def nodes(self) -> set[str]:
        return set(self.cardinalities)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        transformed = dataframe.copy()
        for node, cardinality in self.cardinalities.items():
            values = transformed[node].to_numpy(dtype=float)
            internal_edges = self.bin_edges[node][1:-1]
            states = _rq1_s3_np.searchsorted(internal_edges, values, side='left').astype(_rq1_s3_np.int64)
            if states.min(initial=0) < 0 or states.max(initial=0) >= cardinality:
                raise ValueError(f'Saved discretisation produced an invalid state for {node!r}.')
            transformed[node] = states
        return transformed

@_rq1_s3_dataclass
class _rq1_s3_GaussianConditional:
    child: str
    parents: tuple[str, ...]
    coefficients: np.ndarray
    standard_deviation: float

@_rq1_s3_dataclass
class _rq1_s3_CPTConditional:
    child: str
    parents: tuple[str, ...]
    cardinality: int
    marginal_probabilities: np.ndarray
    tables: dict[tuple[int, ...], np.ndarray]

@_rq1_s3_dataclass
class _rq1_s3_LogisticConditional:
    kind: str
    parents: tuple[str, ...]
    cardinality: int
    prior_probabilities: np.ndarray
    classifier: Any = None
    feature_means: np.ndarray | None = None
    feature_stds: np.ndarray | None = None
    classes: np.ndarray | None = None
    converged: bool = True

@_rq1_s3_dataclass
class _rq1_s3_MixtureConditional:
    """Evaluation-time fitted V2 mixture-of-linear-experts conditional."""
    child: str
    parents: tuple[str, ...]
    n_components: int
    feature_means: np.ndarray
    feature_stds: np.ndarray
    child_mean: float
    child_std: float
    betas: np.ndarray
    variances: np.ndarray
    gate_coefficients: np.ndarray
    converged: bool
    iterations: int
    gradient_norm: float
    min_effective_component_size: float

class _rq1_s3_LearnedDAGDistribution:
    """Fit Q for the continuous, discrete, or hybrid learned DAG."""

    def __init__(self, graph: nx.DiGraph, nodes: Sequence[str], paradigm: str, *, discrete_nodes: Sequence[str]=(), continuous_nodes: Sequence[str]=(), transform: _rq1_s3_SavedQuantileTransform | None=None, cpt_alpha: float=1.0, min_regime_size: int=5, logistic_c: float=1000.0, hybrid_selected_k_by_configuration: Mapping[str, Mapping[str, int]] | None=None, hybrid_continuous_scaling: Mapping[str, Any] | None=None, hybrid_prior_hyperparameters: Mapping[str, float] | None=None) -> None:
        if paradigm not in _rq1_s3_PARADIGMS:
            raise ValueError(f'Unknown paradigm: {paradigm!r}')
        if cpt_alpha <= 0:
            raise ValueError('cpt_alpha must be positive.')
        if min_regime_size < 1:
            raise ValueError('min_regime_size must be positive.')
        if logistic_c <= 0:
            raise ValueError('logistic_c must be positive.')
        self.graph = graph.copy()
        self.nodes = [str(node) for node in nodes]
        self.paradigm = paradigm
        self.discrete_nodes = set((str(node) for node in discrete_nodes))
        self.continuous_nodes = set((str(node) for node in continuous_nodes))
        self.transformer = transform
        self.cpt_alpha = float(cpt_alpha)
        self.min_regime_size = int(min_regime_size)
        self.logistic_c = float(logistic_c)
        self.hybrid_selected_k_by_configuration = {str(child): {str(key): int(value) for key, value in choices.items()} for child, choices in (hybrid_selected_k_by_configuration or {}).items()}
        self.hybrid_continuous_scaling = dict(hybrid_continuous_scaling or {})
        self.hybrid_prior_hyperparameters = {str(key): float(value) for key, value in (hybrid_prior_hyperparameters or {}).items()}
        self.models: dict[str, Any] = {}
        self.diagnostics = {'gaussian_models': 0, 'mixture_models_attempted': 0, 'mixture_models_fitted': 0, 'mixture_models_nonconverged': 0, 'mixture_k2_models': 0, 'mixture_k3_models': 0, 'cpt_models': 0, 'logistic_models_attempted': 0, 'logistic_models_fitted': 0, 'logistic_models_nonconverged': 0, 'prior_fallbacks': 0, 'regime_fallbacks': 0}
        if set(self.graph.nodes) != set(self.nodes):
            raise ValueError('Learned graph nodes do not match the saved node list.')
        if self.paradigm == 'continuous':
            if self.transformer is not None:
                raise ValueError('The continuous model must not have a discretiser.')
        elif self.paradigm == 'discrete':
            if self.transformer is None or self.transformer.nodes != set(self.nodes):
                raise ValueError('The discrete model requires saved boundaries for every node.')
        else:
            if self.discrete_nodes | self.continuous_nodes != set(self.nodes):
                raise ValueError('Hybrid node sets must cover all nodes.')
            if self.discrete_nodes & self.continuous_nodes:
                raise ValueError('Hybrid node sets overlap.')
            if self.transformer is None or self.transformer.nodes != self.discrete_nodes:
                raise ValueError('Hybrid saved boundaries must match the designated discrete nodes.')
            if set(self.hybrid_selected_k_by_configuration) != self.continuous_nodes:
                raise ValueError('Hybrid V2 evaluation requires saved final K choices for every designated continuous child.')
            means = self.hybrid_continuous_scaling.get('continuous_means', {})
            stds = self.hybrid_continuous_scaling.get('continuous_stds', {})
            if set(means) != self.continuous_nodes or set(stds) != self.continuous_nodes:
                raise ValueError('Hybrid V2 evaluation requires the saved structure-learning continuous means/stds for every continuous node.')
            if any((float(stds[node]) <= 0.0 for node in self.continuous_nodes)):
                raise ValueError('Saved Hybrid V2 continuous standard deviations must be positive.')
            required_priors = {'continuous_coefficient_prior_variance', 'continuous_intercept_prior_variance', 'continuous_inverse_gamma_shape', 'continuous_inverse_gamma_scale', 'clg_coefficient_prior_variance', 'clg_intercept_prior_variance', 'clg_inverse_gamma_shape', 'clg_inverse_gamma_scale', 'mixture_gate_coefficient_prior_variance', 'mixture_gate_intercept_prior_variance'}
            missing_priors = required_priors - set(self.hybrid_prior_hyperparameters)
            if missing_priors:
                raise ValueError('Hybrid V2 evaluation is missing saved frozen priors: ' + ', '.join(sorted(missing_priors)))
            if any((self.hybrid_prior_hyperparameters[name] <= 0.0 for name in required_priors)):
                raise ValueError('Saved Hybrid V2 prior hyperparameters must be positive.')

    @staticmethod
    def _configuration_key(values: Sequence[Any]) -> tuple[int, ...]:
        return tuple((int(value) for value in values))

    @staticmethod
    def _design_matrix(dataframe: pd.DataFrame, parents: Sequence[str]) -> _rq1_s3_np.ndarray:
        if not parents:
            return _rq1_s3_np.ones((len(dataframe), 1), dtype=float)
        return _rq1_s3_np.column_stack([_rq1_s3_np.ones(len(dataframe), dtype=float), dataframe[list(parents)].to_numpy(dtype=float)])

    def _fit_gaussian(self, dataframe: pd.DataFrame, child: str, parents: Sequence[str]) -> _rq1_s3_GaussianConditional:
        parent_tuple = tuple(parents)
        y = dataframe[child].to_numpy(dtype=float)
        design = self._design_matrix(dataframe, parent_tuple)
        coefficients, _, _, _ = _rq1_s3_np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coefficients
        standard_deviation = max(float(_rq1_s3_np.sqrt(_rq1_s3_np.mean(residuals ** 2))), _rq1_s3__MIN_STANDARD_DEVIATION)
        self.diagnostics['gaussian_models'] += 1
        return _rq1_s3_GaussianConditional(child=child, parents=parent_tuple, coefficients=coefficients, standard_deviation=standard_deviation)

    @staticmethod
    def _eval_gaussian(model: _rq1_s3_GaussianConditional, dataframe: pd.DataFrame) -> _rq1_s3_np.ndarray:
        if model.parents:
            design = _rq1_s3_np.column_stack([_rq1_s3_np.ones(len(dataframe), dtype=float), dataframe[list(model.parents)].to_numpy(dtype=float)])
        else:
            design = _rq1_s3_np.ones((len(dataframe), 1), dtype=float)
        location = design @ model.coefficients
        return _rq1_s3_norm.logpdf(dataframe[model.child].to_numpy(dtype=float), loc=location, scale=model.standard_deviation)

    def _smoothed_probabilities(self, values: pd.Series, cardinality: int) -> np.ndarray:
        counts = values.astype(int).value_counts().reindex(range(cardinality), fill_value=0).to_numpy(dtype=float)
        return (counts + self.cpt_alpha) / (counts.sum() + self.cpt_alpha * cardinality)

    def _fit_cpt(self, dataframe: pd.DataFrame, child: str, parents: Sequence[str], cardinality: int) -> _rq1_s3_CPTConditional:
        parent_tuple = tuple(parents)
        marginal = self._smoothed_probabilities(dataframe[child], cardinality)
        tables: dict[tuple[int, ...], np.ndarray] = {}
        if parent_tuple:
            group_key: Any = parent_tuple[0] if len(parent_tuple) == 1 else list(parent_tuple)
            for key, group in dataframe.groupby(group_key, sort=False, observed=True):
                normalized = key if isinstance(key, tuple) else (key,)
                tables[self._configuration_key(normalized)] = self._smoothed_probabilities(group[child], cardinality)
        self.diagnostics['cpt_models'] += 1
        return _rq1_s3_CPTConditional(child=child, parents=parent_tuple, cardinality=cardinality, marginal_probabilities=marginal, tables=tables)

    @staticmethod
    def _eval_cpt(model: _rq1_s3_CPTConditional, dataframe: pd.DataFrame) -> _rq1_s3_np.ndarray:
        child_values = dataframe[model.child].to_numpy(dtype=int)
        if not model.parents:
            probabilities = model.marginal_probabilities[child_values]
            return _rq1_s3_np.log(_rq1_s3_np.clip(probabilities, _rq1_s3__MIN_PROBABILITY, None))
        parent_values = dataframe[list(model.parents)].to_numpy(dtype=int)
        result = _rq1_s3_np.empty(len(dataframe), dtype=float)
        for index, (configuration, child_value) in enumerate(zip(parent_values, child_values)):
            probabilities = model.tables.get(tuple((int(value) for value in configuration)), model.marginal_probabilities)
            result[index] = _rq1_s3_np.log(max(float(probabilities[child_value]), _rq1_s3__MIN_PROBABILITY))
        return result

    def _fit_logistic(self, dataframe: pd.DataFrame, child: str, parents: Sequence[str], cardinality: int, fallback_prior: _rq1_s3_np.ndarray) -> _rq1_s3_LogisticConditional:
        parent_tuple = tuple(parents)
        y = dataframe[child].to_numpy(dtype=int)
        local_prior = self._smoothed_probabilities(dataframe[child], cardinality) if len(dataframe) else fallback_prior
        minimum_size = max(self.min_regime_size, cardinality * 3, len(parent_tuple) + 2)
        if len(dataframe) < minimum_size or _rq1_s3_np.unique(y).size < 2:
            self.diagnostics['prior_fallbacks'] += 1
            return _rq1_s3_LogisticConditional(kind='prior', parents=parent_tuple, cardinality=cardinality, prior_probabilities=local_prior)
        self.diagnostics['logistic_models_attempted'] += 1
        features = dataframe[list(parent_tuple)].to_numpy(dtype=float)
        feature_means = features.mean(axis=0)
        feature_stds = features.std(axis=0, ddof=0)
        feature_stds = _rq1_s3_np.where(feature_stds < _rq1_s3__MIN_STANDARD_DEVIATION, 1.0, feature_stds)
        standardized = (features - feature_means) / feature_stds
        classifier = _rq1_s3_LogisticRegression(solver='lbfgs', C=self.logistic_c, max_iter=1000)
        converged = True
        try:
            with _rq1_s3_warnings.catch_warnings(record=True) as captured:
                _rq1_s3_warnings.simplefilter('always', _rq1_s3_ConvergenceWarning)
                classifier.fit(standardized, y)
                converged = not any((issubclass(warning.category, _rq1_s3_ConvergenceWarning) for warning in captured))
        except Exception:
            self.diagnostics['prior_fallbacks'] += 1
            return _rq1_s3_LogisticConditional(kind='prior', parents=parent_tuple, cardinality=cardinality, prior_probabilities=local_prior)
        self.diagnostics['logistic_models_fitted'] += 1
        if not converged:
            self.diagnostics['logistic_models_nonconverged'] += 1
        return _rq1_s3_LogisticConditional(kind='logistic', parents=parent_tuple, cardinality=cardinality, prior_probabilities=local_prior, classifier=classifier, feature_means=feature_means, feature_stds=feature_stds, classes=classifier.classes_.astype(int), converged=converged)

    @staticmethod
    def _eval_logistic(model: _rq1_s3_LogisticConditional, dataframe: pd.DataFrame, indices: _rq1_s3_np.ndarray, child_values: _rq1_s3_np.ndarray) -> _rq1_s3_np.ndarray:
        if model.kind == 'prior':
            return _rq1_s3_np.log(_rq1_s3_np.clip(model.prior_probabilities[child_values], _rq1_s3__MIN_PROBABILITY, None))
        features = dataframe.iloc[indices][list(model.parents)].to_numpy(dtype=float)
        standardized = (features - model.feature_means) / model.feature_stds
        predicted = model.classifier.predict_proba(standardized)
        full_probabilities = _rq1_s3_np.full((len(indices), model.cardinality), _rq1_s3__MIN_PROBABILITY, dtype=float)
        full_probabilities[:, model.classes] = predicted
        full_probabilities /= full_probabilities.sum(axis=1, keepdims=True)
        return _rq1_s3_np.log(_rq1_s3_np.clip(full_probabilities[_rq1_s3_np.arange(len(indices)), child_values], _rq1_s3__MIN_PROBABILITY, None))

    @staticmethod
    def _configuration_storage_key(values: Sequence[Any]) -> str:
        if len(values) == 0:
            return 'none'
        return '|'.join((str(int(value)) for value in values))

    def _selected_components(self, child: str, configuration: Sequence[Any]) -> int:
        child_choices = self.hybrid_selected_k_by_configuration.get(str(child))
        if child_choices is None:
            raise ValueError(f'Missing saved Hybrid V2 K choices for {child!r}.')
        key = self._configuration_storage_key(configuration)
        if key not in child_choices:
            raise ValueError(f'Missing saved Hybrid V2 K choice for {child!r}, configuration {key!r}.')
        value = int(child_choices[key])
        if value not in (1, 2, 3):
            raise ValueError(f'Invalid saved Hybrid V2 component count K={value} for {child!r}.')
        return value

    def _saved_hybrid_continuous_scaling(self, child: str, parents: Sequence[str]) -> tuple[_rq1_s3_np.ndarray, _rq1_s3_np.ndarray, float, float]:
        means = self.hybrid_continuous_scaling['continuous_means']
        stds = self.hybrid_continuous_scaling['continuous_stds']
        parent_tuple = tuple(parents)
        feature_means = _rq1_s3_np.asarray([float(means[parent]) for parent in parent_tuple], dtype=float)
        feature_stds = _rq1_s3_np.asarray([float(stds[parent]) for parent in parent_tuple], dtype=float)
        child_mean = float(means[child])
        child_std = float(stds[child])
        return (feature_means, feature_stds, child_mean, child_std)

    def _hybrid_mixture_prior_values(self, *, prior_family: str, p: int) -> tuple[_rq1_s3_np.ndarray, _rq1_s3_np.ndarray, float, float]:
        if prior_family not in {'continuous', 'clg'}:
            raise ValueError(f'Unknown Hybrid V2 mixture prior family: {prior_family!r}.')
        priors = self.hybrid_prior_hyperparameters
        expert_variances = _rq1_s3_np.full(p, priors[f'{prior_family}_coefficient_prior_variance'], dtype=float)
        expert_variances[0] = priors[f'{prior_family}_intercept_prior_variance']
        a0 = priors[f'{prior_family}_inverse_gamma_shape']
        b0 = priors[f'{prior_family}_inverse_gamma_scale']
        return (expert_variances, 1.0 / expert_variances, a0, b0)

    @staticmethod
    def _mixture_initializations(X: _rq1_s3_np.ndarray, y: _rq1_s3_np.ndarray, n_components: int) -> list[_rq1_s3_np.ndarray]:
        n, p = X.shape
        beta_global, *_ = _rq1_s3_np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ beta_global
        residual_variance = max(float(_rq1_s3_np.var(residual)), 0.05)
        response_scale = max(float(_rq1_s3_np.std(y, ddof=0)), 0.25)

        def pack(betas: _rq1_s3_np.ndarray, log_variances: _rq1_s3_np.ndarray, gamma: _rq1_s3_np.ndarray) -> _rq1_s3_np.ndarray:
            return _rq1_s3_np.concatenate([betas.ravel(), log_variances.ravel(), gamma.ravel()])
        starts: list[_rq1_s3_np.ndarray] = []
        offsets = _rq1_s3_np.linspace(-0.75, 0.75, n_components) * response_scale
        betas = _rq1_s3_np.tile(beta_global, (n_components, 1))
        betas[:, 0] += offsets
        log_variances = _rq1_s3_np.full(n_components, _rq1_s3_np.log(residual_variance))
        starts.append(pack(betas, log_variances, _rq1_s3_np.zeros((n_components - 1, p))))
        if n >= max(10, n_components * (p + 2)):
            boundaries = _rq1_s3_np.quantile(residual, _rq1_s3_np.linspace(0.0, 1.0, n_components + 1)[1:-1])
            labels = _rq1_s3_np.digitize(residual, boundaries, right=True)
            q_betas = _rq1_s3_np.tile(beta_global, (n_components, 1))
            q_log_variances = _rq1_s3_np.full(n_components, _rq1_s3_np.log(residual_variance))
            for component in range(n_components):
                mask = labels == component
                if int(mask.sum()) >= p + 2:
                    beta_component, *_ = _rq1_s3_np.linalg.lstsq(X[mask], y[mask], rcond=None)
                    q_betas[component] = beta_component
                    component_residual = y[mask] - X[mask] @ beta_component
                    q_log_variances[component] = _rq1_s3_np.log(max(float(_rq1_s3_np.var(component_residual)), 0.03))
            starts.append(pack(q_betas, q_log_variances, _rq1_s3_np.zeros((n_components - 1, p))))
        base = starts[0]
        while len(starts) < _rq1_s3__V2_Q_MIXTURE_RESTARTS:
            index = len(starts)
            perturbation = 0.04 * _rq1_s3_np.sin(_rq1_s3_np.arange(base.size, dtype=float) + 1.7 * index)
            starts.append(base + perturbation)
        return starts[:_rq1_s3__V2_Q_MIXTURE_RESTARTS]

    def _fit_mixture(self, dataframe: pd.DataFrame, child: str, parents: Sequence[str], n_components: int, *, prior_family: str) -> _rq1_s3_MixtureConditional:
        if n_components not in (2, 3):
            raise ValueError('Mixture fitting requires K=2 or K=3.')
        parent_tuple = tuple(parents)
        self.diagnostics['mixture_models_attempted'] += 1
        self.diagnostics[f'mixture_k{n_components}_models'] += 1
        feature_means, feature_stds, child_mean, child_std = self._saved_hybrid_continuous_scaling(child, parent_tuple)
        n = len(dataframe)
        if parent_tuple:
            raw_features = dataframe[list(parent_tuple)].to_numpy(dtype=float)
            features = (raw_features - feature_means) / feature_stds
            X = _rq1_s3_np.column_stack([_rq1_s3_np.ones(n), features])
        else:
            X = _rq1_s3_np.ones((n, 1), dtype=float)
        raw_y = dataframe[child].to_numpy(dtype=float)
        y = (raw_y - child_mean) / child_std
        _, p = X.shape
        H = _rq1_s3_helmert(n_components, full=False).T
        free_components = n_components - 1
        beta_size = n_components * p
        variance_size = n_components
        expert_variances, expert_precision, a0, b0 = self._hybrid_mixture_prior_values(prior_family=prior_family, p=p)
        expert_logdet = float(_rq1_s3_np.sum(_rq1_s3_np.log(expert_variances)))
        gate_variances = _rq1_s3_np.full((free_components, p), self.hybrid_prior_hyperparameters['mixture_gate_coefficient_prior_variance'], dtype=float)
        gate_variances[:, 0] = self.hybrid_prior_hyperparameters['mixture_gate_intercept_prior_variance']

        def unpack(theta: _rq1_s3_np.ndarray):
            theta = _rq1_s3_np.asarray(theta, dtype=float)
            betas = theta[:beta_size].reshape(n_components, p)
            start = beta_size
            log_variances = theta[start:start + variance_size]
            gamma = theta[start + variance_size:].reshape(free_components, p)
            return (betas, log_variances, gamma)
        cached_theta: _rq1_s3_np.ndarray | None = None
        cached_state: Any = None

        def state(theta: _rq1_s3_np.ndarray):
            nonlocal cached_theta, cached_state
            theta = _rq1_s3_np.asarray(theta, dtype=float)
            if cached_theta is not None and _rq1_s3_np.array_equal(theta, cached_theta):
                return cached_state
            betas, log_variances, gamma = unpack(theta)
            if _rq1_s3_np.any(_rq1_s3_np.abs(log_variances) > 30.0):
                raise FloatingPointError('V2 Q mixture log-variance left the stable range.')
            variances = _rq1_s3_np.exp(log_variances)
            gate_coefficients = H @ gamma
            logits = X @ gate_coefficients.T
            log_gate = logits - _rq1_s3_logsumexp(logits, axis=1, keepdims=True)
            means = X @ betas.T
            residuals = y[:, None] - means
            log_normal = -0.5 * (_rq1_s3_np.log(2.0 * _rq1_s3_np.pi) + log_variances[None, :] + residuals ** 2 / variances[None, :])
            component_log_joint = log_gate + log_normal
            log_density = _rq1_s3_logsumexp(component_log_joint, axis=1)
            responsibilities = _rq1_s3_np.exp(component_log_joint - log_density[:, None])
            gate_probabilities = _rq1_s3_np.exp(log_gate)
            result = (betas, log_variances, gamma, variances, gate_coefficients, residuals, log_density, responsibilities, gate_probabilities)
            cached_theta = theta.copy()
            cached_state = result
            return result

        def log_joint(theta: _rq1_s3_np.ndarray) -> float:
            betas, log_variances, gamma, variances, _, _, log_density, _, _ = state(theta)
            value = float(_rq1_s3_np.sum(log_density))
            beta_quadratic = _rq1_s3_np.sum(betas ** 2 * expert_precision[None, :], axis=1)
            value += float(_rq1_s3_np.sum(-0.5 * (p * _rq1_s3_np.log(2.0 * _rq1_s3_np.pi) + expert_logdet + p * log_variances + beta_quadratic / variances)))
            value += float(_rq1_s3_np.sum(a0 * _rq1_s3_np.log(b0) - _rq1_s3_gammaln(a0) - (a0 + 1.0) * log_variances - b0 / variances + log_variances))
            value += float(_rq1_s3_np.sum(-0.5 * (gamma ** 2 / gate_variances + _rq1_s3_np.log(2.0 * _rq1_s3_np.pi * gate_variances))))
            return value

        def objective(theta: _rq1_s3_np.ndarray) -> float:
            try:
                value = -log_joint(theta)
            except (FloatingPointError, OverflowError, ValueError):
                return 1e+100
            return float(value) if _rq1_s3_np.isfinite(value) else 1e+100

        def gradient(theta: _rq1_s3_np.ndarray) -> _rq1_s3_np.ndarray:
            betas, log_variances, gamma, variances, _, residuals, _, responsibilities, gate_probabilities = state(theta)
            del log_variances, gamma
            beta_gradient = _rq1_s3_np.empty_like(betas)
            log_variance_gradient = _rq1_s3_np.empty(n_components, dtype=float)
            for component in range(n_components):
                weights = responsibilities[:, component]
                beta_gradient[component] = -X.T @ (weights * residuals[:, component]) / variances[component] + betas[component] * expert_precision / variances[component]
                beta_quadratic = float(_rq1_s3_np.sum(betas[component] ** 2 * expert_precision))
                log_variance_gradient[component] = float(_rq1_s3_np.sum(weights * (0.5 - 0.5 * residuals[:, component] ** 2 / variances[component]))) + 0.5 * p + a0 - 0.5 * beta_quadratic / variances[component] - b0 / variances[component]
            gate_gradient = H.T @ ((gate_probabilities - responsibilities).T @ X)
            gate_gradient += unpack(theta)[2] / gate_variances
            result = _rq1_s3_np.concatenate([beta_gradient.ravel(), log_variance_gradient.ravel(), gate_gradient.ravel()])
            if not _rq1_s3_np.all(_rq1_s3_np.isfinite(result)):
                raise FloatingPointError('Non-finite V2 Q mixture MAP gradient.')
            return result
        starts = self._mixture_initializations(X, y, n_components)
        bounds = [(None, None)] * beta_size + [(-8.0, 4.0)] * variance_size + [(None, None)] * ((n_components - 1) * p)
        candidates: list[tuple[Any, float, float]] = []
        for initial in starts:
            result = _rq1_s3_minimize(objective, initial, method='L-BFGS-B', jac=gradient, bounds=bounds, options={'maxiter': _rq1_s3__V2_Q_MIXTURE_MAX_ITER, 'ftol': 1e-10, 'gtol': 1e-07, 'maxls': 50})
            if not _rq1_s3_np.isfinite(result.fun):
                continue
            theta = _rq1_s3_np.asarray(result.x, dtype=float)
            try:
                effective_counts = state(theta)[7].sum(axis=0)
                min_effective_count = float(_rq1_s3_np.min(effective_counts))
                grad_norm = float(_rq1_s3_np.linalg.norm(gradient(theta), ord=_rq1_s3_np.inf))
            except (FloatingPointError, OverflowError, ValueError):
                continue
            if _rq1_s3_np.isfinite(min_effective_count) and min_effective_count >= _rq1_s3__V2_Q_MIXTURE_MIN_EFFECTIVE_COUNT and _rq1_s3_np.isfinite(grad_norm):
                candidates.append((result, min_effective_count, grad_norm))
        if not candidates:
            raise FloatingPointError(f'All evaluation-time V2 MAP mixture fits failed the finite/interior checks for {child!r} | {parent_tuple}, K={n_components}.')
        best_result, min_effective_count, grad_norm = min(candidates, key=lambda item: float(item[0].fun))
        theta = _rq1_s3_np.asarray(best_result.x, dtype=float)
        betas, _, _, variances, gate_coefficients, _, _, _, _ = state(theta)
        converged = bool(best_result.success or grad_norm <= 0.001)
        self.diagnostics['mixture_models_fitted'] += 1
        if not converged:
            self.diagnostics['mixture_models_nonconverged'] += 1
        return _rq1_s3_MixtureConditional(child=child, parents=parent_tuple, n_components=int(n_components), feature_means=feature_means, feature_stds=feature_stds, child_mean=child_mean, child_std=child_std, betas=_rq1_s3_np.asarray(betas, dtype=float), variances=_rq1_s3_np.asarray(variances, dtype=float), gate_coefficients=_rq1_s3_np.asarray(gate_coefficients, dtype=float), converged=converged, iterations=int(getattr(best_result, 'nit', 0)), gradient_norm=grad_norm, min_effective_component_size=min_effective_count)

    @staticmethod
    def _eval_mixture(model: _rq1_s3_MixtureConditional, dataframe: pd.DataFrame) -> _rq1_s3_np.ndarray:
        n = len(dataframe)
        if model.parents:
            raw_features = dataframe[list(model.parents)].to_numpy(dtype=float)
            features = (raw_features - model.feature_means) / model.feature_stds
            X = _rq1_s3_np.column_stack([_rq1_s3_np.ones(n), features])
        else:
            X = _rq1_s3_np.ones((n, 1), dtype=float)
        raw_y = dataframe[model.child].to_numpy(dtype=float)
        y = (raw_y - model.child_mean) / model.child_std
        logits = X @ model.gate_coefficients.T
        log_gate = logits - _rq1_s3_logsumexp(logits, axis=1, keepdims=True)
        means = X @ model.betas.T
        log_components = -0.5 * (_rq1_s3_np.log(2.0 * _rq1_s3_np.pi) + _rq1_s3_np.log(model.variances)[None, :] + (y[:, None] - means) ** 2 / model.variances[None, :])
        return _rq1_s3_logsumexp(log_gate + log_components, axis=1) - _rq1_s3_np.log(model.child_std)

    def _eval_regime_v2_continuous(self, model: Mapping[str, Any], raw_data: pd.DataFrame, model_data: pd.DataFrame) -> _rq1_s3_np.ndarray:
        result = _rq1_s3_np.empty(len(raw_data), dtype=float)
        discrete_parents = list(model['discrete_parents'])
        key_matrix = model_data[discrete_parents].to_numpy(dtype=int)
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, row in enumerate(key_matrix):
            groups.setdefault(tuple((int(value) for value in row)), []).append(index)
        for key, indices_list in groups.items():
            indices = _rq1_s3_np.asarray(indices_list, dtype=int)
            local_model = model['regimes'].get(key, model['fallback'])
            local_data = raw_data.iloc[indices]
            if isinstance(local_model, _rq1_s3_GaussianConditional):
                values = self._eval_gaussian(local_model, local_data)
            elif isinstance(local_model, _rq1_s3_MixtureConditional):
                values = self._eval_mixture(local_model, local_data)
            else:
                raise ValueError(f"Unknown Hybrid V2 continuous Q model for {model['child']!r}.")
            result[indices] = values
        return result

    def fit(self, raw_training_data: pd.DataFrame) -> '_rq1_s3_LearnedDAGDistribution':
        raw_training_data = raw_training_data[self.nodes].copy().reset_index(drop=True)
        if not _rq1_s3_np.all(_rq1_s3_np.isfinite(raw_training_data.to_numpy(dtype=float))):
            raise ValueError('Training data contain non-finite values.')
        model_training_data = self.transformer.transform(raw_training_data) if self.transformer else raw_training_data.copy()
        self.models.clear()
        for child in self.nodes:
            parents = sorted((str(parent) for parent in self.graph.predecessors(child)))
            if self.paradigm == 'continuous':
                self.models[child] = self._fit_gaussian(raw_training_data, child, parents)
                continue
            if self.paradigm == 'discrete':
                cardinality = self.transformer.cardinalities[child]
                self.models[child] = self._fit_cpt(model_training_data, child, parents, cardinality)
                continue
            continuous_parents = [parent for parent in parents if parent in self.continuous_nodes]
            discrete_parents = [parent for parent in parents if parent in self.discrete_nodes]
            if child in self.continuous_nodes:
                if not discrete_parents:
                    selected_k = self._selected_components(child, ())
                    if selected_k == 1:
                        self.models[child] = self._fit_gaussian(raw_training_data, child, continuous_parents)
                    else:
                        self.models[child] = self._fit_mixture(raw_training_data, child, continuous_parents, selected_k, prior_family='continuous')
                    continue
                fallback = self._fit_gaussian(raw_training_data, child, continuous_parents)
                regimes: dict[tuple[int, ...], _rq1_s3_GaussianConditional | _rq1_s3_MixtureConditional] = {}
                group_key: Any = discrete_parents[0] if len(discrete_parents) == 1 else discrete_parents
                for key, group in model_training_data.groupby(group_key, sort=False, observed=True):
                    normalized_values = key if isinstance(key, tuple) else (key,)
                    normalized = self._configuration_key(normalized_values)
                    minimum_size = max(self.min_regime_size, len(continuous_parents) + 2)
                    if len(group) < minimum_size:
                        self.diagnostics['regime_fallbacks'] += 1
                        continue
                    selected_k = self._selected_components(child, normalized)
                    raw_group = raw_training_data.loc[group.index]
                    if selected_k == 1:
                        local_model: _rq1_s3_GaussianConditional | _rq1_s3_MixtureConditional = self._fit_gaussian(raw_group, child, continuous_parents)
                    else:
                        local_model = self._fit_mixture(raw_group, child, continuous_parents, selected_k, prior_family='clg')
                    regimes[normalized] = local_model
                self.models[child] = {'kind': 'regime_v2_continuous', 'child': child, 'discrete_parents': tuple(discrete_parents), 'regimes': regimes, 'fallback': fallback}
                continue
            cardinality = self.transformer.cardinalities[child]
            if not continuous_parents:
                self.models[child] = self._fit_cpt(model_training_data, child, discrete_parents, cardinality)
                continue
            global_prior = self._smoothed_probabilities(model_training_data[child], cardinality)
            fallback = self._fit_logistic(model_training_data, child, continuous_parents, cardinality, global_prior)
            regimes: dict[tuple[int, ...], _rq1_s3_LogisticConditional] = {}
            if discrete_parents:
                group_key = discrete_parents[0] if len(discrete_parents) == 1 else discrete_parents
                grouped = model_training_data.groupby(group_key, sort=False, observed=True)
            else:
                grouped = [((), model_training_data)]
            for key, group in grouped:
                if discrete_parents:
                    normalized_values = key if isinstance(key, tuple) else (key,)
                    normalized = self._configuration_key(normalized_values)
                else:
                    normalized = ()
                regimes[normalized] = self._fit_logistic(group, child, continuous_parents, cardinality, global_prior)
            self.models[child] = {'kind': 'regime_logistic', 'child': child, 'discrete_parents': tuple(discrete_parents), 'cardinality': cardinality, 'regimes': regimes, 'fallback': fallback}
        return self

    def _eval_regime_gaussian(self, model: Mapping[str, Any], raw_data: pd.DataFrame, model_data: pd.DataFrame) -> _rq1_s3_np.ndarray:
        result = _rq1_s3_np.empty(len(raw_data), dtype=float)
        discrete_parents = list(model['discrete_parents'])
        key_matrix = model_data[discrete_parents].to_numpy(dtype=int)
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, row in enumerate(key_matrix):
            groups.setdefault(tuple((int(value) for value in row)), []).append(index)
        for key, indices_list in groups.items():
            indices = _rq1_s3_np.asarray(indices_list, dtype=int)
            local_model = model['regimes'].get(key, model['fallback'])
            result[indices] = self._eval_gaussian(local_model, raw_data.iloc[indices])
        return result

    def _eval_regime_logistic(self, model: Mapping[str, Any], model_data: pd.DataFrame) -> _rq1_s3_np.ndarray:
        result = _rq1_s3_np.empty(len(model_data), dtype=float)
        child_values = model_data[model['child']].to_numpy(dtype=int)
        discrete_parents = list(model['discrete_parents'])
        if discrete_parents:
            key_matrix = model_data[discrete_parents].to_numpy(dtype=int)
        else:
            key_matrix = _rq1_s3_np.empty((len(model_data), 0), dtype=int)
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, row in enumerate(key_matrix):
            groups.setdefault(tuple((int(value) for value in row)), []).append(index)
        for key, indices_list in groups.items():
            indices = _rq1_s3_np.asarray(indices_list, dtype=int)
            local_model = model['regimes'].get(key, model['fallback'])
            result[indices] = self._eval_logistic(local_model, model_data, indices, child_values[indices])
        return result

    def joint_logscore(self, raw_evaluation_data: pd.DataFrame, *, return_nodewise: bool=False) -> _rq1_s3_np.ndarray | tuple[_rq1_s3_np.ndarray, dict[str, _rq1_s3_np.ndarray]]:
        if not self.models:
            raise RuntimeError('Fit Q before evaluating it.')
        raw_data = raw_evaluation_data[self.nodes].copy().reset_index(drop=True)
        model_data = self.transformer.transform(raw_data) if self.transformer else raw_data.copy()
        total = _rq1_s3_np.zeros(len(raw_data), dtype=float)
        nodewise: dict[str, _rq1_s3_np.ndarray] = {}
        for child in self.nodes:
            model = self.models[child]
            if isinstance(model, _rq1_s3_GaussianConditional):
                values = self._eval_gaussian(model, raw_data)
            elif isinstance(model, _rq1_s3_MixtureConditional):
                values = self._eval_mixture(model, raw_data)
            elif isinstance(model, _rq1_s3_CPTConditional):
                values = self._eval_cpt(model, model_data)
            elif isinstance(model, _rq1_s3_Mapping) and model['kind'] == 'regime_gaussian':
                values = self._eval_regime_gaussian(model, raw_data, model_data)
            elif isinstance(model, _rq1_s3_Mapping) and model['kind'] == 'regime_v2_continuous':
                values = self._eval_regime_v2_continuous(model, raw_data, model_data)
            elif isinstance(model, _rq1_s3_Mapping) and model['kind'] == 'regime_logistic':
                values = self._eval_regime_logistic(model, model_data)
            else:
                raise ValueError(f'Unknown fitted Q model for {child!r}.')
            if values.shape != (len(raw_data),) or not _rq1_s3_np.all(_rq1_s3_np.isfinite(values)):
                raise FloatingPointError(f'Invalid learned Q score for {child!r}.')
            nodewise[child] = values
            total += values
        if return_nodewise:
            return (total, nodewise)
        return total

class _rq1_s3_HybridV3LearnedDAGDistribution(_rq1_s3_LearnedDAGDistribution):
    """Established Hybrid V2 Q evaluator with corrected V3 predictive CC."""

    def __init__(self, *, corrected_cc_module, cd_v3_engine, graph: nx.DiGraph, nodes: Sequence[str], discrete_nodes: Sequence[str], continuous_nodes: Sequence[str], transform: _rq1_s3_SavedQuantileTransform, dc_selected_k_by_configuration: Mapping[str, Mapping[str, int]], hybrid_continuous_scaling: Mapping[str, Any], hybrid_prior_hyperparameters: Mapping[str, float], cpt_alpha: float, min_regime_size: int, logistic_c: float) -> None:
        self.corrected_cc_module = corrected_cc_module
        self.cd_v3_engine = cd_v3_engine
        self.cc_engine = None
        self.cc_children: set[str] = set()
        selected = {str(child): {str(k): int(v) for k, v in choices.items()} for child, choices in dc_selected_k_by_configuration.items()}
        for child in map(str, continuous_nodes):
            selected.setdefault(child, {'none': 1})
        super().__init__(graph=graph, nodes=nodes, paradigm='hybrid', discrete_nodes=discrete_nodes, continuous_nodes=continuous_nodes, transform=transform, cpt_alpha=cpt_alpha, min_regime_size=min_regime_size, logistic_c=logistic_c, hybrid_selected_k_by_configuration=selected, hybrid_continuous_scaling=hybrid_continuous_scaling, hybrid_prior_hyperparameters=hybrid_prior_hyperparameters)
        self.diagnostics['cc_v3_models'] = 0
        self.diagnostics['cd_v3_bma_models'] = 0

    def fit(self, raw_training_data: pd.DataFrame) -> '_rq1_s3_HybridV3LearnedDAGDistribution':
        raw_training_data = raw_training_data[self.nodes].copy().reset_index(drop=True)
        if not _rq1_s3_np.all(_rq1_s3_np.isfinite(raw_training_data.to_numpy(dtype=float))):
            raise ValueError('Training data contain non-finite values.')
        model_training_data = self.transformer.transform(raw_training_data)
        self.models.clear()
        self.cc_children.clear()
        continuous_order = [n for n in self.nodes if n in self.continuous_nodes]
        needs_cc = any((child in self.continuous_nodes and (not any((p in self.discrete_nodes for p in self.graph.predecessors(child)))) for child in self.nodes))
        if needs_cc:
            if self.corrected_cc_module is None:
                raise RuntimeError('A corrected Final V3 module is required for final CC families.')
            self.cc_engine = self.corrected_cc_module.V3CScoringEngine(raw_training_data[continuous_order].copy())
        else:
            self.cc_engine = None
        for child in self.nodes:
            parents = sorted((str(p) for p in self.graph.predecessors(child)))
            continuous_parents = [p for p in parents if p in self.continuous_nodes]
            discrete_parents = [p for p in parents if p in self.discrete_nodes]
            if child in self.continuous_nodes and (not discrete_parents):
                self.cc_engine.score_local(child, continuous_parents)
                self.models[child] = {'kind': 'corrected_v3_cc', 'child': child, 'parents': tuple(continuous_parents)}
                self.cc_children.add(child)
                self.diagnostics['cc_v3_models'] += 1
                continue
            if child in self.continuous_nodes:
                fallback = self._fit_gaussian(raw_training_data, child, continuous_parents)
                regimes = {}
                group_key: Any = discrete_parents[0] if len(discrete_parents) == 1 else discrete_parents
                for key, group in model_training_data.groupby(group_key, sort=False, observed=True):
                    normalized_values = key if isinstance(key, tuple) else (key,)
                    normalized = self._configuration_key(normalized_values)
                    minimum_size = max(self.min_regime_size, len(continuous_parents) + 2)
                    if len(group) < minimum_size:
                        self.diagnostics['regime_fallbacks'] += 1
                        continue
                    selected_k = self._selected_components(child, normalized)
                    raw_group = raw_training_data.loc[group.index]
                    if selected_k == 1:
                        local_model = self._fit_gaussian(raw_group, child, continuous_parents)
                    else:
                        local_model = self._fit_mixture(raw_group, child, continuous_parents, selected_k, prior_family='clg')
                    regimes[normalized] = local_model
                self.models[child] = {'kind': 'regime_v2_continuous', 'child': child, 'discrete_parents': tuple(discrete_parents), 'regimes': regimes, 'fallback': fallback}
                continue
            cardinality = self.transformer.cardinalities[child]
            if not continuous_parents:
                self.models[child] = self._fit_cpt(model_training_data, child, discrete_parents, cardinality)
                continue
            if self.cd_v3_engine is None or getattr(self.cd_v3_engine, 'cd', None) is None:
                raise RuntimeError('CD V3 predictive engine is unavailable.')
            predictive = self.cd_v3_engine.cd.fit_predictive_family(child=child, c_parents=continuous_parents, d_parents=discrete_parents, all_parents=parents)
            self.models[child] = {'kind': 'cd_v3_bma', 'child': child, 'parents': tuple(parents), 'predictive': predictive}
            self.diagnostics['cd_v3_bma_models'] += 1
        return self

    def joint_logscore(self, raw_evaluation_data: pd.DataFrame, *, return_nodewise: bool=False):
        if not self.models:
            raise RuntimeError('Fit Q before evaluating it.')
        raw_data = raw_evaluation_data[self.nodes].copy().reset_index(drop=True)
        model_data = self.transformer.transform(raw_data)
        total = _rq1_s3_np.zeros(len(raw_data), dtype=float)
        nodewise: dict[str, _rq1_s3_np.ndarray] = {}
        continuous_order = [n for n in self.nodes if n in self.continuous_nodes]
        cc_eval = raw_data[continuous_order]
        cd_v3_eval = self.cd_v3_engine.transform_new_data(model_data.copy()) if self.cd_v3_engine is not None else None
        for child in _rq1_s3_nx.topological_sort(self.graph):
            child = str(child)
            model = self.models[child]
            if isinstance(model, _rq1_s3_GaussianConditional):
                values = self._eval_gaussian(model, raw_data)
            elif isinstance(model, _rq1_s3_MixtureConditional):
                values = self._eval_mixture(model, raw_data)
            elif isinstance(model, _rq1_s3_CPTConditional):
                values = self._eval_cpt(model, model_data)
            elif isinstance(model, _rq1_s3_Mapping) and model['kind'] == 'corrected_v3_cc':
                values = self.cc_engine.local_predictive_logpdf(child, model['parents'], cc_eval)
            elif isinstance(model, _rq1_s3_Mapping) and model['kind'] == 'regime_v2_continuous':
                values = self._eval_regime_v2_continuous(model, raw_data, model_data)
            elif isinstance(model, _rq1_s3_Mapping) and model['kind'] == 'cd_v3_bma':
                if cd_v3_eval is None:
                    raise RuntimeError('CD V3 evaluation transform unavailable.')
                values = model['predictive'].log_prob(cd_v3_eval)
            elif isinstance(model, _rq1_s3_Mapping) and model['kind'] == 'regime_logistic':
                values = self._eval_regime_logistic(model, model_data)
            else:
                raise ValueError(f'Unknown fitted Q model for {child!r}.')
            values = _rq1_s3_np.asarray(values, dtype=float)
            if values.shape != (len(raw_data),) or not _rq1_s3_np.all(_rq1_s3_np.isfinite(values)):
                raise FloatingPointError(f'Invalid learned Q score for {child!r}.')
            nodewise[child] = values
            total += values
        return (total, nodewise) if return_nodewise else total

def _rq1_s3__frozen_prior_hyperparameters_from_scorer(scorer) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if getattr(scorer, 'dd', None) is not None:
        out['equivalent_sample_size'] = float(scorer.dd.equivalent_sample_size)
    if getattr(scorer, 'cd', None) is not None:
        out.update({'softmax_coefficient_prior_variance': float(scorer.cd.coefficient_prior_variance), 'softmax_intercept_prior_variance': float(scorer.cd.intercept_prior_variance), 'logistic_max_iter': int(scorer.cd.max_iter)})
    if getattr(scorer, 'dc', None) is not None:
        dc = scorer.dc
        for name in ('clg_coefficient_prior_variance', 'clg_intercept_prior_variance', 'clg_inverse_gamma_shape', 'clg_inverse_gamma_scale', 'continuous_coefficient_prior_variance', 'continuous_intercept_prior_variance', 'continuous_inverse_gamma_shape', 'continuous_inverse_gamma_scale', 'mixture_gate_coefficient_prior_variance', 'mixture_gate_intercept_prior_variance'):
            out[name] = float(getattr(dc, name))
        for name in ('mixture_max_components', 'mixture_max_iter', 'mixture_restarts'):
            out[name] = int(getattr(dc, name))
        out['mixture_hessian_step'] = float(dc.mixture_hessian_step)
        out['mixture_min_effective_count'] = float(dc.mixture_min_effective_count)
    return out

def _rq1_s3_reconstruct_final_evaluation_metadata(*, model_module, cc_scorer_file: Path, raw_training: pd.DataFrame, transformer: _rq1_s3_SavedQuantileTransform, learned_graph: nx.DiGraph, nodes: Sequence[str], discrete_nodes: Sequence[str], continuous_nodes: Sequence[str], cardinalities: Mapping[str, int], saved_final_score: float | None, score_tolerance: float) -> dict[str, Any]:
    """Rebuild metadata for pre-KL-ready 02 cells and parity-gate it."""
    hybrid_training = transformer.transform(raw_training)
    scorer = model_module.HybridScoringEngine(hybrid_training.copy(), discrete_nodes, continuous_nodes, cardinalities, cc_scorer_file=cc_scorer_file)
    total = 0.0
    local_rows = []
    for child in nodes:
        parents = tuple(sorted(map(str, learned_graph.predecessors(child))))
        total += float(scorer.score_local(str(child), parents))
        local_rows.append(_rq1_s3_asdict(scorer.get_score_info(str(child), parents)))
    if saved_final_score is not None:
        diff = abs(float(total) - float(saved_final_score))
        if diff > score_tolerance:
            raise AssertionError(f'Reconstructed training/scorer does not reproduce the saved final DAG score: reconstructed={total:.12f}, saved={saved_final_score:.12f}, abs_diff={diff:.3e} > {score_tolerance:.3e}. Do not compute KL from a non-identical reconstruction.')
    else:
        diff = None
    dc_diag = scorer.mixture_diagnostics_dataframe().to_dict(orient='records') if getattr(scorer, 'dc', None) is not None else []
    cd_diag = scorer.laplace_diagnostics_dataframe().to_dict(orient='records') if getattr(scorer, 'cd', None) is not None else []
    selected_k: dict[str, dict[str, int]] = {}
    for row in dc_diag:
        if bool(row.get('selected', False)):
            selected_k.setdefault(str(row['child']), {})[_rq1_s3__configuration_storage_key(row.get('discrete_configuration'))] = int(row['n_components'])
    return {'schema_version': 'hybrid-v3-predictive-q-metadata-v1-reconstructed', 'metadata_source': 'reconstructed_from_pre_kl_ready_structure_cell', 'reconstructed_final_dag_score': float(total), 'saved_final_dag_score': saved_final_score, 'reconstruction_score_abs_diff': diff, 'hybrid_continuous_scaling': scorer.preprocessing_metadata(), 'frozen_prior_hyperparameters': _rq1_s3__frozen_prior_hyperparameters_from_scorer(scorer), 'final_local_score_info': local_rows, 'dc_selected_k_by_configuration': selected_k, 'dc_mixture_diagnostics': dc_diag, 'cd_laplace_diagnostics': cd_diag}

def _rq1_s3__load_manifest_base_sample_size(structure_json: Path) -> int | None:
    for parent in structure_json.parents:
        manifest = parent / 'manifest.json'
        if not manifest.exists():
            continue
        try:
            payload = _rq1_s3_json.loads(manifest.read_text(encoding='utf-8'))
        except Exception:
            continue
        sizes = payload.get('experiment_scope', {}).get('sample_sizes')
        if sizes:
            return int(max(map(int, sizes)))
    return None

def _rq1_s3_reconstruct_generator_payload_and_training(*, root: Path, cell: Mapping[str, Any], structure_json: Path, explicit_base_sample_size: int | None) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    definition = cell.get('experiment_definition', {})
    generator_name = str(definition['generator'])
    seed = int(definition['seed'])
    n = int(definition['sample_size'])
    base_n = explicit_base_sample_size or definition.get('training_base_sample_size') or cell.get('settings', {}).get('training_base_sample_size') or _rq1_s3__load_manifest_base_sample_size(structure_json) or n
    base_n = int(base_n)
    if base_n < n:
        raise ValueError(f'Reconstruction base sample size {base_n} < cell n={n}.')
    Generator = _rq1_s3_import_generator(root, generator_name)
    generator = Generator(num_samples=base_n, seed=seed)
    generator.generate_dag()
    base_data = generator.generate_data().reset_index(drop=True)
    payload = generator.get_payload()
    nodes = list(map(str, payload['nodes']))
    raw_training = base_data[nodes].iloc[:n].copy().reset_index(drop=True)
    expected_hash = cell.get('dataset_sha256')
    if expected_hash:
        actual_hash = _rq1_s3_structure_dataframe_sha256(raw_training)
        if actual_hash != expected_hash:
            raise ValueError(f'Deterministic training-data reconstruction failed the structure-cell hash check (base_n={base_n}). For an older multi-size run, pass --reconstruction-base-sample-size equal to the maximum n used by that 02 invocation, or rerun the cell with the updated KL-ready 02.')
    return (raw_training, payload)

def _rq1_s3_load_cell_inputs(*, root: Path, cell: _rq1_s3_Mapping[str, Any], structure_json: Path, explicit_base_sample_size: int | None) -> tuple[_rq1_s3_pd.DataFrame, _rq1_s3_Mapping[str, Any], str]:
    raw_meta = cell.get('raw_training_data', {})
    records = raw_meta.get('records') if isinstance(raw_meta, _rq1_s3_Mapping) else None
    ground_truth = cell.get('ground_truth')
    if records is not None and ground_truth and ground_truth.get('true_parameters'):
        columns = list(map(str, raw_meta['column_order']))
        raw = _rq1_s3_pd.DataFrame.from_records(records, columns=columns).reset_index(drop=True)
        expected_hash = raw_meta.get('sha256') or cell.get('dataset_sha256')
        if expected_hash and _rq1_s3_structure_dataframe_sha256(raw) != expected_hash:
            raise ValueError('Embedded raw training records failed their saved hash check.')
        payload = {'nodes': list(map(str, ground_truth['nodes'])), 'discrete_nodes': list(map(str, ground_truth['discrete_nodes'])), 'continuous_nodes': list(map(str, ground_truth['continuous_nodes'])), 'binary_nodes': list(map(str, ground_truth.get('binary_nodes', []))), 'multiclass_nodes': list(map(str, ground_truth.get('multiclass_nodes', []))), 'node_regimes': ground_truth.get('node_regimes', {}), 'true_parameters': ground_truth['true_parameters'], 'graph': _rq1_s3_build_graph(ground_truth['nodes'], ground_truth['true_edges'])}
        return (raw, payload, 'embedded_kl_ready_cell')
    raw, payload = _rq1_s3_reconstruct_generator_payload_and_training(root=root, cell=cell, structure_json=structure_json, explicit_base_sample_size=explicit_base_sample_size)
    return (raw, payload, 'deterministic_generator_reconstruction')

def _rq1_s3_discover_structure_files(structure_dir: Path, *, generators: set[str] | None, seeds: set[int] | None, sample_sizes: set[int] | None) -> list[Path]:
    files = sorted((structure_dir / 'runs').rglob('hybrid_v3_structure.json'))
    selected = []
    for path in files:
        try:
            cell = _rq1_s3_json.loads(path.read_text(encoding='utf-8'))
            d = cell['experiment_definition']
            g, s, n = (str(d['generator']), int(d['seed']), int(d['sample_size']))
        except Exception:
            continue
        if generators is not None and g not in generators:
            continue
        if seeds is not None and s not in seeds:
            continue
        if sample_sizes is not None and n not in sample_sizes:
            continue
        selected.append(path)
    return selected

def _rq1_s3_output_path(output_dir: Path, generator: str, seed: int, n: int) -> Path:
    return output_dir / 'runs' / generator / f'seed_{seed:02d}' / f'n_{n:04d}' / 'hybrid_v3_kl.json'

def _rq1_s3_evaluate_cell(*, root: Path, structure_json: Path, cell: _rq1_s3_Mapping[str, Any], mc_samples: int, cpt_alpha: float, min_regime_size: int, logistic_c: float, reconstruction_base_sample_size: int | None, metadata_score_tolerance: float, particles: int=256, particle_replicates: int=2, particle_batch_size: int=64) -> dict[str, Any]:
    definition = cell['experiment_definition']
    generator_name = str(definition['generator'])
    seed = int(definition['seed'])
    n = int(definition['sample_size'])
    raw_training, payload, training_source = _rq1_s3_load_cell_inputs(root=root, cell=cell, structure_json=structure_json, explicit_base_sample_size=reconstruction_base_sample_size)
    nodes = list(map(str, payload['nodes']))
    discrete_nodes = list(map(str, payload['discrete_nodes']))
    continuous_nodes = list(map(str, payload['continuous_nodes']))
    cardinalities = _rq1_s3_default_cardinalities(payload)
    true_graph = payload['graph'].copy()
    learned_graph = _rq1_s3_build_graph(nodes, cell['learned_edges'])
    transformer = _rq1_s3_SavedQuantileTransform.from_metadata(cell['discretizer_metadata'])
    model_file = (root / 'src' / 'hybrid_bn' / 'SLhybrid.py').resolve()
    cc_file = (root / 'src' / 'hybrid_bn' / 'SLhybrid.py').resolve()
    model_module = _rq1_inline_load(model_file, f'_hv3_kl_model_{seed}_{n}_{_rq1_s3_os.getpid()}')
    evaluation_metadata = cell.get('evaluation_metadata')
    metadata_source = 'saved_kl_ready_structure_cell'
    if not isinstance(evaluation_metadata, _rq1_s3_Mapping):
        saved_score = cell.get('learned_dag_score_recomputed_main')
        evaluation_metadata = _rq1_s3_reconstruct_final_evaluation_metadata(model_module=model_module, cc_scorer_file=cc_file, raw_training=raw_training, transformer=transformer, learned_graph=learned_graph, nodes=nodes, discrete_nodes=discrete_nodes, continuous_nodes=continuous_nodes, cardinalities=cardinalities, saved_final_score=float(saved_score) if saved_score is not None else None, score_tolerance=metadata_score_tolerance)
        metadata_source = 'reconstructed_and_final_score_parity_checked'
    selected_k = evaluation_metadata.get('dc_selected_k_by_configuration', {})
    scaling = evaluation_metadata.get('hybrid_continuous_scaling', {})
    priors = evaluation_metadata.get('frozen_prior_hyperparameters', {})
    discrete_set = set(discrete_nodes)
    final_quadrants = {}
    for child in nodes:
        parents = list(map(str, learned_graph.predecessors(child)))
        d_parents = [p for p in parents if p in discrete_set]
        c_parents = [p for p in parents if p not in discrete_set]
        if child in discrete_set:
            quadrant = 'CD' if c_parents else 'DD'
        else:
            quadrant = 'DC' if d_parents else 'CC'
        final_quadrants[child] = quadrant
        if quadrant == 'DC' and child not in selected_k:
            raise ValueError(f'Missing DC selected-K metadata for final child {child!r}.')
    cc_module = None
    if any((q == 'CC' for q in final_quadrants.values())):
        cc_module = _rq1_inline_load(cc_file, f'_hv3_kl_cc_{seed}_{n}_{_rq1_s3_os.getpid()}')
    true_model = _rq1_s3_TrueContinuousDAG(true_graph, payload['true_parameters'], nodes)
    mc_seed = _rq1_s3_deterministic_mc_seed(generator_name, seed)
    evaluation_sample = true_model.sample(mc_samples, mc_seed)
    truth_estimator = _rq1_s3_TransformedTruthParticleLikelihood(true_model, transformer, discrete_nodes, particles=particles, replicates=particle_replicates, batch_size=particle_batch_size)
    identity_check = truth_estimator.validate_continuous_identity(evaluation_sample.iloc[:min(128, len(evaluation_sample))].copy())
    if identity_check > 1e-10:
        raise AssertionError(f'Continuous special-case identity check failed: {identity_check:.3e}')
    log_p, particle_diagnostics = truth_estimator.loglikelihood(evaluation_sample, mc_seed)
    hybrid_training_for_cd = transformer.transform(raw_training.copy())
    cd_v3_engine = model_module.HybridScoringEngine(hybrid_training_for_cd, discrete_nodes, continuous_nodes, cardinalities, cc_scorer_file=cc_file)
    q = _rq1_s3_HybridV3LearnedDAGDistribution(corrected_cc_module=cc_module, cd_v3_engine=cd_v3_engine, graph=learned_graph, nodes=nodes, discrete_nodes=discrete_nodes, continuous_nodes=continuous_nodes, transform=transformer, dc_selected_k_by_configuration=selected_k, hybrid_continuous_scaling=scaling, hybrid_prior_hyperparameters=priors, cpt_alpha=cpt_alpha, min_regime_size=min_regime_size, logistic_c=logistic_c).fit(raw_training)
    log_q, q_nodewise = q.joint_logscore(evaluation_sample, return_nodewise=True)
    ratio = log_p - log_q
    estimate = float(_rq1_s3_np.mean(ratio))
    se = float(_rq1_s3_np.std(ratio, ddof=1) / _rq1_s3_np.sqrt(len(ratio)))
    node_details = [{'node': node, 'final_quadrant': final_quadrants[node], 'kl_contribution': None, 'mean_log_p': None, 'mean_log_q': float(_rq1_s3_np.mean(q_nodewise[node])), 'note': 'undefined: transformed P_T need not factorise over original DAG'} for node in nodes]
    structure = cell.get('structure_metrics', {})
    return {'schema_version': _rq1_s3_KL_SCHEMA_VERSION, 'status': 'complete', 'source_structure_json': str(structure_json), 'experiment_definition': {'generator': generator_name, 'seed': seed, 'sample_size': n}, 'input_reconstruction': {'training_data_source': training_source, 'evaluation_metadata_source': metadata_source}, 'structure_metrics': structure, 'kl': {'estimate': estimate, 'standard_error': se, 'lower_95': float(estimate - 1.96 * se), 'upper_95': float(estimate + 1.96 * se), 'mean_log_p': float(_rq1_s3_np.mean(log_p)), 'mean_log_q': float(_rq1_s3_np.mean(log_q)), 'mc_samples': int(mc_samples), 'mc_seed': int(mc_seed), 'materially_negative': bool(estimate < -3.0 * se), 'definition': 'mean(log P_T(T(x)) - log Q(T(x))) for fresh x~P', 'true_transformed_density': 'joint p_T(x_C,d_D) estimated by SMC marginalisation of latent raw values inside saved training-quantile intervals', 'dequantisation': False, 'particle_likelihood': particle_diagnostics, 'continuous_identity_max_abs_diff': float(identity_check), 'nodewise_kl_decomposition_available': False}, 'node_contributions': node_details, 'q_fit_diagnostics': dict(q.diagnostics), 'final_family_quadrants': final_quadrants, 'evaluation_settings': {'cpt_alpha': float(cpt_alpha), 'min_regime_size': int(min_regime_size), 'logistic_c': float(logistic_c), 'cc_q': 'corrected Final V3 posterior predictive', 'dd_q': 'additive-smoothed CPT', 'cd_q': 'score-aligned CD V3 BMA predictive: linear/quadratic/cubic/spline5', 'dc_q': 'saved-K Gaussian/MAP MoE, V2 evaluator convention', 'p_side': 'representation-correct SMC marginal transformed truth', 'particles': int(particles), 'particle_replicates': int(particle_replicates), 'particle_batch_size': int(particle_batch_size)}}

def _rq1_s3_rebuild_csvs(output_dir: Path) -> tuple[_rq1_s3_pd.DataFrame, _rq1_s3_pd.DataFrame, _rq1_s3_pd.DataFrame]:
    rows = []
    node_rows = []
    for path in sorted((output_dir / 'runs').rglob('hybrid_v3_kl.json')):
        try:
            r = _rq1_s3_json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if r.get('schema_version') != _rq1_s3_KL_SCHEMA_VERSION or r.get('status') != 'complete':
            continue
        d = r['experiment_definition']
        k = r['kl']
        s = r.get('structure_metrics', {})
        rows.append({'Generator': d['generator'], 'Seed': int(d['seed']), 'Sample_Size': int(d['sample_size']), 'SHD': s.get('shd'), 'Directed_Precision': s.get('directed_precision'), 'Directed_Recall': s.get('directed_recall'), 'Directed_F1': s.get('directed_f1'), 'Learned_Edges': s.get('learned_edge_count'), 'KL_Estimate': k['estimate'], 'KL_Standard_Error': k['standard_error'], 'KL_CI_Lower': k['lower_95'], 'KL_CI_Upper': k['upper_95'], 'Mean_Log_P': k['mean_log_p'], 'Mean_Log_Q': k['mean_log_q'], 'KL_MC_Samples': k['mc_samples'], 'KL_MC_Seed': k['mc_seed'], 'KL_Materially_Negative': k['materially_negative'], 'Training_Data_Source': r['input_reconstruction']['training_data_source'], 'Evaluation_Metadata_Source': r['input_reconstruction']['evaluation_metadata_source'], 'Result_JSON': str(path)})
        for nr in r.get('node_contributions', []):
            node_rows.append({'Generator': d['generator'], 'Seed': int(d['seed']), 'Sample_Size': int(d['sample_size']), 'Node': nr['node'], 'Final_Quadrant': nr['final_quadrant'], 'KL_Contribution': nr.get('kl_contribution'), 'Mean_Log_P': nr.get('mean_log_p'), 'Mean_Log_Q': nr.get('mean_log_q'), 'Note': nr.get('note', '')})
    runs = _rq1_s3_pd.DataFrame(rows)
    nodes = _rq1_s3_pd.DataFrame(node_rows)
    if not runs.empty:
        runs = runs.sort_values(['Generator', 'Seed', 'Sample_Size']).reset_index(drop=True)
    if not nodes.empty:
        nodes = nodes.sort_values(['Generator', 'Seed', 'Sample_Size', 'Node']).reset_index(drop=True)
    summary_rows = []
    if not runs.empty:
        for (g, n), group in runs.groupby(['Generator', 'Sample_Size'], sort=True):
            vals = group['KL_Estimate'].astype(float)
            summary_rows.append({'Generator': g, 'Sample_Size': int(n), 'Runs': int(len(group)), 'KL_Mean': float(vals.mean()), 'KL_Std': float(vals.std(ddof=1)) if len(vals) > 1 else _rq1_s3_np.nan, 'SHD_Mean': float(_rq1_s3_pd.to_numeric(group['SHD'], errors='coerce').mean()), 'Directed_F1_Mean': float(_rq1_s3_pd.to_numeric(group['Directed_F1'], errors='coerce').mean())})
    summary = _rq1_s3_pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / 'hybrid_v3_kl_results.csv', index=False)
    nodes.to_csv(output_dir / 'hybrid_v3_kl_node_contributions.csv', index=False)
    nodes.to_csv(output_dir / 'hybrid_v3_q_nodewise_diagnostics.csv', index=False)
    summary.to_csv(output_dir / 'hybrid_v3_kl_summary.csv', index=False)
    return (runs, nodes, summary)

def _rq1_s3_parse_args() -> _rq1_s3_argparse.Namespace:
    p = _rq1_s3_argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=_rq1_s3_Path, default=None)
    p.add_argument('--structure-results-dir', type=_rq1_s3_Path, default=None)
    p.add_argument('--output-dir', type=_rq1_s3_Path, default=None)
    p.add_argument('--generators', nargs='+', choices=tuple(_rq1_s3_GENERATOR_SPECS), default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    p.add_argument('--sample-sizes', nargs='+', type=int, default=None)
    p.add_argument('--mc-samples', type=int, default=5000)
    p.add_argument('--cpt-alpha', type=float, default=1.0)
    p.add_argument('--min-regime-size', type=int, default=5)
    p.add_argument('--logistic-c', type=float, default=1000.0)
    p.add_argument('--metadata-score-tolerance', type=float, default=1e-06)
    p.add_argument('--particles', type=int, default=256, help='Particles per transformed-truth likelihood replicate.')
    p.add_argument('--particle-replicates', type=int, default=2, help='Independent particle likelihood replicates.')
    p.add_argument('--particle-batch-size', type=int, default=64, help='Fresh rows processed together by the particle estimator.')
    p.add_argument('--reconstruction-base-sample-size', type=int, default=None, help='Only needed for older 02 cells when the original invocation used multiple sample sizes.')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--stop-on-error', action='store_true')
    args = p.parse_args()
    if args.mc_samples < 2:
        p.error('--mc-samples must be >=2')
    if args.cpt_alpha <= 0 or args.min_regime_size < 1 or args.logistic_c <= 0:
        p.error('Invalid Q-fit settings.')
    if args.metadata_score_tolerance <= 0:
        p.error('--metadata-score-tolerance must be positive.')
    if args.particles < 8 or args.particle_replicates < 1 or args.particle_batch_size < 1:
        p.error('Invalid particle settings.')
    return args

def _rq1_s3_main() -> int:
    args = _rq1_s3_parse_args()
    here = _rq1_s3_Path(__file__).resolve().parent
    root = _rq1_s3_discover_root(here, args.project_root)
    if args.structure_results_dir is None:
        preferred = root / 'Outputs' / 'HybridV3_CDv3Candidate_StructureLearning_NoAdd2'
        fallback = root / 'Outputs' / 'HybridV3_StructureLearning'
        structure_dir = preferred if preferred.exists() else fallback
    else:
        structure_dir = args.structure_results_dir.expanduser().resolve() if args.structure_results_dir.is_absolute() else (root / args.structure_results_dir).resolve()
    if not structure_dir.exists():
        raise FileNotFoundError(structure_dir)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None and args.output_dir.is_absolute() else (root / args.output_dir).resolve() if args.output_dir is not None else root / 'Outputs' / 'HybridV3_CDv3Candidate_KL_RepresentationCorrect'
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _rq1_s3_discover_structure_files(structure_dir, generators=set(args.generators) if args.generators else None, seeds=set(args.seeds) if args.seeds else None, sample_sizes=set(args.sample_sizes) if args.sample_sizes else None)
    if not files:
        raise RuntimeError(f'No matching hybrid_v3_structure.json files under {structure_dir}')
    print('=' * 124)
    print('HYBRID V3 + CD V3 — REPRESENTATION-CORRECT FRESH-P KL')
    print('=' * 124)
    print(f'Project root:      {root}')
    print(f'Structure results: {structure_dir}')
    print(f'KL output:         {output_dir}')
    print(f'Cells:             {len(files)}')
    print(f'MC samples:        {args.mc_samples}')
    print('P: joint transformed truth via latent-bin particle marginalisation')
    print(f'Particles:          {args.particles} x {args.particle_replicates} replicates | batch={args.particle_batch_size}')
    print('Q: DD CPT | CD V3 BMA | DC saved-K V2 predictive | CC corrected V3 predictive')
    print('=' * 124, flush=True)
    completed = skipped = failed = 0
    started = _rq1_s3_time.perf_counter()
    failure_path = output_dir / 'failures.jsonl'
    for index, path in enumerate(files, 1):
        cell = _rq1_s3_json.loads(path.read_text(encoding='utf-8'))
        d = cell['experiment_definition']
        g, seed, n = (str(d['generator']), int(d['seed']), int(d['sample_size']))
        out = _rq1_s3_output_path(output_dir, g, seed, n)
        print(f'\n[{index}/{len(files)}] {g} seed={seed} n={n}', flush=True)
        if out.exists() and (not args.overwrite):
            try:
                old = _rq1_s3_json.loads(out.read_text(encoding='utf-8'))
                if old.get('schema_version') == _rq1_s3_KL_SCHEMA_VERSION and old.get('status') == 'complete' and (int(old.get('kl', {}).get('mc_samples', -1)) == args.mc_samples) and (int(old.get('evaluation_settings', {}).get('particles', -1)) == args.particles) and (int(old.get('evaluation_settings', {}).get('particle_replicates', -1)) == args.particle_replicates) and (int(old.get('evaluation_settings', {}).get('particle_batch_size', -1)) == args.particle_batch_size):
                    skipped += 1
                    print('    matching KL result exists; skipping.', flush=True)
                    continue
            except Exception:
                pass
        try:
            cell_started = _rq1_s3_time.perf_counter()
            result = _rq1_s3_evaluate_cell(root=root, structure_json=path, cell=cell, mc_samples=int(args.mc_samples), cpt_alpha=float(args.cpt_alpha), min_regime_size=int(args.min_regime_size), logistic_c=float(args.logistic_c), reconstruction_base_sample_size=args.reconstruction_base_sample_size, metadata_score_tolerance=float(args.metadata_score_tolerance), particles=int(args.particles), particle_replicates=int(args.particle_replicates), particle_batch_size=int(args.particle_batch_size))
            result['runtime_seconds'] = float(_rq1_s3_time.perf_counter() - cell_started)
            _rq1_s3_atomic_write_json(out, result)
            completed += 1
            k = result['kl']
            s = result.get('structure_metrics', {})
            print(f"    KL={k['estimate']:.6f} ± {1.96 * k['standard_error']:.6f} | SHD={s.get('shd', 'NA')} | F1={s.get('directed_f1', 'NA')} | metadata={result['input_reconstruction']['evaluation_metadata_source']} | runtime={result['runtime_seconds']:.1f}s", flush=True)
        except Exception as exc:
            failed += 1
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with failure_path.open('a', encoding='utf-8') as f:
                f.write(_rq1_s3_json.dumps(_rq1_s3_json_safe({'source': str(path), 'generator': g, 'seed': seed, 'sample_size': n, 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': _rq1_s3_traceback.format_exc()})) + '\n')
            print(f'    FAILED: {type(exc).__name__}: {exc}', flush=True)
            if args.stop_on_error:
                _rq1_s3_rebuild_csvs(output_dir)
                raise
        finally:
            _rq1_s3_gc.collect()
        _rq1_s3_rebuild_csvs(output_dir)
    runs, _, _ = _rq1_s3_rebuild_csvs(output_dir)
    print('\n' + '=' * 124)
    print('HYBRID V3 KL COMPLETE')
    print('=' * 124)
    print(f'Completed now: {completed}')
    print(f'Skipped existing: {skipped}')
    print(f'Failed now: {failed}')
    print(f'Rows available: {len(runs)}')
    if not runs.empty:
        print(f"Mean KL={runs['KL_Estimate'].mean():.6f}")
    print(f"Results: {output_dir / 'hybrid_v3_kl_results.csv'}")
    print(f"Summary: {output_dir / 'hybrid_v3_kl_summary.csv'}")
    print(f"Node contributions: {output_dir / 'hybrid_v3_kl_node_contributions.csv'}")
    print(f'Elapsed: {_rq1_s3_time.perf_counter() - started:.1f}s')
    print('=' * 124)
    return 0 if failed == 0 else 1


# ======================================================================
# Evaluation implementation: 18_three_seed_model_comparison.py
# ======================================================================
import argparse as _rq1_s4_argparse
import importlib.util as _rq1_s4_importlib
import json as _rq1_s4_json
import math as _rq1_s4_math
import os as _rq1_s4_os
import subprocess as _rq1_s4_subprocess
import sys as _rq1_s4_sys
import time as _rq1_s4_time
import traceback as _rq1_s4_traceback
from pathlib import Path as _rq1_s4_Path
from typing import Any as _rq1_s4_Any, Iterable as _rq1_s4_Iterable, Mapping as _rq1_s4_Mapping, Sequence as _rq1_s4_Sequence
_rq1_s4_sys.dont_write_bytecode = True
import networkx as _rq1_s4_nx
import numpy as _rq1_s4_np
import pandas as _rq1_s4_pd
from scipy.stats import norm as _rq1_s4_norm
_rq1_s4_GENERATOR_SPECS = {'unimodal': ('generateUnimodalGT.py', 'StochasticContinuousGenerator'), 'skew': ('generateSkewGT.py', 'StochasticContinuousSkewedGenerator'), 'multimodal': ('generateMultimodalGT.py', 'StochasticContinuousMultimodalGenerator'), 'skew_multi': ('generateSkewMultiGT.py', 'StochasticContinuousSkewedMultimodalGenerator')}
_rq1_s4_SCHEMA_VERSION = 'three-model-representation-correct-comparison-v1'
_rq1_s4_MIN_VARIANCE = 1e-12

def _rq1_s4_json_safe(value: Any) -> Any:
    if isinstance(value, _rq1_s4_Mapping):
        return {str(k): _rq1_s4_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_rq1_s4_json_safe(v) for v in value]
    if isinstance(value, _rq1_s4_np.ndarray):
        return _rq1_s4_json_safe(value.tolist())
    if isinstance(value, _rq1_s4_np.generic):
        return _rq1_s4_json_safe(value.item())
    if isinstance(value, _rq1_s4_Path):
        return str(value)
    if isinstance(value, float):
        if _rq1_s4_math.isnan(value):
            return 'NaN'
        if _rq1_s4_math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
    return value

def _rq1_s4_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(_rq1_s4_json.dumps(_rq1_s4_json_safe(payload), indent=2), encoding='utf-8')
    _rq1_s4_os.replace(tmp, path)

def _rq1_s4_discover_root(here: _rq1_s4_Path, explicit: _rq1_s4_Path | None) -> _rq1_s4_Path:
    return _rq1_find_root()

def _rq1_s4_find_script(root: Path, here: Path, filename: str) -> Path:
    for p in (here / filename, root / 'experiments' / 'rq1' / filename):
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f'Could not find {filename}. Place 18 beside it in Experiments.')

def _rq1_s4_import_generator(root: Path, generator: str):
    filename, class_name = _rq1_s4_GENERATOR_SPECS[generator]
    module = _rq1_inline_load(root / 'src' / 'generators' / filename, f'_comparison_generator_{generator}_{_rq1_s4_os.getpid()}')
    return getattr(module, class_name)

def _rq1_s4_run_subprocess(command: list[str], cwd: Path) -> None:
    print('\n$', ' '.join(command), flush=True)
    completed = _rq1_s4_subprocess.run(command, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f'Subprocess failed with exit code {completed.returncode}: ' + ' '.join(command))

def _rq1_s4_structure_metrics(true_edges: Iterable[tuple[str, str]], learned_edges: Iterable[tuple[str, str]]) -> dict[str, Any]:
    true_d = {(str(u), str(v)) for u, v in true_edges}
    learned_d = {(str(u), str(v)) for u, v in learned_edges}
    true_s = {frozenset((u, v)) for u, v in true_d}
    learned_s = {frozenset((u, v)) for u, v in learned_d}
    stp = len(true_s & learned_s)
    sfp = len(learned_s - true_s)
    sfn = len(true_s - learned_s)
    sp = stp / (stp + sfp) if stp + sfp else 0.0
    sr = stp / (stp + sfn) if stp + sfn else 0.0
    sf1 = 2 * sp * sr / (sp + sr) if sp + sr else 0.0
    dtp = len(true_d & learned_d)
    dfp = len(learned_d - true_d)
    dfn = len(true_d - learned_d)
    dp = dtp / (dtp + dfp) if dtp + dfp else 0.0
    dr = dtp / (dtp + dfn) if dtp + dfn else 0.0
    df1 = 2 * dp * dr / (dp + dr) if dp + dr else 0.0
    reversals = sum(((v, u) in true_d and (u, v) not in true_d for u, v in learned_d))
    extra = len(learned_s - true_s)
    missing = len(true_s - learned_s)
    return {'true_edge_count': len(true_d), 'learned_edge_count': len(learned_d), 'skeleton_true_positives': stp, 'skeleton_false_positives': sfp, 'skeleton_false_negatives': sfn, 'skeleton_precision': float(sp), 'skeleton_recall': float(sr), 'skeleton_f1': float(sf1), 'directed_true_positives': dtp, 'directed_false_positives': dfp, 'directed_false_negatives': dfn, 'directed_precision': float(dp), 'directed_recall': float(dr), 'directed_f1': float(df1), 'reversal_count': int(reversals), 'extra_adjacencies': int(extra), 'missing_adjacencies': int(missing), 'shd': int(reversals + extra + missing)}

class _rq1_s4_LearnedGaussianDAG:

    def __init__(self, graph: nx.DiGraph, nodes: Sequence[str]) -> None:
        self.graph = graph.copy()
        self.nodes = list(map(str, nodes))
        self.models: dict[str, dict[str, Any]] = {}

    def fit(self, training: pd.DataFrame) -> '_rq1_s4_LearnedGaussianDAG':
        self.models = {}
        for child in self.nodes:
            parents = tuple(sorted(map(str, self.graph.predecessors(child))))
            y = training[child].to_numpy(float)
            if parents:
                X = training[list(parents)].to_numpy(float)
                design = _rq1_s4_np.column_stack([_rq1_s4_np.ones(len(training)), X])
            else:
                design = _rq1_s4_np.ones((len(training), 1), dtype=float)
            beta, *_ = _rq1_s4_np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ beta
            variance = max(float(_rq1_s4_np.dot(residual, residual) / len(y)), _rq1_s4_MIN_VARIANCE)
            self.models[child] = {'parents': parents, 'beta': _rq1_s4_np.asarray(beta, dtype=float), 'variance': variance}
        return self

    def joint_logpdf(self, frame: pd.DataFrame) -> tuple[_rq1_s4_np.ndarray, dict[str, _rq1_s4_np.ndarray]]:
        total = _rq1_s4_np.zeros(len(frame), dtype=float)
        nodewise: dict[str, _rq1_s4_np.ndarray] = {}
        for child in self.nodes:
            model = self.models[child]
            parents = model['parents']
            if parents:
                X = frame[list(parents)].to_numpy(float)
                design = _rq1_s4_np.column_stack([_rq1_s4_np.ones(len(frame)), X])
            else:
                design = _rq1_s4_np.ones((len(frame), 1), dtype=float)
            mean = design @ model['beta']
            logs = _rq1_s4_norm.logpdf(frame[child].to_numpy(float), loc=mean, scale=_rq1_s4_math.sqrt(model['variance']))
            nodewise[child] = _rq1_s4_np.asarray(logs, dtype=float)
            total += logs
        return (total, nodewise)

def _rq1_s4_run_continuous_cell(*, root: Path, corrected_kl_module, generator_name: str, seed: int, n: int, mc_samples: int, output_root: Path, max_parents: int, tabu_length: int, max_iter: int, overwrite: bool) -> dict[str, Any]:
    run_dir = output_root / 'continuous' / 'runs' / generator_name / f'seed_{seed:02d}' / f'n_{n:04d}'
    result_path = run_dir / 'continuous_kl.json'
    if result_path.exists() and (not overwrite):
        print(f'[continuous seed={seed}] reuse {result_path}', flush=True)
        return _rq1_s4_json.loads(result_path.read_text(encoding='utf-8'))
    continuous_module = _rq1_inline_load(root / 'src' / 'hybrid_bn' / 'SLcontinuous.py', f'_continuous_baseline_{seed}_{_rq1_s4_os.getpid()}')
    Generator = _rq1_s4_import_generator(root, generator_name)
    generator = Generator(num_samples=n, seed=seed)
    generator.generate_dag()
    training = generator.generate_data()
    payload = generator.get_payload()
    nodes = list(map(str, payload['nodes']))
    training = training[nodes].copy().reset_index(drop=True)
    true_graph = payload['graph'].copy()
    started = _rq1_s4_time.perf_counter()
    scorer = continuous_module.ContinuousScoringEngine(training)
    search = continuous_module.TabuGreedyHillClimbing(nodes, scorer, tabu_length=tabu_length, max_iter=max_iter, max_parents=max_parents, improvement_tolerance=1e-10)
    learned_edges = search.search()
    structure_runtime = _rq1_s4_time.perf_counter() - started
    learned_graph = _rq1_s4_nx.DiGraph()
    learned_graph.add_nodes_from(nodes)
    learned_graph.add_edges_from(((str(u), str(v)) for u, v in learned_edges))
    metrics = _rq1_s4_structure_metrics(true_graph.edges(), learned_graph.edges())
    learned_score = float(sum((scorer.score_local(child, list(learned_graph.predecessors(child))) for child in nodes)))
    score_parity = abs(learned_score - float(search.best_score_)) if search.best_score_ is not None else float('nan')
    if _rq1_s4_np.isfinite(score_parity) and score_parity > 1e-08:
        raise AssertionError(f'Continuous returned-score parity failed seed {seed}: {score_parity:.3e}')
    true_model = corrected_kl_module.TrueContinuousDAG(true_graph, payload['true_parameters'], nodes)
    mc_seed = corrected_kl_module.deterministic_mc_seed(generator_name, seed)
    fresh = true_model.sample(mc_samples, mc_seed)
    log_p = _rq1_s4_np.zeros(mc_samples, dtype=float)
    for node in nodes:
        log_p += true_model.node_logpdf(node, fresh)
    q = _rq1_s4_LearnedGaussianDAG(learned_graph, nodes).fit(training)
    log_q, q_nodewise = q.joint_logpdf(fresh)
    ratio = log_p - log_q
    estimate = float(_rq1_s4_np.mean(ratio))
    se = float(_rq1_s4_np.std(ratio, ddof=1) / _rq1_s4_np.sqrt(mc_samples))
    run_dir.mkdir(parents=True, exist_ok=True)
    qdiag = _rq1_s4_pd.DataFrame([{'Node': node, 'Parents': '|'.join(sorted(map(str, learned_graph.predecessors(node)))), 'Parent_Count': int(learned_graph.in_degree(node)), 'Mean_Log_P': float(_rq1_s4_np.mean(true_model.node_logpdf(node, fresh))), 'Mean_Log_Q': float(_rq1_s4_np.mean(q_nodewise[node])), 'Signed_Local_LogRatio_Mean': float(_rq1_s4_np.mean(true_model.node_logpdf(node, fresh) - q_nodewise[node]))} for node in nodes])
    qdiag.to_csv(run_dir / 'continuous_node_diagnostics.csv', index=False)
    result = {'schema_version': _rq1_s4_SCHEMA_VERSION, 'status': 'complete', 'model': 'continuous_baseline', 'experiment_definition': {'generator': generator_name, 'seed': int(seed), 'sample_size': int(n), 'representation': 'raw_continuous_identity'}, 'structure_metrics': metrics, 'learned_edges': sorted([[str(u), str(v)] for u, v in learned_graph.edges()]), 'best_structure_score': learned_score, 'returned_score_parity_abs_diff': score_parity, 'structure_runtime_seconds': float(structure_runtime), 'kl': {'estimate': estimate, 'standard_error': se, 'lower_95': float(estimate - 1.96 * se), 'upper_95': float(estimate + 1.96 * se), 'mean_log_p': float(_rq1_s4_np.mean(log_p)), 'mean_log_q': float(_rq1_s4_np.mean(log_q)), 'mc_samples': int(mc_samples), 'mc_seed': int(mc_seed), 'definition': 'mean(log p_X(x) - log q_X(x)) for fresh x~P'}, 'runtime_seconds': float(_rq1_s4_time.perf_counter() - started)}
    _rq1_s4_atomic_json(run_dir / 'continuous_structure.json', result)
    _rq1_s4_atomic_json(result_path, result)
    print(f"[continuous seed={seed}] KL={estimate:.6f} ± {1.96 * se:.6f} | SHD={metrics['shd']} | F1={metrics['directed_f1']:.3f}", flush=True)
    return result

def _rq1_s4_load_discrete_result(root: Path, generator: str, seed: int, n: int) -> dict[str, Any]:
    p = root / 'Outputs' / 'DiscreteBaseline_RepresentationCorrect' / 'runs' / generator / f'seed_{seed:02d}' / f'n_{n:04d}' / 'discrete_kl.json'
    if not p.exists():
        raise FileNotFoundError(p)
    return _rq1_s4_json.loads(p.read_text(encoding='utf-8'))

def _rq1_s4_load_hybrid_result(root: Path, generator: str, seed: int, n: int) -> dict[str, Any]:
    p = root / 'Outputs' / 'HybridV3_CDv3Candidate_KL_RepresentationCorrect' / 'runs' / generator / f'seed_{seed:02d}' / f'n_{n:04d}' / 'hybrid_v3_kl.json'
    if not p.exists():
        raise FileNotFoundError(p)
    return _rq1_s4_json.loads(p.read_text(encoding='utf-8'))

def _rq1_s4_result_row(model: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    exp = payload['experiment_definition']
    kl = payload['kl']
    sm = payload['structure_metrics']
    return {'Model': model, 'Generator': exp['generator'], 'Seed': int(exp['seed']), 'Sample_Size': int(exp['sample_size']), 'KL': float(kl['estimate']), 'KL_SE': float(kl['standard_error']), 'KL_Lower95': float(kl['lower_95']), 'KL_Upper95': float(kl['upper_95']), 'SHD': int(sm['shd']), 'Precision': float(sm['directed_precision']), 'Recall': float(sm['directed_recall']), 'F1': float(sm['directed_f1']), 'Edges': int(sm['learned_edge_count']), 'Runtime_S': float(payload.get('runtime_seconds', _rq1_s4_np.nan))}

def _rq1_s4_build_comparison(*, output_root: Path, generator: str, seeds: Sequence[int], n: int, continuous_results: Mapping[int, Mapping[str, Any]], discrete_results: Mapping[int, Mapping[str, Any]], hybrid_results: Mapping[int, Mapping[str, Any]]) -> tuple[_rq1_s4_pd.DataFrame, _rq1_s4_pd.DataFrame, _rq1_s4_pd.DataFrame]:
    rows = []
    for seed in seeds:
        rows.append(_rq1_s4_result_row('Continuous', continuous_results[seed]))
        rows.append(_rq1_s4_result_row('Discrete', discrete_results[seed]))
        rows.append(_rq1_s4_result_row('Hybrid_CDv3', hybrid_results[seed]))
    df = _rq1_s4_pd.DataFrame(rows).sort_values(['Seed', 'Model']).reset_index(drop=True)
    df.to_csv(output_root / 'three_model_results.csv', index=False)
    summary = df.groupby('Model', as_index=False).agg(Runs=('KL', 'size'), Mean_KL=('KL', 'mean'), SD_KL=('KL', 'std'), Mean_KL_SE=('KL_SE', 'mean'), Mean_SHD=('SHD', 'mean'), Mean_Precision=('Precision', 'mean'), Mean_Recall=('Recall', 'mean'), Mean_F1=('F1', 'mean'), Mean_Edges=('Edges', 'mean'), Mean_Runtime_S=('Runtime_S', 'mean')).sort_values('Mean_KL').reset_index(drop=True)
    summary.to_csv(output_root / 'three_model_summary.csv', index=False)
    pivot = df.pivot(index='Seed', columns='Model', values='KL')
    pair_rows = []
    comparisons = [('Hybrid_CDv3', 'Discrete'), ('Hybrid_CDv3', 'Continuous'), ('Discrete', 'Continuous')]
    for a, b in comparisons:
        if a not in pivot.columns or b not in pivot.columns:
            continue
        diffs = pivot[a] - pivot[b]
        for seed, diff in diffs.items():
            pair_rows.append({'Model_A': a, 'Model_B': b, 'Seed': int(seed), 'KL_A_minus_B': float(diff), 'A_Better': bool(diff < 0)})
        pair_rows.append({'Model_A': a, 'Model_B': b, 'Seed': 'MEAN', 'KL_A_minus_B': float(diffs.mean()), 'A_Better': bool(diffs.mean() < 0)})
    pair_df = _rq1_s4_pd.DataFrame(pair_rows)
    pair_df.to_csv(output_root / 'pairwise_kl_differences.csv', index=False)
    lines = ['THREE-MODEL REPRESENTATION-CORRECT COMPARISON', '=' * 120, f'Generator={generator} | seeds={list(seeds)} | n={n}', '', 'PER-SEED', '-' * 120, df.to_string(index=False), '', 'SUMMARY', '-' * 120, summary.to_string(index=False), '', 'PAIRWISE KL DIFFERENCES (A-B; negative means A better)', '-' * 120, pair_df.to_string(index=False), '', 'PRIMARY DECISION RULE', '-' * 120, '1. KL is primary.', '2. Hybrid competitive with Discrete across seeds -> proceed to skew_multi gate.', '3. If skew_multi also holds -> unleash full run.', '4. Structural metrics remain secondary diagnostics.']
    (output_root / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')
    return (df, summary, pair_df)

def _rq1_s4_parse_args() -> _rq1_s4_argparse.Namespace:
    p = _rq1_s4_argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=_rq1_s4_Path, default=None)
    p.add_argument('--generator', choices=tuple(_rq1_s4_GENERATOR_SPECS), default='multimodal')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--n', type=int, default=5000)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--mc-samples', type=int, default=5000)
    p.add_argument('--particles', type=int, default=256)
    p.add_argument('--particle-replicates', type=int, default=2)
    p.add_argument('--particle-batch-size', type=int, default=64)
    p.add_argument('--max-parents', type=int, default=3)
    p.add_argument('--tabu-length', type=int, default=10)
    p.add_argument('--baseline-max-iter', type=int, default=300, help='Authoritative continuous/discrete GHC budget.')
    p.add_argument('--hybrid-max-iter', type=int, default=150, help='Authoritative Hybrid no-add2 GHC budget.')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def _rq1_s4_main() -> int:
    args = _rq1_s4_parse_args()
    here = _rq1_s4_Path(__file__).resolve().parent
    root = _rq1_s4_discover_root(here, args.project_root)
    experiments = root / 'experiments' / 'rq1'
    script15 = _rq1_s4_find_script(root, here, '15_discrete_baseline_representation_correct_kl.py')
    script16 = _rq1_s4_find_script(root, here, '16_hybrid_v3_cd_v3_structure_learning.py')
    script17 = _rq1_s4_find_script(root, here, '17_hybrid_v3_cd_v3_representation_correct_kl.py')
    script07v2 = _rq1_s4_find_script(root, here, '07_v2_hybrid_v3_representation_correct_kl.py')
    corrected_kl_module = _rq1_inline_load(script07v2, f'_comparison_kl07v2_{_rq1_s4_os.getpid()}')
    output_root = root / 'Outputs' / 'ThreeModel_RepresentationCorrect_Comparison'
    output_root.mkdir(parents=True, exist_ok=True)
    print('=' * 124)
    print('THREE-MODEL REPRESENTATION-CORRECT GATE')
    print('=' * 124)
    print(f'Generator: {args.generator}')
    print(f'Seeds:     {args.seeds}')
    print(f'n:         {args.n}')
    print('Models:    Continuous baseline | Discrete baseline | Hybrid + CD V3')
    print(f'Search caps: continuous/discrete={args.baseline_max_iter} | Hybrid={args.hybrid_max_iter}')
    print('=' * 124, flush=True)
    continuous_results: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        continuous_results[seed] = _rq1_s4_run_continuous_cell(root=root, corrected_kl_module=corrected_kl_module, generator_name=args.generator, seed=seed, n=args.n, mc_samples=args.mc_samples, output_root=output_root, max_parents=args.max_parents, tabu_length=args.tabu_length, max_iter=args.baseline_max_iter, overwrite=args.overwrite)
    for seed in args.seeds:
        cmd = [_rq1_s4_sys.executable, '-u', str(script15), '--generator', args.generator, '--seed', str(seed), '--n', str(args.n), '--mc-samples', str(args.mc_samples), '--particles', str(args.particles), '--particle-replicates', str(args.particle_replicates), '--particle-batch-size', str(args.particle_batch_size), '--max-parents', str(args.max_parents), '--tabu-length', str(args.tabu_length), '--max-iter', str(args.baseline_max_iter)]
        if args.overwrite:
            cmd.append('--overwrite')
        _rq1_s4_run_subprocess(cmd, experiments)
    cmd16 = [_rq1_s4_sys.executable, '-u', str(script16), '--generators', args.generator, '--seeds', *[str(s) for s in args.seeds], '--sample-sizes', str(args.n), '--workers', str(args.workers), '--max-parents', str(args.max_parents), '--tabu-length', str(args.tabu_length), '--max-iter', str(args.hybrid_max_iter)]
    if args.overwrite:
        cmd16.append('--overwrite')
    _rq1_s4_run_subprocess(cmd16, experiments)
    cmd17 = [_rq1_s4_sys.executable, '-u', str(script17), '--generators', args.generator, '--seeds', *[str(s) for s in args.seeds], '--sample-sizes', str(args.n), '--mc-samples', str(args.mc_samples), '--particles', str(args.particles), '--particle-replicates', str(args.particle_replicates), '--particle-batch-size', str(args.particle_batch_size)]
    if args.overwrite:
        cmd17.append('--overwrite')
    _rq1_s4_run_subprocess(cmd17, experiments)
    discrete_results = {seed: _rq1_s4_load_discrete_result(root, args.generator, seed, args.n) for seed in args.seeds}
    hybrid_results = {seed: _rq1_s4_load_hybrid_result(root, args.generator, seed, args.n) for seed in args.seeds}
    df, summary, pair_df = _rq1_s4_build_comparison(output_root=output_root, generator=args.generator, seeds=args.seeds, n=args.n, continuous_results=continuous_results, discrete_results=discrete_results, hybrid_results=hybrid_results)
    print('\n' + '=' * 124)
    print('THREE-MODEL GATE COMPLETE')
    print('=' * 124)
    print('\nPER-SEED')
    print(df.to_string(index=False))
    print('\nSUMMARY')
    print(summary.to_string(index=False))
    print('\nPAIRWISE KL')
    print(pair_df.to_string(index=False))
    print(f"\nReport: {output_root / 'report.txt'}")
    print('=' * 124)
    if args.generator == 'multimodal':
        print('\nIf this passes, run the identical skew-multi gate:')
        print('python -u 18_three_seed_model_comparison.py --generator skew_multi')
    return 0


# ======================================================================
# Evaluation implementation: 21_common_resolution_fully_discrete_kl.py
# ======================================================================
import argparse as _rq1_s5_argparse
import hashlib as _rq1_s5_hashlib
import importlib.util as _rq1_s5_importlib
import json as _rq1_s5_json
import math as _rq1_s5_math
import os as _rq1_s5_os
import sys as _rq1_s5_sys
import time as _rq1_s5_time
import traceback as _rq1_s5_traceback
from pathlib import Path as _rq1_s5_Path
from typing import Any as _rq1_s5_Any, Iterable as _rq1_s5_Iterable, Mapping as _rq1_s5_Mapping, Sequence as _rq1_s5_Sequence
_rq1_s5_sys.dont_write_bytecode = True
import networkx as _rq1_s5_nx
import numpy as _rq1_s5_np
import pandas as _rq1_s5_pd
from scipy.special import logsumexp as _rq1_s5_logsumexp
from scipy.stats import norm as _rq1_s5_norm
_rq1_s5_GENERATOR_SPECS = {'unimodal': ('generateUnimodalGT.py', 'StochasticContinuousGenerator'), 'skew': ('generateSkewGT.py', 'StochasticContinuousSkewedGenerator'), 'multimodal': ('generateMultimodalGT.py', 'StochasticContinuousMultimodalGenerator'), 'skew_multi': ('generateSkewMultiGT.py', 'StochasticContinuousSkewedMultimodalGenerator')}
_rq1_s5_SCHEMA_VERSION = 'common-resolution-fully-discrete-kl-v1'
_rq1_s5_MIN_PROB = 1e-300

def _rq1_s5_discover_root(here: _rq1_s5_Path, explicit: _rq1_s5_Path | None) -> _rq1_s5_Path:
    return _rq1_find_root()

def _rq1_s5_find_script(root: Path, here: Path, filename: str) -> Path:
    for p in (here / filename, root / 'experiments' / 'rq1' / filename):
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f'Could not find {filename}. Put this script in Experiments.')

def _rq1_s5_import_generator(root: Path, generator: str):
    filename, class_name = _rq1_s5_GENERATOR_SPECS[generator]
    module = _rq1_inline_load(root / 'src' / 'generators' / filename, f'_common_fd_generator_{generator}_{_rq1_s5_os.getpid()}')
    return getattr(module, class_name)

def _rq1_s5_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(_rq1_s5_json.dumps(_rq1_s5_json_safe(payload), indent=2), encoding='utf-8')
    _rq1_s5_os.replace(tmp, path)

def _rq1_s5_json_safe(value: Any) -> Any:
    if isinstance(value, _rq1_s5_Mapping):
        return {str(k): _rq1_s5_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_rq1_s5_json_safe(v) for v in value]
    if isinstance(value, _rq1_s5_np.ndarray):
        return _rq1_s5_json_safe(value.tolist())
    if isinstance(value, _rq1_s5_np.generic):
        return _rq1_s5_json_safe(value.item())
    if isinstance(value, _rq1_s5_Path):
        return str(value)
    if isinstance(value, float):
        if _rq1_s5_math.isnan(value):
            return 'NaN'
        if _rq1_s5_math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
    return value

def _rq1_s5_graph_from_edges(nodes: Sequence[str], edges: Iterable[Sequence[str]]) -> _rq1_s5_nx.DiGraph:
    g = _rq1_s5_nx.DiGraph()
    g.add_nodes_from(map(str, nodes))
    g.add_edges_from(((str(u), str(v)) for u, v in edges))
    if not _rq1_s5_nx.is_directed_acyclic_graph(g):
        raise ValueError('Saved learned graph is not a DAG.')
    return g

def _rq1_s5_deterministic_q_seed(generator: str, seed: int, model: str, replicate: int) -> int:
    token = f'common-fd|{generator}|{seed}|{model}|{replicate}'.encode()
    digest = _rq1_s5_hashlib.sha256(token).digest()
    return int.from_bytes(digest[:4], 'little', signed=False)

def _rq1_s5_stable_normal_interval_probability(lower: _rq1_s5_np.ndarray, upper: _rq1_s5_np.ndarray, location: _rq1_s5_np.ndarray | float, scale: float) -> _rq1_s5_np.ndarray:
    fl = _rq1_s5_norm.cdf(lower, loc=location, scale=scale)
    fu = _rq1_s5_norm.cdf(upper, loc=location, scale=scale)
    sl = _rq1_s5_norm.sf(lower, loc=location, scale=scale)
    su = _rq1_s5_norm.sf(upper, loc=location, scale=scale)
    return _rq1_s5_np.clip(_rq1_s5_np.maximum(fu - fl, sl - su), _rq1_s5_MIN_PROB, 1.0)

def _rq1_s5_truncated_normal(lower: _rq1_s5_np.ndarray, upper: _rq1_s5_np.ndarray, location: _rq1_s5_np.ndarray, scale: float, rng: _rq1_s5_np.random.Generator) -> _rq1_s5_np.ndarray:
    lower, upper, location = _rq1_s5_np.broadcast_arrays(_rq1_s5_np.asarray(lower, float), _rq1_s5_np.asarray(upper, float), _rq1_s5_np.asarray(location, float))
    fl = _rq1_s5_norm.cdf(lower, loc=location, scale=scale)
    fu = _rq1_s5_norm.cdf(upper, loc=location, scale=scale)
    sl = _rq1_s5_norm.sf(lower, loc=location, scale=scale)
    su = _rq1_s5_norm.sf(upper, loc=location, scale=scale)
    u = rng.random(lower.shape)
    use_sf = fu > 0.9
    result = _rq1_s5_np.empty(lower.shape, dtype=float)
    cdf_mask = ~use_sf
    if _rq1_s5_np.any(cdf_mask):
        q = fl[cdf_mask] + u[cdf_mask] * _rq1_s5_np.maximum(fu[cdf_mask] - fl[cdf_mask], 0.0)
        q = _rq1_s5_np.clip(q, _rq1_s5_np.finfo(float).tiny, 1.0 - _rq1_s5_np.finfo(float).eps)
        result[cdf_mask] = _rq1_s5_norm.ppf(q, loc=location[cdf_mask], scale=scale)
    if _rq1_s5_np.any(use_sf):
        s = su[use_sf] + u[use_sf] * _rq1_s5_np.maximum(sl[use_sf] - su[use_sf], 0.0)
        s = _rq1_s5_np.clip(s, _rq1_s5_np.finfo(float).tiny, 1.0 - _rq1_s5_np.finfo(float).eps)
        result[use_sf] = _rq1_s5_norm.isf(s, loc=location[use_sf], scale=scale)
    return result

def _rq1_s5_interval_bounds(transform, node: str, states: _rq1_s5_np.ndarray) -> tuple[_rq1_s5_np.ndarray, _rq1_s5_np.ndarray]:
    edges = _rq1_s5_np.asarray(transform.bin_edges[node], dtype=float)
    states = _rq1_s5_np.asarray(states, dtype=_rq1_s5_np.int64)
    return (edges[states], edges[states + 1])

def _rq1_s5_combine_replicate_log_estimates(replicate_logs: Sequence[_rq1_s5_np.ndarray]) -> tuple[_rq1_s5_np.ndarray, dict[str, float]]:
    matrix = _rq1_s5_np.vstack([_rq1_s5_np.asarray(row, dtype=float) for row in replicate_logs])
    combined = _rq1_s5_logsumexp(matrix, axis=0) - _rq1_s5_math.log(len(matrix))
    if len(matrix) > 1:
        row_sd = _rq1_s5_np.std(matrix, axis=0, ddof=1)
        diagnostics = {'replicate_mean_logq': [float(_rq1_s5_np.mean(matrix[i])) for i in range(len(matrix))], 'replicate_mean_logq_sd': float(_rq1_s5_np.std([_rq1_s5_np.mean(matrix[i]) for i in range(len(matrix))], ddof=1)), 'median_row_logq_replicate_sd': float(_rq1_s5_np.median(row_sd)), 'p95_row_logq_replicate_sd': float(_rq1_s5_np.quantile(row_sd, 0.95)), 'max_row_logq_replicate_sd': float(_rq1_s5_np.max(row_sd))}
    else:
        diagnostics = {'replicate_mean_logq': [float(_rq1_s5_np.mean(matrix[0]))], 'replicate_mean_logq_sd': float('nan'), 'median_row_logq_replicate_sd': float('nan'), 'p95_row_logq_replicate_sd': float('nan'), 'max_row_logq_replicate_sd': float('nan')}
    return (_rq1_s5_np.asarray(combined, dtype=float), diagnostics)

class _rq1_s5_ContinuousGaussianBinMassEstimator:
    """Sequential exact-local integration for a learned linear-Gaussian DAG.

    The joint bin mass is estimated by likelihood weighting along the learned
    DAG. At each node:
      * multiply by Q(child in target bin | sampled parents);
      * sample child from that conditional truncated to the target bin.

    This is the Q-side analogue of the representation-correct latent-bin
    estimator, but every learned local conditional is Gaussian.
    """

    def __init__(self, *, graph: _rq1_s5_nx.DiGraph, models: Mapping[str, Mapping[str, Any]], nodes: Sequence[str], transform, particles: int, replicates: int, batch_size: int, generator: str, seed: int) -> None:
        self.graph = graph.copy()
        self.models = models
        self.nodes = list(map(str, nodes))
        self.transform = transform
        self.particles = int(particles)
        self.replicates = int(replicates)
        self.batch_size = int(batch_size)
        self.generator = str(generator)
        self.seed = int(seed)
        self.order = list(map(str, _rq1_s5_nx.topological_sort(self.graph)))

    def _one_replicate(self, target_states: pd.DataFrame, rng: _rq1_s5_np.random.Generator) -> _rq1_s5_np.ndarray:
        n = len(target_states)
        out = _rq1_s5_np.empty(n, dtype=float)
        for start in range(0, n, self.batch_size):
            stop = min(n, start + self.batch_size)
            b = stop - start
            state_batch = target_states.iloc[start:stop]
            log_weight = _rq1_s5_np.zeros((b, self.particles), dtype=float)
            raw: dict[str, _rq1_s5_np.ndarray] = {}
            for child in self.order:
                model = self.models[child]
                parents = list(model['parents'])
                beta = _rq1_s5_np.asarray(model['beta'], dtype=float)
                sd = _rq1_s5_math.sqrt(float(model['variance']))
                if parents:
                    mean = _rq1_s5_np.full((b, self.particles), float(beta[0]), dtype=float)
                    for j, parent in enumerate(parents, start=1):
                        mean += float(beta[j]) * raw[parent]
                else:
                    mean = _rq1_s5_np.full((b, self.particles), float(beta[0]), dtype=float)
                states = state_batch[child].to_numpy(_rq1_s5_np.int64)
                lower_1d, upper_1d = _rq1_s5_interval_bounds(self.transform, child, states)
                lower = _rq1_s5_np.broadcast_to(lower_1d[:, None], (b, self.particles))
                upper = _rq1_s5_np.broadcast_to(upper_1d[:, None], (b, self.particles))
                probability = _rq1_s5_stable_normal_interval_probability(lower, upper, mean, sd)
                log_weight += _rq1_s5_np.log(probability)
                raw[child] = _rq1_s5_truncated_normal(lower, upper, mean, sd, rng)
            out[start:stop] = _rq1_s5_logsumexp(log_weight, axis=1) - _rq1_s5_math.log(self.particles)
        return out

    def logmass(self, target_states: pd.DataFrame) -> tuple[_rq1_s5_np.ndarray, dict[str, Any]]:
        reps = []
        for r in range(self.replicates):
            rng = _rq1_s5_np.random.default_rng(_rq1_s5_deterministic_q_seed(self.generator, self.seed, 'continuous', r))
            reps.append(self._one_replicate(target_states, rng))
        combined, diagnostics = _rq1_s5_combine_replicate_log_estimates(reps)
        diagnostics.update({'method': 'sequential_conditional_gaussian_bin_integration', 'particles': self.particles, 'replicates': self.replicates, 'batch_size': self.batch_size})
        return (combined, diagnostics)

class _rq1_s5_HybridCommonResolutionImportanceEstimator:
    """Integrate Hybrid's continuous variables inside the common bins.

    The fitted native Hybrid Q is a density in its designated continuous
    variables and a probability mass in its designated discrete variables.

    For a fully-discrete target state d_all:

        Q_TD(d_all)
          = integral_{x_C in common bins(d_C)}
                q_H(x_C, d_D) dx_C.

    We estimate this integral with an independent truncated-normal proposal
    over only the Hybrid continuous variables.  There are far fewer latent
    dimensions here than for the fully continuous model, and the proposal has
    full support over each target bin.

    Existing Hybrid D variables are represented by deterministic in-bin raw
    representatives; Q only consumes their transformed states.
    """

    def __init__(self, *, q, raw_training: _rq1_s5_pd.DataFrame, common_transform, native_transform, continuous_nodes: Sequence[str], discrete_nodes: Sequence[str], nodes: Sequence[str], particles: int, replicates: int, batch_size: int, generator: str, seed: int) -> None:
        self.q = q
        self.raw_training = raw_training.reset_index(drop=True)
        self.common_transform = common_transform
        self.native_transform = native_transform
        self.continuous_nodes = list(map(str, continuous_nodes))
        self.discrete_nodes = list(map(str, discrete_nodes))
        self.nodes = list(map(str, nodes))
        self.particles = int(particles)
        self.replicates = int(replicates)
        self.batch_size = int(batch_size)
        self.generator = str(generator)
        self.seed = int(seed)
        self.proposal: dict[str, tuple[float, float]] = {}
        for node in self.continuous_nodes:
            values = self.raw_training[node].to_numpy(float)
            mu = float(_rq1_s5_np.mean(values))
            sd = max(float(_rq1_s5_np.std(values, ddof=0)), 1e-08)
            self.proposal[node] = (mu, sd)
        for node in self.discrete_nodes:
            a = _rq1_s5_np.asarray(self.native_transform.bin_edges[node], dtype=float)
            b = _rq1_s5_np.asarray(self.common_transform.bin_edges[node], dtype=float)
            if a.shape != b.shape or not _rq1_s5_np.array_equal(a, b):
                raise ValueError(f"Hybrid native bins for {node!r} do not exactly match the discrete baseline's common bins.")
        common_training_states = self.common_transform.transform(self.raw_training.copy())
        self.representatives: dict[str, _rq1_s5_np.ndarray] = {}
        for node in self.discrete_nodes:
            k = int(self.common_transform.cardinalities[node])
            reps = _rq1_s5_np.empty(k, dtype=float)
            raw_values = self.raw_training[node].to_numpy(float)
            states = common_training_states[node].to_numpy(_rq1_s5_np.int64)
            for state in range(k):
                vals = raw_values[states == state]
                if len(vals):
                    reps[state] = float(_rq1_s5_np.median(vals))
                else:
                    lo = float(self.common_transform.bin_edges[node][state])
                    hi = float(self.common_transform.bin_edges[node][state + 1])
                    mu = float(_rq1_s5_np.mean(raw_values))
                    sd = max(float(_rq1_s5_np.std(raw_values)), 1e-08)
                    if _rq1_s5_np.isfinite(lo) and _rq1_s5_np.isfinite(hi):
                        reps[state] = 0.5 * (lo + hi)
                    elif _rq1_s5_np.isfinite(lo):
                        reps[state] = lo + 0.5 * sd
                    elif _rq1_s5_np.isfinite(hi):
                        reps[state] = hi - 0.5 * sd
                    else:
                        reps[state] = mu
            check = self.native_transform.transform(_rq1_s5_pd.DataFrame({node: reps, **{other: _rq1_s5_np.repeat(float(self.raw_training[other].iloc[0]), k) for other in self.native_transform.nodes if other != node}})[list(self.native_transform.cardinalities)])[node].to_numpy(_rq1_s5_np.int64)
            if not _rq1_s5_np.array_equal(check, _rq1_s5_np.arange(k, dtype=_rq1_s5_np.int64)):
                raise AssertionError(f'Could not construct valid state representatives for {node!r}.')
            self.representatives[node] = reps

    def _one_replicate(self, target_states: _rq1_s5_pd.DataFrame, rng: _rq1_s5_np.random.Generator) -> tuple[_rq1_s5_np.ndarray, _rq1_s5_np.ndarray]:
        n = len(target_states)
        out = _rq1_s5_np.empty(n, dtype=float)
        ess_fraction = _rq1_s5_np.empty(n, dtype=float)
        for start in range(0, n, self.batch_size):
            stop = min(n, start + self.batch_size)
            b = stop - start
            states = target_states.iloc[start:stop]
            total = b * self.particles
            raw_frame = _rq1_s5_pd.DataFrame(index=_rq1_s5_np.arange(total), columns=self.nodes, dtype=float)
            log_r = _rq1_s5_np.zeros(total, dtype=float)
            for node in self.discrete_nodes:
                s = states[node].to_numpy(_rq1_s5_np.int64)
                values = self.representatives[node][s]
                raw_frame[node] = _rq1_s5_np.repeat(values, self.particles)
            for node in self.continuous_nodes:
                mu, sd = self.proposal[node]
                s = states[node].to_numpy(_rq1_s5_np.int64)
                lo_1d, hi_1d = _rq1_s5_interval_bounds(self.common_transform, node, s)
                lo = _rq1_s5_np.repeat(lo_1d, self.particles)
                hi = _rq1_s5_np.repeat(hi_1d, self.particles)
                loc = _rq1_s5_np.full(total, mu, dtype=float)
                bin_prob = _rq1_s5_stable_normal_interval_probability(lo, hi, loc, sd)
                sampled = _rq1_s5_truncated_normal(lo, hi, loc, sd, rng)
                raw_frame[node] = sampled
                log_r += _rq1_s5_norm.logpdf(sampled, loc=mu, scale=sd) - _rq1_s5_np.log(bin_prob)
            raw_frame = raw_frame[self.nodes]
            log_q_native = _rq1_s5_np.asarray(self.q.joint_logscore(raw_frame), dtype=float)
            if log_q_native.shape != (total,) or not _rq1_s5_np.all(_rq1_s5_np.isfinite(log_q_native)):
                raise FloatingPointError('Hybrid native Q returned invalid values during common-resolution integration.')
            log_w = (log_q_native - log_r).reshape(b, self.particles)
            out[start:stop] = _rq1_s5_logsumexp(log_w, axis=1) - _rq1_s5_math.log(self.particles)
            log_sum_w = _rq1_s5_logsumexp(log_w, axis=1)
            log_sum_w2 = _rq1_s5_logsumexp(2.0 * log_w, axis=1)
            ess = _rq1_s5_np.exp(2.0 * log_sum_w - log_sum_w2)
            ess_fraction[start:stop] = _rq1_s5_np.clip(ess / self.particles, 0.0, 1.0)
        return (out, ess_fraction)

    def logmass(self, target_states: pd.DataFrame) -> tuple[_rq1_s5_np.ndarray, dict[str, Any]]:
        replicate_logs = []
        ess_rows = []
        for r in range(self.replicates):
            rng = _rq1_s5_np.random.default_rng(_rq1_s5_deterministic_q_seed(self.generator, self.seed, 'hybrid', r))
            logs, ess = self._one_replicate(target_states, rng)
            replicate_logs.append(logs)
            ess_rows.append(ess)
        combined, diagnostics = _rq1_s5_combine_replicate_log_estimates(replicate_logs)
        all_ess = _rq1_s5_np.concatenate(ess_rows)
        diagnostics.update({'method': 'importance_integration_of_native_hybrid_Q_over_common_bins', 'proposal': 'independent training-marginal Gaussian truncated to each common bin for Hybrid continuous variables', 'particles': self.particles, 'replicates': self.replicates, 'batch_size': self.batch_size, 'continuous_integration_dimensions': len(self.continuous_nodes), 'median_ess_fraction': float(_rq1_s5_np.median(all_ess)), 'p10_ess_fraction': float(_rq1_s5_np.quantile(all_ess, 0.1)), 'p01_ess_fraction': float(_rq1_s5_np.quantile(all_ess, 0.01)), 'min_ess_fraction': float(_rq1_s5_np.min(all_ess))})
        return (combined, diagnostics)

def _rq1_s5_existing_paths(root: Path, generator: str, seed: int, n: int) -> dict[str, Path]:
    base_disc = root / 'Outputs' / 'DiscreteBaseline_RepresentationCorrect' / 'runs' / generator / f'seed_{seed:02d}' / f'n_{n:04d}'
    base_cont = root / 'Outputs' / 'ThreeModel_RepresentationCorrect_Comparison' / 'continuous' / 'runs' / generator / f'seed_{seed:02d}' / f'n_{n:04d}'
    base_h_struct = root / 'Outputs' / 'HybridV3_CDv3Candidate_StructureLearning_NoAdd2' / 'runs' / generator / f'seed_{seed:02d}' / f'n_{n:04d}'
    base_h_kl = root / 'Outputs' / 'HybridV3_CDv3Candidate_KL_RepresentationCorrect' / 'runs' / generator / f'seed_{seed:02d}' / f'n_{n:04d}'
    return {'discrete_structure': base_disc / 'discrete_structure.json', 'discrete_kl': base_disc / 'discrete_kl.json', 'continuous_structure': base_cont / 'continuous_structure.json', 'continuous_kl': base_cont / 'continuous_kl.json', 'hybrid_structure': base_h_struct / 'hybrid_v3_structure.json', 'hybrid_kl': base_h_kl / 'hybrid_v3_kl.json'}

def _rq1_s5_require_paths(paths: Mapping[str, Path]) -> None:
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError('Required learned-model artifacts are missing:\n  ' + '\n  '.join(missing))

def _rq1_s5_fit_hybrid_q(*, root: Path, hybrid_module, structure_cell: _rq1_s5_Mapping[str, Any], structure_path: Path, raw_training: pd.DataFrame, payload: _rq1_s5_Mapping[str, Any], metadata_score_tolerance: float, cpt_alpha: float, min_regime_size: int, logistic_c: float, seed: int, n: int):
    nodes = list(map(str, payload['nodes']))
    discrete_nodes = list(map(str, payload['discrete_nodes']))
    continuous_nodes = list(map(str, payload['continuous_nodes']))
    cardinalities = hybrid_module.default_cardinalities(payload)
    learned_graph = hybrid_module.build_graph(nodes, structure_cell['learned_edges'])
    native_transform = hybrid_module.SavedQuantileTransform.from_metadata(structure_cell['discretizer_metadata'])
    model_file = (root / 'src' / 'hybrid_bn' / 'SLhybrid.py').resolve()
    cc_file = (root / 'src' / 'hybrid_bn' / 'SLhybrid.py').resolve()
    model_module = _rq1_inline_load(model_file, f'_common_fd_hybrid_model_{seed}_{n}_{_rq1_s5_os.getpid()}')
    evaluation_metadata = structure_cell.get('evaluation_metadata')
    if not isinstance(evaluation_metadata, _rq1_s5_Mapping):
        saved_score = structure_cell.get('learned_dag_score_recomputed_main')
        evaluation_metadata = hybrid_module.reconstruct_final_evaluation_metadata(model_module=model_module, cc_scorer_file=cc_file, raw_training=raw_training, transformer=native_transform, learned_graph=learned_graph, nodes=nodes, discrete_nodes=discrete_nodes, continuous_nodes=continuous_nodes, cardinalities=cardinalities, saved_final_score=float(saved_score) if saved_score is not None else None, score_tolerance=metadata_score_tolerance)
    selected_k = evaluation_metadata.get('dc_selected_k_by_configuration', {})
    scaling = evaluation_metadata.get('hybrid_continuous_scaling', {})
    priors = evaluation_metadata.get('frozen_prior_hyperparameters', {})
    discrete_set = set(discrete_nodes)
    needs_cc = False
    for child in nodes:
        parents = list(map(str, learned_graph.predecessors(child)))
        if child not in discrete_set and (not any((p in discrete_set for p in parents))):
            needs_cc = True
            break
    cc_module = _rq1_inline_load(cc_file, f'_common_fd_cc_{seed}_{n}_{_rq1_s5_os.getpid()}') if needs_cc else None
    hybrid_training_for_cd = native_transform.transform(raw_training.copy())
    cd_v3_engine = model_module.HybridScoringEngine(hybrid_training_for_cd, discrete_nodes, continuous_nodes, cardinalities, cc_scorer_file=cc_file)
    q = hybrid_module.HybridV3LearnedDAGDistribution(corrected_cc_module=cc_module, cd_v3_engine=cd_v3_engine, graph=learned_graph, nodes=nodes, discrete_nodes=discrete_nodes, continuous_nodes=continuous_nodes, transform=native_transform, dc_selected_k_by_configuration=selected_k, hybrid_continuous_scaling=scaling, hybrid_prior_hyperparameters=priors, cpt_alpha=cpt_alpha, min_regime_size=min_regime_size, logistic_c=logistic_c).fit(raw_training)
    return (q, native_transform, learned_graph, discrete_nodes, continuous_nodes)

def _rq1_s5_evaluate_cell(*, root: Path, generator_name: str, seed: int, n: int, args: argparse.Namespace, kl07, disc15, comparison18, hybrid17, discrete_model_module) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths = _rq1_s5_existing_paths(root, generator_name, seed, n)
    _rq1_s5_require_paths(paths)
    discrete_structure = _rq1_s5_json.loads(paths['discrete_structure'].read_text(encoding='utf-8'))
    discrete_native = _rq1_s5_json.loads(paths['discrete_kl'].read_text(encoding='utf-8'))
    continuous_native = _rq1_s5_json.loads(paths['continuous_kl'].read_text(encoding='utf-8'))
    hybrid_structure = _rq1_s5_json.loads(paths['hybrid_structure'].read_text(encoding='utf-8'))
    hybrid_native = _rq1_s5_json.loads(paths['hybrid_kl'].read_text(encoding='utf-8'))
    Generator = _rq1_s5_import_generator(root, generator_name)
    generator = Generator(num_samples=n, seed=seed)
    generator.generate_dag()
    raw_training = generator.generate_data()
    payload = generator.get_payload()
    nodes = list(map(str, payload['nodes']))
    raw_training = raw_training[nodes].copy().reset_index(drop=True)
    true_graph = payload['graph'].copy()
    common_transform = kl07.SavedQuantileTransform.from_metadata(discrete_structure['discretizer_metadata'])
    cardinalities = discrete_model_module.default_cardinalities(nodes)
    refit = discrete_model_module.QuantileDiscretizer(cardinalities)
    refit.fit_transform(raw_training.copy())
    for node in nodes:
        a = _rq1_s5_np.asarray(refit.bin_edges[node], dtype=float)
        b = _rq1_s5_np.asarray(common_transform.bin_edges[node], dtype=float)
        if not _rq1_s5_np.array_equal(a, b):
            raise AssertionError(f'Common-bin training-only parity failed for {node!r}.')
    saved_mc = int(discrete_native['kl']['mc_samples'])
    saved_mc_seed = int(discrete_native['kl']['mc_seed'])
    expected_mc_seed = int(kl07.deterministic_mc_seed(generator_name, seed))
    if saved_mc_seed != expected_mc_seed:
        raise AssertionError('Saved discrete KL uses a different fresh MC seed.')
    if args.mc_samples is not None and args.mc_samples != saved_mc:
        raise ValueError(f'Requested --mc-samples={args.mc_samples}, but the saved common P term uses {saved_mc}. Cheap mode requires the same fresh sample. Omit --mc-samples or use the saved value.')
    mc_samples = saved_mc
    true_model = kl07.TrueContinuousDAG(true_graph, payload['true_parameters'], nodes)
    fresh = true_model.sample(mc_samples, saved_mc_seed)[nodes]
    common_states = common_transform.transform(fresh.copy()).reset_index(drop=True)
    shared_mean_log_p = float(discrete_native['kl']['mean_log_p'])
    discrete_graph = _rq1_s5_graph_from_edges(nodes, discrete_structure['learned_edges'])
    dtrain = common_transform.transform(raw_training.copy())
    q_discrete = disc15.SmoothedDiscreteDAG(discrete_graph, nodes, cardinalities, alpha=args.cpt_alpha).fit(dtrain)
    logq_discrete, _ = q_discrete.joint_logscore(fresh, common_transform)
    saved_disc_mean_q = float(discrete_native['kl']['mean_log_q'])
    disc_q_parity = abs(float(_rq1_s5_np.mean(logq_discrete)) - saved_disc_mean_q)
    if disc_q_parity > 1e-10:
        raise AssertionError(f'Discrete common-Q parity failed: {disc_q_parity:.3e}')
    common_disc_kl = shared_mean_log_p - float(_rq1_s5_np.mean(logq_discrete))
    disc_kl_parity = abs(common_disc_kl - float(discrete_native['kl']['estimate']))
    if disc_kl_parity > 1e-08:
        raise AssertionError(f'Discrete common-KL parity failed: {disc_kl_parity:.3e}')
    continuous_graph = _rq1_s5_graph_from_edges(nodes, continuous_native['learned_edges'])
    continuous_q = comparison18.LearnedGaussianDAG(continuous_graph, nodes).fit(raw_training)
    continuous_estimator = _rq1_s5_ContinuousGaussianBinMassEstimator(graph=continuous_graph, models=continuous_q.models, nodes=nodes, transform=common_transform, particles=args.continuous_particles, replicates=args.replicates, batch_size=args.batch_size, generator=generator_name, seed=seed)
    t0 = _rq1_s5_time.perf_counter()
    logq_continuous, continuous_diag = continuous_estimator.logmass(common_states)
    continuous_runtime = _rq1_s5_time.perf_counter() - t0
    hybrid_q, hybrid_native_transform, _, hybrid_discrete_nodes, hybrid_continuous_nodes = _rq1_s5_fit_hybrid_q(root=root, hybrid_module=hybrid17, structure_cell=hybrid_structure, structure_path=paths['hybrid_structure'], raw_training=raw_training, payload=payload, metadata_score_tolerance=args.metadata_score_tolerance, cpt_alpha=args.cpt_alpha, min_regime_size=args.min_regime_size, logistic_c=args.logistic_c, seed=seed, n=n)
    hybrid_estimator = _rq1_s5_HybridCommonResolutionImportanceEstimator(q=hybrid_q, raw_training=raw_training, common_transform=common_transform, native_transform=hybrid_native_transform, continuous_nodes=hybrid_continuous_nodes, discrete_nodes=hybrid_discrete_nodes, nodes=nodes, particles=args.hybrid_particles, replicates=args.replicates, batch_size=args.batch_size, generator=generator_name, seed=seed)
    t1 = _rq1_s5_time.perf_counter()
    logq_hybrid, hybrid_diag = hybrid_estimator.logmass(common_states)
    hybrid_runtime = _rq1_s5_time.perf_counter() - t1
    q_logs = {'Discrete': _rq1_s5_np.asarray(logq_discrete, dtype=float), 'Hybrid_CDv3': _rq1_s5_np.asarray(logq_hybrid, dtype=float), 'Continuous': _rq1_s5_np.asarray(logq_continuous, dtype=float)}
    native_payloads = {'Discrete': discrete_native, 'Hybrid_CDv3': hybrid_native, 'Continuous': continuous_native}
    model_rows = []
    for model, logq in q_logs.items():
        native = native_payloads[model]
        common_kl = shared_mean_log_p - float(_rq1_s5_np.mean(logq))
        native_kl = float(native['kl']['estimate'])
        sm = native['structure_metrics']
        model_rows.append({'Model': model, 'Generator': generator_name, 'Seed': int(seed), 'Sample_Size': int(n), 'Fresh_MC_Rows': int(mc_samples), 'Common_FD_KL': float(common_kl), 'Common_Mean_Log_P': shared_mean_log_p, 'Common_Mean_Log_Q': float(_rq1_s5_np.mean(logq)), 'Native_KL': native_kl, 'Resolution_Penalty_NativeMinusCommon': float(native_kl - common_kl), 'SHD': int(sm['shd']), 'Precision': float(sm['directed_precision']), 'Recall': float(sm['directed_recall']), 'F1': float(sm['directed_f1']), 'Edges': int(sm['learned_edge_count'])})
    pair_rows = []
    pairs = [('Hybrid_CDv3', 'Discrete'), ('Hybrid_CDv3', 'Continuous'), ('Discrete', 'Continuous')]
    for a, b in pairs:
        paired = q_logs[b] - q_logs[a]
        diff = float(_rq1_s5_np.mean(paired))
        se = float(_rq1_s5_np.std(paired, ddof=1) / _rq1_s5_math.sqrt(len(paired)))
        pair_rows.append({'Generator': generator_name, 'Seed': int(seed), 'Sample_Size': int(n), 'Model_A': a, 'Model_B': b, 'Common_KL_A_minus_B': diff, 'Paired_SE': se, 'Lower95': float(diff - 1.96 * se), 'Upper95': float(diff + 1.96 * se), 'A_Better': bool(diff < 0), 'Significant_95': bool(diff - 1.96 * se > 0 or diff + 1.96 * se < 0)})
    diagnostics = {'schema_version': _rq1_s5_SCHEMA_VERSION, 'generator': generator_name, 'seed': int(seed), 'sample_size': int(n), 'fresh_mc_rows': int(mc_samples), 'fresh_mc_seed': int(saved_mc_seed), 'shared_mean_log_p_source': str(paths['discrete_kl']), 'shared_mean_log_p': shared_mean_log_p, 'common_transform_source': str(paths['discrete_structure']), 'discrete_parity': {'mean_log_q_abs_diff': float(disc_q_parity), 'kl_abs_diff': float(disc_kl_parity)}, 'continuous_q_integration': {**continuous_diag, 'runtime_seconds': float(continuous_runtime)}, 'hybrid_q_integration': {**hybrid_diag, 'runtime_seconds': float(hybrid_runtime)}, 'flags': []}
    if hybrid_diag['p10_ess_fraction'] < args.ess_warning:
        diagnostics['flags'].append('HYBRID_IMPORTANCE_ESS_LOW_RERUN_WITH_MORE_PARTICLES')
    rep_sd = hybrid_diag.get('replicate_mean_logq_sd', float('nan'))
    if _rq1_s5_np.isfinite(rep_sd) and rep_sd > args.replicate_logq_warning:
        diagnostics['flags'].append('HYBRID_REPLICATE_LOGQ_DRIFT_HIGH')
    return (model_rows, pair_rows, diagnostics)

def _rq1_s5_build_reports(output_dir: Path, model_rows: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]], diagnostics: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _rq1_s5_pd.DataFrame(model_rows)
    pairs = _rq1_s5_pd.DataFrame(pair_rows)
    df.to_csv(output_dir / 'common_resolution_results.csv', index=False)
    pairs.to_csv(output_dir / 'common_resolution_pairwise.csv', index=False)
    if len(df):
        summary = df.groupby(['Generator', 'Model'], as_index=False).agg(Runs=('Common_FD_KL', 'size'), Mean_Common_FD_KL=('Common_FD_KL', 'mean'), SD_Common_FD_KL=('Common_FD_KL', 'std'), Mean_Native_KL=('Native_KL', 'mean'), Mean_Resolution_Penalty=('Resolution_Penalty_NativeMinusCommon', 'mean'), Mean_SHD=('SHD', 'mean'), Mean_F1=('F1', 'mean')).sort_values(['Generator', 'Mean_Common_FD_KL']).reset_index(drop=True)
    else:
        summary = _rq1_s5_pd.DataFrame()
    summary.to_csv(output_dir / 'common_resolution_summary.csv', index=False)
    if len(pairs):
        pair_summary = pairs.groupby(['Generator', 'Model_A', 'Model_B'], as_index=False).agg(Runs=('Common_KL_A_minus_B', 'size'), Mean_Difference=('Common_KL_A_minus_B', 'mean'), SD_Difference=('Common_KL_A_minus_B', 'std'), A_Wins=('A_Better', 'sum'), Significant_Runs=('Significant_95', 'sum'))
    else:
        pair_summary = _rq1_s5_pd.DataFrame()
    pair_summary.to_csv(output_dir / 'common_resolution_pairwise_summary.csv', index=False)
    _rq1_s5_atomic_json(output_dir / 'diagnostics.json', {'cells': list(diagnostics)})
    lines = ['COMMON-RESOLUTION FULLY-DISCRETE KL', '=' * 132, '', 'Interpretation:', '  Native KL = each paradigm solves its own native-resolution task.', '  Common FD KL = every learned model is transformed to the exact same', '                 fully-discrete quantile representation.', '  Native-Common = extra native-resolution burden; it is diagnostic,', '                  not a KL decomposition theorem.', '', 'IMPORTANT:', '  Pairwise common-KL differences are paired on the same fresh P sample', '  and do not require rowwise log P because the common P term cancels.', '', 'PER-SEED', '-' * 132, df.to_string(index=False) if len(df) else '(none)', '', 'SUMMARY', '-' * 132, summary.to_string(index=False) if len(summary) else '(none)', '', 'PAIRWISE COMMON-KL', '-' * 132, pairs.to_string(index=False) if len(pairs) else '(none)', '', 'PAIRWISE SUMMARY', '-' * 132, pair_summary.to_string(index=False) if len(pair_summary) else '(none)', '', 'DIAGNOSTIC FLAGS', '-' * 132]
    any_flags = False
    for cell in diagnostics:
        if cell.get('flags'):
            any_flags = True
            lines.append(f"{cell['generator']} seed={cell['seed']}: " + ', '.join(cell['flags']))
    if not any_flags:
        lines.append('none')
    (output_dir / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def _rq1_s5_parse_args() -> _rq1_s5_argparse.Namespace:
    p = _rq1_s5_argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=_rq1_s5_Path, default=None)
    p.add_argument('--generators', nargs='+', choices=tuple(_rq1_s5_GENERATOR_SPECS), default=['multimodal', 'skew_multi'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--sample-size', type=int, default=5000)
    p.add_argument('--mc-samples', type=int, default=None)
    p.add_argument('--continuous-particles', type=int, default=128)
    p.add_argument('--hybrid-particles', type=int, default=128)
    p.add_argument('--replicates', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--cpt-alpha', type=float, default=1.0)
    p.add_argument('--min-regime-size', type=int, default=5)
    p.add_argument('--logistic-c', type=float, default=1000.0)
    p.add_argument('--metadata-score-tolerance', type=float, default=1e-06)
    p.add_argument('--ess-warning', type=float, default=0.02, help='Flag Hybrid if p10 ESS fraction drops below this.')
    p.add_argument('--replicate-logq-warning', type=float, default=0.03, help='Flag Hybrid if replicate mean-logQ SD exceeds this.')
    p.add_argument('--output-dir', type=_rq1_s5_Path, default=_rq1_s5_Path('Outputs/CommonResolution_FullyDiscreteKL'))
    return p.parse_args()

def _rq1_s5_main() -> int:
    args = _rq1_s5_parse_args()
    here = _rq1_s5_Path(__file__).resolve().parent
    root = _rq1_s5_discover_root(here, args.project_root)
    kl07_path = _rq1_s5_find_script(root, here, '07_v2_hybrid_v3_representation_correct_kl.py')
    disc15_path = _rq1_s5_find_script(root, here, '15_discrete_baseline_representation_correct_kl.py')
    comparison18_path = _rq1_s5_find_script(root, here, '18_three_seed_model_comparison.py')
    hybrid17_path = _rq1_s5_find_script(root, here, '17_hybrid_v3_cd_v3_representation_correct_kl.py')
    kl07 = _rq1_inline_load(kl07_path, f'_common_fd_kl07_{_rq1_s5_os.getpid()}')
    disc15 = _rq1_inline_load(disc15_path, f'_common_fd_disc15_{_rq1_s5_os.getpid()}')
    comparison18 = _rq1_inline_load(comparison18_path, f'_common_fd_comparison18_{_rq1_s5_os.getpid()}')
    hybrid17 = _rq1_inline_load(hybrid17_path, f'_common_fd_hybrid17_{_rq1_s5_os.getpid()}')
    discrete_model_module = _rq1_inline_load(root / 'src' / 'hybrid_bn' / 'SLdiscrete.py', f'_common_fd_discrete_model_{_rq1_s5_os.getpid()}')
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 132)
    print('COMMON-RESOLUTION FULLY-DISCRETE KL')
    print('=' * 132)
    print(f'Generators:           {args.generators}')
    print(f'Seeds:                {args.seeds}')
    print(f'n:                    {args.sample_size}')
    print("Common bins:           discrete baseline's training-fitted all-node quantiles")
    print('Shared P term:         reused from existing discrete representation-correct KL')
    print(f'Continuous Q:          {args.continuous_particles} particles x {args.replicates} reps')
    print(f'Hybrid Q:              {args.hybrid_particles} particles x {args.replicates} reps')
    print('Structure learning:    NONE')
    print('=' * 132, flush=True)
    model_rows = []
    pair_rows = []
    diagnostics = []
    failures = []
    total_cells = len(args.generators) * len(args.seeds)
    counter = 0
    for generator_name in args.generators:
        for seed in args.seeds:
            counter += 1
            print(f'\n[{counter}/{total_cells}] {generator_name} seed={seed} n={args.sample_size}', flush=True)
            started = _rq1_s5_time.perf_counter()
            try:
                rows, pairs, diag = _rq1_s5_evaluate_cell(root=root, generator_name=generator_name, seed=seed, n=args.sample_size, args=args, kl07=kl07, disc15=disc15, comparison18=comparison18, hybrid17=hybrid17, discrete_model_module=discrete_model_module)
                model_rows.extend(rows)
                pair_rows.extend(pairs)
                diagnostics.append(diag)
                row_map = {r['Model']: r for r in rows}
                h = row_map['Hybrid_CDv3']
                d = row_map['Discrete']
                c = row_map['Continuous']
                hd_pair = next((p for p in pairs if p['Model_A'] == 'Hybrid_CDv3' and p['Model_B'] == 'Discrete'))
                print(f"    Common FD KL | D={d['Common_FD_KL']:.6f} | H={h['Common_FD_KL']:.6f} | C={c['Common_FD_KL']:.6f}", flush=True)
                print(f"    H-D paired diff={hd_pair['Common_KL_A_minus_B']:+.6f} ± {1.96 * hd_pair['Paired_SE']:.6f} | Hybrid better={hd_pair['A_Better']}", flush=True)
                print(f"    Hybrid native/common={h['Native_KL']:.6f}/{h['Common_FD_KL']:.6f} | p10 ESS={diag['hybrid_q_integration']['p10_ess_fraction']:.3f} | flags={diag['flags']}", flush=True)
                print(f'    elapsed={_rq1_s5_time.perf_counter() - started:.1f}s', flush=True)
                cache_path = output_dir / 'runs' / generator_name / f'seed_{seed:02d}' / f'n_{args.sample_size:04d}' / 'diagnostics.json'
                _rq1_s5_atomic_json(cache_path, diag)
            except Exception as exc:
                message = f'{type(exc).__name__}: {exc}'
                failures.append({'generator': generator_name, 'seed': seed, 'error': message})
                print(f'    FAILED: {message}', flush=True)
                _rq1_s5_traceback.print_exc()
    _rq1_s5_build_reports(output_dir, model_rows, pair_rows, diagnostics)
    if failures:
        _rq1_s5_atomic_json(output_dir / 'failures.json', {'failures': failures})
    print('\n' + '=' * 132)
    print('COMMON-RESOLUTION EVALUATION COMPLETE')
    print('=' * 132)
    print(f'Completed cells: {len(diagnostics)}/{total_cells}')
    print(f'Failed cells:    {len(failures)}')
    print(f"Results:  {output_dir / 'common_resolution_results.csv'}")
    print(f"Summary:  {output_dir / 'common_resolution_summary.csv'}")
    print(f"Pairwise: {output_dir / 'common_resolution_pairwise.csv'}")
    print(f"Report:   {output_dir / 'report.txt'}")
    print('=' * 132)
    return 1 if failures else 0


# ======================================================================
# Evaluation implementation: 21b_common_resolution_fully_discrete_kl_smc.py
# ======================================================================
import argparse as _rq1_s6_argparse
import hashlib as _rq1_s6_hashlib
import importlib.util as _rq1_s6_importlib
import json as _rq1_s6_json
import math as _rq1_s6_math
import os as _rq1_s6_os
import sys as _rq1_s6_sys
import time as _rq1_s6_time
import traceback as _rq1_s6_traceback
from pathlib import Path as _rq1_s6_Path
from typing import Any as _rq1_s6_Any, Mapping as _rq1_s6_Mapping, Sequence as _rq1_s6_Sequence
_rq1_s6_sys.dont_write_bytecode = True
import numpy as _rq1_s6_np
import pandas as _rq1_s6_pd
from scipy.special import logsumexp as _rq1_s6_logsumexp
from scipy.stats import norm as _rq1_s6_norm
_rq1_s6_BASE_SCRIPT = '21_common_resolution_fully_discrete_kl.py'
_rq1_s6_HYBRID17_SCRIPT = '17_hybrid_v3_cd_v3_representation_correct_kl.py'
_rq1_s6_SCHEMA_VERSION = 'common-resolution-fd-sequential-smc-v1'
_rq1_s6_MIN_PROB = 1e-300

def _rq1_s6_discover_root(here: _rq1_s6_Path, explicit: _rq1_s6_Path | None) -> _rq1_s6_Path:
    return _rq1_find_root()

def _rq1_s6_deterministic_seed(generator: str, seed: int, rep: int) -> int:
    token = f'common-fd-smc-v1|{generator}|{seed}|{rep}'.encode()
    return int.from_bytes(_rq1_s6_hashlib.sha256(token).digest()[:4], 'little')

def _rq1_s6_normal_interval_prob(lower, upper, mean, sd):
    lower, upper, mean, sd = _rq1_s6_np.broadcast_arrays(_rq1_s6_np.asarray(lower, float), _rq1_s6_np.asarray(upper, float), _rq1_s6_np.asarray(mean, float), _rq1_s6_np.asarray(sd, float))
    c = _rq1_s6_norm.cdf(upper, loc=mean, scale=sd) - _rq1_s6_norm.cdf(lower, loc=mean, scale=sd)
    s = _rq1_s6_norm.sf(lower, loc=mean, scale=sd) - _rq1_s6_norm.sf(upper, loc=mean, scale=sd)
    return _rq1_s6_np.clip(_rq1_s6_np.maximum(c, s), _rq1_s6_MIN_PROB, 1.0)

def _rq1_s6_truncated_normal(lower, upper, mean, sd, rng):
    lower, upper, mean, sd = _rq1_s6_np.broadcast_arrays(_rq1_s6_np.asarray(lower, float), _rq1_s6_np.asarray(upper, float), _rq1_s6_np.asarray(mean, float), _rq1_s6_np.asarray(sd, float))
    fl = _rq1_s6_norm.cdf(lower, loc=mean, scale=sd)
    fu = _rq1_s6_norm.cdf(upper, loc=mean, scale=sd)
    sl = _rq1_s6_norm.sf(lower, loc=mean, scale=sd)
    su = _rq1_s6_norm.sf(upper, loc=mean, scale=sd)
    u = rng.random(lower.shape)
    use_sf = fu > 0.9
    out = _rq1_s6_np.empty(lower.shape, dtype=float)
    m = ~use_sf
    if _rq1_s6_np.any(m):
        q = fl[m] + u[m] * _rq1_s6_np.maximum(fu[m] - fl[m], 0.0)
        q = _rq1_s6_np.clip(q, _rq1_s6_np.finfo(float).tiny, 1.0 - _rq1_s6_np.finfo(float).eps)
        out[m] = _rq1_s6_norm.ppf(q, loc=mean[m], scale=sd[m])
    if _rq1_s6_np.any(use_sf):
        q = su[use_sf] + u[use_sf] * _rq1_s6_np.maximum(sl[use_sf] - su[use_sf], 0.0)
        q = _rq1_s6_np.clip(q, _rq1_s6_np.finfo(float).tiny, 1.0 - _rq1_s6_np.finfo(float).eps)
        out[use_sf] = _rq1_s6_norm.isf(q, loc=mean[use_sf], scale=sd[use_sf])
    return out

def _rq1_s6_interval_bounds(transform, node: str, states: _rq1_s6_np.ndarray):
    edges = _rq1_s6_np.asarray(transform.bin_edges[node], dtype=float)
    s = _rq1_s6_np.asarray(states, dtype=_rq1_s6_np.int64)
    return (edges[s], edges[s + 1])

def _rq1_s6_resample_indices(log_weights: _rq1_s6_np.ndarray, rng: _rq1_s6_np.random.Generator):
    """Multinomial resampling row by row."""
    lw = _rq1_s6_np.asarray(log_weights, float)
    lw = lw - _rq1_s6_logsumexp(lw, axis=1, keepdims=True)
    w = _rq1_s6_np.exp(lw)
    b, n = w.shape
    idx = _rq1_s6_np.empty((b, n), dtype=_rq1_s6_np.int64)
    for i in range(b):
        idx[i] = rng.choice(n, size=n, replace=True, p=w[i])
    return (idx, w)

def _rq1_s6_gather_particles(array: _rq1_s6_np.ndarray, indices: _rq1_s6_np.ndarray):
    return _rq1_s6_np.take_along_axis(array, indices, axis=1)

class _rq1_s6_SequentialHybridPushforward:

    def __init__(self, *, base, hybrid17, q, raw_training: pd.DataFrame, common_transform, native_transform, continuous_nodes: Sequence[str], discrete_nodes: Sequence[str], nodes: Sequence[str], generator: str, seed: int, outer_particles: int, cc_inner_samples: int, replicates: int, batch_size: int):
        self.base = base
        self.hybrid17 = hybrid17
        self.q = q
        self.training = raw_training.reset_index(drop=True)
        self.common = common_transform
        self.native = native_transform
        self.continuous_nodes = list(map(str, continuous_nodes))
        self.discrete_nodes = list(map(str, discrete_nodes))
        self.cset = set(self.continuous_nodes)
        self.dset = set(self.discrete_nodes)
        self.nodes = list(map(str, nodes))
        self.generator = str(generator)
        self.seed = int(seed)
        self.np = int(outer_particles)
        self.inner = int(cc_inner_samples)
        self.replicates = int(replicates)
        self.batch_size = int(batch_size)
        self.order = [str(x) for x in __import__('networkx').topological_sort(q.graph)]
        for node in self.discrete_nodes:
            a = _rq1_s6_np.asarray(self.common.bin_edges[node], float)
            b = _rq1_s6_np.asarray(self.native.bin_edges[node], float)
            if not _rq1_s6_np.array_equal(a, b):
                raise ValueError(f'Common/native Hybrid bins differ for discrete node {node!r}.')
        native_train_states = self.native.transform(self.training.copy())
        self.d_reps = {}
        for node in self.discrete_nodes:
            k = int(self.native.cardinalities[node])
            vals = self.training[node].to_numpy(float)
            st = native_train_states[node].to_numpy(_rq1_s6_np.int64)
            reps = _rq1_s6_np.empty(k, float)
            for s in range(k):
                x = vals[st == s]
                if len(x):
                    reps[s] = float(_rq1_s6_np.median(x))
                else:
                    lo = float(self.native.bin_edges[node][s])
                    hi = float(self.native.bin_edges[node][s + 1])
                    scale = max(float(_rq1_s6_np.std(vals, ddof=0)), 1e-06)
                    if _rq1_s6_np.isfinite(lo) and _rq1_s6_np.isfinite(hi):
                        reps[s] = 0.5 * (lo + hi)
                    elif _rq1_s6_np.isfinite(lo):
                        reps[s] = lo + scale
                    elif _rq1_s6_np.isfinite(hi):
                        reps[s] = hi - scale
                    else:
                        reps[s] = float(_rq1_s6_np.median(vals))
            self.d_reps[node] = reps
        self.defaults = {node: float(_rq1_s6_np.median(self.training[node].to_numpy(float))) for node in self.nodes}
        self.cc_proposal = {}
        for child in self.continuous_nodes:
            model = self.q.models[child]
            if isinstance(model, _rq1_s6_Mapping) and model.get('kind') == 'corrected_v3_cc':
                parents = list(model['parents'])
                y = self.training[child].to_numpy(float)
                if parents:
                    X = _rq1_s6_np.column_stack([_rq1_s6_np.ones(len(self.training)), self.training[parents].to_numpy(float)])
                else:
                    X = _rq1_s6_np.ones((len(self.training), 1), float)
                beta, *_ = _rq1_s6_np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                sd = max(float(_rq1_s6_np.sqrt(_rq1_s6_np.mean(resid ** 2))), 0.0001)
                self.cc_proposal[child] = (tuple(parents), beta, sd)

    def _frame(self, raw: Mapping[str, _rq1_s6_np.ndarray], target_states: _rq1_s6_pd.DataFrame, *, extra_child: str | None=None, extra_values: _rq1_s6_np.ndarray | None=None, repeat_inner: int=1) -> _rq1_s6_pd.DataFrame:
        b = len(target_states)
        n = self.np
        mult = int(repeat_inner)
        total = b * n * mult
        data = {node: _rq1_s6_np.full(total, self.defaults[node], dtype=float) for node in self.nodes}
        for node in self.discrete_nodes:
            s = target_states[node].to_numpy(_rq1_s6_np.int64)
            vals = self.d_reps[node][s]
            data[node] = _rq1_s6_np.repeat(vals, n * mult)
        for node, arr in raw.items():
            a = _rq1_s6_np.asarray(arr, float)
            if a.shape != (b, n):
                raise ValueError(f'bad particle shape for {node}: {a.shape}')
            data[node] = _rq1_s6_np.repeat(a[:, :, None], mult, axis=2).reshape(-1)
        if extra_child is not None:
            v = _rq1_s6_np.asarray(extra_values, float)
            if v.shape != (b, n, mult):
                raise ValueError(f'bad extra child shape {extra_child}: {v.shape}')
            data[extra_child] = v.reshape(-1)
        return _rq1_s6_pd.DataFrame(data, columns=self.nodes)

    def _d_log_probability(self, child: str, raw: _rq1_s6_Mapping[str, _rq1_s6_np.ndarray], target_states: pd.DataFrame) -> _rq1_s6_np.ndarray:
        model = self.q.models[child]
        b = len(target_states)
        if hasattr(model, 'marginal_probabilities') and hasattr(model, 'tables'):
            child_state = target_states[child].to_numpy(_rq1_s6_np.int64)
            parents = list(model.parents)
            out = _rq1_s6_np.empty(b, float)
            if not parents:
                probs = _rq1_s6_np.asarray(model.marginal_probabilities, float)
                out = _rq1_s6_np.log(_rq1_s6_np.clip(probs[child_state], _rq1_s6_MIN_PROB, 1.0))
            else:
                pm = target_states[parents].to_numpy(_rq1_s6_np.int64)
                for i in range(b):
                    key = tuple((int(x) for x in pm[i]))
                    probs = model.tables.get(key, model.marginal_probabilities)
                    out[i] = _rq1_s6_math.log(max(float(probs[int(child_state[i])]), _rq1_s6_MIN_PROB))
            return _rq1_s6_np.repeat(out[:, None], self.np, axis=1)
        if isinstance(model, _rq1_s6_Mapping) and model.get('kind') == 'cd_v3_bma':
            frame = self._frame(raw, target_states)
            model_data = self.q.transformer.transform(frame)
            eval_data = self.q.cd_v3_engine.transform_new_data(model_data.copy())
            lp = _rq1_s6_np.asarray(model['predictive'].log_prob(eval_data), float)
            if lp.shape != (b * self.np,):
                raise ValueError('CD local predictive returned wrong shape.')
            return lp.reshape(b, self.np)
        if isinstance(model, _rq1_s6_Mapping) and model.get('kind') == 'regime_logistic':
            frame = self._frame(raw, target_states)
            model_data = self.q.transformer.transform(frame)
            lp = _rq1_s6_np.asarray(self.q._eval_regime_logistic(model, model_data), float)
            return lp.reshape(b, self.np)
        raise TypeError(f'Unsupported discrete Hybrid local model for {child}: {type(model)}')

    def _gaussian_dc(self, model, lower: _rq1_s6_np.ndarray, upper: _rq1_s6_np.ndarray, raw: Mapping[str, _rq1_s6_np.ndarray], rng: _rq1_s6_np.random.Generator):
        b, n = lower.shape
        if model.parents:
            mean = _rq1_s6_np.full((b, n), float(model.coefficients[0]), float)
            for j, p in enumerate(model.parents, start=1):
                mean += float(model.coefficients[j]) * raw[p]
        else:
            mean = _rq1_s6_np.full((b, n), float(model.coefficients[0]), float)
        sd = float(model.standard_deviation)
        pbin = _rq1_s6_normal_interval_prob(lower, upper, mean, sd)
        sampled = _rq1_s6_truncated_normal(lower, upper, mean, sd, rng)
        return (_rq1_s6_np.log(pbin), sampled)

    def _mixture_dc(self, model, lower: _rq1_s6_np.ndarray, upper: _rq1_s6_np.ndarray, raw: Mapping[str, _rq1_s6_np.ndarray], rng: _rq1_s6_np.random.Generator):
        b, n = lower.shape
        if model.parents:
            cols = _rq1_s6_np.stack([raw[p] for p in model.parents], axis=-1)
            feat = (cols - _rq1_s6_np.asarray(model.feature_means)[None, None, :]) / _rq1_s6_np.asarray(model.feature_stds)[None, None, :]
            X = _rq1_s6_np.concatenate([_rq1_s6_np.ones((b, n, 1)), feat], axis=2)
        else:
            X = _rq1_s6_np.ones((b, n, 1), float)
        logits = _rq1_s6_np.einsum('bnp,kp->bnk', X, _rq1_s6_np.asarray(model.gate_coefficients))
        logits -= _rq1_s6_logsumexp(logits, axis=2, keepdims=True)
        gate = _rq1_s6_np.exp(logits)
        mean_std = _rq1_s6_np.einsum('bnp,kp->bnk', X, _rq1_s6_np.asarray(model.betas))
        means = float(model.child_mean) + float(model.child_std) * mean_std
        sds = float(model.child_std) * _rq1_s6_np.sqrt(_rq1_s6_np.asarray(model.variances))[None, None, :]
        lo = lower[:, :, None]
        hi = upper[:, :, None]
        comp_prob = _rq1_s6_normal_interval_prob(lo, hi, means, sds)
        joint = gate * comp_prob
        pbin = _rq1_s6_np.clip(_rq1_s6_np.sum(joint, axis=2), _rq1_s6_MIN_PROB, 1.0)
        post = joint / pbin[:, :, None]
        u = rng.random((b, n))
        cs = _rq1_s6_np.cumsum(post, axis=2)
        comp = _rq1_s6_np.sum(u[:, :, None] > cs, axis=2)
        comp = _rq1_s6_np.minimum(comp, model.n_components - 1)
        mean_sel = _rq1_s6_np.take_along_axis(means, comp[:, :, None], axis=2)[:, :, 0]
        sd_full = _rq1_s6_np.broadcast_to(sds, means.shape)
        sd_sel = _rq1_s6_np.take_along_axis(sd_full, comp[:, :, None], axis=2)[:, :, 0]
        sampled = _rq1_s6_truncated_normal(lower, upper, mean_sel, sd_sel, rng)
        return (_rq1_s6_np.log(pbin), sampled)

    def _dc_step(self, child: str, raw: Mapping[str, _rq1_s6_np.ndarray], target_states: pd.DataFrame, rng: _rq1_s6_np.random.Generator):
        model = self.q.models[child]
        dparents = list(model['discrete_parents'])
        b = len(target_states)
        states = target_states[child].to_numpy(_rq1_s6_np.int64)
        lo1, hi1 = _rq1_s6_interval_bounds(self.common, child, states)
        lower = _rq1_s6_np.repeat(lo1[:, None], self.np, axis=1)
        upper = _rq1_s6_np.repeat(hi1[:, None], self.np, axis=1)
        logp = _rq1_s6_np.empty((b, self.np), float)
        proposed = _rq1_s6_np.empty((b, self.np), float)
        for i in range(b):
            key = tuple((int(target_states.iloc[i][p]) for p in dparents))
            local = model['regimes'].get(key, model['fallback'])
            local_raw = {p: arr[i:i + 1] for p, arr in raw.items()}
            if hasattr(local, 'coefficients') and hasattr(local, 'standard_deviation'):
                lp, y = self._gaussian_dc(local, lower[i:i + 1], upper[i:i + 1], local_raw, rng)
            elif hasattr(local, 'gate_coefficients') and hasattr(local, 'betas'):
                lp, y = self._mixture_dc(local, lower[i:i + 1], upper[i:i + 1], local_raw, rng)
            else:
                raise TypeError(f'Unsupported DC local model {type(local)}')
            logp[i] = lp[0]
            proposed[i] = y[0]
        return (logp, proposed, {'kind': 'DC_exact'})

    def _cc_step(self, child: str, raw: Mapping[str, _rq1_s6_np.ndarray], target_states: pd.DataFrame, rng: _rq1_s6_np.random.Generator):
        b = len(target_states)
        n = self.np
        m = self.inner
        parents, beta, sd = self.cc_proposal[child]
        mean = _rq1_s6_np.full((b, n), float(beta[0]), float)
        for j, p in enumerate(parents, start=1):
            mean += float(beta[j]) * raw[p]
        states = target_states[child].to_numpy(_rq1_s6_np.int64)
        lo1, hi1 = _rq1_s6_interval_bounds(self.common, child, states)
        lower = _rq1_s6_np.repeat(lo1[:, None], n, axis=1)
        upper = _rq1_s6_np.repeat(hi1[:, None], n, axis=1)
        r_bin = _rq1_s6_normal_interval_prob(lower, upper, mean, sd)
        lower3 = _rq1_s6_np.repeat(lower[:, :, None], m, axis=2)
        upper3 = _rq1_s6_np.repeat(upper[:, :, None], m, axis=2)
        mean3 = _rq1_s6_np.repeat(mean[:, :, None], m, axis=2)
        cand = _rq1_s6_truncated_normal(lower3, upper3, mean3, sd, rng)
        frame = self._frame(raw, target_states, extra_child=child, extra_values=cand, repeat_inner=m)
        cc_eval = frame[self.continuous_nodes]
        qlog = _rq1_s6_np.asarray(self.q.cc_engine.local_predictive_logpdf(child, tuple(parents), cc_eval), float).reshape(b, n, m)
        log_r_trunc = _rq1_s6_norm.logpdf(cand, loc=mean3, scale=sd) - _rq1_s6_np.log(r_bin)[:, :, None]
        logiw = qlog - log_r_trunc
        logz = _rq1_s6_logsumexp(logiw, axis=2) - _rq1_s6_math.log(m)
        lw = logiw - _rq1_s6_logsumexp(logiw, axis=2, keepdims=True)
        w = _rq1_s6_np.exp(lw)
        u = rng.random((b, n))
        cs = _rq1_s6_np.cumsum(w, axis=2)
        which = _rq1_s6_np.sum(u[:, :, None] > cs, axis=2)
        which = _rq1_s6_np.minimum(which, m - 1)
        proposed = _rq1_s6_np.take_along_axis(cand, which[:, :, None], axis=2)[:, :, 0]
        ess = 1.0 / _rq1_s6_np.sum(w ** 2, axis=2)
        ess_frac = _rq1_s6_np.clip(ess / m, 0.0, 1.0)
        return (logz, proposed, {'kind': 'CC_inner_importance', 'median_inner_ess_fraction': float(_rq1_s6_np.median(ess_frac)), 'p10_inner_ess_fraction': float(_rq1_s6_np.quantile(ess_frac, 0.1))})

    def _one_replicate(self, states: pd.DataFrame, rng):
        nrows = len(states)
        result = _rq1_s6_np.empty(nrows, float)
        step_ess = []
        cc_ess = []
        for start in range(0, nrows, self.batch_size):
            stop = min(nrows, start + self.batch_size)
            target = states.iloc[start:stop].reset_index(drop=True)
            b = len(target)
            raw = {}
            log_mass = _rq1_s6_np.zeros(b, float)
            for child in self.order:
                if child in self.dset:
                    log_inc = self._d_log_probability(child, raw, target)
                    proposed = None
                else:
                    model = self.q.models[child]
                    if isinstance(model, _rq1_s6_Mapping) and model.get('kind') == 'corrected_v3_cc':
                        log_inc, proposed, diag = self._cc_step(child, raw, target, rng)
                        cc_ess.append(diag['p10_inner_ess_fraction'])
                    elif isinstance(model, _rq1_s6_Mapping) and model.get('kind') == 'regime_v2_continuous':
                        log_inc, proposed, diag = self._dc_step(child, raw, target, rng)
                    else:
                        raise TypeError(f'Unsupported continuous Hybrid model for {child}: {model}')
                log_mass += _rq1_s6_logsumexp(log_inc, axis=1) - _rq1_s6_math.log(self.np)
                idx, weights = _rq1_s6_resample_indices(log_inc, rng)
                ess = 1.0 / _rq1_s6_np.sum(weights ** 2, axis=1)
                step_ess.extend((ess / self.np).tolist())
                for node in raw:
                    raw[node] = _rq1_s6_gather_particles(raw[node], idx)
                if proposed is not None:
                    raw[child] = _rq1_s6_gather_particles(proposed, idx)
            result[start:stop] = log_mass
        return (result, {'median_step_ess_fraction': float(_rq1_s6_np.median(step_ess)), 'p10_step_ess_fraction': float(_rq1_s6_np.quantile(step_ess, 0.1)), 'min_step_ess_fraction': float(_rq1_s6_np.min(step_ess)), 'p10_cc_inner_ess_fraction': float(_rq1_s6_np.quantile(cc_ess, 0.1)) if cc_ess else float('nan')})

    def logmass(self, states: pd.DataFrame):
        reps = []
        rep_diag = []
        for r in range(self.replicates):
            rng = _rq1_s6_np.random.default_rng(_rq1_s6_deterministic_seed(self.generator, self.seed, r))
            x, d = self._one_replicate(states, rng)
            reps.append(x)
            rep_diag.append(d)
        mat = _rq1_s6_np.vstack(reps)
        combined = _rq1_s6_logsumexp(mat, axis=0) - _rq1_s6_math.log(len(reps))
        mean_logs = _rq1_s6_np.mean(mat, axis=1)
        return (combined, {'method': 'learned-DAG sequential SMC with local bin conditioning', 'outer_particles': self.np, 'cc_inner_samples': self.inner, 'replicates': self.replicates, 'replicate_mean_logq': mean_logs.tolist(), 'replicate_mean_logq_sd': float(_rq1_s6_np.std(mean_logs, ddof=1)) if len(mean_logs) > 1 else float('nan'), 'median_step_ess_fraction': float(_rq1_s6_np.median([d['median_step_ess_fraction'] for d in rep_diag])), 'p10_step_ess_fraction': float(_rq1_s6_np.min([d['p10_step_ess_fraction'] for d in rep_diag])), 'min_step_ess_fraction': float(_rq1_s6_np.min([d['min_step_ess_fraction'] for d in rep_diag])), 'p10_cc_inner_ess_fraction': float(_rq1_s6_np.nanmin([d['p10_cc_inner_ess_fraction'] for d in rep_diag]))})

def _rq1_s6_parse_args():
    p = _rq1_s6_argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=_rq1_s6_Path, default=None)
    p.add_argument('--generators', nargs='+', choices=('unimodal', 'skew', 'multimodal', 'skew_multi'), default=['multimodal', 'skew_multi'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--sample-size', type=int, default=5000)
    p.add_argument('--comparison-rows', type=int, default=1000)
    p.add_argument('--outer-particles', type=int, default=128)
    p.add_argument('--cc-inner-samples', type=int, default=16)
    p.add_argument('--replicates', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--cpt-alpha', type=float, default=1.0)
    p.add_argument('--min-regime-size', type=int, default=5)
    p.add_argument('--logistic-c', type=float, default=1000.0)
    p.add_argument('--metadata-score-tolerance', type=float, default=1e-06)
    p.add_argument('--dpi-tolerance', type=float, default=0.03)
    p.add_argument('--output-dir', type=_rq1_s6_Path, default=_rq1_s6_Path('Outputs/CommonResolution_FullyDiscreteKL_SMC'))
    return p.parse_args()

def _rq1_s6_main():
    args = _rq1_s6_parse_args()
    here = _rq1_s6_Path(__file__).resolve().parent
    root = _rq1_s6_discover_root(here, args.project_root)
    base = _rq1_inline_load(root / 'experiments' / 'rq1' / _rq1_s6_BASE_SCRIPT, f'_common_fd_base_{_rq1_s6_os.getpid()}')
    hybrid17 = _rq1_inline_load(root / 'experiments' / 'rq1' / _rq1_s6_HYBRID17_SCRIPT, f'_common_fd_h17_{_rq1_s6_os.getpid()}')
    discrete_model = _rq1_inline_load(root / 'src' / 'hybrid_bn' / 'SLdiscrete.py', f'_common_fd_discmodel_{_rq1_s6_os.getpid()}')
    disc15 = _rq1_inline_load(root / 'experiments' / 'rq1' / '15_discrete_baseline_representation_correct_kl.py', f'_common_fd_disc15_{_rq1_s6_os.getpid()}')
    comparison18 = _rq1_inline_load(root / 'experiments' / 'rq1' / '18_three_seed_model_comparison.py', f'_common_fd_18_{_rq1_s6_os.getpid()}')
    kl07 = _rq1_inline_load(root / 'experiments' / 'rq1' / '07_v2_hybrid_v3_representation_correct_kl.py', f'_common_fd_07_{_rq1_s6_os.getpid()}')
    output = args.output_dir
    if not output.is_absolute():
        output = (root / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    pair_rows = []
    diagnostics = []
    failures = []
    print('=' * 126)
    print('COMMON-RESOLUTION FULLY-DISCRETE KL — SEQUENTIAL SMC')
    print('=' * 126)
    print(f'Generators:        {args.generators}')
    print(f'Seeds:             {args.seeds}')
    print(f'n:                 {args.sample_size}')
    print(f'Comparison rows:   {args.comparison_rows}')
    print(f'Hybrid outer SMC:  {args.outer_particles} x {args.replicates}')
    print(f'CC inner samples:  {args.cc_inner_samples}')
    print('P recomputation:   NONE')
    print('Structure learning:NONE')
    print('=' * 126)
    cell_no = 0
    total = len(args.generators) * len(args.seeds)
    for gen_name in args.generators:
        for seed in args.seeds:
            cell_no += 1
            t0 = _rq1_s6_time.perf_counter()
            print(f'\n[{cell_no}/{total}] {gen_name} seed={seed} n={args.sample_size}', flush=True)
            try:
                paths = base.existing_paths(root, gen_name, seed, args.sample_size)
                base.require_paths(paths)
                ds = _rq1_s6_json.loads(paths['discrete_structure'].read_text(encoding='utf-8'))
                dk = _rq1_s6_json.loads(paths['discrete_kl'].read_text(encoding='utf-8'))
                ck = _rq1_s6_json.loads(paths['continuous_kl'].read_text(encoding='utf-8'))
                hs = _rq1_s6_json.loads(paths['hybrid_structure'].read_text(encoding='utf-8'))
                hk = _rq1_s6_json.loads(paths['hybrid_kl'].read_text(encoding='utf-8'))
                Generator = base.import_generator(root, gen_name)
                generator = Generator(num_samples=args.sample_size, seed=seed)
                generator.generate_dag()
                raw_train = generator.generate_data()
                payload = generator.get_payload()
                nodes = list(map(str, payload['nodes']))
                raw_train = raw_train[nodes].copy().reset_index(drop=True)
                common = kl07.SavedQuantileTransform.from_metadata(ds['discretizer_metadata'])
                cards = discrete_model.default_cardinalities(nodes)
                refit = discrete_model.QuantileDiscretizer(cards)
                refit.fit_transform(raw_train.copy())
                for node in nodes:
                    if not _rq1_s6_np.array_equal(_rq1_s6_np.asarray(refit.bin_edges[node], float), _rq1_s6_np.asarray(common.bin_edges[node], float)):
                        raise AssertionError(f'common-bin parity failed for {node}')
                mc_seed = int(dk['kl']['mc_seed'])
                true_model = kl07.TrueContinuousDAG(payload['graph'], payload['true_parameters'], nodes)
                full_fresh = true_model.sample(int(dk['kl']['mc_samples']), mc_seed)[nodes].reset_index(drop=True)
                m = min(args.comparison_rows, len(full_fresh))
                fresh = full_fresh.iloc[:m].copy().reset_index(drop=True)
                states = common.transform(fresh.copy()).reset_index(drop=True)
                dgraph = base.graph_from_edges(nodes, ds['learned_edges'])
                dtrain = common.transform(raw_train.copy())
                qd = disc15.SmoothedDiscreteDAG(dgraph, nodes, cards, alpha=args.cpt_alpha).fit(dtrain)
                logqd, _ = qd.joint_logscore(fresh, common)
                cgraph = base.graph_from_edges(nodes, ck['learned_edges'])
                qc = comparison18.LearnedGaussianDAG(cgraph, nodes).fit(raw_train)
                cest = base.ContinuousGaussianBinMassEstimator(graph=cgraph, models=qc.models, nodes=nodes, transform=common, particles=args.outer_particles, replicates=args.replicates, batch_size=args.batch_size, generator=gen_name, seed=seed)
                logqc, cdiag = cest.logmass(states)
                qh, native_transform, _, dnodes, cnodes = base.fit_hybrid_q(root=root, hybrid_module=hybrid17, structure_cell=hs, structure_path=paths['hybrid_structure'], raw_training=raw_train, payload=payload, metadata_score_tolerance=args.metadata_score_tolerance, cpt_alpha=args.cpt_alpha, min_regime_size=args.min_regime_size, logistic_c=args.logistic_c, seed=seed, n=args.sample_size)
                hest = _rq1_s6_SequentialHybridPushforward(base=base, hybrid17=hybrid17, q=qh, raw_training=raw_train, common_transform=common, native_transform=native_transform, continuous_nodes=cnodes, discrete_nodes=dnodes, nodes=nodes, generator=gen_name, seed=seed, outer_particles=args.outer_particles, cc_inner_samples=args.cc_inner_samples, replicates=args.replicates, batch_size=args.batch_size)
                logqh, hdiag = hest.logmass(states)
                discrete_kl = float(dk['kl']['estimate'])
                native_h = float(hk['kl']['estimate'])
                native_c = float(ck['kl']['estimate'])
                hd = _rq1_s6_np.asarray(logqd) - _rq1_s6_np.asarray(logqh)
                cd = _rq1_s6_np.asarray(logqd) - _rq1_s6_np.asarray(logqc)
                hdiff = float(_rq1_s6_np.mean(hd))
                hse = float(_rq1_s6_np.std(hd, ddof=1) / _rq1_s6_math.sqrt(len(hd)))
                cdiff = float(_rq1_s6_np.mean(cd))
                cse = float(_rq1_s6_np.std(cd, ddof=1) / _rq1_s6_math.sqrt(len(cd)))
                common_h = discrete_kl + hdiff
                common_c = discrete_kl + cdiff
                common_d = discrete_kl
                dpi_excess = common_h - native_h
                dpi_threshold = max(args.dpi_tolerance, 2.0 * 1.96 * hse)
                flags = []
                if dpi_excess > dpi_threshold:
                    flags.append('DATA_PROCESSING_VIOLATION')
                if hdiag['p10_step_ess_fraction'] < 0.05:
                    flags.append('LOW_OUTER_STEP_ESS')
                inner_ess = hdiag['p10_cc_inner_ess_fraction']
                if _rq1_s6_np.isfinite(inner_ess) and inner_ess < 0.05:
                    flags.append('LOW_CC_INNER_ESS')
                rep_sd = hdiag['replicate_mean_logq_sd']
                if _rq1_s6_np.isfinite(rep_sd) and rep_sd > 0.03:
                    flags.append('HYBRID_REPLICATE_DRIFT')
                rows.extend([{'Model': 'Discrete', 'Generator': gen_name, 'Seed': seed, 'Sample_Size': args.sample_size, 'Comparison_Rows': m, 'Common_FD_KL': common_d, 'Native_KL': discrete_kl, 'Native_Minus_Common': 0.0, 'SHD': int(dk['structure_metrics']['shd']), 'F1': float(dk['structure_metrics']['directed_f1'])}, {'Model': 'Hybrid_CDv3', 'Generator': gen_name, 'Seed': seed, 'Sample_Size': args.sample_size, 'Comparison_Rows': m, 'Common_FD_KL': common_h, 'Native_KL': native_h, 'Native_Minus_Common': native_h - common_h, 'SHD': int(hk['structure_metrics']['shd']), 'F1': float(hk['structure_metrics']['directed_f1'])}, {'Model': 'Continuous', 'Generator': gen_name, 'Seed': seed, 'Sample_Size': args.sample_size, 'Comparison_Rows': m, 'Common_FD_KL': common_c, 'Native_KL': native_c, 'Native_Minus_Common': native_c - common_c, 'SHD': int(ck['structure_metrics']['shd']), 'F1': float(ck['structure_metrics']['directed_f1'])}])
                pair_rows.extend([{'Generator': gen_name, 'Seed': seed, 'Model_A': 'Hybrid_CDv3', 'Model_B': 'Discrete', 'Common_KL_A_minus_B': hdiff, 'Paired_SE': hse, 'Lower95': hdiff - 1.96 * hse, 'Upper95': hdiff + 1.96 * hse, 'A_Better': hdiff < 0}, {'Generator': gen_name, 'Seed': seed, 'Model_A': 'Continuous', 'Model_B': 'Discrete', 'Common_KL_A_minus_B': cdiff, 'Paired_SE': cse, 'Lower95': cdiff - 1.96 * cse, 'Upper95': cdiff + 1.96 * cse, 'A_Better': cdiff < 0}])
                diagnostics.append({'generator': gen_name, 'seed': seed, 'common_discrete_kl': common_d, 'common_hybrid_kl': common_h, 'native_hybrid_kl': native_h, 'hybrid_dpi_excess_common_minus_native': dpi_excess, 'hybrid_minus_discrete': hdiff, 'hybrid_minus_discrete_paired_se': hse, 'hybrid_smc': hdiag, 'continuous_smc': cdiag, 'flags': flags})
                print(f'    Common FD KL | D={common_d:.6f} | H={common_h:.6f} | C={common_c:.6f}')
                print(f'    H-D={hdiff:+.6f} ± {1.96 * hse:.6f} | H native={native_h:.6f} | native-common={native_h - common_h:+.6f}')
                print(f"    SMC p10 step ESS={hdiag['p10_step_ess_fraction']:.3f} | CC inner p10 ESS={hdiag['p10_cc_inner_ess_fraction']:.3f} | rep mean-logQ SD={hdiag['replicate_mean_logq_sd']:.4f} | flags={flags}")
                print(f'    elapsed={_rq1_s6_time.perf_counter() - t0:.1f}s', flush=True)
            except Exception as exc:
                failures.append({'generator': gen_name, 'seed': seed, 'error': f'{type(exc).__name__}: {exc}'})
                print(f'    FAILED: {type(exc).__name__}: {exc}', flush=True)
                _rq1_s6_traceback.print_exc()
    rdf = _rq1_s6_pd.DataFrame(rows)
    pdf = _rq1_s6_pd.DataFrame(pair_rows)
    rdf.to_csv(output / 'common_resolution_smc_results.csv', index=False)
    pdf.to_csv(output / 'common_resolution_smc_pairwise.csv', index=False)
    if len(rdf):
        summary = rdf.groupby(['Generator', 'Model'], as_index=False).agg(Runs=('Common_FD_KL', 'size'), Mean_Common_FD_KL=('Common_FD_KL', 'mean'), SD_Common_FD_KL=('Common_FD_KL', 'std'), Mean_Native_KL=('Native_KL', 'mean'), Mean_Native_Minus_Common=('Native_Minus_Common', 'mean'), Mean_SHD=('SHD', 'mean'), Mean_F1=('F1', 'mean'))
    else:
        summary = _rq1_s6_pd.DataFrame()
    summary.to_csv(output / 'common_resolution_smc_summary.csv', index=False)
    (output / 'diagnostics.json').write_text(_rq1_s6_json.dumps({'cells': diagnostics, 'failures': failures}, indent=2), encoding='utf-8')
    report = ['COMMON-RESOLUTION FULLY-DISCRETE KL — SEQUENTIAL SMC', '=' * 120, '', rdf.to_string(index=False) if len(rdf) else '(no results)', '', 'PAIRWISE', '-' * 120, pdf.to_string(index=False) if len(pdf) else '(none)', '', 'SUMMARY', '-' * 120, summary.to_string(index=False) if len(summary) else '(none)', '', 'FLAGS', '-' * 120]
    any_flags = False
    for d in diagnostics:
        if d['flags']:
            any_flags = True
            report.append(f"{d['generator']} seed={d['seed']}: {', '.join(d['flags'])}")
    if not any_flags:
        report.append('none')
    (output / 'report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n' + '=' * 126)
    print('SEQUENTIAL COMMON-RESOLUTION EVALUATION COMPLETE')
    print('=' * 126)
    print(f'Completed: {len(diagnostics)}/{total}')
    print(f'Failed:    {len(failures)}')
    print(f"Results:   {output / 'common_resolution_smc_results.csv'}")
    print(f"Pairwise:  {output / 'common_resolution_smc_pairwise.csv'}")
    print(f"Summary:   {output / 'common_resolution_smc_summary.csv'}")
    print(f"Report:    {output / 'report.txt'}")
    print('=' * 126)
    return 1 if failures else 0


# ======================================================================
# Evaluation implementation: 01_rq1_5_native_mixed_probe.py
# ======================================================================
import argparse as _rq1_s7_argparse
import contextlib as _rq1_s7_contextlib
import hashlib as _rq1_s7_hashlib
import importlib as _rq1_s7_importlib
import importlib.util as _rq1_s7_importlib
import json as _rq1_s7_json
import math as _rq1_s7_math
import os as _rq1_s7_os
import sys as _rq1_s7_sys
import time as _rq1_s7_time
import traceback as _rq1_s7_traceback
from pathlib import Path as _rq1_s7_Path
from typing import Any as _rq1_s7_Any, Iterable as _rq1_s7_Iterable, Mapping as _rq1_s7_Mapping, Sequence as _rq1_s7_Sequence
_rq1_s7_sys.dont_write_bytecode = True
import networkx as _rq1_s7_nx
import numpy as _rq1_s7_np
import pandas as _rq1_s7_pd
from scipy.special import logsumexp as _rq1_s7_logsumexp
_rq1_s7_SCHEMA_VERSION = 'rq1.5-native-mixed-probe-v1'
_rq1_s7_GENERATOR_SPECS = {'unimodal': ('generateMixedUnimodalGT.py', 'NativeMixedUnimodalGenerator'), 'skew': ('generateMixedSkewGT.py', 'NativeMixedSkewedGenerator'), 'multimodal': ('generateMixedMultimodalGT.py', 'NativeMixedMultimodalGenerator'), 'skew_multi': ('generateMixedSkewMultiGT.py', 'NativeMixedSkewedMultimodalGenerator')}
_rq1_s7_DEFAULT_GENERATORS = list(_rq1_s7_GENERATOR_SPECS)
_rq1_s7_DEFAULT_SEEDS = [0, 1, 2]
_rq1_s7_SCRIPT_15 = '15_discrete_baseline_representation_correct_kl.py'
_rq1_s7_SCRIPT_16 = '16_hybrid_v3_cd_v3_structure_learning.py'
_rq1_s7_SCRIPT_17 = '17_hybrid_v3_cd_v3_representation_correct_kl.py'
_rq1_s7_SCRIPT_18 = '18_three_seed_model_comparison.py'
_rq1_s7_SCRIPT_21 = '21_common_resolution_fully_discrete_kl.py'
_rq1_s7_SCRIPT_21B = '21b_common_resolution_fully_discrete_kl_smc.py'
_rq1_s7_SCRIPT_07 = '07_v2_hybrid_v3_representation_correct_kl.py'

def _rq1_s7_discover_root(here: _rq1_s7_Path, explicit: _rq1_s7_Path | None) -> _rq1_s7_Path:
    return _rq1_find_root()

def _rq1_s7_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(_rq1_s7_json.dumps(payload, indent=2, default=_rq1_s7_json_default), encoding='utf-8')
    _rq1_s7_os.replace(tmp, path)

def _rq1_s7_json_default(x: Any):
    if isinstance(x, _rq1_s7_np.generic):
        return x.item()
    if isinstance(x, _rq1_s7_np.ndarray):
        return x.tolist()
    if isinstance(x, _rq1_s7_Path):
        return str(x)
    if isinstance(x, set):
        return sorted(x)
    if isinstance(x, float) and (not _rq1_s7_np.isfinite(x)):
        return None
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(type(x).__name__)

def _rq1_s7_dataframe_sha256(frame: _rq1_s7_pd.DataFrame) -> str:
    h = _rq1_s7_hashlib.sha256()
    h.update('|'.join(map(str, frame.columns)).encode('utf-8'))
    hashed = _rq1_s7_pd.util.hash_pandas_object(frame, index=True).to_numpy(_rq1_s7_np.uint64)
    h.update(_rq1_s7_np.ascontiguousarray(hashed).tobytes())
    return h.hexdigest()

def _rq1_s7_deterministic_seed(*parts: Any) -> int:
    text = '|'.join(map(str, parts))
    return int.from_bytes(_rq1_s7_hashlib.sha256(text.encode('utf-8')).digest()[:4], 'little')

def _rq1_s7_graph_from_edges(nodes: Sequence[str], edges: Iterable[Sequence[str]]) -> _rq1_s7_nx.DiGraph:
    g = _rq1_s7_nx.DiGraph()
    g.add_nodes_from(map(str, nodes))
    g.add_edges_from(((str(u), str(v)) for u, v in edges))
    if not _rq1_s7_nx.is_directed_acyclic_graph(g):
        raise ValueError('Edge set is not a DAG.')
    return g

def _rq1_s7_fmt_seconds(seconds: float) -> str:
    if not _rq1_s7_np.isfinite(seconds):
        return '--'
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f'{seconds:.1f}s'
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f'{m:02d}:{s:02d}'
    h, m = divmod(m, 60)
    return f'{h}:{m:02d}:{s:02d}'

class _rq1_s7_Progress:

    def __init__(self, total: int, label: str):
        self.total = max(1, int(total))
        self.label = label
        self.done = 0
        self.started = _rq1_s7_time.perf_counter()

    def update(self, n: int=1):
        self.done = min(self.total, self.done + n)
        elapsed = _rq1_s7_time.perf_counter() - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float('nan')
        print(f'\r    {self.label:<22} {self.done:4d}/{self.total:<4d} | elapsed {_rq1_s7_fmt_seconds(elapsed)} | ETA {_rq1_s7_fmt_seconds(eta)}', end='\n' if self.done == self.total else '', flush=True)

def _rq1_s7_load_generator(root: Path, name: str):
    filename, classname = _rq1_s7_GENERATOR_SPECS[name]
    path = root / 'src' / 'generators' / filename
    mod = _rq1_inline_load(path, f'_rq15_generator_{name}')
    return getattr(mod, classname)

def _rq1_s7_fixed_native_edges(cardinality: int) -> _rq1_s7_np.ndarray:
    if cardinality == 2:
        return _rq1_s7_np.asarray([-_rq1_s7_np.inf, 0.5, _rq1_s7_np.inf], dtype=float)
    if cardinality == 3:
        return _rq1_s7_np.asarray([-_rq1_s7_np.inf, 0.5, 1.5, _rq1_s7_np.inf], dtype=float)
    internal = _rq1_s7_np.arange(0.5, cardinality - 0.5, 1.0)
    return _rq1_s7_np.concatenate(([-_rq1_s7_np.inf], internal, [_rq1_s7_np.inf])).astype(float)

class _rq1_s7_NativeStateDiscretizer:
    """Drop-in QuantileDiscretizer replacement for already-native D states."""

    def __init__(self, cardinalities: Mapping[str, int]) -> None:
        self.cardinalities = {str(k): int(v) for k, v in cardinalities.items()}
        self.bin_edges = {node: _rq1_s7_fixed_native_edges(card) for node, card in self.cardinalities.items()}
        self.is_fitted = False

    @property
    def categories(self):
        return {node: tuple(range(card)) for node, card in self.cardinalities.items()}

    def fit(self, data: pd.DataFrame):
        self._validate(data)
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError('NativeStateDiscretizer must be fitted first.')
        self._validate(data)
        out = data.copy().reset_index(drop=True)
        for node, card in self.cardinalities.items():
            out[node] = _rq1_s7_np.rint(out[node].to_numpy(float)).astype(_rq1_s7_np.int64)
            values = out[node].to_numpy(_rq1_s7_np.int64)
            if values.min(initial=0) < 0 or values.max(initial=0) >= card:
                raise ValueError(f'Invalid native state in {node}.')
        return out

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)

    def to_metadata(self):
        return {'cardinalities': dict(self.cardinalities), 'bin_edges': {node: edges.copy() for node, edges in self.bin_edges.items()}}

    def _validate(self, data: _rq1_s7_pd.DataFrame):
        for node, card in self.cardinalities.items():
            raw = _rq1_s7_pd.to_numeric(data[node], errors='raise').to_numpy(float)
            if not _rq1_s7_np.all(_rq1_s7_np.isfinite(raw)):
                raise ValueError(f'Non-finite native state in {node}.')
            rounded = _rq1_s7_np.rint(raw)
            if _rq1_s7_np.max(_rq1_s7_np.abs(raw - rounded), initial=0.0) > 1e-09:
                raise ValueError(f'{node} is not integer-valued native discrete data.')
            if rounded.min(initial=0) < 0 or rounded.max(initial=0) >= card:
                raise ValueError(f'{node} contains a state outside 0..{card - 1}.')

@_rq1_s7_contextlib.contextmanager
def _rq1_s7_native_discretizer_patch(model_module):
    old = getattr(model_module, 'QuantileDiscretizer', None)
    model_module.QuantileDiscretizer = _rq1_s7_NativeStateDiscretizer
    try:
        yield
    finally:
        if old is None:
            delattr(model_module, 'QuantileDiscretizer')
        else:
            model_module.QuantileDiscretizer = old

def _rq1_s7_fit_common_transform(*, training: pd.DataFrame, continuous_nodes: Sequence[str], discrete_nodes: Sequence[str], cards: Mapping[str, int], SavedTransform):
    edges: dict[str, _rq1_s7_np.ndarray] = {}
    all_cards: dict[str, int] = {}
    for node in continuous_nodes:
        q = _rq1_s7_np.quantile(training[node].to_numpy(float), [1.0 / 3.0, 2.0 / 3.0])
        if q[0] >= q[1]:
            center = float(q[0])
            q = _rq1_s7_np.asarray([center - 1e-09, center + 1e-09])
        edges[node] = _rq1_s7_np.asarray([-_rq1_s7_np.inf, q[0], q[1], _rq1_s7_np.inf], float)
        all_cards[node] = 3
    for node in discrete_nodes:
        all_cards[node] = int(cards[node])
        edges[node] = _rq1_s7_fixed_native_edges(int(cards[node]))
    return SavedTransform(all_cards, edges)

def _rq1_s7_dequantize(frame: pd.DataFrame, discrete_nodes: Sequence[str], *, generator: str, seed: int, purpose: str) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    for node in discrete_nodes:
        rng = _rq1_s7_np.random.default_rng(_rq1_s7_deterministic_seed('rq1.5-dequant-v1', generator, seed, purpose, node))
        out[node] = out[node].to_numpy(float) + rng.uniform(-0.5, 0.5, size=len(out))
    return out

def _rq1_s7_structure_metrics(true_edges, learned_edges) -> dict[str, Any]:
    true_set = {(str(u), str(v)) for u, v in true_edges}
    learned_set = {(str(u), str(v)) for u, v in learned_edges}
    correct = true_set & learned_set
    reversed_true = {(u, v) for u, v in true_set if (u, v) not in learned_set and (v, u) in learned_set}
    reversed_learned = {(v, u) for u, v in reversed_true}
    deletions = {e for e in true_set if e not in learned_set and (e[1], e[0]) not in learned_set}
    additions = {e for e in learned_set if e not in true_set and e not in reversed_learned and ((e[1], e[0]) not in true_set)}
    tp = len(correct)
    p = tp / len(learned_set) if learned_set else 0.0
    r = tp / len(true_set) if true_set else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    ts = {frozenset(e) for e in true_set}
    ls = {frozenset(e) for e in learned_set}
    stp = len(ts & ls)
    sp = stp / len(ls) if ls else 0.0
    sr = stp / len(ts) if ts else 0.0
    sf1 = 2 * sp * sr / (sp + sr) if sp + sr else 0.0
    return {'SHD': len(additions) + len(deletions) + len(reversed_true), 'SHD_Additions': len(additions), 'SHD_Deletions': len(deletions), 'SHD_Reversals': len(reversed_true), 'Directed_Precision': p, 'Directed_Recall': r, 'Directed_F1': f1, 'Skeleton_Precision': sp, 'Skeleton_Recall': sr, 'Skeleton_F1': sf1, 'True_Edges': len(true_set), 'Learned_Edges': len(learned_set)}

def _rq1_s7_quadrant(parent: str, child: str, dset: set[str]) -> str:
    pd = parent in dset
    cd = child in dset
    if not pd and (not cd):
        return 'CC'
    if not pd and cd:
        return 'CD'
    if pd and (not cd):
        return 'DC'
    return 'DD'

def _rq1_s7_quadrant_metrics(true_edges, learned_edges, discrete_nodes: Sequence[str]) -> list[dict[str, Any]]:
    dset = set(map(str, discrete_nodes))
    true_set = {(str(u), str(v)) for u, v in true_edges}
    learned_set = {(str(u), str(v)) for u, v in learned_edges}
    rows = []
    for q in ('CC', 'CD', 'DC', 'DD'):
        tq = {(u, v) for u, v in true_set if _rq1_s7_quadrant(u, v, dset) == q}
        lq = {(u, v) for u, v in learned_set if _rq1_s7_quadrant(u, v, dset) == q}
        correct = tq & lq
        p = len(correct) / len(lq) if lq else 0.0
        r = len(correct) / len(tq) if tq else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        adj = 0
        orient = 0
        for u, v in tq:
            if (u, v) in learned_set:
                adj += 1
                orient += 1
            elif (v, u) in learned_set:
                adj += 1
        rows.append({'Quadrant': q, 'True_Edges': len(tq), 'Learned_Edges': len(lq), 'Correct_Directed': len(correct), 'Precision': p, 'Recall': r, 'F1': f1, 'Recovered_True_Adjacencies': adj, 'Orientation_Accuracy_Given_Adjacency': orient / adj if adj else float('nan')})
    return rows

def _rq1_s7__resample_indices(weights: _rq1_s7_np.ndarray, rng: _rq1_s7_np.random.Generator) -> _rq1_s7_np.ndarray:
    b, particles = weights.shape
    cdf = _rq1_s7_np.cumsum(weights, axis=1)
    u = rng.random((b, particles))
    idx = _rq1_s7_np.empty((b, particles), dtype=_rq1_s7_np.int64)
    for i in range(b):
        idx[i] = _rq1_s7_np.searchsorted(cdf[i], u[i], side='right')
    _rq1_s7_np.minimum(idx, particles - 1, out=idx)
    return idx

def _rq1_s7__gather(arr: _rq1_s7_np.ndarray, idx: _rq1_s7_np.ndarray) -> _rq1_s7_np.ndarray:
    return _rq1_s7_np.take_along_axis(arr, idx, axis=1)

class _rq1_s7_MixedTruthCommonSMC:

    def __init__(self, *, generator_obj, transform, particles: int, replicates: int, batch_size: int, seed: int, generator_name: str):
        self.gt = generator_obj
        self.transform = transform
        self.particles = int(particles)
        self.replicates = int(replicates)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.generator_name = generator_name
        self.order = list(_rq1_s7_nx.topological_sort(self.gt.graph))
        self.dset = set(self.gt.discrete_nodes)

    def logmass(self, states: pd.DataFrame):
        reps = []
        rep_diags = []
        total_batches = self.replicates * _rq1_s7_math.ceil(len(states) / self.batch_size)
        progress = _rq1_s7_Progress(total_batches, 'true common SMC')
        for rep in range(self.replicates):
            rng = _rq1_s7_np.random.default_rng(_rq1_s7_deterministic_seed('rq1.5-true-common-smc', self.generator_name, self.seed, rep))
            out = _rq1_s7_np.empty(len(states), dtype=float)
            ess_all = []
            for start in range(0, len(states), self.batch_size):
                stop = min(len(states), start + self.batch_size)
                target = states.iloc[start:stop].reset_index(drop=True)
                b = len(target)
                raw: dict[str, _rq1_s7_np.ndarray] = {}
                logmass = _rq1_s7_np.zeros(b, dtype=float)
                for child in self.order:
                    parents = list(self.gt.graph.predecessors(child))
                    parent_values = {p: raw[p].reshape(-1) for p in parents}
                    if child in self.dset:
                        probs = self.gt.discrete_probabilities(child, parent_values)
                        if probs.shape[0] == 1 and b * self.particles != 1:
                            probs = _rq1_s7_np.repeat(probs, b * self.particles, axis=0)
                        y = _rq1_s7_np.repeat(target[child].to_numpy(_rq1_s7_np.int64), self.particles)
                        inc = probs[_rq1_s7_np.arange(b * self.particles), y].reshape(b, self.particles)
                    else:
                        edges = _rq1_s7_np.asarray(self.transform.bin_edges[child], float)
                        s = target[child].to_numpy(_rq1_s7_np.int64)
                        low = _rq1_s7_np.repeat(edges[s], self.particles)
                        high = _rq1_s7_np.repeat(edges[s + 1], self.particles)
                        inc = self.gt.continuous_interval_probability(child, low, high, parent_values).reshape(b, self.particles)
                    inc = _rq1_s7_np.maximum(inc, 1e-300)
                    loginc = _rq1_s7_np.log(inc)
                    logmass += _rq1_s7_logsumexp(loginc, axis=1) - _rq1_s7_math.log(self.particles)
                    weights = inc / inc.sum(axis=1, keepdims=True)
                    ess = 1.0 / _rq1_s7_np.sum(weights * weights, axis=1)
                    ess_all.extend((ess / self.particles).tolist())
                    idx = _rq1_s7__resample_indices(weights, rng)
                    for node in raw:
                        raw[node] = _rq1_s7__gather(raw[node], idx)
                    if child in self.dset:
                        raw[child] = _rq1_s7_np.repeat(target[child].to_numpy(_rq1_s7_np.int64)[:, None], self.particles, axis=1)
                    else:
                        parents_after = {p: raw[p].reshape(-1) for p in parents}
                        edges = _rq1_s7_np.asarray(self.transform.bin_edges[child], float)
                        s = target[child].to_numpy(_rq1_s7_np.int64)
                        low = _rq1_s7_np.repeat(edges[s], self.particles)
                        high = _rq1_s7_np.repeat(edges[s + 1], self.particles)
                        proposed = self.gt.sample_continuous_truncated(child, low, high, parents_after, rng)
                        raw[child] = proposed.reshape(b, self.particles)
                out[start:stop] = logmass
                progress.update()
            reps.append(out)
            rep_diags.append({'p10_step_ess_fraction': float(_rq1_s7_np.quantile(ess_all, 0.1)), 'median_step_ess_fraction': float(_rq1_s7_np.median(ess_all)), 'min_step_ess_fraction': float(_rq1_s7_np.min(ess_all))})
        matrix = _rq1_s7_np.vstack(reps)
        combined = _rq1_s7_logsumexp(matrix, axis=0) - _rq1_s7_math.log(self.replicates)
        rep_means = _rq1_s7_np.mean(matrix, axis=1)
        return (combined, {'replicate_mean_logp': rep_means.tolist(), 'replicate_mean_logp_sd': float(_rq1_s7_np.std(rep_means, ddof=1)) if self.replicates > 1 else float('nan'), 'p10_step_ess_fraction': float(min((d['p10_step_ess_fraction'] for d in rep_diags))), 'median_step_ess_fraction': float(_rq1_s7_np.median([d['median_step_ess_fraction'] for d in rep_diags])), 'min_step_ess_fraction': float(min((d['min_step_ess_fraction'] for d in rep_diags))), 'particles': self.particles, 'replicates': self.replicates, 'batch_size': self.batch_size})

class _rq1_s7_RQ15Probe:

    def __init__(self, root: Path, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.models = root / 'src' / 'hybrid_bn'
        self.exp = root / 'experiments' / 'rq1'
        self.out = args.output_dir.resolve() if args.output_dir.is_absolute() else (root / args.output_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.cont = _rq1_inline_load(self.models / 'SLcontinuous.py', '_rq15_cont')
        self.disc = _rq1_inline_load(self.models / 'SLdiscrete.py', '_rq15_disc')
        self.h16 = _rq1_inline_load(self.exp / _rq1_s7_SCRIPT_16)
        self.h17 = _rq1_inline_load(self.exp / _rq1_s7_SCRIPT_17, '_rq15_h17')
        self.c18 = _rq1_inline_load(self.exp / _rq1_s7_SCRIPT_18, '_rq15_c18')
        self.d15 = _rq1_inline_load(self.exp / _rq1_s7_SCRIPT_15, '_rq15_d15')
        self.base21 = _rq1_inline_load(self.exp / _rq1_s7_SCRIPT_21, '_rq15_base21')
        self.smc21b = _rq1_inline_load(self.exp / _rq1_s7_SCRIPT_21B, '_rq15_smc21b')
        self.kl07 = _rq1_inline_load(self.exp / _rq1_s7_SCRIPT_07, '_rq15_kl07')
        self.hybrid_model_file = (self.models / 'SLhybrid.py').resolve()
        self.hybrid_model = _rq1_inline_load(self.hybrid_model_file)
        self.cc_file = (self.models / 'SLhybrid.py').resolve()
        self._validate_contract()

    def _validate_contract(self):
        if self.args.max_parents != 3:
            raise ValueError('RQ1.5 contract requires max_parents=3.')
        if self.args.max_iter != 300:
            raise ValueError('RQ1.5 contract requires max_iter=300.')
        print('RQ1.5 contract: max_parents=3, max_iter=300, ADD2=OFF')

    def run_dir(self, gen: str, seed: int) -> Path:
        return self.out / 'runs' / gen / f'seed_{seed:02d}'

    def fit_continuous_structure(self, train_z, nodes, true_graph):
        scorer = self.cont.ContinuousScoringEngine(train_z)
        search = self.cont.TabuGreedyHillClimbing(nodes, scorer, tabu_length=self.args.tabu_length, max_iter=self.args.max_iter, max_parents=self.args.max_parents, improvement_tolerance=1e-10)
        t = _rq1_s7_time.perf_counter()
        edges = search.search()
        runtime = _rq1_s7_time.perf_counter() - t
        graph = _rq1_s7_graph_from_edges(nodes, edges)
        return (graph, runtime)

    def fit_discrete_structure(self, train_states, nodes, cards):
        scorer = self.disc.DiscreteScoringEngine(train_states, cards)
        search = self.disc.TabuGreedyHillClimbing(nodes, scorer, tabu_length=self.args.tabu_length, max_iter=self.args.max_iter, max_parents=self.args.max_parents, improvement_tolerance=1e-10)
        t = _rq1_s7_time.perf_counter()
        edges = search.search()
        runtime = _rq1_s7_time.perf_counter() - t
        return (_rq1_s7_graph_from_edges(nodes, edges), runtime)

    def fit_hybrid_structure(self, *, raw_train, payload, gen, seed):
        nodes = list(map(str, payload['nodes']))
        dnodes = list(map(str, payload['discrete_nodes']))
        cnodes = list(map(str, payload['continuous_nodes']))
        cards = {str(k): int(v) for k, v in payload['cardinalities'].items()}
        true_graph = payload['graph']
        with _rq1_s7_native_discretizer_patch(self.hybrid_model):
            result, parity_df = self.h16.run_one_cell(model_module=self.hybrid_model, model_file=self.hybrid_model_file, cc_scorer_file=self.cc_file, raw_training=raw_train, nodes=nodes, discrete_nodes=dnodes, continuous_nodes=cnodes, cardinalities=cards, true_graph=true_graph, workers=self.args.workers, tabu_length=self.args.tabu_length, max_iter=self.args.max_iter, max_parents=self.args.max_parents, enable_add2=False, parity_tolerance=self.args.parity_tolerance)
        selected_add2 = int(result.get('diagnostics', {}).get('selected_add2_action_count', 0))
        if selected_add2 != 0:
            raise AssertionError('ADD2 action selected despite hard-off contract.')
        cell = {'schema_version': _rq1_s7_SCHEMA_VERSION, 'status': 'complete', 'experiment_definition': {'generator': gen, 'seed': int(seed), 'sample_size': len(raw_train), 'native_mixed_ground_truth': True, 'all_ground_truth_variables_continuous': False, 'hybrid_discrete_nodes': dnodes, 'hybrid_continuous_nodes': cnodes}, 'settings': {'workers': self.args.workers, 'tabu_length': self.args.tabu_length, 'max_iter': self.args.max_iter, 'max_parents': self.args.max_parents, 'enable_add2': False, 'representation_adapter': 'NativeStateDiscretizer identity mapping'}, 'dataset_sha256': _rq1_s7_dataframe_sha256(raw_train), 'true_edges': sorted([[str(u), str(v)] for u, v in true_graph.edges()]), **result}
        path = self.run_dir(gen, seed) / 'hybrid_structure.json'
        _rq1_s7_atomic_json(path, cell)
        parity_df.to_csv(path.parent / 'hybrid_worker_main_parity.csv', index=False)
        return (cell, path)

    def run_cell(self, gen: str, seed: int) -> dict[str, Any]:
        run_dir = self.run_dir(gen, seed)
        result_path = run_dir / 'result.json'
        if result_path.exists() and (not self.args.overwrite):
            print(f'  REUSE complete {gen} seed={seed}')
            return _rq1_s7_json.loads(result_path.read_text(encoding='utf-8'))
        Generator = _rq1_s7_load_generator(self.root, gen)
        gt = Generator(num_samples=self.args.sample_size, seed=seed)
        gt.generate_dag()
        raw_train = _rq1_s7_pd.DataFrame(gt.generate_data(), columns=gt.nodes)
        payload = gt.get_payload()
        nodes = list(map(str, payload['nodes']))
        dnodes = list(map(str, payload['discrete_nodes']))
        cnodes = list(map(str, payload['continuous_nodes']))
        cards_native = {str(k): int(v) for k, v in payload['cardinalities'].items()}
        cards_all = {node: cards_native[node] if node in cards_native else 3 for node in nodes}
        true_graph = payload['graph']
        fresh = gt.sample(self.args.native_rows, sample_seed=_rq1_s7_deterministic_seed('rq1.5-native-fresh', gen, seed))[nodes].reset_index(drop=True)
        native_logp = _rq1_s7_np.asarray(gt.joint_log_prob(fresh), float)
        train_z = _rq1_s7_dequantize(raw_train, dnodes, generator=gen, seed=seed, purpose='train')
        fresh_z = _rq1_s7_dequantize(fresh, dnodes, generator=gen, seed=seed, purpose='fresh-native')
        common = _rq1_s7_fit_common_transform(training=raw_train, continuous_nodes=cnodes, discrete_nodes=dnodes, cards=cards_native, SavedTransform=self.kl07.SavedQuantileTransform)
        dtrain = common.transform(raw_train.copy())
        print('    [1/6] Continuous structure (dequantised native D)')
        cgraph, c_runtime = self.fit_continuous_structure(train_z, nodes, true_graph)
        print('    [2/6] Discrete structure (native D + K3 true C)')
        dgraph, d_runtime = self.fit_discrete_structure(dtrain, nodes, cards_all)
        print('    [3/6] Hybrid structure (native mixed, ADD2 OFF)')
        hcell, hpath = self.fit_hybrid_structure(raw_train=raw_train, payload=payload, gen=gen, seed=seed)
        hgraph = _rq1_s7_graph_from_edges(nodes, hcell['learned_edges'])
        h_runtime = float(hcell.get('runtime_seconds', float('nan')))
        qc = self.c18.LearnedGaussianDAG(cgraph, nodes).fit(train_z)
        qd = self.d15.SmoothedDiscreteDAG(dgraph, nodes, cards_all, alpha=self.args.cpt_alpha).fit(dtrain)
        qh, native_h_transform, hgraph2, hd, hc = self.base21.fit_hybrid_q(root=self.root, hybrid_module=self.h17, structure_cell=hcell, structure_path=hpath, raw_training=raw_train, payload=payload, metadata_score_tolerance=self.args.metadata_score_tolerance, cpt_alpha=self.args.cpt_alpha, min_regime_size=self.args.min_regime_size, logistic_c=self.args.logistic_c, seed=seed, n=self.args.sample_size)
        if set(hgraph2.edges()) != set(hgraph.edges()):
            raise AssertionError('Hybrid Q graph mismatch after reconstruction.')
        print('    [4/6] Native KL')
        logqc_native, _ = qc.joint_logpdf(fresh_z)
        logqh_native = _rq1_s7_np.asarray(qh.joint_logscore(fresh), float)
        c_ratio = native_logp - _rq1_s7_np.asarray(logqc_native, float)
        h_ratio = native_logp - logqh_native

        def kl_stats(ratio):
            ratio = _rq1_s7_np.asarray(ratio, float)
            mean = float(_rq1_s7_np.mean(ratio))
            se = float(_rq1_s7_np.std(ratio, ddof=1) / _rq1_s7_math.sqrt(len(ratio)))
            return {'estimate': mean, 'se': se, 'lower95': mean - 1.96 * se, 'upper95': mean + 1.96 * se}
        c_native = kl_stats(c_ratio)
        h_native = kl_stats(h_ratio)
        print('    [5/6] Common K3 / Discrete-native truth SMC')
        common_fresh = gt.sample(self.args.common_rows, sample_seed=_rq1_s7_deterministic_seed('rq1.5-common-fresh', gen, seed))[nodes].reset_index(drop=True)
        common_states = common.transform(common_fresh.copy())
        truth = _rq1_s7_MixedTruthCommonSMC(generator_obj=gt, transform=common, particles=self.args.truth_particles, replicates=self.args.truth_replicates, batch_size=self.args.common_batch_size, seed=seed, generator_name=gen)
        logp_common, pdiag = truth.logmass(common_states)
        logqd, _ = qd.joint_logscore(common_fresh, common)
        d_ratio = _rq1_s7_np.asarray(logp_common) - _rq1_s7_np.asarray(logqd)
        d_native = kl_stats(d_ratio)
        cpush = self.base21.ContinuousGaussianBinMassEstimator(graph=cgraph, models=qc.models, nodes=nodes, transform=common, particles=self.args.common_particles, replicates=self.args.common_replicates, batch_size=self.args.common_batch_size, generator=gen, seed=seed)
        logqc_common, cdiag = cpush.logmass(common_states)
        hpush = self.smc21b.SequentialHybridPushforward(base=self.base21, hybrid17=self.h17, q=qh, raw_training=raw_train, common_transform=common, native_transform=native_h_transform, continuous_nodes=hc, discrete_nodes=hd, nodes=nodes, generator=gen, seed=seed, outer_particles=self.args.common_particles, cc_inner_samples=self.args.common_cc_inner, replicates=self.args.common_replicates, batch_size=self.args.common_batch_size)
        logqh_common, hdiag = hpush.logmass(common_states)

        def paired(a, b):
            values = _rq1_s7_np.asarray(b, float) - _rq1_s7_np.asarray(a, float)
            mean = float(_rq1_s7_np.mean(values))
            se = float(_rq1_s7_np.std(values, ddof=1) / _rq1_s7_math.sqrt(len(values)))
            return {'estimate': mean, 'se': se, 'lower95': mean - 1.96 * se, 'upper95': mean + 1.96 * se}
        hd_diff = paired(logqh_common, logqd)
        cd_diff = paired(logqc_common, logqd)
        hc_diff = paired(logqh_common, logqc_common)
        h_common = d_native['estimate'] + hd_diff['estimate']
        c_common = d_native['estimate'] + cd_diff['estimate']
        print('    [6/6] Structural diagnostics')
        model_graphs = {'Continuous': (cgraph, c_runtime), 'Discrete': (dgraph, d_runtime), 'Hybrid': (hgraph, h_runtime)}
        structure = {}
        quadrants = {}
        for model, (graph, runtime) in model_graphs.items():
            structure[model] = {**_rq1_s7_structure_metrics(true_graph.edges(), graph.edges()), 'Structure_Runtime_S': float(runtime)}
            quadrants[model] = _rq1_s7_quadrant_metrics(true_graph.edges(), graph.edges(), dnodes)
        flags = []
        if pdiag['p10_step_ess_fraction'] < 0.05:
            flags.append('LOW_TRUE_COMMON_SMC_ESS')
        if hdiag.get('p10_step_ess_fraction', 1.0) < 0.05:
            flags.append('LOW_HYBRID_COMMON_SMC_ESS')
        inner = hdiag.get('p10_cc_inner_ess_fraction', float('nan'))
        if _rq1_s7_np.isfinite(inner) and inner < 0.05:
            flags.append('LOW_HYBRID_CC_INNER_ESS')
        result = {'schema_version': _rq1_s7_SCHEMA_VERSION, 'status': 'complete', 'generator': gen, 'seed': seed, 'sample_size': self.args.sample_size, 'ground_truth': {'native_mixed': True, 'nodes': nodes, 'continuous_nodes': cnodes, 'discrete_nodes': dnodes, 'cardinalities': cards_native, 'true_edges': sorted([[str(u), str(v)] for u, v in true_graph.edges()])}, 'representation': {'Continuous': 'native C + unit-width dequantised native D', 'Discrete': 'native D + training-quantile K3 native C', 'Hybrid': 'native mixed identity representation', 'Common': 'native D + training-quantile K3 native C'}, 'native_kl': {'Continuous': c_native, 'Discrete': d_native, 'Hybrid': h_native}, 'common_k3': {'Discrete_KL': d_native['estimate'], 'Hybrid_KL': h_common, 'Continuous_KL': c_common, 'Hybrid_Minus_Discrete': hd_diff, 'Continuous_Minus_Discrete': cd_diff, 'Hybrid_Minus_Continuous': hc_diff, 'True_SMC_Diagnostics': pdiag, 'Hybrid_SMC_Diagnostics': hdiag, 'Continuous_Pushforward_Diagnostics': cdiag}, 'structure': structure, 'quadrants': quadrants, 'flags': flags, 'dataset_sha256': _rq1_s7_dataframe_sha256(raw_train)}
        _rq1_s7_atomic_json(result_path, result)
        return result

def _rq1_s7_aggregate(output_dir: Path):
    model_rows = []
    common_rows = []
    quadrant_rows = []
    for path in sorted(output_dir.glob('runs/*/seed_*/result.json')):
        r = _rq1_s7_json.loads(path.read_text(encoding='utf-8'))
        gen = r['generator']
        seed = int(r['seed'])
        for model in ('Continuous', 'Discrete', 'Hybrid'):
            native = r['native_kl'][model]
            s = r['structure'][model]
            common_key = f'{model}_KL'
            if model == 'Discrete':
                common_kl = r['common_k3']['Discrete_KL']
            else:
                common_kl = r['common_k3'][common_key]
            model_rows.append({'Generator': gen, 'Seed': seed, 'Model': model, 'Native_KL': native['estimate'], 'Native_KL_SE': native['se'], 'Common_KL_K3': common_kl, **s})
            for q in r['quadrants'][model]:
                quadrant_rows.append({'Generator': gen, 'Seed': seed, 'Model': model, **q})
        cm = r['common_k3']
        common_rows.append({'Generator': gen, 'Seed': seed, 'Discrete_Common_KL': cm['Discrete_KL'], 'Hybrid_Common_KL': cm['Hybrid_KL'], 'Continuous_Common_KL': cm['Continuous_KL'], 'Hybrid_Minus_Discrete': cm['Hybrid_Minus_Discrete']['estimate'], 'Hybrid_Minus_Discrete_SE': cm['Hybrid_Minus_Discrete']['se'], 'Hybrid_Minus_Continuous': cm['Hybrid_Minus_Continuous']['estimate'], 'Hybrid_Minus_Continuous_SE': cm['Hybrid_Minus_Continuous']['se'], 'Continuous_Minus_Discrete': cm['Continuous_Minus_Discrete']['estimate'], 'Continuous_Minus_Discrete_SE': cm['Continuous_Minus_Discrete']['se'], 'True_p10_Step_ESS': cm['True_SMC_Diagnostics']['p10_step_ess_fraction'], 'Hybrid_p10_Step_ESS': cm['Hybrid_SMC_Diagnostics'].get('p10_step_ess_fraction', _rq1_s7_np.nan), 'Flags': '|'.join(r.get('flags', []))})
    m = _rq1_s7_pd.DataFrame(model_rows)
    c = _rq1_s7_pd.DataFrame(common_rows)
    q = _rq1_s7_pd.DataFrame(quadrant_rows)
    if len(m):
        m = m.sort_values(['Generator', 'Seed', 'Model'])
    if len(c):
        c = c.sort_values(['Generator', 'Seed'])
    if len(q):
        q = q.sort_values(['Generator', 'Seed', 'Model', 'Quadrant'])
    m.to_csv(output_dir / 'mixed_model_results.csv', index=False)
    c.to_csv(output_dir / 'mixed_common_pairwise.csv', index=False)
    q.to_csv(output_dir / 'mixed_quadrant_diagnostics.csv', index=False)
    if len(m):
        ms = m.groupby(['Generator', 'Model'], as_index=False).agg(Runs=('Seed', 'size'), Mean_Native_KL=('Native_KL', 'mean'), SD_Native_KL=('Native_KL', 'std'), Mean_Common_KL_K3=('Common_KL_K3', 'mean'), SD_Common_KL_K3=('Common_KL_K3', 'std'), Mean_SHD=('SHD', 'mean'), Mean_SHD_Additions=('SHD_Additions', 'mean'), Mean_SHD_Deletions=('SHD_Deletions', 'mean'), Mean_SHD_Reversals=('SHD_Reversals', 'mean'), Mean_Directed_F1=('Directed_F1', 'mean'), Mean_Skeleton_F1=('Skeleton_F1', 'mean'), Mean_Structure_Runtime_S=('Structure_Runtime_S', 'mean'))
    else:
        ms = _rq1_s7_pd.DataFrame()
    if len(c):
        cs = c.groupby('Generator', as_index=False).agg(Runs=('Seed', 'size'), Mean_Discrete_Common_KL=('Discrete_Common_KL', 'mean'), Mean_Hybrid_Common_KL=('Hybrid_Common_KL', 'mean'), Mean_Continuous_Common_KL=('Continuous_Common_KL', 'mean'), Mean_Hybrid_Minus_Discrete=('Hybrid_Minus_Discrete', 'mean'), SD_Hybrid_Minus_Discrete=('Hybrid_Minus_Discrete', 'std'), Mean_Hybrid_Minus_Continuous=('Hybrid_Minus_Continuous', 'mean'), SD_Hybrid_Minus_Continuous=('Hybrid_Minus_Continuous', 'std'), Mean_Continuous_Minus_Discrete=('Continuous_Minus_Discrete', 'mean'), SD_Continuous_Minus_Discrete=('Continuous_Minus_Discrete', 'std'), Mean_True_p10_Step_ESS=('True_p10_Step_ESS', 'mean'), Mean_Hybrid_p10_Step_ESS=('Hybrid_p10_Step_ESS', 'mean'))
    else:
        cs = _rq1_s7_pd.DataFrame()
    if len(q):
        qs = q.groupby(['Generator', 'Model', 'Quadrant'], as_index=False).agg(Runs=('Seed', 'size'), Mean_True_Edges=('True_Edges', 'mean'), Mean_Learned_Edges=('Learned_Edges', 'mean'), Mean_Precision=('Precision', 'mean'), Mean_Recall=('Recall', 'mean'), Mean_F1=('F1', 'mean'), Mean_Orientation_Accuracy=('Orientation_Accuracy_Given_Adjacency', 'mean'))
    else:
        qs = _rq1_s7_pd.DataFrame()
    ms.to_csv(output_dir / 'mixed_model_summary.csv', index=False)
    cs.to_csv(output_dir / 'mixed_common_summary.csv', index=False)
    qs.to_csv(output_dir / 'mixed_quadrant_summary.csv', index=False)
    report = ['RQ1.5 NATIVE-MIXED PROBE', '=' * 130, '', 'MODEL SUMMARY', '-' * 130, ms.to_string(index=False) if len(ms) else '(none)', '', 'COMMON K3 SUMMARY', '-' * 130, cs.to_string(index=False) if len(cs) else '(none)', '', 'QUADRANT SUMMARY', '-' * 130, qs.to_string(index=False) if len(qs) else '(none)', '', 'Contract: native mixed GT; Hybrid identity representation; Discrete bins only true C; Continuous dequantises only true D; ADD2 OFF; max_parents=3; max_iter=300.']
    (output_dir / 'report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return (m, c, q)

def _rq1_s7_parse_args():
    p = _rq1_s7_argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=_rq1_s7_Path, default=None)
    p.add_argument('--generators', nargs='+', choices=tuple(_rq1_s7_GENERATOR_SPECS), default=_rq1_s7_DEFAULT_GENERATORS)
    p.add_argument('--seeds', nargs='+', type=int, default=_rq1_s7_DEFAULT_SEEDS)
    p.add_argument('--sample-size', type=int, default=5000)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--tabu-length', type=int, default=10)
    p.add_argument('--max-iter', type=int, default=300)
    p.add_argument('--max-parents', type=int, default=3)
    p.add_argument('--parity-tolerance', type=float, default=1e-08)
    p.add_argument('--cpt-alpha', type=float, default=1.0)
    p.add_argument('--min-regime-size', type=int, default=5)
    p.add_argument('--logistic-c', type=float, default=1000.0)
    p.add_argument('--metadata-score-tolerance', type=float, default=1e-06)
    p.add_argument('--native-rows', type=int, default=5000)
    p.add_argument('--common-rows', type=int, default=1000)
    p.add_argument('--truth-particles', type=int, default=128)
    p.add_argument('--truth-replicates', type=int, default=2)
    p.add_argument('--common-particles', type=int, default=128)
    p.add_argument('--common-cc-inner', type=int, default=16)
    p.add_argument('--common-replicates', type=int, default=2)
    p.add_argument('--common-batch-size', type=int, default=16)
    p.add_argument('--output-dir', type=_rq1_s7_Path, default=_rq1_s7_Path('MixedResults/RQ1_5_NativeMixed_Probe'))
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def _rq1_s7_main() -> int:
    args = _rq1_s7_parse_args()
    here = _rq1_s7_Path(__file__).resolve().parent
    root = _rq1_s7_discover_root(here, args.project_root)
    args.generators = list(dict.fromkeys(args.generators))
    args.seeds = sorted(set(args.seeds))
    if args.sample_size != 5000:
        print(f'NOTE: requested n={args.sample_size}; canonical RQ1.5 probe is n=5000.')
    probe = _rq1_s7_RQ15Probe(root, args)
    cells = [(g, s) for g in args.generators for s in args.seeds]
    failures = []
    times = []
    print('=' * 140)
    print('RQ1.5 NATIVE-MIXED PROBE')
    print('=' * 140)
    print(f'Generators: {args.generators}')
    print(f'Seeds:      {args.seeds}')
    print(f'n:          {args.sample_size}')
    print(f'Cells:      {len(cells)} datasets / {len(cells) * 3} model fits')
    print('Hybrid:     native mixed, ADD2 OFF')
    print('Discrete:   native D + K3 true C')
    print('Continuous: native C + unit-width dequantised true D')
    print(f'Output:     {probe.out}')
    print('=' * 140)
    for i, (gen, seed) in enumerate(cells, start=1):
        mean_t = float(_rq1_s7_np.mean(times)) if times else float('nan')
        eta = mean_t * (len(cells) - i + 1)
        print(f'\n[CELL {i}/{len(cells)}] {gen} seed={seed} | rolling ETA {_rq1_s7_fmt_seconds(eta)}')
        t = _rq1_s7_time.perf_counter()
        try:
            result = probe.run_cell(gen, seed)
            c = result['native_kl']['Continuous']['estimate']
            d = result['native_kl']['Discrete']['estimate']
            h = result['native_kl']['Hybrid']['estimate']
            cm = result['common_k3']
            print(f'  Native KL: C={c:.4f} D={d:.4f} H={h:.4f}')
            print(f"  Common K3: C={cm['Continuous_KL']:.4f} D={cm['Discrete_KL']:.4f} H={cm['Hybrid_KL']:.4f} | H-D={cm['Hybrid_Minus_Discrete']['estimate']:+.4f}")
            print('  SHD: ' + ' '.join((f"{m[0]}={result['structure'][m]['SHD']}" for m in ('Continuous', 'Discrete', 'Hybrid'))))
            if result['flags']:
                print(f"  FLAGS: {result['flags']}")
        except Exception as exc:
            failure = {'generator': gen, 'seed': seed, 'error': f'{type(exc).__name__}: {exc}', 'traceback': _rq1_s7_traceback.format_exc()}
            failures.append(failure)
            _rq1_s7_atomic_json(probe.run_dir(gen, seed) / 'failure.json', failure)
            print(f'  FAILED: {type(exc).__name__}: {exc}')
            _rq1_s7_traceback.print_exc()
        times.append(_rq1_s7_time.perf_counter() - t)
        _rq1_s7_aggregate(probe.out)
    _, c, _ = _rq1_s7_aggregate(probe.out)
    _rq1_s7_atomic_json(probe.out / 'manifest.json', {'schema_version': _rq1_s7_SCHEMA_VERSION, 'requested_cells': len(cells), 'complete_cells': int(len(c)), 'failures': failures, 'settings': vars(args)})
    print('\n' + '=' * 140)
    print('RQ1.5 COMPLETE')
    print('=' * 140)
    print(f'Complete cells: {len(c)}/{len(cells)}')
    print(f'Failures:       {len(failures)}')
    print(f"Model results:  {probe.out / 'mixed_model_results.csv'}")
    print(f"Model summary:  {probe.out / 'mixed_model_summary.csv'}")
    print(f"Common:         {probe.out / 'mixed_common_summary.csv'}")
    print(f"Quadrants:      {probe.out / 'mixed_quadrant_summary.csv'}")
    print(f"Report:         {probe.out / 'report.txt'}")
    print('=' * 140)
    return 1 if failures else 0


# Namespaced exports from the original evaluation modules.

_rq1_inline_modules['07_v2_hybrid_v3_representation_correct_kl.py'] = _rq1_types.SimpleNamespace(**{
    'Any': _rq1_s0_Any,
    'CPTConditional': _rq1_s0_CPTConditional,
    'ConvergenceWarning': _rq1_s0_ConvergenceWarning,
    'GENERATOR_SPECS': _rq1_s0_GENERATOR_SPECS,
    'GaussianConditional': _rq1_s0_GaussianConditional,
    'HybridV3LearnedDAGDistribution': _rq1_s0_HybridV3LearnedDAGDistribution,
    'Iterable': _rq1_s0_Iterable,
    'KL_SCHEMA_VERSION': _rq1_s0_KL_SCHEMA_VERSION,
    'LearnedDAGDistribution': _rq1_s0_LearnedDAGDistribution,
    'LogisticConditional': _rq1_s0_LogisticConditional,
    'LogisticRegression': _rq1_s0_LogisticRegression,
    'Mapping': _rq1_s0_Mapping,
    'MixtureConditional': _rq1_s0_MixtureConditional,
    'PARADIGMS': _rq1_s0_PARADIGMS,
    'Path': _rq1_s0_Path,
    'SUPPORTED_TRUE_FAMILIES': _rq1_s0_SUPPORTED_TRUE_FAMILIES,
    'SavedQuantileTransform': _rq1_s0_SavedQuantileTransform,
    'Sequence': _rq1_s0_Sequence,
    'TransformedTruthParticleLikelihood': _rq1_s0_TransformedTruthParticleLikelihood,
    'TrueContinuousDAG': _rq1_s0_TrueContinuousDAG,
    '_MIN_PROBABILITY': _rq1_s0__MIN_PROBABILITY,
    '_MIN_STANDARD_DEVIATION': _rq1_s0__MIN_STANDARD_DEVIATION,
    '_V2_Q_MIXTURE_MAX_ITER': _rq1_s0__V2_Q_MIXTURE_MAX_ITER,
    '_V2_Q_MIXTURE_MIN_EFFECTIVE_COUNT': _rq1_s0__V2_Q_MIXTURE_MIN_EFFECTIVE_COUNT,
    '_V2_Q_MIXTURE_RESTARTS': _rq1_s0__V2_Q_MIXTURE_RESTARTS,
    '_configuration_storage_key': _rq1_s0__configuration_storage_key,
    '_frozen_prior_hyperparameters_from_scorer': _rq1_s0__frozen_prior_hyperparameters_from_scorer,
    '_load_manifest_base_sample_size': _rq1_s0__load_manifest_base_sample_size,
    'argparse': _rq1_s0_argparse,
    'asdict': _rq1_s0_asdict,
    'atomic_write_json': _rq1_s0_atomic_write_json,
    'build_graph': _rq1_s0_build_graph,
    'dataclass': _rq1_s0_dataclass,
    'datetime': _rq1_s0_datetime,
    'decode_float': _rq1_s0_decode_float,
    'default_cardinalities': _rq1_s0_default_cardinalities,
    'deterministic_mc_seed': _rq1_s0_deterministic_mc_seed,
    'discover_root': _rq1_s0_discover_root,
    'discover_structure_files': _rq1_s0_discover_structure_files,
    'evaluate_cell': _rq1_s0_evaluate_cell,
    'gammaln': _rq1_s0_gammaln,
    'gc': _rq1_s0_gc,
    'hashlib': _rq1_s0_hashlib,
    'helmert': _rq1_s0_helmert,
    'import_generator': _rq1_s0_import_generator,
    'importlib': _rq1_s0_importlib,
    'json': _rq1_s0_json,
    'json_safe': _rq1_s0_json_safe,
    'load_cell_inputs': _rq1_s0_load_cell_inputs,
    'logsumexp': _rq1_s0_logsumexp,
    'main': _rq1_s0_main,
    'math': _rq1_s0_math,
    'minimize': _rq1_s0_minimize,
    'norm': _rq1_s0_norm,
    'np': _rq1_s0_np,
    'nx': _rq1_s0_nx,
    'os': _rq1_s0_os,
    'output_path': _rq1_s0_output_path,
    'owens_t': _rq1_s0_owens_t,
    'parse_args': _rq1_s0_parse_args,
    'pd': _rq1_s0_pd,
    'rebuild_csvs': _rq1_s0_rebuild_csvs,
    'reconstruct_final_evaluation_metadata': _rq1_s0_reconstruct_final_evaluation_metadata,
    'reconstruct_generator_payload_and_training': _rq1_s0_reconstruct_generator_payload_and_training,
    'skewnorm': _rq1_s0_skewnorm,
    'structure_dataframe_sha256': _rq1_s0_structure_dataframe_sha256,
    'sys': _rq1_s0_sys,
    'time': _rq1_s0_time,
    'timezone': _rq1_s0_timezone,
    'traceback': _rq1_s0_traceback,
    'warnings': _rq1_s0_warnings,
})

_rq1_inline_modules['15_discrete_baseline_representation_correct_kl.py'] = _rq1_types.SimpleNamespace(**{
    'Any': _rq1_s1_Any,
    'GENERATOR_SPECS': _rq1_s1_GENERATOR_SPECS,
    'Iterable': _rq1_s1_Iterable,
    'Mapping': _rq1_s1_Mapping,
    'Path': _rq1_s1_Path,
    'SCHEMA_VERSION': _rq1_s1_SCHEMA_VERSION,
    'Sequence': _rq1_s1_Sequence,
    'SmoothedDiscreteDAG': _rq1_s1_SmoothedDiscreteDAG,
    '_MIN_PROB': _rq1_s1__MIN_PROB,
    'argparse': _rq1_s1_argparse,
    'atomic_json': _rq1_s1_atomic_json,
    'discover_root': _rq1_s1_discover_root,
    'find_corrected_kl_script': _rq1_s1_find_corrected_kl_script,
    'graph_from_edges': _rq1_s1_graph_from_edges,
    'import_generator': _rq1_s1_import_generator,
    'importlib': _rq1_s1_importlib,
    'json': _rq1_s1_json,
    'json_safe': _rq1_s1_json_safe,
    'main': _rq1_s1_main,
    'math': _rq1_s1_math,
    'np': _rq1_s1_np,
    'nx': _rq1_s1_nx,
    'os': _rq1_s1_os,
    'parse_args': _rq1_s1_parse_args,
    'pd': _rq1_s1_pd,
    'rebuild_csvs': _rq1_s1_rebuild_csvs,
    'run': _rq1_s1_run,
    'structure_metrics': _rq1_s1_structure_metrics,
    'sys': _rq1_s1_sys,
    'time': _rq1_s1_time,
    'traceback': _rq1_s1_traceback,
})

_rq1_inline_modules['16_hybrid_v3_cd_v3_structure_learning.py'] = _rq1_types.SimpleNamespace(**{
    'Any': _rq1_s2_Any,
    'Candidate': _rq1_s2_Candidate,
    'Counter': _rq1_s2_Counter,
    'DEFAULT_GENERATORS': _rq1_s2_DEFAULT_GENERATORS,
    'DEFAULT_SAMPLE_SIZES': _rq1_s2_DEFAULT_SAMPLE_SIZES,
    'DEFAULT_SEEDS': _rq1_s2_DEFAULT_SEEDS,
    'Dict': _rq1_s2_Dict,
    'GENERATOR_SPECS': _rq1_s2_GENERATOR_SPECS,
    'Iterable': _rq1_s2_Iterable,
    'Mapping': _rq1_s2_Mapping,
    'ParallelAtomicAdd2TabuGreedyHillClimbing': _rq1_s2_ParallelAtomicAdd2TabuGreedyHillClimbing,
    'Path': _rq1_s2_Path,
    'ProcessPoolExecutor': _rq1_s2_ProcessPoolExecutor,
    'SCHEMA_VERSION': _rq1_s2_SCHEMA_VERSION,
    'Sequence': _rq1_s2_Sequence,
    'Tuple': _rq1_s2_Tuple,
    '_WORKER_SCORER': _rq1_s2__WORKER_SCORER,
    '_WORKER_TASK_COUNT': _rq1_s2__WORKER_TASK_COUNT,
    '_evict_worker_family': _rq1_s2__evict_worker_family,
    '_worker_initializer': _rq1_s2__worker_initializer,
    '_worker_score_family': _rq1_s2__worker_score_family,
    'argparse': _rq1_s2_argparse,
    'as_completed': _rq1_s2_as_completed,
    'atomic_write_json': _rq1_s2_atomic_write_json,
    'build_main_scorer': _rq1_s2_build_main_scorer,
    'build_summary': _rq1_s2_build_summary,
    'calculate_structure_metrics': _rq1_s2_calculate_structure_metrics,
    'cell_matches': _rq1_s2_cell_matches,
    'dataclass': _rq1_s2_dataclass,
    'dataframe_sha256': _rq1_s2_dataframe_sha256,
    'default_cardinalities': _rq1_s2_default_cardinalities,
    'discover_root': _rq1_s2_discover_root,
    'edge_type': _rq1_s2_edge_type,
    'family_key': _rq1_s2_family_key,
    'file_sha256': _rq1_s2_file_sha256,
    'final_family_quadrant': _rq1_s2_final_family_quadrant,
    'gc': _rq1_s2_gc,
    'graph_from_edges': _rq1_s2_graph_from_edges,
    'hashlib': _rq1_s2_hashlib,
    'import_generator': _rq1_s2_import_generator,
    'importlib': _rq1_s2_importlib,
    'itertools': _rq1_s2_itertools,
    'json': _rq1_s2_json,
    'json_safe': _rq1_s2_json_safe,
    'main': _rq1_s2_main,
    'math': _rq1_s2_math,
    'mp': _rq1_s2_mp,
    'np': _rq1_s2_np,
    'nx': _rq1_s2_nx,
    'os': _rq1_s2_os,
    'package_sha256': _rq1_s2_package_sha256,
    'parity_path': _rq1_s2_parity_path,
    'parse_args': _rq1_s2_parse_args,
    'pd': _rq1_s2_pd,
    'platform': _rq1_s2_platform,
    'reconcile_final_families': _rq1_s2_reconcile_final_families,
    'result_path': _rq1_s2_result_path,
    'run_one_cell': _rq1_s2_run_one_cell,
    'sys': _rq1_s2_sys,
    'time': _rq1_s2_time,
    'traceback': _rq1_s2_traceback,
    'write_report': _rq1_s2_write_report,
})

_rq1_inline_modules['17_hybrid_v3_cd_v3_representation_correct_kl.py'] = _rq1_types.SimpleNamespace(**{
    'Any': _rq1_s3_Any,
    'CPTConditional': _rq1_s3_CPTConditional,
    'ConvergenceWarning': _rq1_s3_ConvergenceWarning,
    'GENERATOR_SPECS': _rq1_s3_GENERATOR_SPECS,
    'GaussianConditional': _rq1_s3_GaussianConditional,
    'HybridV3LearnedDAGDistribution': _rq1_s3_HybridV3LearnedDAGDistribution,
    'Iterable': _rq1_s3_Iterable,
    'KL_SCHEMA_VERSION': _rq1_s3_KL_SCHEMA_VERSION,
    'LearnedDAGDistribution': _rq1_s3_LearnedDAGDistribution,
    'LogisticConditional': _rq1_s3_LogisticConditional,
    'LogisticRegression': _rq1_s3_LogisticRegression,
    'Mapping': _rq1_s3_Mapping,
    'MixtureConditional': _rq1_s3_MixtureConditional,
    'PARADIGMS': _rq1_s3_PARADIGMS,
    'Path': _rq1_s3_Path,
    'SUPPORTED_TRUE_FAMILIES': _rq1_s3_SUPPORTED_TRUE_FAMILIES,
    'SavedQuantileTransform': _rq1_s3_SavedQuantileTransform,
    'Sequence': _rq1_s3_Sequence,
    'TransformedTruthParticleLikelihood': _rq1_s3_TransformedTruthParticleLikelihood,
    'TrueContinuousDAG': _rq1_s3_TrueContinuousDAG,
    '_MIN_PROBABILITY': _rq1_s3__MIN_PROBABILITY,
    '_MIN_STANDARD_DEVIATION': _rq1_s3__MIN_STANDARD_DEVIATION,
    '_V2_Q_MIXTURE_MAX_ITER': _rq1_s3__V2_Q_MIXTURE_MAX_ITER,
    '_V2_Q_MIXTURE_MIN_EFFECTIVE_COUNT': _rq1_s3__V2_Q_MIXTURE_MIN_EFFECTIVE_COUNT,
    '_V2_Q_MIXTURE_RESTARTS': _rq1_s3__V2_Q_MIXTURE_RESTARTS,
    '_configuration_storage_key': _rq1_s3__configuration_storage_key,
    '_frozen_prior_hyperparameters_from_scorer': _rq1_s3__frozen_prior_hyperparameters_from_scorer,
    '_load_manifest_base_sample_size': _rq1_s3__load_manifest_base_sample_size,
    'argparse': _rq1_s3_argparse,
    'asdict': _rq1_s3_asdict,
    'atomic_write_json': _rq1_s3_atomic_write_json,
    'build_graph': _rq1_s3_build_graph,
    'dataclass': _rq1_s3_dataclass,
    'datetime': _rq1_s3_datetime,
    'decode_float': _rq1_s3_decode_float,
    'default_cardinalities': _rq1_s3_default_cardinalities,
    'deterministic_mc_seed': _rq1_s3_deterministic_mc_seed,
    'discover_root': _rq1_s3_discover_root,
    'discover_structure_files': _rq1_s3_discover_structure_files,
    'evaluate_cell': _rq1_s3_evaluate_cell,
    'gammaln': _rq1_s3_gammaln,
    'gc': _rq1_s3_gc,
    'hashlib': _rq1_s3_hashlib,
    'helmert': _rq1_s3_helmert,
    'import_generator': _rq1_s3_import_generator,
    'importlib': _rq1_s3_importlib,
    'json': _rq1_s3_json,
    'json_safe': _rq1_s3_json_safe,
    'load_cell_inputs': _rq1_s3_load_cell_inputs,
    'logsumexp': _rq1_s3_logsumexp,
    'main': _rq1_s3_main,
    'math': _rq1_s3_math,
    'minimize': _rq1_s3_minimize,
    'norm': _rq1_s3_norm,
    'np': _rq1_s3_np,
    'nx': _rq1_s3_nx,
    'os': _rq1_s3_os,
    'output_path': _rq1_s3_output_path,
    'owens_t': _rq1_s3_owens_t,
    'parse_args': _rq1_s3_parse_args,
    'pd': _rq1_s3_pd,
    'rebuild_csvs': _rq1_s3_rebuild_csvs,
    'reconstruct_final_evaluation_metadata': _rq1_s3_reconstruct_final_evaluation_metadata,
    'reconstruct_generator_payload_and_training': _rq1_s3_reconstruct_generator_payload_and_training,
    'skewnorm': _rq1_s3_skewnorm,
    'structure_dataframe_sha256': _rq1_s3_structure_dataframe_sha256,
    'sys': _rq1_s3_sys,
    'time': _rq1_s3_time,
    'timezone': _rq1_s3_timezone,
    'traceback': _rq1_s3_traceback,
    'warnings': _rq1_s3_warnings,
})

_rq1_inline_modules['18_three_seed_model_comparison.py'] = _rq1_types.SimpleNamespace(**{
    'Any': _rq1_s4_Any,
    'GENERATOR_SPECS': _rq1_s4_GENERATOR_SPECS,
    'Iterable': _rq1_s4_Iterable,
    'LearnedGaussianDAG': _rq1_s4_LearnedGaussianDAG,
    'MIN_VARIANCE': _rq1_s4_MIN_VARIANCE,
    'Mapping': _rq1_s4_Mapping,
    'Path': _rq1_s4_Path,
    'SCHEMA_VERSION': _rq1_s4_SCHEMA_VERSION,
    'Sequence': _rq1_s4_Sequence,
    'argparse': _rq1_s4_argparse,
    'atomic_json': _rq1_s4_atomic_json,
    'build_comparison': _rq1_s4_build_comparison,
    'discover_root': _rq1_s4_discover_root,
    'find_script': _rq1_s4_find_script,
    'import_generator': _rq1_s4_import_generator,
    'importlib': _rq1_s4_importlib,
    'json': _rq1_s4_json,
    'json_safe': _rq1_s4_json_safe,
    'load_discrete_result': _rq1_s4_load_discrete_result,
    'load_hybrid_result': _rq1_s4_load_hybrid_result,
    'main': _rq1_s4_main,
    'math': _rq1_s4_math,
    'norm': _rq1_s4_norm,
    'np': _rq1_s4_np,
    'nx': _rq1_s4_nx,
    'os': _rq1_s4_os,
    'parse_args': _rq1_s4_parse_args,
    'pd': _rq1_s4_pd,
    'result_row': _rq1_s4_result_row,
    'run_continuous_cell': _rq1_s4_run_continuous_cell,
    'run_subprocess': _rq1_s4_run_subprocess,
    'structure_metrics': _rq1_s4_structure_metrics,
    'subprocess': _rq1_s4_subprocess,
    'sys': _rq1_s4_sys,
    'time': _rq1_s4_time,
    'traceback': _rq1_s4_traceback,
})

_rq1_inline_modules['21_common_resolution_fully_discrete_kl.py'] = _rq1_types.SimpleNamespace(**{
    'Any': _rq1_s5_Any,
    'ContinuousGaussianBinMassEstimator': _rq1_s5_ContinuousGaussianBinMassEstimator,
    'GENERATOR_SPECS': _rq1_s5_GENERATOR_SPECS,
    'HybridCommonResolutionImportanceEstimator': _rq1_s5_HybridCommonResolutionImportanceEstimator,
    'Iterable': _rq1_s5_Iterable,
    'MIN_PROB': _rq1_s5_MIN_PROB,
    'Mapping': _rq1_s5_Mapping,
    'Path': _rq1_s5_Path,
    'SCHEMA_VERSION': _rq1_s5_SCHEMA_VERSION,
    'Sequence': _rq1_s5_Sequence,
    'argparse': _rq1_s5_argparse,
    'atomic_json': _rq1_s5_atomic_json,
    'build_reports': _rq1_s5_build_reports,
    'combine_replicate_log_estimates': _rq1_s5_combine_replicate_log_estimates,
    'deterministic_q_seed': _rq1_s5_deterministic_q_seed,
    'discover_root': _rq1_s5_discover_root,
    'evaluate_cell': _rq1_s5_evaluate_cell,
    'existing_paths': _rq1_s5_existing_paths,
    'find_script': _rq1_s5_find_script,
    'fit_hybrid_q': _rq1_s5_fit_hybrid_q,
    'graph_from_edges': _rq1_s5_graph_from_edges,
    'hashlib': _rq1_s5_hashlib,
    'import_generator': _rq1_s5_import_generator,
    'importlib': _rq1_s5_importlib,
    'interval_bounds': _rq1_s5_interval_bounds,
    'json': _rq1_s5_json,
    'json_safe': _rq1_s5_json_safe,
    'logsumexp': _rq1_s5_logsumexp,
    'main': _rq1_s5_main,
    'math': _rq1_s5_math,
    'norm': _rq1_s5_norm,
    'np': _rq1_s5_np,
    'nx': _rq1_s5_nx,
    'os': _rq1_s5_os,
    'parse_args': _rq1_s5_parse_args,
    'pd': _rq1_s5_pd,
    'require_paths': _rq1_s5_require_paths,
    'stable_normal_interval_probability': _rq1_s5_stable_normal_interval_probability,
    'sys': _rq1_s5_sys,
    'time': _rq1_s5_time,
    'traceback': _rq1_s5_traceback,
    'truncated_normal': _rq1_s5_truncated_normal,
})

_rq1_inline_modules['21b_common_resolution_fully_discrete_kl_smc.py'] = _rq1_types.SimpleNamespace(**{
    'Any': _rq1_s6_Any,
    'BASE_SCRIPT': _rq1_s6_BASE_SCRIPT,
    'HYBRID17_SCRIPT': _rq1_s6_HYBRID17_SCRIPT,
    'MIN_PROB': _rq1_s6_MIN_PROB,
    'Mapping': _rq1_s6_Mapping,
    'Path': _rq1_s6_Path,
    'SCHEMA_VERSION': _rq1_s6_SCHEMA_VERSION,
    'Sequence': _rq1_s6_Sequence,
    'SequentialHybridPushforward': _rq1_s6_SequentialHybridPushforward,
    'argparse': _rq1_s6_argparse,
    'deterministic_seed': _rq1_s6_deterministic_seed,
    'discover_root': _rq1_s6_discover_root,
    'gather_particles': _rq1_s6_gather_particles,
    'hashlib': _rq1_s6_hashlib,
    'importlib': _rq1_s6_importlib,
    'interval_bounds': _rq1_s6_interval_bounds,
    'json': _rq1_s6_json,
    'logsumexp': _rq1_s6_logsumexp,
    'main': _rq1_s6_main,
    'math': _rq1_s6_math,
    'norm': _rq1_s6_norm,
    'normal_interval_prob': _rq1_s6_normal_interval_prob,
    'np': _rq1_s6_np,
    'os': _rq1_s6_os,
    'parse_args': _rq1_s6_parse_args,
    'pd': _rq1_s6_pd,
    'resample_indices': _rq1_s6_resample_indices,
    'sys': _rq1_s6_sys,
    'time': _rq1_s6_time,
    'traceback': _rq1_s6_traceback,
    'truncated_normal': _rq1_s6_truncated_normal,
})

_rq1_inline_modules['01_rq1_5_native_mixed_probe.py'] = _rq1_types.SimpleNamespace(**{
    'Any': _rq1_s7_Any,
    'DEFAULT_GENERATORS': _rq1_s7_DEFAULT_GENERATORS,
    'DEFAULT_SEEDS': _rq1_s7_DEFAULT_SEEDS,
    'GENERATOR_SPECS': _rq1_s7_GENERATOR_SPECS,
    'Iterable': _rq1_s7_Iterable,
    'Mapping': _rq1_s7_Mapping,
    'MixedTruthCommonSMC': _rq1_s7_MixedTruthCommonSMC,
    'NativeStateDiscretizer': _rq1_s7_NativeStateDiscretizer,
    'Path': _rq1_s7_Path,
    'Progress': _rq1_s7_Progress,
    'RQ15Probe': _rq1_s7_RQ15Probe,
    'SCHEMA_VERSION': _rq1_s7_SCHEMA_VERSION,
    'SCRIPT_07': _rq1_s7_SCRIPT_07,
    'SCRIPT_15': _rq1_s7_SCRIPT_15,
    'SCRIPT_16': _rq1_s7_SCRIPT_16,
    'SCRIPT_17': _rq1_s7_SCRIPT_17,
    'SCRIPT_18': _rq1_s7_SCRIPT_18,
    'SCRIPT_21': _rq1_s7_SCRIPT_21,
    'SCRIPT_21B': _rq1_s7_SCRIPT_21B,
    'Sequence': _rq1_s7_Sequence,
    '_gather': _rq1_s7__gather,
    '_resample_indices': _rq1_s7__resample_indices,
    'aggregate': _rq1_s7_aggregate,
    'argparse': _rq1_s7_argparse,
    'atomic_json': _rq1_s7_atomic_json,
    'contextlib': _rq1_s7_contextlib,
    'dataframe_sha256': _rq1_s7_dataframe_sha256,
    'dequantize': _rq1_s7_dequantize,
    'deterministic_seed': _rq1_s7_deterministic_seed,
    'discover_root': _rq1_s7_discover_root,
    'fit_common_transform': _rq1_s7_fit_common_transform,
    'fixed_native_edges': _rq1_s7_fixed_native_edges,
    'fmt_seconds': _rq1_s7_fmt_seconds,
    'graph_from_edges': _rq1_s7_graph_from_edges,
    'hashlib': _rq1_s7_hashlib,
    'importlib': _rq1_s7_importlib,
    'json': _rq1_s7_json,
    'json_default': _rq1_s7_json_default,
    'load_generator': _rq1_s7_load_generator,
    'logsumexp': _rq1_s7_logsumexp,
    'main': _rq1_s7_main,
    'math': _rq1_s7_math,
    'native_discretizer_patch': _rq1_s7_native_discretizer_patch,
    'np': _rq1_s7_np,
    'nx': _rq1_s7_nx,
    'os': _rq1_s7_os,
    'parse_args': _rq1_s7_parse_args,
    'pd': _rq1_s7_pd,
    'quadrant': _rq1_s7_quadrant,
    'quadrant_metrics': _rq1_s7_quadrant_metrics,
    'structure_metrics': _rq1_s7_structure_metrics,
    'sys': _rq1_s7_sys,
    'time': _rq1_s7_time,
    'traceback': _rq1_s7_traceback,
})


# ======================================================================
# Main benchmark runner
# ======================================================================
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence, Iterable
import networkx as nx
import numpy as np
import pandas as pd
SCHEMA_VERSION = 'rq1.5-native-mixed-learning-curves-v1'
DEFAULT_GENERATORS = ['unimodal', 'skew', 'multimodal', 'skew_multi']
DEFAULT_SEEDS = list(range(10))
DEFAULT_SAMPLE_SIZES = [100, 500, 2000, 5000]
BASE_PROBE_NAME = '01_rq1_5_native_mixed_probe.py'

def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    raise TypeError(type(value).__name__)

def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=json_default), encoding='utf-8')
    os.replace(tmp, path)

def fmt_seconds(seconds: float) -> str:
    if not np.isfinite(seconds):
        return '--'
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f'{minutes:02d}:{sec:02d}'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02d}:{sec:02d}'

def locate_base_probe(root: Path, this_file: Path) -> Path:
    return _rq1_Path('01_rq1_5_native_mixed_probe.py')

def discover_root(this_file: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    probes = [this_file.parent, *this_file.parents, Path.cwd(), *Path.cwd().resolve().parents]
    for p in probes:
        p = p.resolve()
        if (p / 'src' / 'hybrid_bn' / 'SLcontinuous.py').exists() and (p / 'src' / 'hybrid_bn' / 'SLdiscrete.py').exists() and (p / 'src' / 'hybrid_bn' / 'SLhybrid.py').exists() and (p / 'src' / 'generators').exists():
            return p
    raise FileNotFoundError('Could not discover RQ1 project root. Pass --project-root.')

def curve_run_dir(output_dir: Path, gen: str, seed: int, n: int) -> Path:
    return output_dir / 'runs' / gen / f'seed_{seed:02d}' / f'n_{n:04d}'

def run_prepared_cell(*, base: ModuleType, probe: Any, gen: str, seed: int, n: int, gt: Any, raw_base: pd.DataFrame, payload: Mapping[str, Any]) -> dict[str, Any]:
    probe.args.sample_size = int(n)
    run_dir = probe.run_dir(gen, seed)
    result_path = run_dir / 'result.json'
    raw_train = raw_base.iloc[:n].copy().reset_index(drop=True)
    current_hash = base.dataframe_sha256(raw_train)
    if result_path.exists() and (not probe.args.overwrite):
        saved = json.loads(result_path.read_text(encoding='utf-8'))
        saved_hash = saved.get('dataset_sha256')
        if saved_hash != current_hash:
            raise ValueError(f'Existing result hash mismatch for {gen} seed={seed} n={n}: saved={saved_hash}, current={current_hash}. Use --overwrite only after checking why the dataset changed.')
        print(f'    REUSE complete n={n}', flush=True)
        return saved
    nodes = list(map(str, payload['nodes']))
    dnodes = list(map(str, payload['discrete_nodes']))
    cnodes = list(map(str, payload['continuous_nodes']))
    cards_native = {str(k): int(v) for k, v in payload['cardinalities'].items()}
    cards_all = {node: cards_native[node] if node in cards_native else 3 for node in nodes}
    true_graph = payload['graph']
    fresh = gt.sample(probe.args.native_rows, sample_seed=base.deterministic_seed('rq1.5-native-fresh', gen, seed))[nodes].reset_index(drop=True)
    native_logp = np.asarray(gt.joint_log_prob(fresh), float)
    train_z = base.dequantize(raw_train, dnodes, generator=gen, seed=seed, purpose='train')
    fresh_z = base.dequantize(fresh, dnodes, generator=gen, seed=seed, purpose='fresh-native')
    common = base.fit_common_transform(training=raw_train, continuous_nodes=cnodes, discrete_nodes=dnodes, cards=cards_native, SavedTransform=probe.kl07.SavedQuantileTransform)
    dtrain = common.transform(raw_train.copy())
    print(f'    [n={n} | 1/6] Continuous structure', flush=True)
    cgraph, c_runtime = probe.fit_continuous_structure(train_z, nodes, true_graph)
    print(f'    [n={n} | 2/6] Discrete structure', flush=True)
    dgraph, d_runtime = probe.fit_discrete_structure(dtrain, nodes, cards_all)
    print(f'    [n={n} | 3/6] Hybrid structure', flush=True)
    hcell, hpath = probe.fit_hybrid_structure(raw_train=raw_train, payload=payload, gen=gen, seed=seed)
    hgraph = base.graph_from_edges(nodes, hcell['learned_edges'])
    h_runtime = float(hcell.get('runtime_seconds', float('nan')))
    qc = probe.c18.LearnedGaussianDAG(cgraph, nodes).fit(train_z)
    qd = probe.d15.SmoothedDiscreteDAG(dgraph, nodes, cards_all, alpha=probe.args.cpt_alpha).fit(dtrain)
    qh, native_h_transform, hgraph2, hd, hc = probe.base21.fit_hybrid_q(root=probe.root, hybrid_module=probe.h17, structure_cell=hcell, structure_path=hpath, raw_training=raw_train, payload=payload, metadata_score_tolerance=probe.args.metadata_score_tolerance, cpt_alpha=probe.args.cpt_alpha, min_regime_size=probe.args.min_regime_size, logistic_c=probe.args.logistic_c, seed=seed, n=n)
    if set(hgraph2.edges()) != set(hgraph.edges()):
        raise AssertionError('Hybrid Q graph mismatch after reconstruction.')
    print(f'    [n={n} | 4/6] Native KL', flush=True)
    logqc_native, _ = qc.joint_logpdf(fresh_z)
    logqh_native = np.asarray(qh.joint_logscore(fresh), float)

    def kl_stats(ratio: np.ndarray) -> dict[str, float]:
        ratio = np.asarray(ratio, float)
        mean = float(np.mean(ratio))
        se = float(np.std(ratio, ddof=1) / math.sqrt(len(ratio)))
        return {'estimate': mean, 'se': se, 'lower95': mean - 1.96 * se, 'upper95': mean + 1.96 * se}
    c_native = kl_stats(native_logp - np.asarray(logqc_native, float))
    h_native = kl_stats(native_logp - logqh_native)
    print(f'    [n={n} | 5/6] Common K3 / truth SMC', flush=True)
    common_fresh = gt.sample(probe.args.common_rows, sample_seed=base.deterministic_seed('rq1.5-common-fresh', gen, seed))[nodes].reset_index(drop=True)
    common_states = common.transform(common_fresh.copy())
    truth = base.MixedTruthCommonSMC(generator_obj=gt, transform=common, particles=probe.args.truth_particles, replicates=probe.args.truth_replicates, batch_size=probe.args.common_batch_size, seed=seed, generator_name=gen)
    logp_common, pdiag = truth.logmass(common_states)
    logqd, _ = qd.joint_logscore(common_fresh, common)
    d_native = kl_stats(np.asarray(logp_common) - np.asarray(logqd))
    cpush = probe.base21.ContinuousGaussianBinMassEstimator(graph=cgraph, models=qc.models, nodes=nodes, transform=common, particles=probe.args.common_particles, replicates=probe.args.common_replicates, batch_size=probe.args.common_batch_size, generator=gen, seed=seed)
    logqc_common, cdiag = cpush.logmass(common_states)
    hpush = probe.smc21b.SequentialHybridPushforward(base=probe.base21, hybrid17=probe.h17, q=qh, raw_training=raw_train, common_transform=common, native_transform=native_h_transform, continuous_nodes=hc, discrete_nodes=hd, nodes=nodes, generator=gen, seed=seed, outer_particles=probe.args.common_particles, cc_inner_samples=probe.args.common_cc_inner, replicates=probe.args.common_replicates, batch_size=probe.args.common_batch_size)
    logqh_common, hdiag = hpush.logmass(common_states)

    def paired(logqa: np.ndarray, logqb: np.ndarray) -> dict[str, float]:
        values = np.asarray(logqb, float) - np.asarray(logqa, float)
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        return {'estimate': mean, 'se': se, 'lower95': mean - 1.96 * se, 'upper95': mean + 1.96 * se}
    hd_diff = paired(logqh_common, logqd)
    cd_diff = paired(logqc_common, logqd)
    hc_diff = paired(logqh_common, logqc_common)
    h_common = d_native['estimate'] + hd_diff['estimate']
    c_common = d_native['estimate'] + cd_diff['estimate']
    print(f'    [n={n} | 6/6] Structural diagnostics', flush=True)
    model_graphs = {'Continuous': (cgraph, c_runtime), 'Discrete': (dgraph, d_runtime), 'Hybrid': (hgraph, h_runtime)}
    structure = {}
    quadrants = {}
    for model, (graph, runtime) in model_graphs.items():
        structure[model] = {**base.structure_metrics(true_graph.edges(), graph.edges()), 'Structure_Runtime_S': float(runtime)}
        quadrants[model] = base.quadrant_metrics(true_graph.edges(), graph.edges(), dnodes)
    flags = []
    if pdiag['p10_step_ess_fraction'] < 0.05:
        flags.append('LOW_TRUE_COMMON_SMC_ESS')
    if hdiag.get('p10_step_ess_fraction', 1.0) < 0.05:
        flags.append('LOW_HYBRID_COMMON_SMC_ESS')
    inner = hdiag.get('p10_cc_inner_ess_fraction', float('nan'))
    if np.isfinite(inner) and inner < 0.05:
        flags.append('LOW_HYBRID_CC_INNER_ESS')
    result = {'schema_version': SCHEMA_VERSION, 'status': 'complete', 'generator': gen, 'seed': int(seed), 'sample_size': int(n), 'ground_truth': {'native_mixed': True, 'nodes': nodes, 'continuous_nodes': cnodes, 'discrete_nodes': dnodes, 'cardinalities': cards_native, 'true_edges': sorted([[str(u), str(v)] for u, v in true_graph.edges()])}, 'representation': {'Continuous': 'native C + unit-width dequantised native D', 'Discrete': 'native D + training-quantile K3 native C', 'Hybrid': 'native mixed identity representation', 'Common': 'native D + training-quantile K3 native C'}, 'native_kl': {'Continuous': c_native, 'Discrete': d_native, 'Hybrid': h_native}, 'common_k3': {'Discrete_KL': d_native['estimate'], 'Hybrid_KL': h_common, 'Continuous_KL': c_common, 'Hybrid_Minus_Discrete': hd_diff, 'Continuous_Minus_Discrete': cd_diff, 'Hybrid_Minus_Continuous': hc_diff, 'True_SMC_Diagnostics': pdiag, 'Hybrid_SMC_Diagnostics': hdiag, 'Continuous_Pushforward_Diagnostics': cdiag}, 'structure': structure, 'quadrants': quadrants, 'flags': flags, 'dataset_sha256': current_hash, 'nested_prefix': True, 'max_dataset_size': int(len(raw_base))}
    atomic_json(result_path, result)
    return result

def maybe_import_legacy_result(*, base: ModuleType, output_dir: Path, legacy_dir: Path, gen: str, seed: int, raw_base: pd.DataFrame, overwrite: bool) -> bool:
    if overwrite:
        return False
    target = curve_run_dir(output_dir, gen, seed, 5000) / 'result.json'
    if target.exists():
        return False
    source_dir = legacy_dir / 'runs' / gen / f'seed_{seed:02d}'
    source = source_dir / 'result.json'
    if not source.exists():
        return False
    try:
        payload = json.loads(source.read_text(encoding='utf-8'))
    except Exception:
        return False
    if payload.get('status') != 'complete':
        return False
    if str(payload.get('generator')) != str(gen):
        return False
    if int(payload.get('seed', -1)) != int(seed):
        return False
    if int(payload.get('sample_size', -1)) != 5000:
        return False
    current_hash = base.dataframe_sha256(raw_base.iloc[:5000].reset_index(drop=True))
    if payload.get('dataset_sha256') != current_hash:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    imported = dict(payload)
    imported['imported_from_legacy_probe'] = str(source.resolve())
    imported['nested_prefix'] = True
    imported['max_dataset_size'] = int(len(raw_base))
    atomic_json(target, imported)
    for name in ('hybrid_structure.json', 'hybrid_worker_main_parity.csv'):
        src = source_dir / name
        dst = target.parent / name
        if src.exists() and (not dst.exists()):
            shutil.copy2(src, dst)
    return True

def worker_group(config: Mapping[str, Any], gen: str, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(config['root']).resolve()
    base_probe_path = Path(config['base_probe_path']).resolve()
    output_dir = Path(config['output_dir']).resolve()
    legacy_dir = Path(config['legacy_dir']).resolve()
    sample_sizes = list(map(int, config['sample_sizes']))
    max_n = max(sample_sizes)
    log_path = output_dir / 'logs' / f'{gen}_seed_{seed:02d}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, Any]] = []
    group_error = None
    with log_path.open('a', encoding='utf-8', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print('=' * 120, flush=True)
        print(f'GROUP START {gen} seed={seed} | pid={os.getpid()}', flush=True)
        print(f'sample sizes: {sample_sizes}', flush=True)
        try:
            base = _rq1_inline_load(base_probe_path, f'_rq15_base_{os.getpid()}_{gen}_{seed}')

            class CurveProbe(base.RQ15Probe):

                def run_dir(self, g: str, s: int) -> Path:
                    return self.out / 'runs' / g / f'seed_{s:02d}' / f'n_{int(self.args.sample_size):04d}'
            ns = argparse.Namespace(project_root=root, generators=list(config['generators']), seeds=list(map(int, config['seeds'])), sample_size=max_n, workers=int(config['hybrid_inner_workers']), tabu_length=int(config['tabu_length']), max_iter=int(config['max_iter']), max_parents=int(config['max_parents']), parity_tolerance=float(config['parity_tolerance']), cpt_alpha=float(config['cpt_alpha']), min_regime_size=int(config['min_regime_size']), logistic_c=float(config['logistic_c']), metadata_score_tolerance=float(config['metadata_score_tolerance']), native_rows=int(config['native_rows']), common_rows=int(config['common_rows']), truth_particles=int(config['truth_particles']), truth_replicates=int(config['truth_replicates']), common_particles=int(config['common_particles']), common_cc_inner=int(config['common_cc_inner']), common_replicates=int(config['common_replicates']), common_batch_size=int(config['common_batch_size']), output_dir=output_dir, overwrite=bool(config['overwrite']))
            probe = CurveProbe(root, ns)
            Generator = base.load_generator(root, gen)
            gt = Generator(num_samples=max_n, seed=seed)
            gt.generate_dag()
            raw_base = pd.DataFrame(gt.generate_data(), columns=gt.nodes).reset_index(drop=True)
            payload = gt.get_payload()
            nodes = list(map(str, payload['nodes']))
            raw_base = raw_base[nodes].copy().reset_index(drop=True)
            if 5000 in sample_sizes and max_n >= 5000 and bool(config['legacy_reuse']):
                reused = maybe_import_legacy_result(base=base, output_dir=output_dir, legacy_dir=legacy_dir, gen=gen, seed=seed, raw_base=raw_base, overwrite=bool(config['overwrite']))
                if reused:
                    print('Imported hash-matched legacy n=5000 result.', flush=True)
            for n in sample_sizes:
                cell_start = time.perf_counter()
                try:
                    result = run_prepared_cell(base=base, probe=probe, gen=gen, seed=seed, n=n, gt=gt, raw_base=raw_base, payload=payload)
                    statuses.append({'sample_size': int(n), 'status': 'complete', 'seconds': float(time.perf_counter() - cell_start), 'flags': list(result.get('flags', []))})
                except Exception as exc:
                    failure = {'schema_version': SCHEMA_VERSION, 'generator': gen, 'seed': int(seed), 'sample_size': int(n), 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
                    failure_dir = curve_run_dir(output_dir, gen, seed, n)
                    atomic_json(failure_dir / 'failure.json', failure)
                    statuses.append({'sample_size': int(n), 'status': 'failed', 'seconds': float(time.perf_counter() - cell_start), 'error': failure['error']})
                    print(f"FAILED n={n}: {failure['error']}", flush=True)
                    traceback.print_exc()
        except Exception as exc:
            group_error = f'{type(exc).__name__}: {exc}'
            print(f'GROUP FAILED: {group_error}', flush=True)
            traceback.print_exc()
        print(f'GROUP END {gen} seed={seed} | elapsed {fmt_seconds(time.perf_counter() - started)}', flush=True)
        print('=' * 120, flush=True)
    return {'generator': gen, 'seed': int(seed), 'seconds': float(time.perf_counter() - started), 'statuses': statuses, 'group_error': group_error, 'log': str(log_path)}

def aggregate(output_dir: Path):
    model_rows = []
    common_rows = []
    quadrant_rows = []
    failures = []
    for path in sorted(output_dir.glob('runs/*/seed_*/n_*/result.json')):
        try:
            r = json.loads(path.read_text(encoding='utf-8'))
            if r.get('status') != 'complete':
                continue
            gen = str(r['generator'])
            seed = int(r['seed'])
            n = int(r['sample_size'])
            for model in ('Continuous', 'Discrete', 'Hybrid'):
                native = r['native_kl'][model]
                s = r['structure'][model]
                common_kl = r['common_k3']['Discrete_KL'] if model == 'Discrete' else r['common_k3'][f'{model}_KL']
                model_rows.append({'Generator': gen, 'Seed': seed, 'Sample_Size': n, 'Model': model, 'Native_KL': native['estimate'], 'Native_KL_SE': native['se'], 'Common_KL_K3': common_kl, **s, 'Flags': '|'.join(r.get('flags', []))})
                for q in r['quadrants'][model]:
                    quadrant_rows.append({'Generator': gen, 'Seed': seed, 'Sample_Size': n, 'Model': model, **q})
            cm = r['common_k3']
            common_rows.append({'Generator': gen, 'Seed': seed, 'Sample_Size': n, 'Discrete_Common_KL': cm['Discrete_KL'], 'Hybrid_Common_KL': cm['Hybrid_KL'], 'Continuous_Common_KL': cm['Continuous_KL'], 'Hybrid_Minus_Discrete': cm['Hybrid_Minus_Discrete']['estimate'], 'Hybrid_Minus_Discrete_SE': cm['Hybrid_Minus_Discrete']['se'], 'Hybrid_Minus_Continuous': cm['Hybrid_Minus_Continuous']['estimate'], 'Hybrid_Minus_Continuous_SE': cm['Hybrid_Minus_Continuous']['se'], 'Continuous_Minus_Discrete': cm['Continuous_Minus_Discrete']['estimate'], 'Continuous_Minus_Discrete_SE': cm['Continuous_Minus_Discrete']['se'], 'True_p10_Step_ESS': cm['True_SMC_Diagnostics']['p10_step_ess_fraction'], 'Hybrid_p10_Step_ESS': cm['Hybrid_SMC_Diagnostics'].get('p10_step_ess_fraction', np.nan), 'Flags': '|'.join(r.get('flags', []))})
        except Exception as exc:
            failures.append({'path': str(path), 'error': f'{type(exc).__name__}: {exc}'})
    for path in sorted(output_dir.glob('runs/*/seed_*/n_*/failure.json')):
        try:
            f = json.loads(path.read_text(encoding='utf-8'))
            failures.append({'path': str(path), 'generator': f.get('generator'), 'seed': f.get('seed'), 'sample_size': f.get('sample_size'), 'error': f.get('error')})
        except Exception:
            pass
    m = pd.DataFrame(model_rows)
    c = pd.DataFrame(common_rows)
    q = pd.DataFrame(quadrant_rows)
    if len(m):
        m = m.sort_values(['Generator', 'Sample_Size', 'Seed', 'Model']).reset_index(drop=True)
    if len(c):
        c = c.sort_values(['Generator', 'Sample_Size', 'Seed']).reset_index(drop=True)
    if len(q):
        q = q.sort_values(['Generator', 'Sample_Size', 'Seed', 'Model', 'Quadrant']).reset_index(drop=True)
    m.to_csv(output_dir / 'mixed_model_results.csv', index=False)
    c.to_csv(output_dir / 'mixed_common_pairwise.csv', index=False)
    q.to_csv(output_dir / 'mixed_quadrant_diagnostics.csv', index=False)
    if len(m):
        ms = m.groupby(['Generator', 'Sample_Size', 'Model'], as_index=False).agg(Runs=('Seed', 'size'), Mean_Native_KL=('Native_KL', 'mean'), SD_Native_KL=('Native_KL', 'std'), Mean_Common_KL_K3=('Common_KL_K3', 'mean'), SD_Common_KL_K3=('Common_KL_K3', 'std'), Mean_SHD=('SHD', 'mean'), SD_SHD=('SHD', 'std'), Mean_SHD_Additions=('SHD_Additions', 'mean'), Mean_SHD_Deletions=('SHD_Deletions', 'mean'), Mean_SHD_Reversals=('SHD_Reversals', 'mean'), Mean_Directed_F1=('Directed_F1', 'mean'), Mean_Skeleton_F1=('Skeleton_F1', 'mean'), Mean_Structure_Runtime_S=('Structure_Runtime_S', 'mean')).sort_values(['Generator', 'Sample_Size', 'Model']).reset_index(drop=True)
    else:
        ms = pd.DataFrame()
    if len(c):
        cs = c.groupby(['Generator', 'Sample_Size'], as_index=False).agg(Runs=('Seed', 'size'), Mean_Discrete_Common_KL=('Discrete_Common_KL', 'mean'), Mean_Hybrid_Common_KL=('Hybrid_Common_KL', 'mean'), Mean_Continuous_Common_KL=('Continuous_Common_KL', 'mean'), Mean_Hybrid_Minus_Discrete=('Hybrid_Minus_Discrete', 'mean'), SD_Hybrid_Minus_Discrete=('Hybrid_Minus_Discrete', 'std'), Mean_Hybrid_Minus_Continuous=('Hybrid_Minus_Continuous', 'mean'), SD_Hybrid_Minus_Continuous=('Hybrid_Minus_Continuous', 'std'), Mean_Continuous_Minus_Discrete=('Continuous_Minus_Discrete', 'mean'), SD_Continuous_Minus_Discrete=('Continuous_Minus_Discrete', 'std'), Mean_True_p10_Step_ESS=('True_p10_Step_ESS', 'mean'), Mean_Hybrid_p10_Step_ESS=('Hybrid_p10_Step_ESS', 'mean')).sort_values(['Generator', 'Sample_Size']).reset_index(drop=True)
    else:
        cs = pd.DataFrame()
    if len(q):
        qs = q.groupby(['Generator', 'Sample_Size', 'Model', 'Quadrant'], as_index=False).agg(Runs=('Seed', 'size'), Mean_True_Edges=('True_Edges', 'mean'), Mean_Learned_Edges=('Learned_Edges', 'mean'), Mean_Precision=('Precision', 'mean'), Mean_Recall=('Recall', 'mean'), Mean_F1=('F1', 'mean'), Mean_Orientation_Accuracy=('Orientation_Accuracy_Given_Adjacency', 'mean')).sort_values(['Generator', 'Sample_Size', 'Model', 'Quadrant']).reset_index(drop=True)
    else:
        qs = pd.DataFrame()
    ms.to_csv(output_dir / 'mixed_model_summary.csv', index=False)
    cs.to_csv(output_dir / 'mixed_common_summary.csv', index=False)
    qs.to_csv(output_dir / 'mixed_quadrant_summary.csv', index=False)
    pd.DataFrame(failures).to_csv(output_dir / 'aggregation_failures.csv', index=False)
    lines = ['RQ1.5 NATIVE-MIXED LEARNING CURVES', '=' * 140, '', 'MODEL SUMMARY', '-' * 140, ms.to_string(index=False) if len(ms) else '(none)', '', 'COMMON K3 SUMMARY', '-' * 140, cs.to_string(index=False) if len(cs) else '(none)', '', 'QUADRANT SUMMARY', '-' * 140, qs.to_string(index=False) if len(qs) else '(none)', '', 'Contract: native mixed GT; nested training prefixes; Hybrid identity representation; Discrete bins only true C; Continuous dequantises only true D; ADD2 OFF; max_parents=3; max_iter=300.']
    (output_dir / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return (m, ms, c, cs, q, qs, failures)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=None)
    p.add_argument('--generators', nargs='+', choices=tuple(DEFAULT_GENERATORS), default=DEFAULT_GENERATORS)
    p.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--sample-sizes', nargs='+', type=int, default=DEFAULT_SAMPLE_SIZES)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--hybrid-inner-workers', type=int, default=1, help='Internal workers used by one Hybrid structure fit. Keep 1 when outer workers >1.')
    p.add_argument('--tabu-length', type=int, default=10)
    p.add_argument('--max-iter', type=int, default=300)
    p.add_argument('--max-parents', type=int, default=3)
    p.add_argument('--parity-tolerance', type=float, default=1e-08)
    p.add_argument('--cpt-alpha', type=float, default=1.0)
    p.add_argument('--min-regime-size', type=int, default=5)
    p.add_argument('--logistic-c', type=float, default=1000.0)
    p.add_argument('--metadata-score-tolerance', type=float, default=1e-06)
    p.add_argument('--native-rows', type=int, default=5000)
    p.add_argument('--common-rows', type=int, default=1000)
    p.add_argument('--truth-particles', type=int, default=128)
    p.add_argument('--truth-replicates', type=int, default=2)
    p.add_argument('--common-particles', type=int, default=128)
    p.add_argument('--common-cc-inner', type=int, default=16)
    p.add_argument('--common-replicates', type=int, default=2)
    p.add_argument('--common-batch-size', type=int, default=16)
    p.add_argument('--output-dir', type=Path, default=Path('MixedResults/RQ1_5_NativeMixed_LearningCurves'))
    p.add_argument('--legacy-results-dir', type=Path, default=Path('MixedResults/RQ1_5_NativeMixed_Probe'))
    p.add_argument('--no-legacy-reuse', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    return p.parse_args()

def main() -> int:
    args = parse_args()
    this_file = Path(__file__).resolve()
    root = discover_root(this_file, args.project_root)
    base_probe_path = locate_base_probe(root, this_file)
    args.generators = list(dict.fromkeys(map(str, args.generators)))
    args.seeds = sorted(set(map(int, args.seeds)))
    args.sample_sizes = sorted(set(map(int, args.sample_sizes)))
    if args.workers < 1:
        raise ValueError('--workers must be >=1.')
    if args.hybrid_inner_workers < 1:
        raise ValueError('--hybrid-inner-workers must be >=1.')
    if args.max_parents != 3:
        raise ValueError('RQ1.5 contract requires max_parents=3.')
    if args.max_iter != 300:
        raise ValueError('RQ1.5 contract requires max_iter=300.')
    if not args.sample_sizes or min(args.sample_sizes) < 30:
        raise ValueError('All sample sizes must be >=30.')
    if max(args.sample_sizes) > 5000:
        raise ValueError('RQ1.5 generators are capped at n=5000 for this benchmark.')
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (root / args.output_dir).resolve()
    legacy_dir = args.legacy_results_dir.resolve() if args.legacy_results_dir.is_absolute() else (root / args.legacy_results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = [(g, s) for g in args.generators for s in args.seeds]
    requested_cells = len(groups) * len(args.sample_sizes)
    config = {'root': str(root), 'base_probe_path': str(base_probe_path), 'output_dir': str(output_dir), 'legacy_dir': str(legacy_dir), 'legacy_reuse': not args.no_legacy_reuse, 'generators': args.generators, 'seeds': args.seeds, 'sample_sizes': args.sample_sizes, 'hybrid_inner_workers': args.hybrid_inner_workers, 'tabu_length': args.tabu_length, 'max_iter': args.max_iter, 'max_parents': args.max_parents, 'parity_tolerance': args.parity_tolerance, 'cpt_alpha': args.cpt_alpha, 'min_regime_size': args.min_regime_size, 'logistic_c': args.logistic_c, 'metadata_score_tolerance': args.metadata_score_tolerance, 'native_rows': args.native_rows, 'common_rows': args.common_rows, 'truth_particles': args.truth_particles, 'truth_replicates': args.truth_replicates, 'common_particles': args.common_particles, 'common_cc_inner': args.common_cc_inner, 'common_replicates': args.common_replicates, 'common_batch_size': args.common_batch_size, 'overwrite': args.overwrite}
    print('=' * 132)
    print('RQ1.5 NATIVE-MIXED LEARNING CURVES — FROZEN CONFIRMATION')
    print('=' * 132)
    print(f'Project root:          {root}')
    print(f'Base probe:            {base_probe_path}')
    print(f'Generators:            {args.generators}')
    print(f'Seeds:                 {args.seeds}')
    print(f'Sample sizes:          {args.sample_sizes}')
    print(f'Dataset cells:         {requested_cells}')
    print(f'Model fits max:        {requested_cells * 3}')
    print(f'Parallel groups:       {args.workers}')
    print(f'Hybrid inner workers:  {args.hybrid_inner_workers}')
    print(f'Legacy n=5000 reuse:   {not args.no_legacy_reuse}')
    print(f'Output:                 {output_dir}')
    print('Hybrid ADD2:           OFF — inherited hard contract')
    print('Nested data:           one max-N dataset per generator/seed, exact prefixes')
    print('=' * 132, flush=True)
    if args.preflight_only:
        base = _rq1_inline_load(base_probe_path, f'_rq15_preflight_{os.getpid()}')
        ns = argparse.Namespace(project_root=root, generators=args.generators, seeds=args.seeds, sample_size=max(args.sample_sizes), workers=args.hybrid_inner_workers, tabu_length=args.tabu_length, max_iter=args.max_iter, max_parents=args.max_parents, parity_tolerance=args.parity_tolerance, cpt_alpha=args.cpt_alpha, min_regime_size=args.min_regime_size, logistic_c=args.logistic_c, metadata_score_tolerance=args.metadata_score_tolerance, native_rows=args.native_rows, common_rows=args.common_rows, truth_particles=args.truth_particles, truth_replicates=args.truth_replicates, common_particles=args.common_particles, common_cc_inner=args.common_cc_inner, common_replicates=args.common_replicates, common_batch_size=args.common_batch_size, output_dir=output_dir, overwrite=args.overwrite)
        base.RQ15Probe(root, ns)
        print('PREFLIGHT PASSED: frozen RQ1.5 dependencies loaded successfully.')
        print(f'Planned groups: {len(groups)} | planned cells: {requested_cells}')
        return 0
    overall_start = time.perf_counter()
    group_results = []
    completed = 0
    group_times = []
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        future_map = {pool.submit(worker_group, config, gen, seed): (gen, seed) for gen, seed in groups}
        for fut in as_completed(future_map):
            gen, seed = future_map[fut]
            completed += 1
            try:
                result = fut.result()
            except Exception as exc:
                result = {'generator': gen, 'seed': seed, 'seconds': float('nan'), 'statuses': [], 'group_error': f'{type(exc).__name__}: {exc}', 'log': '(worker crashed before reporting log)'}
            group_results.append(result)
            if np.isfinite(result.get('seconds', float('nan'))):
                group_times.append(float(result['seconds']))
            done_cells = sum((1 for gr in group_results for st in gr.get('statuses', []) if st.get('status') == 'complete'))
            failed_cells = sum((1 for gr in group_results for st in gr.get('statuses', []) if st.get('status') == 'failed'))
            mean_group = float(np.mean(group_times)) if group_times else float('nan')
            remaining_groups = len(groups) - completed
            eta = mean_group * remaining_groups / max(1, args.workers) if np.isfinite(mean_group) else float('nan')
            status = 'OK' if not result.get('group_error') else 'GROUP-ERROR'
            print(f"[{completed:02d}/{len(groups):02d}] {gen:<11} seed={seed:02d} {status:<11} | complete cells={done_cells}/{requested_cells} failed={failed_cells} | group={fmt_seconds(result.get('seconds', float('nan')))} | ETA~{fmt_seconds(eta)}", flush=True)
            if result.get('group_error'):
                print(f"    {result['group_error']} | log: {result.get('log')}", flush=True)
            aggregate(output_dir)
    m, _, _, _, _, _, aggregation_failures = aggregate(output_dir)
    completed_cells = 0
    if len(m):
        completed_cells = int(m[['Generator', 'Seed', 'Sample_Size']].drop_duplicates().shape[0])
    explicit_failed = sum((1 for gr in group_results for st in gr.get('statuses', []) if st.get('status') == 'failed'))
    group_errors = [gr for gr in group_results if gr.get('group_error')]
    manifest = {'schema_version': SCHEMA_VERSION, 'project_root': str(root), 'base_probe_path': str(base_probe_path), 'generators': args.generators, 'seeds': args.seeds, 'sample_sizes': args.sample_sizes, 'requested_cells': requested_cells, 'completed_cells': completed_cells, 'explicit_failed_cells': explicit_failed, 'group_errors': group_errors, 'aggregation_failures': aggregation_failures, 'workers': args.workers, 'hybrid_inner_workers': args.hybrid_inner_workers, 'legacy_reuse': not args.no_legacy_reuse, 'elapsed_seconds': float(time.perf_counter() - overall_start), 'settings': config}
    atomic_json(output_dir / 'manifest.json', manifest)
    print('\n' + '=' * 132)
    print('RQ1.5 LEARNING-CURVE RUN COMPLETE')
    print('=' * 132)
    print(f'Completed cells:      {completed_cells}/{requested_cells}')
    print(f'Explicit cell fails:  {explicit_failed}')
    print(f'Group errors:         {len(group_errors)}')
    print(f'Elapsed:              {fmt_seconds(time.perf_counter() - overall_start)}')
    print(f"Model results:        {output_dir / 'mixed_model_results.csv'}")
    print(f"Model summary:        {output_dir / 'mixed_model_summary.csv'}")
    print(f"Common summary:       {output_dir / 'mixed_common_summary.csv'}")
    print(f"Quadrant summary:     {output_dir / 'mixed_quadrant_summary.csv'}")
    print(f"Manifest:             {output_dir / 'manifest.json'}")
    print(f"Worker logs:          {output_dir / 'logs'}")
    print('=' * 132)
    return 1 if explicit_failed or group_errors or completed_cells != requested_cells else 0
if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
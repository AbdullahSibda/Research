
"""
Hybrid Bayesian Network Structure Learning.

Purpose
-------
Implements a Bayesian network structure-learning framework for datasets
containing both continuous and discrete variables. The framework preserves
native variable types and supports all four parent-child relationships:
continuous-to-continuous (CC), continuous-to-discrete (CD),
discrete-to-continuous (DC), and discrete-to-discrete (DD).

Scoring Framework
-----------------
1. Continuous children use flexible conditional-density models combining
   nonlinear regression, Bayesian model averaging, skew-normal residuals,
   and Gaussian mixture alternatives. Conjugate marginal likelihoods and
   Laplace approximations are used according to the local model.

2. Discrete children with continuous parents use multinomial logistic
   regression with Laplace-approximated marginal likelihoods. Multiple
   feature representations accommodate linear and nonlinear dependencies.

3. Discrete children with exclusively discrete parents use
   Dirichlet-multinomial marginal likelihoods.

4. Continuous children with discrete parents use conditional regression
   and mixture models fitted within discrete-parent configurations.

Structure Learning
------------------
Local scores are combined into a decomposable network objective.
Tabu-constrained greedy hill climbing searches over directed acyclic
graphs using edge additions, deletions, and reversals, subject to
acyclicity and maximum-parent constraints.

Implementation
--------------
The continuous-family scoring machinery is implemented in this file.
Additional scoring components, preprocessing utilities, and graph-search
algorithms are provided by the accompanying HybridV3_CDv3Candidate
package.

The public HybridScoringEngine interface provides access to the
integrated scoring framework.

This module is intended to be imported by experimental pipelines.
Input datasets must be numerically encoded and preprocessed before
structure learning.
"""

from __future__ import annotations

from builtins import ImportError
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.linalg import helmert
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, fmin_l_bfgs_b, minimize
from scipy.signal import find_peaks
from scipy.special import erfc, gammaln, log_ndtr, logsumexp

try:
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    from pytensor.compile.mode import get_mode
    from pytensor.gradient import grad, jacobian
except Exception as exc:
    raise ImportError(
        "Final V3 C->C scorer requires PyMC and PyTensor for the validated M2 backend.\n"
        "Install with: python -m pip install -U pymc pytensor\n"
        f"Original error: {type(exc).__name__}: {exc}"
    ) from exc

                                               
_dataclass = dataclass
_math = math
_time = time
_minimize = minimize
_logsumexp = logsumexp

                                                                       
SOURCE_PROVENANCE = {'mean': {'sha256': 'ce8e91c95b0c8c7c2ee3622bf0126bb99f35a210de2cf75c33578ee5d1da6186', 'bytes': 52208}, 'base': {'sha256': '38faf4ec1c6148e576d7334e41421d63811d4e70dd34bae894f7f808d82e7eac', 'bytes': 130288}, 'skew_a': {'sha256': 'f2881cabc9488b2e79e783a9ce1a7862275a147dea11336874dae5bdaae56383', 'bytes': 43940}, 'skew_c': {'sha256': 'ee6957782a1eecfe8cc76ce87f273bd83eef2f0fe35b2dadeedcfe4d0a93ada4', 'bytes': 37767}, 'skew_d': {'sha256': 'e7b26b5002ed12eaa7da6abecbdf43b4509ec128fa5f7148987f3922d364601c', 'bytes': 39754}, 'skew_e': {'sha256': '262e4d0bd672857938cea945151403d922c33ddccd1e2998b81efd1921b8c42c', 'bytes': 50647}, 'a2': {'sha256': 'e68c97da3b66dd05381532ed8a72f77f325c2d83fa33bb30ecb9256d5cc781cc', 'bytes': 56424}, 'partb': {'sha256': 'ca086949c9c38b13aa0683016f27753aff10284ca7b0adf74cb38d299b9092df', 'bytes': 54894}, 'partd': {'sha256': '8fa629dfc63afb0cbd0e87d4ec336e98afe02c61d7081539573c3de7e9bf7b53', 'bytes': 56108}, 'pt': {'sha256': 'e21a3836d2b8f4f43a118aee3e26a47dbee5fbeb9cfc18eae5ed56b35a9d760c', 'bytes': 16693}}


                                          
_mean_EPS = 1e-12
_mean_T_GRID: Tuple[float, ...] = tuple((round(float(value), 2) for value in np.arange(0.05, 0.951, 0.01)))
_mean_K_T = len(_mean_T_GRID)
_mean_STATE_L = 0
_mean_STATE_C = 1
_mean_N_STATES = _mean_K_T + 2

@dataclass(frozen=True)
class _mean_NIGPrior:
    intercept_var: float = 4.0
    coeff_var: float = 1.0
    ig_shape: float = 1.5
    ig_scale: float = 0.5

@dataclass
class _mean_StateFit:
    state: Tuple[int, ...]
    columns: Tuple[int, ...]
    beta_mean: np.ndarray
    sigma2_mean: float
    log_evidence: float
    condition_number: float
    jitter: float

@dataclass(frozen=True)
class _mean_LocalScoreDiagnostic:
    child: str
    parents: Tuple[str, ...]
    parent_count: int
    method: str
    score: float
    state_space_size: int
    scored_state_count: int
    runtime_seconds: float
    chib_forward_log_evidence: float = float('nan')
    chib_reverse_log_evidence: float = float('nan')
    chib_order_gap: float = float('nan')
    reference_state: str = ''
    reference_sample_weight: float = float('nan')
    beam_width: int = 0
    beam_max_sweeps: int = 0
    beam_sweeps_completed: int = 0
    beam_converged: bool = False
    beam_visited_state_count: int = 0
    beam_visited_state_fraction: float = float('nan')
    beam_final_logz_increment: float = float('nan')

def _mean_state_q(state: int) -> float:
    if state < 2:
        return float('nan')
    return float(_mean_T_GRID[state - 2])

def _mean_state_label(state: int) -> str:
    if state == _mean_STATE_L:
        return 'L'
    if state == _mean_STATE_C:
        return 'C'
    return f'T{int(round(100.0 * _mean_state_q(state))):02d}'

def _mean_state_tuple_label(state: Tuple[int, ...]) -> str:
    if not state:
        return 'root'
    return '|'.join((_mean_state_label(value) for value in state))

def _mean_log_state_prior(state: int) -> float:
    if state == _mean_STATE_L:
        return math.log(1.0 / 3.0)
    if state == _mean_STATE_C:
        return math.log(1.0 / 3.0)
    return math.log(1.0 / 3.0) - math.log(_mean_K_T)

def _mean_joint_state_log_prior(state: Tuple[int, ...]) -> float:
    return float(sum((_mean_log_state_prior(value) for value in state)))

class _mean__FeatureBank:


















    BLOCK_SIZE = 1 + 3 + _mean_K_T
    LINEAR_INDEX = 0
    C_INDICATOR_INDEX = 1
    C_HINGE25_INDEX = 2
    C_HINGE75_INDEX = 3
    T_START_INDEX = 4

    def __init__(self, standardized_data: pd.DataFrame, nodes: Sequence[str]) -> None:
        self.data = standardized_data
        self.nodes = tuple((str(node) for node in nodes))
        self.n = len(standardized_data)
        self._node_index = {node: index for index, node in enumerate(self.nodes)}
        self._y_matrix = standardized_data[list(self.nodes)].to_numpy(dtype=float)
        self._built = False
        self._global_features: np.ndarray | None = None
        self._global_gram: np.ndarray | None = None
        self._global_cross_y: np.ndarray | None = None
        self._metadata: Dict[str, Dict[str, object]] = {}
        self._block_slices: Dict[str, slice] = {}
        self._block_sums: Dict[str, np.ndarray] = {}
        self._y_sums = {node: float(self._y_matrix[:, index].sum()) for node, index in self._node_index.items()}
        self._yty = {node: float(self._y_matrix[:, index] @ self._y_matrix[:, index]) for node, index in self._node_index.items()}
        self.block_build_count = 0
        self.block_gram_build_count = 0
        self.global_gram_build_count = 0

    def _make_block(self, node: str) -> Tuple[np.ndarray, Dict[str, object]]:
        x = self.data[node].to_numpy(dtype=float)
        levels = np.asarray([0.25, 0.5, 0.75, *_mean_T_GRID], dtype=float)
        quantiles = np.quantile(x, levels)
        q25 = float(quantiles[0])
        q50 = float(quantiles[1])
        q75 = float(quantiles[2])
        t_knots = np.asarray(quantiles[3:], dtype=float)
        columns: List[np.ndarray] = [x]
        feature_meta: List[Tuple[str, float, float, float]] = [('L', float('nan'), 0.0, 1.0)]

        def standardise(raw: np.ndarray, *, kind: str, knot: float) -> np.ndarray:
            mean = float(raw.mean())
            sd = float(raw.std(ddof=0))
            if not np.isfinite(sd) or sd <= _mean_EPS:
                raise ValueError(f'Degenerate {kind} feature for parent {node!r}.')
            feature_meta.append((kind, float(knot), mean, sd))
            return (raw - mean) / sd
        columns.append(standardise((x > q50).astype(float), kind='C_indicator', knot=q50))
        columns.append(standardise(np.maximum(0.0, x - q25), kind='C_hinge25', knot=q25))
        columns.append(standardise(np.maximum(0.0, x - q75), kind='C_hinge75', knot=q75))
        for q_index, knot in enumerate(t_knots):
            columns.append(standardise((x > float(knot)).astype(float), kind=f'T{int(round(100 * _mean_T_GRID[q_index])):02d}', knot=float(knot)))
        block = np.column_stack(columns)
        if block.shape != (self.n, self.BLOCK_SIZE):
            raise RuntimeError(f'Unexpected feature-block shape for {node!r}: {block.shape}.')
        metadata = {'q25': q25, 'q50': q50, 'q75': q75, 't_knots': t_knots, 'feature_meta': tuple(feature_meta)}
        return (block, metadata)

    def _ensure_global(self) -> None:
        if self._built:
            return
        blocks: List[np.ndarray] = []
        offset = 0
        for node in self.nodes:
            block, metadata = self._make_block(node)
            blocks.append(block)
            self._metadata[node] = metadata
            self._block_slices[node] = slice(offset, offset + self.BLOCK_SIZE)
            self._block_sums[node] = block.sum(axis=0)
            offset += self.BLOCK_SIZE
            self.block_build_count += 1
        features = np.column_stack(blocks)
        self._global_features = features
        self._global_gram = features.T @ features
        self._global_cross_y = features.T @ self._y_matrix
        self.global_gram_build_count = 1
        self.block_gram_build_count = 1
        self._built = True

    def metadata(self, node: str) -> Dict[str, object]:
        self._ensure_global()
        return self._metadata[node]

    def block_slice(self, node: str) -> slice:
        self._ensure_global()
        return self._block_slices[node]

    def block_sum(self, node: str) -> np.ndarray:
        self._ensure_global()
        return self._block_sums[node]

    def cross_y(self, parent: str, child: str) -> np.ndarray:
        self._ensure_global()
        sl = self._block_slices[parent]
        return self._global_cross_y[sl, self._node_index[child]]

    def y_sum(self, child: str) -> float:
        return self._y_sums[child]

    def yty(self, child: str) -> float:
        return self._yty[child]

    def block_gram(self, left: str, right: str) -> np.ndarray:
        self._ensure_global()
        left_sl = self._block_slices[left]
        right_sl = self._block_slices[right]
        return self._global_gram[left_sl, right_sl]

    def local_sufficient_statistics(self, *, child: str, parents: Tuple[str, ...], coefficient_prior_variance: float, intercept_prior_variance: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, List[int], Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, ...]]]:

        if not parents:
            return (np.asarray([[float(self.n)]], dtype=float), np.asarray([self.y_sum(child)], dtype=float), self.yty(child), np.asarray([float(intercept_prior_variance)], dtype=float), [0], {}, {})
        self._ensure_global()
        p = len(parents)
        block_size = self.BLOCK_SIZE
        total_columns = 1 + p * block_size
        gram = np.empty((total_columns, total_columns), dtype=float)
        cross_y = np.empty(total_columns, dtype=float)
        prior_vars = np.full(total_columns, float(coefficient_prior_variance), dtype=float)
        prior_vars[0] = float(intercept_prior_variance)
        gram[0, 0] = float(self.n)
        cross_y[0] = self.y_sum(child)
        for j, parent in enumerate(parents):
            sl = slice(1 + j * block_size, 1 + (j + 1) * block_size)
            sums = self.block_sum(parent)
            gram[0, sl] = sums
            gram[sl, 0] = sums
            cross_y[sl] = self.cross_y(parent, child)
        for j, left in enumerate(parents):
            left_sl = slice(1 + j * block_size, 1 + (j + 1) * block_size)
            for k in range(j, p):
                right = parents[k]
                right_sl = slice(1 + k * block_size, 1 + (k + 1) * block_size)
                block_cross = self.block_gram(left, right)
                gram[left_sl, right_sl] = block_cross
                if j == k:
                    gram[right_sl, left_sl] = block_cross
                else:
                    gram[right_sl, left_sl] = block_cross.T
        base_columns: List[int] = [0]
        c_columns: Dict[int, Tuple[int, int, int]] = {}
        t_columns: Dict[int, Tuple[int, ...]] = {}
        for j in range(p):
            start = 1 + j * block_size
            base_columns.append(start + self.LINEAR_INDEX)
            c_columns[j] = (start + self.C_INDICATOR_INDEX, start + self.C_HINGE25_INDEX, start + self.C_HINGE75_INDEX)
            t_columns[j] = tuple((start + self.T_START_INDEX + q_index for q_index in range(_mean_K_T)))
        return (gram, cross_y, self.yty(child), prior_vars, base_columns, c_columns, t_columns)

    def evaluation_block(self, node: str, standardized_eval: pd.DataFrame) -> np.ndarray:

        metadata = self.metadata(node)
        x = standardized_eval[node].to_numpy(dtype=float)
        m = len(x)
        block = np.empty((m, self.BLOCK_SIZE), dtype=float)
        block[:, self.LINEAR_INDEX] = x
        feature_meta = metadata['feature_meta']
        _, knot, mean, sd = feature_meta[self.C_INDICATOR_INDEX]
        block[:, self.C_INDICATOR_INDEX] = ((x > knot).astype(float) - mean) / sd
        _, knot, mean, sd = feature_meta[self.C_HINGE25_INDEX]
        block[:, self.C_HINGE25_INDEX] = (np.maximum(0.0, x - knot) - mean) / sd
        _, knot, mean, sd = feature_meta[self.C_HINGE75_INDEX]
        block[:, self.C_HINGE75_INDEX] = (np.maximum(0.0, x - knot) - mean) / sd
        for q_index in range(_mean_K_T):
            column = self.T_START_INDEX + q_index
            _, knot, mean, sd = feature_meta[column]
            block[:, column] = ((x > knot).astype(float) - mean) / sd
        return block

class _mean__LocalFamily:


    def __init__(self, *, child: str, parents: Tuple[str, ...], feature_bank: _mean__FeatureBank, prior: _mean_NIGPrior) -> None:
        self.child = child
        self.parents = parents
        self.p = len(parents)
        self.prior = prior
        self.bank = feature_bank
        self.n = feature_bank.n
        self.gram, self.cross_y, self.yty, self.prior_vars, self.base_columns, self.c_columns, self.t_columns = feature_bank.local_sufficient_statistics(child=child, parents=parents, coefficient_prior_variance=prior.coeff_var, intercept_prior_variance=prior.intercept_var)
        self.log_evidence_cache: Dict[Tuple[int, ...], float] = {}
        self.fit_cache: Dict[Tuple[int, ...], _mean_StateFit] = {}
        self.conditional_logweight_cache: Dict[Tuple[int, Tuple[int, ...]], np.ndarray] = {}

    def columns_for_state(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(state) != self.p:
            raise ValueError('State dimension does not match the parent count.')
        columns = list(self.base_columns)
        for j, value in enumerate(state):
            if value == _mean_STATE_L:
                continue
            if value == _mean_STATE_C:
                columns.extend(self.c_columns[j])
            else:
                columns.append(self.t_columns[j][value - 2])
        return tuple(columns)

    def _batch_log_evidence_for_columns(self, column_matrix: np.ndarray) -> np.ndarray:

        column_matrix = np.asarray(column_matrix, dtype=np.int64)
        if column_matrix.ndim != 2 or column_matrix.shape[0] == 0:
            raise ValueError('column_matrix must be a non-empty 2D array.')
        m, d = column_matrix.shape
        XtX = self.gram[column_matrix[:, :, None], column_matrix[:, None, :]]
        Xty = self.cross_y[column_matrix]
        variances = self.prior_vars[column_matrix]
        posterior_precision = XtX.copy()
        diag = np.arange(d)
        posterior_precision[:, diag, diag] += 1.0 / variances
        sign, logdet_A = np.linalg.slogdet(posterior_precision)
        if np.any(sign <= 0) or not np.all(np.isfinite(logdet_A)):
            results = np.empty(m, dtype=float)
            for row in range(m):
                cols = tuple((int(value) for value in column_matrix[row]))
                results[row] = self._single_log_evidence_for_columns(cols)
            return results
        beta = np.linalg.solve(posterior_precision, Xty[..., None])[..., 0]
        quad = self.yty - np.einsum('bi,bi->b', beta, Xty)
        a0 = float(self.prior.ig_shape)
        b0 = float(self.prior.ig_scale)
        an = a0 + 0.5 * self.n
        bn = b0 + 0.5 * quad
        bad = ~np.isfinite(bn) | (bn <= 0.0)
        if np.any(bad):
            if np.any(bn[bad] <= -1e-10):
                raise FloatingPointError(f'Invalid posterior IG scale for {self.child!r}.')
            bn = bn.copy()
            bn[bad] = 1e-12
        logdet_V0 = np.sum(np.log(variances), axis=1)
        logdet_Vn = -logdet_A
        result = -0.5 * self.n * math.log(2.0 * math.pi) + 0.5 * (logdet_Vn - logdet_V0) + a0 * math.log(b0) - an * np.log(bn) + float(gammaln(an) - gammaln(a0))
        return np.asarray(result, dtype=float)

    def _single_log_evidence_for_columns(self, columns: Tuple[int, ...]) -> float:
        idx = np.asarray(columns, dtype=int)
        XtX = self.gram[np.ix_(idx, idx)]
        Xty = self.cross_y[idx]
        variances = self.prior_vars[idx]
        posterior_precision = np.diag(1.0 / variances) + XtX
        sign, logdet_A = np.linalg.slogdet(posterior_precision)
        if sign <= 0 or not np.isfinite(logdet_A):
            jitter = 1e-10 * max(1.0, float(np.max(np.diag(posterior_precision))))
            posterior_precision = posterior_precision + jitter * np.eye(len(columns))
            sign, logdet_A = np.linalg.slogdet(posterior_precision)
            if sign <= 0 or not np.isfinite(logdet_A):
                raise FloatingPointError(f'Invalid posterior precision for {self.child!r}.')
        beta = np.linalg.solve(posterior_precision, Xty)
        quad = float(self.yty - beta @ Xty)
        a0 = float(self.prior.ig_shape)
        b0 = float(self.prior.ig_scale)
        an = a0 + 0.5 * self.n
        bn = b0 + 0.5 * quad
        if not np.isfinite(bn) or bn <= 0.0:
            if bn > -1e-10:
                bn = 1e-12
            else:
                raise FloatingPointError(f'Invalid posterior IG scale for {self.child!r}: {bn}.')
        return float(-0.5 * self.n * math.log(2.0 * math.pi) + 0.5 * (-float(logdet_A) - float(np.sum(np.log(variances)))) + a0 * math.log(b0) - an * math.log(bn) + float(gammaln(an) - gammaln(a0)))

    def log_evidence_many(self, states: Sequence[Tuple[int, ...]]) -> np.ndarray:
        normalized = [tuple((int(value) for value in state)) for state in states]
        if not normalized:
            return np.empty(0, dtype=float)
        result = np.empty(len(normalized), dtype=float)
        missing_by_width: Dict[int, List[Tuple[int, Tuple[int, ...], Tuple[int, ...]]]] = {}
        for index, state in enumerate(normalized):
            cached = self.log_evidence_cache.get(state)
            if cached is not None:
                result[index] = cached
                continue
            columns = self.columns_for_state(state)
            missing_by_width.setdefault(len(columns), []).append((index, state, columns))
        for items in missing_by_width.values():
            column_matrix = np.asarray([columns for _, _, columns in items], dtype=np.int64)
            values = self._batch_log_evidence_for_columns(column_matrix)
            for (index, state, _), value in zip(items, values):
                value = float(value)
                self.log_evidence_cache[state] = value
                result[index] = value
        return result

    def log_evidence(self, state: Tuple[int, ...]) -> float:
        state = tuple((int(value) for value in state))
        cached = self.log_evidence_cache.get(state)
        if cached is not None:
            return float(cached)
        return float(self.log_evidence_many([state])[0])

    def fit_state(self, state: Tuple[int, ...]) -> _mean_StateFit:
        state = tuple((int(value) for value in state))
        cached = self.fit_cache.get(state)
        if cached is not None:
            return cached
        columns = self.columns_for_state(state)
        idx = np.asarray(columns, dtype=int)
        XtX = self.gram[np.ix_(idx, idx)]
        Xty = self.cross_y[idx]
        variances = self.prior_vars[idx]
        posterior_precision = np.diag(1.0 / variances) + XtX
        jitter = 0.0
        try:
            np.linalg.cholesky(posterior_precision)
        except np.linalg.LinAlgError:
            jitter = 1e-10 * max(1.0, float(np.max(np.diag(posterior_precision))))
            posterior_precision = posterior_precision + jitter * np.eye(len(columns))
            np.linalg.cholesky(posterior_precision)
        beta_mean = np.linalg.solve(posterior_precision, Xty)
        a0 = float(self.prior.ig_shape)
        b0 = float(self.prior.ig_scale)
        an = a0 + 0.5 * self.n
        quad = float(self.yty - beta_mean @ Xty)
        bn = b0 + 0.5 * quad
        if not np.isfinite(bn) or bn <= 0.0:
            if bn > -1e-10:
                bn = 1e-12
            else:
                raise FloatingPointError(f'Invalid posterior IG scale for {self.child!r}: {bn}.')
        if an <= 1.0:
            raise FloatingPointError('Posterior mean sigma^2 does not exist.')
        sigma2_mean = float(bn / (an - 1.0))
        fit = _mean_StateFit(state=state, columns=columns, beta_mean=beta_mean, sigma2_mean=sigma2_mean, log_evidence=self.log_evidence(state), condition_number=float(np.linalg.cond(posterior_precision)), jitter=float(jitter))
        self.fit_cache[state] = fit
        return fit

    def evaluation_master(self, standardized_eval: pd.DataFrame) -> np.ndarray:
        m = len(standardized_eval)
        if not self.parents:
            return np.ones((m, 1), dtype=float)
        blocks = [self.bank.evaluation_block(parent, standardized_eval) for parent in self.parents]
        return np.column_stack([np.ones(m, dtype=float), *blocks])

    def state_logpdf_standardized(self, state: Tuple[int, ...], standardized_eval: pd.DataFrame, master_eval: np.ndarray | None=None) -> np.ndarray:
        fit = self.fit_state(state)
        if master_eval is None:
            master_eval = self.evaluation_master(standardized_eval)
        X = master_eval[:, np.asarray(fit.columns, dtype=int)]
        mu = X @ fit.beta_mean
        y = standardized_eval[self.child].to_numpy(dtype=float)
        sigma2 = max(float(fit.sigma2_mean), 1e-15)
        return -0.5 * (math.log(2.0 * math.pi * sigma2) + (y - mu) ** 2 / sigma2)

class _mean_V3CScoringEngine:


    def __init__(self, data: pd.DataFrame, *, coefficient_prior_variance: float=1.0, intercept_prior_variance: float=4.0, inverse_gamma_shape: float=1.5, inverse_gamma_scale: float=0.5, beam_width: int=32, beam_max_sweeps: int=4, gibbs_chains: int=4, gibbs_burn_sweeps: int=30, gibbs_sample_sweeps: int=100) -> None:
        if data.empty:
            raise ValueError('V3CScoringEngine requires non-empty data.')
        if data.isna().any().any():
            raise ValueError('Training data contains missing values.')
        self.nodes = tuple((str(column) for column in data.columns))
        raw = data.copy().reset_index(drop=True)
        raw.columns = self.nodes
        self.raw_data = raw
        self.n = len(raw)
        self.prior = _mean_NIGPrior(intercept_var=float(intercept_prior_variance), coeff_var=float(coefficient_prior_variance), ig_shape=float(inverse_gamma_shape), ig_scale=float(inverse_gamma_scale))
        if any((value <= 0.0 for value in asdict(self.prior).values())):
            raise ValueError('All NIG prior hyperparameters must be positive.')
        self.beam_cfg = {'width': int(beam_width), 'max_sweeps': int(beam_max_sweeps)}
        if self.beam_cfg['width'] < 2 or self.beam_cfg['max_sweeps'] < 1:
            raise ValueError('Invalid Part-H beam configuration.')
        self._legacy_gibbs_kwargs = (int(gibbs_chains), int(gibbs_burn_sweeps), int(gibbs_sample_sweeps))
        self.means = {node: float(raw[node].to_numpy(dtype=float).mean()) for node in self.nodes}
        self.stds = {}
        standardized = raw.copy()
        for node in self.nodes:
            values = raw[node].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f'Column {node!r} contains non-finite values.')
            sd = float(values.std(ddof=0))
            if sd <= _mean_EPS:
                sd = 1.0
            self.stds[node] = sd
            standardized[node] = (values - self.means[node]) / sd
        self.data = standardized
        self.feature_bank = _mean__FeatureBank(self.data, self.nodes)
        digest = hashlib.sha256()
        digest.update('|'.join(self.nodes).encode('utf-8'))
        digest.update(np.ascontiguousarray(standardized.to_numpy(dtype=np.float64)).tobytes())
        self.data_fingerprint = digest.hexdigest()
        self.cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}
        self.local_families: Dict[Tuple[str, Tuple[str, ...]], _mean__LocalFamily] = {}
        self.diagnostics: Dict[Tuple[str, Tuple[str, ...]], _mean_LocalScoreDiagnostic] = {}
        self.posterior_state_logweights: Dict[Tuple[str, Tuple[str, ...]], Tuple[List[Tuple[int, ...]], np.ndarray]] = {}

    def prior_hyperparameters(self) -> Dict[str, float]:
        return {'continuous_coefficient_prior_variance': self.prior.coeff_var, 'continuous_intercept_prior_variance': self.prior.intercept_var, 'continuous_inverse_gamma_shape': self.prior.ig_shape, 'continuous_inverse_gamma_scale': self.prior.ig_scale, 'branch_prior_L': 1.0 / 3.0, 'branch_prior_T': 1.0 / 3.0, 'branch_prior_C': 1.0 / 3.0, 'threshold_count': _mean_K_T, 'p3_beam_width': self.beam_cfg['width'], 'p3_beam_max_sweeps': self.beam_cfg['max_sweeps']}

    def preprocessing_metadata(self) -> Dict[str, object]:
        return {'continuous_means': dict(self.means), 'continuous_stds': dict(self.stds), 'data_fingerprint_sha256': self.data_fingerprint, 'feature_bank_block_build_count': self.feature_bank.block_build_count, 'feature_bank_block_gram_build_count': self.feature_bank.block_gram_build_count, 'feature_bank_global_gram_build_count': self.feature_bank.global_gram_build_count}

    def _validate_family(self, child: str, parents: Sequence[str]) -> Tuple[str, ...]:
        child = str(child)
        if child not in self.nodes:
            raise KeyError(f'Unknown child {child!r}.')
        if child in parents:
            raise ValueError('A node cannot be its own parent.')
        if len(set(parents)) != len(parents):
            raise ValueError('Duplicate parents are not allowed.')
        normalized = tuple(sorted((str(parent) for parent in parents)))
        unknown = [parent for parent in normalized if parent not in self.nodes]
        if unknown:
            raise KeyError(f'Unknown parents: {unknown}')
        if len(normalized) > 3:
            raise ValueError('This validated V3 C-only scorer supports at most 3 parents.')
        return normalized

    def _family(self, child: str, parents: Tuple[str, ...]) -> _mean__LocalFamily:
        key = (child, parents)
        family = self.local_families.get(key)
        if family is None:
            family = _mean__LocalFamily(child=child, parents=parents, feature_bank=self.feature_bank, prior=self.prior)
            self.local_families[key] = family
        return family

    @staticmethod
    def _joint_logscore_many(family: _mean__LocalFamily, states: Sequence[Tuple[int, ...]]) -> np.ndarray:
        evidences = family.log_evidence_many(states)
        priors = np.asarray([_mean_joint_state_log_prior(state) for state in states], dtype=float)
        return evidences + priors

    def _best_independent_threshold_states(self, family: _mean__LocalFamily) -> Tuple[int, ...]:

        best: List[int] = []
        p = family.p
        for j in range(p):
            states: List[Tuple[int, ...]] = []
            t_ids: List[int] = []
            for q_index in range(_mean_K_T):
                t_state = 2 + q_index
                state = [_mean_STATE_L] * p
                state[j] = t_state
                states.append(tuple(state))
                t_ids.append(t_state)
            values = self._joint_logscore_many(family, states)
            best.append(int(t_ids[int(np.argmax(values))]))
        return tuple(best)

    def _beam_seed_states(self, family: _mean__LocalFamily) -> List[Tuple[int, ...]]:
        p = family.p
        t50 = 2 + int(np.argmin(np.abs(np.asarray(_mean_T_GRID, dtype=float) - 0.5)))
        best_t = self._best_independent_threshold_states(family)
        seeds = {tuple([_mean_STATE_L] * p), tuple([_mean_STATE_C] * p), tuple([t50] * p), tuple(best_t)}
        for j in range(p):
            c_state = [_mean_STATE_L] * p
            c_state[j] = _mean_STATE_C
            seeds.add(tuple(c_state))
            t_state = [_mean_STATE_L] * p
            t_state[j] = int(best_t[j])
            seeds.add(tuple(t_state))
        return sorted(seeds)

    @staticmethod
    def _expand_beam(beam: Sequence[Tuple[int, ...]], p: int) -> List[Tuple[int, ...]]:
        candidates = set(beam)
        for state in beam:
            for j in range(p):
                for new_state_j in range(_mean_N_STATES):
                    if new_state_j == state[j]:
                        continue
                    candidate = list(state)
                    candidate[j] = int(new_state_j)
                    candidates.add(tuple(candidate))
        return sorted(candidates)

    @staticmethod
    def _top_beam(states: Sequence[Tuple[int, ...]], logjoints: np.ndarray, width: int) -> List[Tuple[int, ...]]:
        if len(states) <= width:
            order = np.argsort(-logjoints)
        else:
            idx = np.argpartition(-logjoints, width - 1)[:width]
            order = idx[np.argsort(-logjoints[idx])]
        return [states[int(index)] for index in order[:width]]

    def _score_joint_beam(self, child: str, parents: Tuple[str, ...], family: _mean__LocalFamily) -> Tuple[float, _mean_LocalScoreDiagnostic]:

        started = time.perf_counter()
        width = self.beam_cfg['width']
        max_sweeps = self.beam_cfg['max_sweeps']
        seeds = self._beam_seed_states(family)
        seed_values = self._joint_logscore_many(family, seeds)
        visited_logjoint: Dict[Tuple[int, ...], float] = {state: float(value) for state, value in zip(seeds, seed_values)}
        beam = self._top_beam(seeds, seed_values, width)
        previous_logz = float(logsumexp(np.asarray(list(visited_logjoint.values()), dtype=float)))
        converged = False
        sweeps_completed = 0
        final_increment = 0.0
        for sweep in range(1, max_sweeps + 1):
            sweeps_completed = sweep
            candidates = self._expand_beam(beam, family.p)
            unseen = [state for state in candidates if state not in visited_logjoint]
            if unseen:
                values = self._joint_logscore_many(family, unseen)
                for state, value in zip(unseen, values):
                    visited_logjoint[state] = float(value)
            candidate_values = np.asarray([visited_logjoint[state] for state in candidates], dtype=float)
            new_beam = self._top_beam(candidates, candidate_values, width)
            current_logz = float(logsumexp(np.asarray(list(visited_logjoint.values()), dtype=float)))
            final_increment = float(current_logz - previous_logz)
            beam_unchanged = set(new_beam) == set(beam)
            beam = new_beam
            previous_logz = current_logz
            if beam_unchanged:
                converged = True
                break
        states = list(visited_logjoint.keys())
        log_joint = np.asarray([visited_logjoint[state] for state in states], dtype=float)
        score = float(logsumexp(log_joint))
        log_weights = log_joint - score
        self.posterior_state_logweights[child, parents] = (states, log_weights)
        map_index = int(np.argmax(log_joint))
        map_state = states[map_index]
        map_weight = float(math.exp(log_weights[map_index]))
        diagnostic = _mean_LocalScoreDiagnostic(child=child, parents=parents, parent_count=family.p, method=f'joint_beam_B{width}_S{max_sweeps}', score=score, state_space_size=int(_mean_N_STATES ** family.p), scored_state_count=len(family.log_evidence_cache), runtime_seconds=time.perf_counter() - started, reference_state=_mean_state_tuple_label(map_state), reference_sample_weight=map_weight, beam_width=int(width), beam_max_sweeps=int(max_sweeps), beam_sweeps_completed=int(sweeps_completed), beam_converged=bool(converged), beam_visited_state_count=int(len(states)), beam_visited_state_fraction=float(len(states) / _mean_N_STATES ** family.p), beam_final_logz_increment=float(final_increment))
        return (score, diagnostic)

    def score_local(self, child: str, parents: Sequence[str]) -> float:
        child = str(child)
        parents = self._validate_family(child, parents)
        key = (child, parents)
        if key in self.cache:
            return self.cache[key]
        started = time.perf_counter()
        family = self._family(child, parents)
        p = len(parents)
        if p == 0:
            state: Tuple[int, ...] = ()
            score = family.log_evidence(state)
            diagnostic = _mean_LocalScoreDiagnostic(child=child, parents=parents, parent_count=0, method='exact_nig_root', score=float(score), state_space_size=1, scored_state_count=1, runtime_seconds=time.perf_counter() - started)
        elif p <= 2:
            states = [tuple((int(value) for value in state)) for state in product(range(_mean_N_STATES), repeat=p)]
            evidences = family.log_evidence_many(states)
            log_priors = np.asarray([_mean_joint_state_log_prior(state) for state in states], dtype=float)
            score = float(logsumexp(evidences + log_priors))
            diagnostic = _mean_LocalScoreDiagnostic(child=child, parents=parents, parent_count=p, method=f'exact_bma_{_mean_N_STATES ** p}_states', score=score, state_space_size=int(_mean_N_STATES ** p), scored_state_count=len(family.log_evidence_cache), runtime_seconds=time.perf_counter() - started)
        else:
            score, diagnostic = self._score_joint_beam(child, parents, family)
        if not np.isfinite(score):
            raise FloatingPointError(f'Non-finite V3 local score for {child!r} | {parents}.')
        self.cache[key] = float(score)
        self.diagnostics[key] = diagnostic
        return float(score)

    def get_score_diagnostic(self, child: str, parents: Sequence[str]) -> _mean_LocalScoreDiagnostic:
        parents = self._validate_family(child, parents)
        key = (str(child), parents)
        if key not in self.diagnostics:
            self.score_local(child, parents)
        return self.diagnostics[key]

    def diagnostics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame((asdict(value) for value in self.diagnostics.values()))

    def true_or_learned_dag_score(self, graph: nx.DiGraph) -> float:
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError('The supplied graph is not a DAG.')
        if set(graph.nodes) != set(self.nodes):
            raise ValueError('Graph nodes do not match scorer columns.')
        return float(sum((self.score_local(node, list(graph.predecessors(node))) for node in self.nodes)))

    def _standardize_eval(self, raw_eval: pd.DataFrame) -> pd.DataFrame:
        if set(raw_eval.columns) != set(self.nodes):
            raise ValueError('Evaluation columns do not match training columns.')
        frame = raw_eval[list(self.nodes)].copy().reset_index(drop=True)
        for node in self.nodes:
            values = frame[node].to_numpy(dtype=float)
            frame[node] = (values - self.means[node]) / self.stds[node]
        return frame

    def _exact_posterior_states(self, family: _mean__LocalFamily) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
        p = family.p
        states = [tuple((int(value) for value in state)) for state in product(range(_mean_N_STATES), repeat=p)]
        evidences = family.log_evidence_many(states)
        log_joint = evidences + np.asarray([_mean_joint_state_log_prior(state) for state in states], dtype=float)
        log_weights = log_joint - logsumexp(log_joint)
        return (states, log_weights)

    def local_predictive_logpdf(self, child: str, parents: Sequence[str], raw_eval: pd.DataFrame) -> np.ndarray:
        child = str(child)
        parents = self._validate_family(child, parents)
        self.score_local(child, parents)
        standardized_eval = self._standardize_eval(raw_eval)
        family = self._family(child, parents)
        master_eval = family.evaluation_master(standardized_eval)
        p = len(parents)
        if p == 0:
            standardized_logq = family.state_logpdf_standardized((), standardized_eval, master_eval)
        elif p <= 2:
            states, log_weights = self._exact_posterior_states(family)
            standardized_logq = np.full(len(standardized_eval), -np.inf, dtype=float)
            for state, log_weight in zip(states, log_weights):
                term = float(log_weight) + family.state_logpdf_standardized(state, standardized_eval, master_eval)
                standardized_logq = np.logaddexp(standardized_logq, term)
        else:
            key = (child, parents)
            posterior = self.posterior_state_logweights.get(key)
            if posterior is None:
                _, _ = self._score_joint_beam(child, parents, family)
                posterior = self.posterior_state_logweights[key]
            states, log_weights = posterior
            standardized_logq = np.full(len(standardized_eval), -np.inf, dtype=float)
            for state, log_weight in zip(states, log_weights):
                term = float(log_weight) + family.state_logpdf_standardized(state, standardized_eval, master_eval)
                standardized_logq = np.logaddexp(standardized_logq, term)
        return standardized_logq - math.log(self.stds[child])

    def joint_logpdf(self, graph: nx.DiGraph, raw_eval: pd.DataFrame, *, return_nodewise: bool=False):
        if set(graph.nodes) != set(self.nodes):
            raise ValueError('Graph nodes do not match scorer columns.')
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError('Evaluation graph is not a DAG.')
        total = np.zeros(len(raw_eval), dtype=float)
        nodewise: Dict[str, np.ndarray] = {}
        for child in nx.topological_sort(graph):
            values = self.local_predictive_logpdf(str(child), list(graph.predecessors(child)), raw_eval)
            if values.shape != (len(raw_eval),) or not np.all(np.isfinite(values)):
                raise FloatingPointError(f'Invalid V3 predictive density for {child!r}.')
            nodewise[str(child)] = values
            total += values
        if return_nodewise:
            return (total, nodewise)
        return total

                                                                           
_ska_LOG_2PI = math.log(2.0 * math.pi)
_ska_LOG2 = math.log(2.0)
_ska_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_ska_TWO_OVER_PI = 2.0 / math.pi

@dataclass(frozen=True)
class _ska_NIGPrior:
    intercept_var: float = 4.0
    coeff_var: float = 1.0
    ig_shape: float = 1.5
    ig_scale: float = 0.5

@dataclass(frozen=True)
class _ska_GaussianAnchor:
    logz: float
    beta_mean: np.ndarray
    sigma2_mean: float
    sigma2_mode: float
    an: float
    bn: float

def _ska_stable_log_cosh(x: float) -> float:
    ax = abs(float(x))
    return ax + math.log1p(math.exp(-2.0 * ax)) - math.log(2.0)

def _ska_delta_to_alpha(delta: np.ndarray | float) -> np.ndarray | float:
    d = np.asarray(delta, dtype=float)
    clipped = np.clip(d, -1.0 + 1e-14, 1.0 - 1e-14)
    value = clipped / np.sqrt(1.0 - clipped * clipped)
    if np.ndim(delta) == 0:
        return float(value)
    return value

def _ska_centered_skew_constants_from_delta(delta: np.ndarray | float) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(delta, dtype=float)
    mean_z = _ska_SQRT_2_OVER_PI * d
    var_z = 1.0 - _ska_TWO_OVER_PI * d * d
    sd_z = np.sqrt(np.maximum(var_z, 1e-14))
    return (mean_z, sd_z)

def _ska_centered_skew_logpdf_standard(u: np.ndarray, delta: float) -> np.ndarray:

    d = float(np.clip(delta, -1.0 + 1e-13, 1.0 - 1e-13))
    alpha = float(_ska_delta_to_alpha(d))
    mean_z, sd_z = _ska_centered_skew_constants_from_delta(d)
    z = float(sd_z) * np.asarray(u, dtype=float) + float(mean_z)
    return _ska_LOG2 + math.log(float(sd_z)) - 0.5 * (_ska_LOG_2PI + z * z) + log_ndtr(alpha * z)

def _ska_centered_skew_logpdf(y: np.ndarray, mean: np.ndarray, sigma: float, delta: float) -> np.ndarray:
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.full_like(np.asarray(y, dtype=float), -np.inf)
    u = (np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)) / sigma
    return _ska_centered_skew_logpdf_standard(u, delta) - math.log(sigma)

def _ska_log_joint_reference_theta(theta: np.ndarray, y: np.ndarray, X: np.ndarray, prior: _ska_NIGPrior) -> float:











    theta = np.asarray(theta, dtype=float)
    p = X.shape[1]
    beta = theta[:p]
    log_sigma2 = float(theta[p])
    kappa = float(theta[p + 1])
    if not np.all(np.isfinite(beta)) or not np.isfinite(log_sigma2) or (not np.isfinite(kappa)) or (log_sigma2 < -20.0) or (log_sigma2 > 20.0) or (abs(kappa) > 8.0):
        return -np.inf
    sigma2 = math.exp(log_sigma2)
    sigma = math.sqrt(sigma2)
    delta = math.tanh(kappa)
    mean = X @ beta
    ll_values = _ska_centered_skew_logpdf(y, mean, sigma, delta)
    if not np.all(np.isfinite(ll_values)):
        return -np.inf
    log_likelihood = float(np.sum(ll_values))
    prior_vars = np.full(p, prior.coeff_var, dtype=float)
    prior_vars[0] = prior.intercept_var
    quad_beta = float(np.sum(beta * beta / prior_vars))
    logdet_v0 = float(np.sum(np.log(prior_vars)))
    log_beta_prior = -0.5 * p * _ska_LOG_2PI - 0.5 * p * log_sigma2 - 0.5 * logdet_v0 - 0.5 * quad_beta / sigma2
    a0 = float(prior.ig_shape)
    b0 = float(prior.ig_scale)
    log_sigma2_prior = a0 * math.log(b0) - gammaln(a0) - (a0 + 1.0) * log_sigma2 - b0 / sigma2
    log_delta_prior_and_jacobian = -_ska_LOG2 - 2.0 * _ska_stable_log_cosh(kappa)
    log_sigma_transform_jacobian = log_sigma2
    return float(log_likelihood + log_beta_prior + log_sigma2_prior + log_delta_prior_and_jacobian + log_sigma_transform_jacobian)

def _ska_finite_difference_hessian(objective, x: np.ndarray, relative_step: float=0.0002) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    d = len(x)
    H = np.empty((d, d), dtype=float)
    f0 = float(objective(x))
    steps = relative_step * (1.0 + np.abs(x))
    for i in range(d):
        hi = steps[i]
        xp = x.copy()
        xm = x.copy()
        xp[i] += hi
        xm[i] -= hi
        H[i, i] = (float(objective(xp)) - 2.0 * f0 + float(objective(xm))) / (hi * hi)
        for j in range(i + 1, d):
            hj = steps[j]
            xpp = x.copy()
            xpm = x.copy()
            xmp = x.copy()
            xmm = x.copy()
            xpp[i] += hi
            xpp[j] += hj
            xpm[i] += hi
            xpm[j] -= hj
            xmp[i] -= hi
            xmp[j] += hj
            xmm[i] -= hi
            xmm[j] -= hj
            value = (float(objective(xpp)) - float(objective(xpm)) - float(objective(xmp)) + float(objective(xmm))) / (4.0 * hi * hj)
            H[i, j] = value
            H[j, i] = value
    return 0.5 * (H + H.T)

                                     
def _skc_zero_start(anchor: _ska_GaussianAnchor) -> np.ndarray:
    return np.concatenate([np.asarray(anchor.beta_mean, dtype=float), [math.log(max(anchor.sigma2_mean, 1e-12)), 0.0]])

def _skc_fast_start(anchor: _ska_GaussianAnchor, fast: Dict[str, object]) -> np.ndarray:
    delta = float(np.clip(float(fast['delta_map']), -0.999999, 0.999999))
    kappa = float(np.arctanh(delta))
    return np.concatenate([np.asarray(anchor.beta_mean, dtype=float), [math.log(max(anchor.sigma2_mean, 1e-12)), kappa]])

                                                   
_ske_PARAMETER_DUPLICATE_TOL = 0.0001

def _ske_strict_same_mode(zero_mode: Dict[str, object], fast_mode: Dict[str, object]) -> bool:
    a = np.asarray(zero_mode['theta_hat'], dtype=float)
    b = np.asarray(fast_mode['theta_hat'], dtype=float)
    return bool(np.max(np.abs(a - b)) <= _ske_PARAMETER_DUPLICATE_TOL)

def _ske_pooled_mahalanobis_distance(first: Dict[str, object], second: Dict[str, object]) -> float:

    theta_a = np.asarray(first['theta_hat'], dtype=float)
    theta_b = np.asarray(second['theta_hat'], dtype=float)
    diff = theta_a - theta_b
    cov_a = np.asarray(first['covariance'], dtype=float)
    cov_b = np.asarray(second['covariance'], dtype=float)
    pooled_cov = 0.5 * (cov_a + cov_b)
    precision = np.linalg.pinv(pooled_cov, rcond=1e-12, hermitian=True)
    d2 = float(diff @ precision @ diff)
    return math.sqrt(max(d2, 0.0))

def _ske_mode_predictive_logpdf(*, y: np.ndarray, X: np.ndarray, mode: Dict[str, object]) -> np.ndarray:
    mean = X @ np.asarray(mode['beta_hat'], dtype=float)
    return _ska_centered_skew_logpdf(y, mean, float(mode['sigma_hat']), float(mode['delta_hat']))

def _ske_multimode_predictive_logpdf(*, y: np.ndarray, X: np.ndarray, result: Dict[str, object]) -> np.ndarray:
    modes = result['predictive_modes']
    log_weights = np.asarray(result['predictive_log_weights'], dtype=float)
    terms = []
    for mode, log_weight in zip(modes, log_weights):
        terms.append(_ske_mode_predictive_logpdf(y=y, X=X, mode=mode) + float(log_weight))
    return logsumexp(np.vstack(terms), axis=0)

                                                    
_a2__LOG_2PI = math.log(2.0 * math.pi)
_a2__LOG2 = math.log(2.0)
_a2__SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_a2__TWO_OVER_PI = 2.0 / math.pi

def _a2__delta_to_alpha(delta: float) -> float:
    d = float(np.clip(delta, -1.0 + 1e-14, 1.0 - 1e-14))
    return d / math.sqrt(1.0 - d * d)

def _a2_centered_skew_logpdf_standard(u: np.ndarray, delta: float) -> np.ndarray:
    d = float(np.clip(delta, -1.0 + 1e-13, 1.0 - 1e-13))
    alpha = _a2__delta_to_alpha(d)
    mean_z = _a2__SQRT_2_OVER_PI * d
    sd_z = math.sqrt(max(1.0 - _a2__TWO_OVER_PI * d * d, 1e-14))
    z = sd_z * np.asarray(u, dtype=float) + mean_z
    return _a2__LOG2 + math.log(sd_z) - 0.5 * (_a2__LOG_2PI + z * z) + log_ndtr(alpha * z)

def _a2_centered_skew_logpdf_and_grad_standard(u: np.ndarray, kappa: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    u = np.asarray(u, dtype=float)
    kappa = float(kappa)
    delta = math.tanh(kappa)
    delta_prime = 1.0 - delta * delta
    alpha = math.sinh(kappa)
    alpha_prime = math.cosh(kappa)
    mean_z = _a2__SQRT_2_OVER_PI * delta
    sd_z = math.sqrt(max(1.0 - _a2__TWO_OVER_PI * delta * delta, 1e-14))
    sd_prime = -(_a2__TWO_OVER_PI * delta * delta_prime) / sd_z
    mean_prime = _a2__SQRT_2_OVER_PI * delta_prime
    z = sd_z * u + mean_z
    z_prime = sd_prime * u + mean_prime
    t = alpha * z
    log_phi_t = -0.5 * (_a2__LOG_2PI + t * t)
    log_Phi_t = log_ndtr(t)
    inv_mills = np.exp(log_phi_t - log_Phi_t)
    logf = _a2__LOG2 + math.log(sd_z) - 0.5 * (_a2__LOG_2PI + z * z) + log_Phi_t
    dlogf_du = sd_z * (-z + alpha * inv_mills)
    t_prime = alpha_prime * z + alpha * z_prime
    dlogf_dkappa = sd_prime / sd_z - z * z_prime + inv_mills * t_prime
    return (logf, dlogf_du, dlogf_dkappa)

def _a2__loglik_and_grad_skew(theta: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    y = np.asarray(y, dtype=float)
    log_sigma, kappa = map(float, theta)
    sigma = math.exp(log_sigma)
    u = y / sigma
    logf, dlog_du, dlog_dkappa = _a2_centered_skew_logpdf_and_grad_standard(u, kappa)
    ll = float(np.sum(logf) - len(y) * log_sigma)
    grad = np.asarray([float(np.sum(-1.0 - u * dlog_du)), float(np.sum(dlog_dkappa))], dtype=float)
    return (ll, grad)

def _a2__repairing_kmeans_1d(y: np.ndarray, k: int, max_iter: int=100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    centers = np.quantile(y, (np.arange(k) + 0.5) / k).astype(float)
    centers.sort()

    def lloyd(current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        current = np.asarray(current, dtype=float).copy()
        for _ in range(int(max_iter)):
            labels = np.argmin(np.abs(y[:, None] - current[None, :]), axis=1)
            updated = current.copy()
            for j in range(k):
                if np.any(labels == j):
                    updated[j] = float(np.mean(y[labels == j]))
            updated.sort()
            if np.allclose(updated, current, atol=1e-10, rtol=0.0):
                current = updated
                break
            current = updated
        labels = np.argmin(np.abs(y[:, None] - current[None, :]), axis=1)
        return (current, labels)
    centers, labels = lloyd(centers)
    counts = np.bincount(labels, minlength=k).astype(float)
    if np.all(counts > 0.0):
        return (centers, counts / counts.sum(), labels)
    occupied = [j for j in range(k) if counts[j] > 0.0]
    empty = [j for j in range(k) if np.isclose(counts[j], 0.0, rtol=1e-09, atol=1e-09)]
    repaired = centers.copy()
    for empty_j in empty:
        donor = max(occupied, key=lambda j: float(np.sum((y[labels == j] - centers[j]) ** 2)))
        values = np.sort(y[labels == donor])
        if len(values) < 2:
            raise FloatingPointError('Cannot repair empty deterministic k-means cluster.')
        split = len(values) // 2
        repaired[donor] = float(np.mean(values[:split]))
        repaired[empty_j] = float(np.mean(values[split:]))
        occupied.append(empty_j)
        repaired.sort()
        repaired, labels = lloyd(repaired)
        counts = np.bincount(labels, minlength=k).astype(float)
        if np.all(counts > 0.0):
            break
    if np.any(np.isclose(counts, 0.0, rtol=1e-09, atol=1e-09)):
        raise FloatingPointError(f'K={k} k-means still has an empty cluster after repair.')
    return (repaired, counts / counts.sum(), labels)

                                                       
_pb__LOG_2PI = math.log(2.0 * math.pi)
_pb__LOG2 = math.log(2.0)
_pb_IG_SHAPE = 1.5
_pb_IG_SCALE = 0.5
_pb_DIRICHLET_ALPHA = 2.0
_pb_LOG_SEPARATION_MEAN = math.log(2.5)
_pb_LOG_SEPARATION_SD = 1.0
_pb_LOG_SIGMA_REF_BOUNDS = (-5.0, 2.5)
_pb_SCALE_CONTRAST_BOUNDS = (-3.0, 3.0)
_pb_KAPPA_BOUNDS = (-6.0, 6.0)
_pb_LOGIT_BOUNDS = (-7.0, 7.0)
_pb_LOG_SEPARATION_RATIO_BOUNDS = (math.log(0.1), math.log(25.0))
_pb_HESSIAN_RELATIVE_STEP = 0.0002
_pb_HESSIAN_MIN_EIGENVALUE = 1e-08
_pb_PARAMETER_DUPLICATE_TOL = 0.0001
_pb_POSTERIOR_MODE_SEPARATION = 3.0
_pb_MAX_ITER = 500

@dataclass(frozen=True)
class _pb_PriorProfile:
    name: str
    scale_contrast_sd: float
    ig_shape: float = _pb_IG_SHAPE
    ig_scale: float = _pb_IG_SCALE
    dirichlet_alpha: float = _pb_DIRICHLET_ALPHA
    log_separation_mean: float = _pb_LOG_SEPARATION_MEAN
    log_separation_sd: float = _pb_LOG_SEPARATION_SD

@dataclass
class _pb_OptimizedMode:
    model: str
    start_name: str
    theta: np.ndarray
    log_joint: float
    iterations: int
    gradient_inf_norm: float
    runtime_s: float
    success: bool
    message: str

@dataclass
class _pb_LaplaceMode:
    optimized: _pb_OptimizedMode
    hessian: np.ndarray
    covariance: np.ndarray
    min_eigenvalue: float
    logdet_hessian: float
    logz: float
    hessian_runtime_s: float

@dataclass
class _pb_EvidenceFit:
    model: str
    logz: float
    dominant_theta: np.ndarray
    dominant_start: str
    classification: str
    separation: float
    retained_modes: int
    candidate_modes: int
    min_effective_count: float
    runtime_s: float
    map_runtime_s: float
    hessian_runtime_s: float

def _pb__helmert_basis(k: int) -> np.ndarray:

    return np.asarray(helmert(int(k), full=False), dtype=float).T

def _pb__softmax_baseline(logits: np.ndarray) -> np.ndarray:
    values = np.concatenate([np.asarray(logits, dtype=float), [0.0]])
    values -= np.max(values)
    values = np.exp(values)
    return values / values.sum()

def _pb__mixture_dimension(k: int) -> int:
    return int(1 + (k - 1) + 1 + (k - 1) + (k - 1))

def _pb__decode_mixture(theta: np.ndarray, k: int) -> Dict[str, Any]:
    theta = np.asarray(theta, dtype=float)
    expected = _pb__mixture_dimension(k)
    if theta.shape != (expected,):
        raise ValueError(f'M{k} expects theta length {expected}; got {theta.shape}.')
    cursor = 0
    log_sigma_ref = float(theta[cursor])
    cursor += 1
    gamma = np.asarray(theta[cursor:cursor + k - 1], dtype=float)
    cursor += k - 1
    kappa = float(theta[cursor])
    cursor += 1
    logits = np.asarray(theta[cursor:cursor + k - 1], dtype=float)
    cursor += k - 1
    eta = np.asarray(theta[cursor:cursor + k - 1], dtype=float)
    H = _pb__helmert_basis(k)
    log_sigmas = log_sigma_ref + H @ gamma
    sigmas = np.exp(log_sigmas)
    sigma_ref = math.exp(log_sigma_ref)
    delta = math.tanh(kappa)
    weights = _pb__softmax_baseline(logits)
    ratios = np.exp(eta)
    b = np.concatenate([[0.0], np.cumsum(ratios)])
    bbar = float(np.dot(weights, b))
    offsets = b - bbar
    locations = sigma_ref * offsets
    return {'log_sigma_ref': log_sigma_ref, 'sigma_ref': sigma_ref, 'H': H, 'gamma': gamma, 'log_sigmas': log_sigmas, 'sigmas': sigmas, 'kappa': kappa, 'delta': delta, 'weights': weights, 'logits': logits, 'eta': eta, 'ratios': ratios, 'b': b, 'bbar': bbar, 'offsets': offsets, 'locations': locations}

def _pb__mixture_bounds(k: int) -> List[Tuple[float, float]]:
    return [_pb_LOG_SIGMA_REF_BOUNDS] + [_pb_SCALE_CONTRAST_BOUNDS] * (k - 1) + [_pb_KAPPA_BOUNDS] + [_pb_LOGIT_BOUNDS] * (k - 1) + [_pb_LOG_SEPARATION_RATIO_BOUNDS] * (k - 1)

def _pb__stable_log_cosh(x: float) -> float:
    ax = abs(float(x))
    return ax + math.log1p(math.exp(-2.0 * ax)) - _pb__LOG2

def _pb__log_ref_scale_prior(log_sigma_ref: float, a0: float, b0: float) -> Tuple[float, float]:

    sigma2 = math.exp(2.0 * float(log_sigma_ref))
    value = a0 * math.log(b0) - gammaln(a0) - (a0 + 1.0) * math.log(sigma2) - b0 / sigma2 + _pb__LOG2 + math.log(sigma2)
    grad = -2.0 * a0 + 2.0 * b0 / sigma2
    return (float(value), float(grad))

def _pb__log_delta_prior_kappa(kappa: float) -> Tuple[float, float]:
    delta = math.tanh(float(kappa))
    value = -_pb__LOG2 - 2.0 * _pb__stable_log_cosh(kappa)
    grad = -2.0 * delta
    return (float(value), float(grad))

def _pb__mixture_log_prior_and_gradient(theta: np.ndarray, k: int, prior: _pb_PriorProfile) -> Tuple[float, np.ndarray]:
    d = _pb__decode_mixture(theta, k)
    grad = np.zeros_like(theta, dtype=float)
    cursor = 0
    lp_scale, g_scale = _pb__log_ref_scale_prior(d['log_sigma_ref'], prior.ig_shape, prior.ig_scale)
    grad[cursor] = g_scale
    cursor += 1
    gamma = np.asarray(d['gamma'], dtype=float)
    tau = float(prior.scale_contrast_sd)
    lp_gamma = float(np.sum(-0.5 * (_pb__LOG_2PI + 2.0 * math.log(tau) + (gamma / tau) ** 2)))
    grad[cursor:cursor + k - 1] = -gamma / (tau * tau)
    cursor += k - 1
    lp_delta, g_delta = _pb__log_delta_prior_kappa(float(d['kappa']))
    grad[cursor] = g_delta
    cursor += 1
    weights = np.asarray(d['weights'], dtype=float)
    alpha = float(prior.dirichlet_alpha)
    lp_weights = float(gammaln(k * alpha) - k * gammaln(alpha) + alpha * np.sum(np.log(weights)))
    grad[cursor:cursor + k - 1] = alpha * (1.0 - k * weights[:-1])
    cursor += k - 1
    eta = np.asarray(d['eta'], dtype=float)
    sd = float(prior.log_separation_sd)
    z = (eta - float(prior.log_separation_mean)) / sd
    lp_eta = float(np.sum(-0.5 * (_pb__LOG_2PI + 2.0 * math.log(sd) + z * z)))
    grad[cursor:cursor + k - 1] = -(eta - float(prior.log_separation_mean)) / (sd * sd)
    value = float(lp_scale + lp_gamma + lp_delta + lp_weights + lp_eta)
    return (value, grad)

def _pb__mixture_loglik_and_gradient(theta: np.ndarray, y: np.ndarray, k: int, a2) -> Tuple[float, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    d = _pb__decode_mixture(theta, k)
    sigmas = np.asarray(d['sigmas'], dtype=float)
    weights = np.asarray(d['weights'], dtype=float)
    locations = np.asarray(d['locations'], dtype=float)
    kappa = float(d['kappa'])
    ref_sigma = float(d['sigma_ref'])
    b = np.asarray(d['b'], dtype=float)
    bbar = float(d['bbar'])
    ratios = np.asarray(d['ratios'], dtype=float)
    u = (y[:, None] - locations[None, :]) / sigmas[None, :]
    logf, dlog_du, dlog_dkappa = a2.centered_skew_logpdf_and_grad_standard(u, kappa)
    terms = np.log(weights)[None, :] + logf - np.log(sigmas)[None, :]
    logmix = logsumexp(terms, axis=1)
    if not np.all(np.isfinite(logmix)):
        return (-np.inf, np.full_like(theta, np.nan), np.full((len(y), k), np.nan))
    responsibilities = np.exp(terms - logmix[:, None])
    G = responsibilities * dlog_du
    rsum = np.sum(responsibilities, axis=0)
    grad_ref = float(-len(y) - np.sum(G * u) - np.sum(G * (locations / sigmas)[None, :]))
    expanded_width_grad = -rsum - np.sum(G * u, axis=0)
    grad_gamma = np.asarray(d['H'], dtype=float).T @ expanded_width_grad
    grad_kappa = float(np.sum(responsibilities * dlog_dkappa))
    weighted_G_scale = float(np.sum(G / sigmas[None, :]))
    grad_logits = np.empty(k - 1, dtype=float)
    for ell in range(k - 1):
        location_shift = ref_sigma * weights[ell] * (b[ell] - bbar)
        grad_logits[ell] = rsum[ell] - len(y) * weights[ell] + location_shift * weighted_G_scale
    grad_eta = np.empty(k - 1, dtype=float)
    for ell in range(k - 1):
        indicator = np.zeros(k, dtype=float)
        indicator[ell + 1:] = 1.0
        tail = float(np.sum(weights[ell + 1:]))
        dloc = ref_sigma * ratios[ell] * (indicator - tail)
        du = -dloc / sigmas
        grad_eta[ell] = float(np.sum(G * du[None, :]))
    grad = np.concatenate([np.asarray([grad_ref]), np.asarray(grad_gamma, dtype=float), np.asarray([grad_kappa]), grad_logits, grad_eta])
    return (float(np.sum(logmix)), grad, responsibilities)

def _pb__mixture_log_joint_and_gradient(theta: np.ndarray, y: np.ndarray, k: int, prior: _pb_PriorProfile, a2) -> Tuple[float, np.ndarray]:
    ll, gl, _ = _pb__mixture_loglik_and_gradient(theta, y, k, a2)
    if not np.isfinite(ll) or not np.all(np.isfinite(gl)):
        return (-np.inf, np.full_like(theta, np.nan))
    lp, gp = _pb__mixture_log_prior_and_gradient(theta, k, prior)
    return (float(ll + lp), np.asarray(gl + gp, dtype=float))

def _pb__gaussian_logz_exact(y: np.ndarray, a0: float=_pb_IG_SHAPE, b0: float=_pb_IG_SCALE) -> float:
    y = np.asarray(y, dtype=float)
    n = len(y)
    an = a0 + 0.5 * n
    bn = b0 + 0.5 * float(y @ y)
    return float(-0.5 * n * _pb__LOG_2PI + a0 * math.log(b0) - gammaln(a0) + gammaln(an) - an * math.log(bn))

def _pb__skew_log_joint_and_gradient(theta: np.ndarray, y: np.ndarray, a2) -> Tuple[float, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2,):
        raise ValueError('S expects [log_sigma, kappa].')
    ll, gl = a2._loglik_and_grad_skew(theta, np.asarray(y, dtype=float))
    lp_s, gp_s = _pb__log_ref_scale_prior(float(theta[0]), _pb_IG_SHAPE, _pb_IG_SCALE)
    lp_d, gp_d = _pb__log_delta_prior_kappa(float(theta[1]))
    return (float(ll + lp_s + lp_d), np.asarray([gl[0] + gp_s, gl[1] + gp_d], dtype=float))

def _pb__skew_initial_theta(y: np.ndarray, kappa_start: float) -> np.ndarray:
    rms = math.sqrt(max(float(np.mean(np.asarray(y, dtype=float) ** 2)), 0.0001))
    return np.asarray([np.clip(math.log(rms), *_pb_LOG_SIGMA_REF_BOUNDS), np.clip(kappa_start, *_pb_KAPPA_BOUNDS)])

def _pb__mixture_initial_theta(y: np.ndarray, k: int, kappa_start: float, a2) -> np.ndarray:
    centers, weights, labels = a2._repairing_kmeans_1d(np.asarray(y, dtype=float), k)
    weights = np.asarray(weights, dtype=float)
    centers = np.asarray(centers, dtype=float)
    labels = np.asarray(labels, dtype=int)
    global_sd = max(float(np.std(y, ddof=0)), 0.1)
    component_sigmas = np.empty(k, dtype=float)
    for j in range(k):
        vals = np.asarray(y, dtype=float)[labels == j]
        if len(vals) >= 2:
            component_sigmas[j] = max(float(np.std(vals, ddof=0)), 0.08 * global_sd, 0.03)
        else:
            component_sigmas[j] = max(0.25 * global_sd, 0.03)
    log_sigmas = np.log(component_sigmas)
    log_sigma_ref = float(np.mean(log_sigmas))
    H = _pb__helmert_basis(k)
    gamma = H.T @ (log_sigmas - log_sigma_ref)
    logits = np.log(np.clip(weights[:-1], 1e-10, 1.0) / np.clip(weights[-1], 1e-10, 1.0))
    sigma_ref = math.exp(log_sigma_ref)
    ratios = np.clip(np.diff(centers) / sigma_ref, math.exp(_pb_LOG_SEPARATION_RATIO_BOUNDS[0]), math.exp(_pb_LOG_SEPARATION_RATIO_BOUNDS[1]))
    eta = np.log(ratios)
    theta = np.concatenate([np.asarray([log_sigma_ref]), np.asarray(gamma, dtype=float), np.asarray([kappa_start]), np.asarray(logits, dtype=float), np.asarray(eta, dtype=float)])
    bounds = _pb__mixture_bounds(k)
    return np.asarray([np.clip(x, lo, hi) for x, (lo, hi) in zip(theta, bounds)], dtype=float)

def _pb__optimize_mode(model: str, start_name: str, theta0: np.ndarray, y: np.ndarray, fg, bounds) -> _pb_OptimizedMode:
    started = time.perf_counter()

    def objective(x: np.ndarray):
        value, grad = fg(x)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            return (1e+300, np.zeros_like(x))
        return (-float(value), -np.asarray(grad, dtype=float))
    result = minimize(objective, np.asarray(theta0, dtype=float), jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': _pb_MAX_ITER, 'ftol': 1e-10, 'gtol': 1e-06, 'maxls': 30})
    if not bool(result.success) or not np.isfinite(result.fun):
        result = minimize(objective, np.asarray(result.x, dtype=float), jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2 * _pb_MAX_ITER, 'ftol': 1e-11, 'gtol': 5e-07, 'maxls': 40})
    theta = np.asarray(result.x, dtype=float)
    log_joint, grad = fg(theta)
    if not np.isfinite(log_joint) or not np.all(np.isfinite(grad)):
        raise FloatingPointError(f'{model} {start_name} returned invalid MAP point.')
    if not bool(result.success) and float(np.linalg.norm(grad, ord=np.inf)) > 0.002:
        raise FloatingPointError(f'{model} {start_name} did not converge: {result.message}')
    return _pb_OptimizedMode(model=model, start_name=start_name, theta=theta, log_joint=float(log_joint), iterations=int(getattr(result, 'nit', -1)), gradient_inf_norm=float(np.linalg.norm(grad, ord=np.inf)), runtime_s=float(time.perf_counter() - started), success=bool(result.success), message=str(result.message))

def _pb__deduplicate_modes(modes: Sequence[_pb_OptimizedMode]) -> List[_pb_OptimizedMode]:
    ordered = sorted(modes, key=lambda m: m.log_joint, reverse=True)
    kept: List[_pb_OptimizedMode] = []
    for mode in ordered:
        if not any((np.max(np.abs(mode.theta - other.theta)) <= _pb_PARAMETER_DUPLICATE_TOL for other in kept)):
            kept.append(mode)
    return kept

def _pb__negative_gradient_hessian(theta: np.ndarray, fg) -> np.ndarray:
    x = np.asarray(theta, dtype=float)
    d = len(x)
    steps = _pb_HESSIAN_RELATIVE_STEP * (1.0 + np.abs(x))
    H = np.empty((d, d), dtype=float)
    for j, step in enumerate(steps):
        xp = x.copy()
        xp[j] += step
        xm = x.copy()
        xm[j] -= step
        _, gp = fg(xp)
        _, gm = fg(xm)
        if not np.all(np.isfinite(gp)) or not np.all(np.isfinite(gm)):
            raise FloatingPointError('Non-finite gradient during Hessian construction.')
        H[:, j] = -(np.asarray(gp) - np.asarray(gm)) / (2.0 * step)
    return 0.5 * (H + H.T)

def _pb__laplace_mode(mode: _pb_OptimizedMode, fg) -> _pb_LaplaceMode:
    started = time.perf_counter()
    H = _pb__negative_gradient_hessian(mode.theta, fg)
    eig = np.linalg.eigvalsh(H)
    if not np.all(np.isfinite(eig)):
        raise FloatingPointError('Laplace Hessian has non-finite eigenvalues.')
    min_eig = float(np.min(eig))
    if min_eig <= _pb_HESSIAN_MIN_EIGENVALUE:
        raise FloatingPointError(f'Laplace Hessian not safely positive definite: min eig={min_eig:.3e}')
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0 or not np.isfinite(logdet):
        raise FloatingPointError('Invalid Laplace Hessian determinant.')
    cov = np.linalg.inv(H)
    dim = len(mode.theta)
    logz = float(mode.log_joint + 0.5 * dim * _pb__LOG_2PI - 0.5 * logdet)
    return _pb_LaplaceMode(optimized=mode, hessian=H, covariance=cov, min_eigenvalue=min_eig, logdet_hessian=float(logdet), logz=logz, hessian_runtime_s=float(time.perf_counter() - started))

def _pb__pooled_mahalanobis(a: _pb_LaplaceMode, b: _pb_LaplaceMode) -> float:
    diff = np.asarray(a.optimized.theta - b.optimized.theta, dtype=float)
    pooled = 0.5 * (a.covariance + b.covariance)
    try:
        value = float(diff @ np.linalg.solve(pooled, diff))
    except np.linalg.LinAlgError:
        value = float(diff @ np.linalg.pinv(pooled) @ diff)
    return math.sqrt(max(value, 0.0))

def _pb__combine_laplace_modes(lap_modes: Sequence[_pb_LaplaceMode]) -> Tuple[float, str, float, List[_pb_LaplaceMode]]:
    if not lap_modes:
        raise FloatingPointError('No valid Laplace modes.')
    ordered = sorted(lap_modes, key=lambda m: m.logz, reverse=True)
    if len(ordered) == 1:
        return (float(ordered[0].logz), 'SINGLE', 0.0, [ordered[0]])
    kept: List[_pb_LaplaceMode] = [ordered[0]]
    separations: List[float] = []
    for candidate in ordered[1:]:
        ds = [_pb__pooled_mahalanobis(candidate, existing) for existing in kept]
        dmin = min(ds)
        separations.append(float(dmin))
        if dmin >= _pb_POSTERIOR_MODE_SEPARATION:
            kept.append(candidate)
    if len(kept) >= 2:
        return (float(logsumexp([m.logz for m in kept])), 'DISTINCT', float(min(separations)), kept)
    return (float(kept[0].logz), 'OVERLAP', float(min(separations) if separations else 0.0), kept)

def _pb__fit_skew_evidence(y: np.ndarray, a2, mode_rows: List[Dict[str, Any]], context: Mapping[str, Any]) -> _pb_EvidenceFit:
    started = time.perf_counter()
    starts = (('ZERO', 0.0), ('POS', math.asinh(5.0)), ('NEG', -math.asinh(5.0)))
    modes: List[_pb_OptimizedMode] = []
    errors: List[str] = []
    for name, kappa in starts:
        try:
            theta0 = _pb__skew_initial_theta(y, kappa)
            fg = lambda x: _pb__skew_log_joint_and_gradient(x, y, a2)
            modes.append(_pb__optimize_mode('S', name, theta0, y, fg, [_pb_LOG_SIGMA_REF_BOUNDS, _pb_KAPPA_BOUNDS]))
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    modes = _pb__deduplicate_modes(modes)
    if not modes:
        raise FloatingPointError('All S posterior starts failed: ' + ' | '.join(errors))
    lap: List[_pb_LaplaceMode] = []
    fg = lambda x: _pb__skew_log_joint_and_gradient(x, y, a2)
    for mode in modes:
        try:
            lm = _pb__laplace_mode(mode, fg)
            lap.append(lm)
            mode_rows.append({**context, 'profile': 'U', 'model': 'S', 'start': mode.start_name, 'log_joint': mode.log_joint, 'logz_mode': lm.logz, 'grad_inf': mode.gradient_inf_norm, 'iterations': mode.iterations, 'map_runtime_s': mode.runtime_s, 'hessian_runtime_s': lm.hessian_runtime_s, 'min_eigenvalue': lm.min_eigenvalue, 'logdet_hessian': lm.logdet_hessian, 'theta': json.dumps(mode.theta.tolist())})
        except Exception as exc:
            mode_rows.append({**context, 'profile': 'U', 'model': 'S', 'start': mode.start_name, 'log_joint': mode.log_joint, 'logz_mode': np.nan, 'grad_inf': mode.gradient_inf_norm, 'iterations': mode.iterations, 'map_runtime_s': mode.runtime_s, 'hessian_runtime_s': np.nan, 'min_eigenvalue': np.nan, 'logdet_hessian': np.nan, 'theta': json.dumps(mode.theta.tolist()), 'laplace_failure': str(exc)})
    logz, classification, separation, kept = _pb__combine_laplace_modes(lap)
    dominant = max(kept, key=lambda m: m.logz)
    return _pb_EvidenceFit(model='S', logz=logz, dominant_theta=dominant.optimized.theta, dominant_start=dominant.optimized.start_name, classification=classification, separation=separation, retained_modes=len(kept), candidate_modes=len(modes), min_effective_count=float(len(y)), runtime_s=float(time.perf_counter() - started), map_runtime_s=float(sum((m.runtime_s for m in modes))), hessian_runtime_s=float(sum((m.hessian_runtime_s for m in lap))))

                                     
_pd_KDE_GRID_SIZE = 512
_pd_KDE_BANDWIDTH_MULTIPLIERS = (0.6, 0.8, 1.0, 1.25, 1.6, 2.0)

def _pd_hartigan_dip_statistic(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4 or x[0] == x[-1]:
        return 0.0
    low = 0
    high = n - 1
    dip_value = 0.0
    mn = np.zeros(n, dtype=np.int32)
    mn[0] = 0
    for j in range(1, n):
        mn[j] = j - 1
        while True:
            mnj = int(mn[j])
            mnmnj = int(mn[mnj])
            if mnj == 0 or (x[j] - x[mnj]) * (mnj - mnmnj) < (x[mnj] - x[mnmnj]) * (j - mnj):
                break
            mn[j] = mnmnj
    mj = np.zeros(n, dtype=np.int32)
    mj[n - 1] = n - 1
    for k in range(n - 2, -1, -1):
        mj[k] = k + 1
        while True:
            mjk = int(mj[k])
            mjmjk = int(mj[mjk])
            if mjk == n - 1 or (x[k] - x[mjk]) * (mjk - mjmjk) < (x[mjk] - x[mjmjk]) * (k - mjk):
                break
            mj[k] = mjmjk
    gcm = np.zeros(n, dtype=np.int32)
    lcm = np.zeros(n, dtype=np.int32)
    while True:
        gcm[0] = high
        i = 0
        while int(gcm[i]) > low:
            gcm[i + 1] = mn[int(gcm[i])]
            i += 1
        ig = i
        l_gcm = i
        ix = ig - 1
        lcm[0] = low
        i = 0
        while int(lcm[i]) < high:
            lcm[i + 1] = mj[int(lcm[i])]
            i += 1
        ih = i
        l_lcm = i
        iv = 1
        d = 0.0
        if l_gcm != 1 or l_lcm != 1:
            while True:
                gcmix = int(gcm[ix])
                lcmiv = int(lcm[iv])
                if gcmix > lcmiv:
                    gcmi1 = int(gcm[ix + 1])
                    denom = x[gcmix] - x[gcmi1]
                    dx = 0.0 if np.isclose(denom, 0.0, rtol=1e-09, atol=1e-09) else lcmiv - gcmi1 + 1 - (x[lcmiv] - x[gcmi1]) * (gcmix - gcmi1) / denom
                    iv += 1
                    if dx >= d:
                        d = float(dx)
                        ig = ix + 1
                        ih = iv - 1
                else:
                    lcmiv1 = int(lcm[iv - 1])
                    denom = x[lcmiv] - x[lcmiv1]
                    dx = 0.0 if np.isclose(denom, 0.0, rtol=1e-09, atol=1e-09) else (x[gcmix] - x[lcmiv1]) * (lcmiv - lcmiv1) / denom - (gcmix - lcmiv1 - 1)
                    ix -= 1
                    if dx >= d:
                        d = float(dx)
                        ig = ix + 1
                        ih = iv
                ix = max(ix, 0)
                iv = min(iv, l_lcm)
                if int(gcm[ix]) == int(lcm[iv]):
                    break
        if d < dip_value:
            break
        dip_l = 0.0
        for j in range(ig, l_gcm):
            max_t = 1.0
            jb = int(gcm[j + 1])
            je = int(gcm[j])
            if je - jb > 1 and x[je] != x[jb]:
                c = (je - jb) / (x[je] - x[jb])
                jj = np.arange(jb, je + 1, dtype=float)
                values_t = jj - jb + 1.0 - (x[jb:je + 1] - x[jb]) * c
                max_t = max(max_t, float(np.max(values_t)))
            dip_l = max(dip_l, max_t)
        dip_u = 0.0
        for j in range(ih, l_lcm):
            max_t = 1.0
            jb = int(lcm[j])
            je = int(lcm[j + 1])
            if je - jb > 1 and x[je] != x[jb]:
                c = (je - jb) / (x[je] - x[jb])
                jj = np.arange(jb, je + 1, dtype=float)
                values_t = (x[jb:je + 1] - x[jb]) * c - (jj - jb - 1.0)
                max_t = max(max_t, float(np.max(values_t)))
            dip_u = max(dip_u, max_t)
        dip_value = max(dip_value, dip_l, dip_u)
        if low == int(gcm[ig]) and high == int(lcm[ih]):
            break
        low = int(gcm[ig])
        high = int(lcm[ih])
    return float(dip_value / (2.0 * n))

def _pd_scaled_dip(values: np.ndarray) -> Tuple[float, float]:
    started = time.perf_counter()
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    value = 0.0 if len(x) < 4 else float(math.sqrt(len(x)) * _pd_hartigan_dip_statistic(x))
    return (value, float(time.perf_counter() - started))

def _pd_normalize_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError('Scout weights shape does not match data.')
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError('Scout weights contain no positive finite mass.')
    return w / total

def _pd_weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    c = np.cumsum(w)
    c /= c[-1]
    return float(np.interp(float(q), c, x, left=x[0], right=x[-1]))

def _pd_weighted_mean_sd(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = float(np.sum(w * x))
    variance = float(np.sum(w * (x - mean) ** 2))
    return (mean, math.sqrt(max(variance, 1e-14)))

def _pd_weighted_effective_n(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    return float(1.0 / np.sum(w * w))

def _pd_robust_standardize(values: np.ndarray, weights: np.ndarray | None=None) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    x = x[finite]
    if len(x) < 4:
        raise ValueError('KDE needs at least four finite observations.')
    raw_weights = None if weights is None else np.asarray(weights, dtype=float)[finite]
    w = _pd_normalize_weights(raw_weights, len(x))
    q25 = _pd_weighted_quantile(x, w, 0.25)
    q50 = _pd_weighted_quantile(x, w, 0.5)
    q75 = _pd_weighted_quantile(x, w, 0.75)
    iqr_scale = (q75 - q25) / 1.349
    _, sd = _pd_weighted_mean_sd(x, w)
    scale = float(iqr_scale)
    if not np.isfinite(scale) or scale <= 1e-08:
        scale = float(sd)
    if not np.isfinite(scale) or scale <= 1e-08:
        scale = 1.0
    return ((x - q50) / scale, w)

def _pd_meaningful_peak_count(smoothed_density: np.ndarray, empirical_bin_mass: np.ndarray, *, minimum_peak_distance_bins: int, relative_prominence: float, minimum_mode_mass: float) -> int:
    density = np.asarray(smoothed_density, dtype=float)
    mass = np.asarray(empirical_bin_mass, dtype=float)
    if len(density) < 3 or not np.any(density > 0.0):
        return 1
    max_density = float(np.max(density))
    prominence = max(float(relative_prominence) * max_density, 1e-14)
    peaks, _ = find_peaks(density, prominence=prominence, distance=max(1, int(minimum_peak_distance_bins)))
    boundaries: List[int] = []
    if density[0] > density[1] and density[0] >= prominence:
        boundaries.append(0)
    if density[-1] > density[-2] and density[-1] >= prominence:
        boundaries.append(len(density) - 1)
    if boundaries:
        peaks = np.unique(np.concatenate([peaks, np.asarray(boundaries, dtype=int)]))
    if len(peaks) <= 1:
        return 1
    peaks = np.sort(peaks)
    valleys = [0]
    for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
        segment = density[left_peak:right_peak + 1]
        valleys.append(left_peak + int(np.argmin(segment)))
    valleys.append(len(density) - 1)
    meaningful = 0
    for i in range(len(peaks)):
        lo = int(valleys[i]) + (1 if i > 0 else 0)
        hi = int(valleys[i + 1])
        if hi < lo:
            continue
        if float(np.sum(mass[lo:hi + 1])) >= float(minimum_mode_mass):
            meaningful += 1
    return max(1, int(meaningful))

def _pd_kde_mode_tree(values: np.ndarray, *, weights: np.ndarray | None=None, relative_prominence: float, minimum_mode_mass: float) -> Dict[str, Any]:
    started = time.perf_counter()
    z, w = _pd_robust_standardize(values, weights)
    n_eff = _pd_weighted_effective_n(w)
    if n_eff < 8.0:
        return {'p2': 0.0, 'p3': 0.0, 'max_modes': 1, 'mode_counts': [1] * len(_pd_KDE_BANDWIDTH_MULTIPLIERS), 'n_eff': float(n_eff), 'runtime_s': float(time.perf_counter() - started)}
    q_low = _pd_weighted_quantile(z, w, 0.0025)
    q_high = _pd_weighted_quantile(z, w, 0.9975)
    low = max(float(np.min(z)), q_low - 1.0)
    high = min(float(np.max(z)), q_high + 1.0)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(z)) - 1.0
        high = float(np.max(z)) + 1.0
    counts, edges = np.histogram(z, bins=_pd_KDE_GRID_SIZE, range=(low, high), weights=w)
    counts = np.asarray(counts, dtype=float)
    total_mass = float(np.sum(counts))
    if total_mass <= 0.0:
        raise FloatingPointError('KDE histogram has zero mass.')
    counts /= total_mass
    dx = float(edges[1] - edges[0])
    q25 = _pd_weighted_quantile(z, w, 0.25)
    q75 = _pd_weighted_quantile(z, w, 0.75)
    _, sd_z = _pd_weighted_mean_sd(z, w)
    robust_sd = min(sd_z, (q75 - q25) / 1.349 if q75 > q25 else sd_z)
    if not np.isfinite(robust_sd) or robust_sd <= 1e-06:
        robust_sd = max(sd_z, 1.0)
    base_bandwidth = max(0.9 * robust_sd * float(n_eff) ** (-0.2), 0.35 * dx)
    mode_counts: List[int] = []
    for multiplier in _pd_KDE_BANDWIDTH_MULTIPLIERS:
        bandwidth = float(multiplier) * base_bandwidth
        sigma_bins = max(bandwidth / dx, 0.35)
        smoothed = gaussian_filter1d(counts, sigma=sigma_bins, mode='nearest')
        min_distance = max(1, int(round(0.5 * bandwidth / dx)))
        mode_counts.append(_pd_meaningful_peak_count(smoothed, counts, minimum_peak_distance_bins=min_distance, relative_prominence=relative_prominence, minimum_mode_mass=minimum_mode_mass))
    arr = np.asarray(mode_counts, dtype=int)
    return {'p2': float(np.mean(arr >= 2)), 'p3': float(np.mean(arr >= 3)), 'max_modes': int(np.max(arr)), 'mode_counts': mode_counts, 'n_eff': float(n_eff), 'runtime_s': float(time.perf_counter() - started)}

                                         
_pt_LOG2 = math.log(2.0)
_pt_LOG_2PI = math.log(2.0 * math.pi)
_pt_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_pt_TWO_OVER_PI = 2.0 / math.pi

@dataclass
class _pt_PTMode:
    start_name: str
    theta: np.ndarray
    log_joint: float
    gradient_inf_norm: float
    iterations: int
    success: bool
    message: str
    optimization_seconds: float
    hessian_seconds: float
    hessian_backend: str
    hessian: np.ndarray
    covariance: np.ndarray
    min_eigenvalue: float
    logdet_hessian: float
    logz: float

class _pt_PyTensorScaleTemplate:

    def __init__(self, k: int, *, ig_shape: float=1.5, ig_scale: float=0.5, dirichlet_alpha: float=2.0, log_separation_mean: float=math.log(2.5), log_separation_sd: float=1.0, scale_contrast_sd: float=0.6):
        if int(k) not in (2, 3):
            raise ValueError('Part F only supports K=2 or K=3.')
        self.k = int(k)
        self.dimension = 1 + (k - 1) + 1 + (k - 1) + (k - 1)
        self.H = np.asarray(helmert(self.k, full=False).T, dtype=np.float64)
        self.ig_shape = float(ig_shape)
        self.ig_scale = float(ig_scale)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.log_separation_mean = float(log_separation_mean)
        self.log_separation_sd = float(log_separation_sd)
        self.scale_contrast_sd = float(scale_contrast_sd)
        self.compile_seconds = 0.0
        self._compile()

    def _compile(self):
        started = time.perf_counter()
        theta = pt.vector(f'theta_k{self.k}', dtype='float64')
        y = pt.vector(f'y_k{self.k}', dtype='float64')
        cursor = 0
        log_sigma_ref = theta[cursor]
        cursor += 1
        gamma = theta[cursor:cursor + self.k - 1]
        cursor += self.k - 1
        kappa = theta[cursor]
        cursor += 1
        logits_free = theta[cursor:cursor + self.k - 1]
        cursor += self.k - 1
        eta = theta[cursor:cursor + self.k - 1]
        sigma_ref = pt.exp(log_sigma_ref)
        delta = pt.tanh(kappa)
        alpha_skew = pt.sinh(kappa)
        logits = pt.concatenate([logits_free, pt.zeros((1,), dtype='float64')])
        log_weights = logits - pt.logsumexp(logits)
        weights = pt.exp(log_weights)
        ratios = pt.exp(eta)
        b = pt.concatenate([pt.zeros((1,), dtype='float64'), pt.cumsum(ratios)])
        bbar = pt.sum(weights * b)
        offsets = b - bbar
        locations = sigma_ref * offsets
        H = pt.constant(self.H)
        log_sigmas = log_sigma_ref + pt.dot(H, gamma)
        sigmas = pt.exp(log_sigmas)
        mean_z = _pt_SQRT_2_OVER_PI * delta
        sd_z = pt.sqrt(1.0 - _pt_TWO_OVER_PI * delta * delta)
        u = (y[:, None] - locations[None, :]) / sigmas[None, :]
        z = sd_z * u + mean_z
        t = alpha_skew * z
        standard_normal = pm.Normal.dist(mu=0.0, sigma=1.0)
        logcdf = pm.logcdf(standard_normal, t)
        component_logpdf = _pt_LOG2 + pt.log(sd_z) - 0.5 * (_pt_LOG_2PI + z * z) + logcdf - log_sigmas[None, :]
        logmix = pt.logsumexp(log_weights[None, :] + component_logpdf, axis=1)
        log_likelihood = pt.sum(logmix)
        sigma2_ref = pt.exp(2.0 * log_sigma_ref)
        a0 = self.ig_shape
        b0 = self.ig_scale
        log_scale_prior = a0 * math.log(b0) - math.lgamma(a0) - (a0 + 1.0) * (2.0 * log_sigma_ref) - b0 / sigma2_ref + _pt_LOG2 + 2.0 * log_sigma_ref
        tau = self.scale_contrast_sd
        gamma_z = gamma / tau
        log_gamma_prior = pt.sum(-0.5 * (_pt_LOG_2PI + 2.0 * math.log(tau) + gamma_z * gamma_z))
        log_delta_prior = -_pt_LOG2 - 2.0 * pt.log(pt.cosh(kappa))
        alpha_w = self.dirichlet_alpha
        log_weight_prior = math.lgamma(self.k * alpha_w) - self.k * math.lgamma(alpha_w) + alpha_w * pt.sum(log_weights)
        eta_sd = self.log_separation_sd
        eta_z = (eta - self.log_separation_mean) / eta_sd
        log_eta_prior = pt.sum(-0.5 * (_pt_LOG_2PI + 2.0 * math.log(eta_sd) + eta_z * eta_z))
        log_joint = log_likelihood + log_scale_prior + log_gamma_prior + log_delta_prior + log_weight_prior + log_eta_prior
        neg = -log_joint
        g = grad(neg, theta)
        Hsym = jacobian(g, theta)
        mode = get_mode('FAST_RUN').excluding('local_add_of_sparse_write')
        self._fg = pytensor.function([theta, y], [neg, g], mode=mode, on_unused_input='raise')
        self._h = pytensor.function([theta, y], Hsym, mode=mode, on_unused_input='raise')
        self.compile_seconds = float(time.perf_counter() - started)

    def objective_gradient(self, theta, y):
        value, gradient = self._fg(np.asarray(theta, dtype=np.float64), np.asarray(y, dtype=np.float64))
        value = float(np.asarray(value))
        gradient = np.asarray(gradient, dtype=float)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return (1e+300, np.zeros(self.dimension, dtype=float))
        return (value, gradient)

    def exact_hessian(self, theta, y):
        H = np.asarray(self._h(np.asarray(theta, dtype=np.float64), np.asarray(y, dtype=np.float64)), dtype=float)
        return 0.5 * (H + H.T)

    def fd_hessian(self, theta, y, relative_step=0.0002):
        theta = np.asarray(theta, dtype=float)
        H = np.empty((self.dimension, self.dimension), dtype=float)
        steps = float(relative_step) * (1.0 + np.abs(theta))
        for j, h in enumerate(steps):
            plus = theta.copy()
            minus = theta.copy()
            plus[j] += h
            minus[j] -= h
            _, gp = self.objective_gradient(plus, y)
            _, gm = self.objective_gradient(minus, y)
            if not np.all(np.isfinite(gp)) or not np.all(np.isfinite(gm)):
                raise FloatingPointError('Non-finite gradient in PyTensor fallback Hessian.')
            H[:, j] = (gp - gm) / (2.0 * h)
        return 0.5 * (H + H.T)

    @staticmethod
    def validate_hessian(H):
        H = np.asarray(H, dtype=float)
        H = 0.5 * (H + H.T)
        if not np.all(np.isfinite(H)):
            raise FloatingPointError('Non-finite Hessian.')
        try:
            eig = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError(f'Hessian eigendecomposition failed: {exc}') from exc
        if not np.all(np.isfinite(eig)):
            raise FloatingPointError('Non-finite Hessian eigenvalues.')
        min_eig = float(np.min(eig))
        if min_eig <= 1e-08:
            raise FloatingPointError(f'Hessian not safely positive definite: min eig={min_eig:.3e}')
        sign, logdet = np.linalg.slogdet(H)
        if sign <= 0 or not np.isfinite(logdet):
            raise FloatingPointError('Invalid Hessian determinant.')
        return (H, min_eig, float(logdet))

    def fit_mode(self, y, theta0, start_name, bounds, max_iter=500):
        y = np.asarray(y, dtype=float)
        theta0 = np.asarray(theta0, dtype=float)
        started = time.perf_counter()
        result = minimize(fun=lambda x: self.objective_gradient(x, y), x0=theta0, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': int(max_iter), 'ftol': 1e-10, 'gtol': 1e-06, 'maxls': 30})
        if not bool(result.success) or not np.isfinite(result.fun):
            result = minimize(fun=lambda x: self.objective_gradient(x, y), x0=np.asarray(result.x, dtype=float), jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': int(max_iter) * 2, 'ftol': 1e-11, 'gtol': 5e-07, 'maxls': 40})
        optimization_seconds = float(time.perf_counter() - started)
        theta_hat = np.asarray(result.x, dtype=float)
        neg_value, neg_grad = self.objective_gradient(theta_hat, y)
        if not np.isfinite(neg_value) or not np.all(np.isfinite(neg_grad)):
            raise FloatingPointError('Invalid PyTensor MAP.')
        if not bool(result.success) and float(np.linalg.norm(neg_grad, ord=np.inf)) > 0.002:
            raise FloatingPointError(f'PyTensor {start_name} did not converge: {result.message}')
        h_started = time.perf_counter()
        if self.k == 3:
            backend = 'gradient_fd_reference_matched'
            H, min_eig, logdet = self.validate_hessian(self.fd_hessian(theta_hat, y, relative_step=0.0002))
        else:
            backend = 'exact_autodiff'
            try:
                H, min_eig, logdet = self.validate_hessian(self.exact_hessian(theta_hat, y))
            except Exception:
                backend = 'gradient_fd_fallback'
                H, min_eig, logdet = self.validate_hessian(self.fd_hessian(theta_hat, y, relative_step=0.0002))
        hessian_seconds = float(time.perf_counter() - h_started)
        covariance = np.linalg.inv(H)
        log_joint = -float(neg_value)
        logz = float(log_joint + 0.5 * self.dimension * _pt_LOG_2PI - 0.5 * logdet)
        return _pt_PTMode(start_name=str(start_name), theta=theta_hat, log_joint=log_joint, gradient_inf_norm=float(np.linalg.norm(neg_grad, ord=np.inf)), iterations=int(getattr(result, 'nit', -1)), success=bool(result.success), message=str(result.message), optimization_seconds=optimization_seconds, hessian_seconds=hessian_seconds, hessian_backend=backend, hessian=H, covariance=covariance, min_eigenvalue=min_eig, logdet_hessian=logdet, logz=logz)

                                                                        

class _MeanAPI:
    V3CScoringEngine = _mean_V3CScoringEngine
    NIGPrior = _mean_NIGPrior
    LocalScoreDiagnostic = _mean_LocalScoreDiagnostic
    T_GRID = _mean_T_GRID
    STATE_L = _mean_STATE_L
    STATE_C = _mean_STATE_C
    joint_state_log_prior = staticmethod(_mean_joint_state_log_prior)

class _SkewAAPI:
    NIGPrior = _ska_NIGPrior
    GaussianAnchor = _ska_GaussianAnchor
    centered_skew_logpdf_standard = _ska_centered_skew_logpdf_standard

class _SkewCAPI:
    zero_start = _skc_zero_start
    fast_start = _skc_fast_start

class _SkewDAPI:
    pass

class _SkewEAPI:
    strict_same_mode = _ske_strict_same_mode
    pooled_mahalanobis_distance = _ske_pooled_mahalanobis_distance
    multimode_predictive_logpdf = _ske_multimode_predictive_logpdf


                                                         
SCORER_VERSION = 'v3-continuous-combined-stage3a-optimized3-candidate-v1'
SCORER_PATCH = 'm3-empty-cluster-init-repair-plus-kl-logweight-underflow-v1'
MEAN_POSTERIOR_MASS_CUTOFF = 0.8
GAUSSIAN_CONFIDENCE_CUTOFF = 0.9
FAST_LOG_BF_GUARD = 0.0
TAU_PREDICTIVE_REGRET = 0.5
G_BRANCH_PRIOR = 0.5
S_BRANCH_PRIOR = 0.5
LOG_HALF = math.log(0.5)
FAST_QUAD_ORDER = 121
MULTI_MAX_ITER = 500
MULTI_SEPARATION_THRESHOLD = 3.0
NEWTON_FAST_DELTA_MAX = 0.94
BOUNDARY_MARGIN = 0.0001
LOG_SIGMA2_BOUNDS = (-16.0, 12.0)
KAPPA_BOUNDS = (-6.0, 6.0)
_LOG_2PI = math.log(2.0 * math.pi)
_LOG2 = math.log(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_TWO_OVER_PI = 2.0 / math.pi
_FAST_NODES, _FAST_WEIGHTS = leggauss(FAST_QUAD_ORDER)
_FAST_LOG_PRIOR_WEIGHTS = np.log(_FAST_WEIGHTS) - _LOG2
_FAST_ALPHA = _FAST_NODES / np.sqrt(1.0 - _FAST_NODES * _FAST_NODES)
_FAST_MEAN_Z = _SQRT_2_OVER_PI * _FAST_NODES
_FAST_SD_Z = np.sqrt(1.0 - _TWO_OVER_PI * _FAST_NODES * _FAST_NODES)
_FAST_LOG_DENSITY_CONSTANT = _LOG2 + np.log(_FAST_SD_Z) - 0.5 * _LOG_2PI

def _load_dependencies():
    provenance = {'integration': 'flat-single-file', 'mean_core': 'embedded-as-normal-prefixed-definitions', 'skew_core': 'embedded-as-normal-prefixed-definitions'}
    return (_MeanAPI, _SkewAAPI, _SkewCAPI, _SkewDAPI, _SkewEAPI, provenance)

def _centered_skew_kl_to_gaussian(delta: float, skew_a) -> float:
    d = float(abs(delta))
    if d < 1e-14:
        return 0.0

    def integrand(u: float) -> float:
        arr = np.asarray([u], dtype=float)
        lp = float(skew_a.centered_skew_logpdf_standard(arr, d)[0])
        lq = -0.5 * (math.log(2.0 * math.pi) + u * u)
        return math.exp(lp) * (lp - lq)
    value, _ = quad(integrand, -12.0, 12.0, epsabs=1e-11, epsrel=1e-09, limit=300)
    return float(max(value, 0.0))

def _practical_delta_boundary(n: int, tau: float, skew_a) -> Tuple[float, float]:
    target = float(tau) / float(n)

    def objective(delta: float) -> float:
        return _centered_skew_kl_to_gaussian(delta, skew_a) - target
    hi = 0.999
    if objective(hi) <= 0.0:
        return (hi, _centered_skew_kl_to_gaussian(hi, skew_a))
    root = float(brentq(objective, 1e-10, hi, xtol=1e-10, rtol=1e-10))
    return (root, _centered_skew_kl_to_gaussian(root, skew_a))

def _retained_indices(logweights: np.ndarray, cutoff: float) -> np.ndarray:
    weights = np.exp(np.asarray(logweights, dtype=float))
    order = np.argsort(-weights)
    cumulative = np.cumsum(weights[order])
    count = int(np.searchsorted(cumulative, float(cutoff), side='left') + 1)
    return np.asarray(order[:count], dtype=int)

def _make_skew_prior(mean_engine, skew_a):
    p = mean_engine.prior
    return skew_a.NIGPrior(intercept_var=float(p.intercept_var), coeff_var=float(p.coeff_var), ig_shape=float(p.ig_shape), ig_scale=float(p.ig_scale))

def _make_gaussian_anchor(stage1_fit, n: int, prior, skew_a):
    an = float(prior.ig_shape) + 0.5 * int(n)
    bn = float(stage1_fit.sigma2_mean) * (an - 1.0)
    sigma2_mode = bn / (an + 1.0)
    return skew_a.GaussianAnchor(logz=float(stage1_fit.log_evidence), beta_mean=np.asarray(stage1_fit.beta_mean, dtype=float), sigma2_mean=float(stage1_fit.sigma2_mean), sigma2_mode=float(sigma2_mode), an=float(an), bn=float(bn))

def _mode_hits_artificial_boundary(theta: np.ndarray, p_design: int) -> bool:
    theta = np.asarray(theta, dtype=float)
    log_s2 = float(theta[p_design])
    kappa = float(theta[p_design + 1])
    lo_s2, hi_s2 = LOG_SIGMA2_BOUNDS
    lo_k, hi_k = KAPPA_BOUNDS
    return bool(log_s2 <= lo_s2 + BOUNDARY_MARGIN or log_s2 >= hi_s2 - BOUNDARY_MARGIN or kappa <= lo_k + BOUNDARY_MARGIN or (kappa >= hi_k - BOUNDARY_MARGIN))

def _fast_skew_correction_cached(y: np.ndarray, X: np.ndarray, anchor) -> Dict[str, object]:






    started = time.perf_counter()
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    mean = X @ np.asarray(anchor.beta_mean, dtype=float)
    sigma = math.sqrt(float(anchor.sigma2_mean))
    residuals = (y - mean) / sigma
    normal_log_sum = -0.5 * (len(residuals) * _LOG_2PI + float(residuals @ residuals))
    log_ratios = np.empty(FAST_QUAD_ORDER, dtype=float)
    for i in range(FAST_QUAD_ORDER):
        z = _FAST_SD_Z[i] * residuals + _FAST_MEAN_Z[i]
        skew_log_sum = float(len(residuals) * _FAST_LOG_DENSITY_CONSTANT[i] - 0.5 * float(z @ z) + np.sum(log_ndtr(_FAST_ALPHA[i] * z)))
        log_ratios[i] = skew_log_sum - normal_log_sum
    log_terms = _FAST_LOG_PRIOR_WEIGHTS + log_ratios
    log_bf = float(logsumexp(log_terms))
    posterior_logw = log_terms - log_bf
    posterior_w = np.exp(posterior_logw)
    map_idx = int(np.argmax(posterior_logw))
    delta_map = float(_FAST_NODES[map_idx])
    alpha_map = float(delta_map / math.sqrt(max(1.0 - delta_map * delta_map, 1e-30)))
    return {'logz': float(anchor.logz + log_bf), 'log_bf_vs_gaussian': log_bf, 'runtime_s': float(time.perf_counter() - started), 'delta_map': delta_map, 'alpha_map': alpha_map, 'delta_mean': float(np.sum(posterior_w * _FAST_NODES)), 'abs_delta_mean': float(np.sum(posterior_w * np.abs(_FAST_NODES))), 'positive_mass': float(np.sum(posterior_w[_FAST_NODES > 0.0])), 'negative_mass': float(np.sum(posterior_w[_FAST_NODES < 0.0])), 'quad_nodes': _FAST_NODES, 'posterior_weights': posterior_w, 'anchor_beta': np.asarray(anchor.beta_mean, dtype=float).copy(), 'anchor_sigma': sigma}
_FAST_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_FAST_LOGCDF_TAIL_SWITCH = -20.0
_FAST_TAIL_COEFFS = (1.0, -1.0, 3.0, -15.0, 105.0, -945.0, 10395.0, -135135.0)

def _fast_log_ndtr_erfc_tail20(x: np.ndarray) -> np.ndarray:







    x = np.asarray(x, dtype=float)
    clipped = np.maximum(x, _FAST_LOGCDF_TAIL_SWITCH)
    out = np.log(0.5 * erfc(-clipped * _FAST_INV_SQRT2))
    tail = x < _FAST_LOGCDF_TAIL_SWITCH
    if np.any(tail):
        a = -x[tail]
        inv2 = 1.0 / (a * a)
        series = np.zeros_like(a)
        power = np.ones_like(a)
        for coefficient in _FAST_TAIL_COEFFS:
            series += coefficient * power
            power *= inv2
        out[tail] = -0.5 * a * a - 0.5 * _LOG_2PI - np.log(a) + np.log(series)
    return out

def _fast_skew_correction_batch(y: np.ndarray, designs: Sequence[np.ndarray], anchors: Sequence[object], *, max_states_per_batch: int=256, target_tensor_elements: int=4000000) -> List[Dict[str, object]]:

    started_all = time.perf_counter()
    y = np.asarray(y, dtype=float)
    count = len(designs)
    if count != len(anchors):
        raise ValueError('designs and anchors must have equal length.')
    if count == 0:
        return []
    outputs: List[Dict[str, object]] = [None] * count
    chunk_size = max(1, int(max_states_per_batch))
    positive = _FAST_NODES > 0.0
    negative = _FAST_NODES < 0.0
    abs_nodes = np.abs(_FAST_NODES)
    for start_idx in range(0, count, chunk_size):
        stop = min(start_idx + chunk_size, count)
        m = stop - start_idx
        chunk_started = time.perf_counter()
        residuals = np.empty((len(y), m), dtype=float)
        sigmas = np.empty(m, dtype=float)
        for j, (X, anchor) in enumerate(zip(designs[start_idx:stop], anchors[start_idx:stop])):
            X = np.asarray(X, dtype=float)
            beta = np.asarray(anchor.beta_mean, dtype=float)
            sigma = math.sqrt(float(anchor.sigma2_mean))
            sigmas[j] = sigma
            residuals[:, j] = (y - X @ beta) / sigma
        normal_log_sum = -0.5 * (len(y) * _LOG_2PI + np.sum(residuals * residuals, axis=0))
        log_ratios = np.empty((FAST_QUAD_ORDER, m), dtype=float)
        denom = max(1, len(y) * m)
        q_block = max(1, min(FAST_QUAD_ORDER, int(target_tensor_elements // denom)))
        for qs in range(0, FAST_QUAD_ORDER, q_block):
            qe = min(qs + q_block, FAST_QUAD_ORDER)
            sd = _FAST_SD_Z[qs:qe, None, None]
            meanz = _FAST_MEAN_Z[qs:qe, None, None]
            alpha = _FAST_ALPHA[qs:qe, None, None]
            z = sd * residuals[None, :, :] + meanz
            skew_log_sum = len(y) * _FAST_LOG_DENSITY_CONSTANT[qs:qe, None] - 0.5 * np.sum(z * z, axis=1) + np.sum(_fast_log_ndtr_erfc_tail20(alpha * z), axis=1)
            log_ratios[qs:qe, :] = skew_log_sum - normal_log_sum[None, :]
        log_terms = _FAST_LOG_PRIOR_WEIGHTS[:, None] + log_ratios
        log_bf = logsumexp(log_terms, axis=0)
        posterior_logw = log_terms - log_bf[None, :]
        posterior_w = np.exp(posterior_logw)
        map_idx = np.argmax(posterior_logw, axis=0)
        chunk_runtime = float(time.perf_counter() - chunk_started)
        per_state_runtime = chunk_runtime / float(m)
        for j in range(m):
            anchor = anchors[start_idx + j]
            idx = int(map_idx[j])
            delta_map = float(_FAST_NODES[idx])
            alpha_map = float(delta_map / math.sqrt(max(1.0 - delta_map * delta_map, 1e-30)))
            w = posterior_w[:, j].copy()
            outputs[start_idx + j] = {'logz': float(anchor.logz + log_bf[j]), 'log_bf_vs_gaussian': float(log_bf[j]), 'runtime_s': per_state_runtime, 'delta_map': delta_map, 'alpha_map': alpha_map, 'delta_mean': float(w @ _FAST_NODES), 'abs_delta_mean': float(w @ abs_nodes), 'positive_mass': float(np.sum(w[positive])), 'negative_mass': float(np.sum(w[negative])), 'quad_nodes': _FAST_NODES, 'posterior_weights': w, 'anchor_beta': np.asarray(anchor.beta_mean, dtype=float).copy(), 'anchor_sigma': float(sigmas[j])}
    outputs[0]['batch_total_runtime_s'] = float(time.perf_counter() - started_all)
    return outputs

def _negative_log_joint_and_gradient(theta: np.ndarray, y: np.ndarray, X: np.ndarray, prior) -> Tuple[float, np.ndarray]:







    theta = np.asarray(theta, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    beta = theta[:p]
    log_sigma2 = float(theta[p])
    kappa = float(theta[p + 1])
    if not np.all(np.isfinite(beta)) or not np.isfinite(log_sigma2) or (not np.isfinite(kappa)) or (log_sigma2 < -20.0) or (log_sigma2 > 20.0) or (abs(kappa) > 8.0):
        return (1e+300, np.zeros_like(theta))
    sigma2 = math.exp(log_sigma2)
    sigma = math.sqrt(sigma2)
    delta = math.tanh(kappa)
    delta_prime = 1.0 - delta * delta
    alpha = math.sinh(kappa)
    alpha_prime = math.cosh(kappa)
    mean_z = _SQRT_2_OVER_PI * delta
    var_z = 1.0 - _TWO_OVER_PI * delta * delta
    sd_z = math.sqrt(max(var_z, 1e-14))
    sd_z_prime = -_TWO_OVER_PI * delta * delta_prime / sd_z
    residual = (y - X @ beta) / sigma
    z = sd_z * residual + mean_z
    t = alpha * z
    log_phi_cdf = log_ndtr(t)
    if not np.all(np.isfinite(log_phi_cdf)):
        return (1e+300, np.zeros_like(theta))
    log_mills = -0.5 * (_LOG_2PI + t * t) - log_phi_cdf
    mills = np.exp(log_mills)
    log_likelihood = float(np.sum(_LOG2 + math.log(sd_z) - 0.5 * (_LOG_2PI + z * z) + log_phi_cdf - math.log(sigma)))
    dlog_du = sd_z * (-z + alpha * mills)
    grad_beta = -(X.T @ dlog_du) / sigma
    grad_log_sigma2 = -0.5 * (float(dlog_du @ residual) + len(y))
    mean_z_prime = _SQRT_2_OVER_PI * delta_prime
    z_prime = sd_z_prime * residual + mean_z_prime
    t_prime = alpha_prime * z + alpha * z_prime
    grad_kappa = float(np.sum(sd_z_prime / sd_z - z * z_prime + mills * t_prime))
    prior_vars = np.full(p, float(prior.coeff_var), dtype=float)
    prior_vars[0] = float(prior.intercept_var)
    quad_beta = float(np.sum(beta * beta / prior_vars))
    logdet_v0 = float(np.sum(np.log(prior_vars)))
    log_beta_prior = -0.5 * p * _LOG_2PI - 0.5 * p * log_sigma2 - 0.5 * logdet_v0 - 0.5 * quad_beta / sigma2
    a0 = float(prior.ig_shape)
    b0 = float(prior.ig_scale)
    log_sigma2_prior = a0 * math.log(b0) - math.lgamma(a0) - (a0 + 1.0) * log_sigma2 - b0 / sigma2
    abs_kappa = abs(kappa)
    stable_log_cosh = abs_kappa + math.log1p(math.exp(-2.0 * abs_kappa)) - _LOG2
    log_delta_prior_and_jacobian = -_LOG2 - 2.0 * stable_log_cosh
    value = float(log_likelihood + log_beta_prior + log_sigma2_prior + log_delta_prior_and_jacobian + log_sigma2)
    grad_beta += -beta / (prior_vars * sigma2)
    grad_log_sigma2 += -0.5 * p + 0.5 * quad_beta / sigma2 - a0 + b0 / sigma2
    grad_kappa += -2.0 * delta
    gradient = np.concatenate([np.asarray(grad_beta, dtype=float), [grad_log_sigma2, grad_kappa]])
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        return (1e+300, np.zeros_like(theta))
    return (-value, -gradient)

def _negative_log_joint_gradient_hessian(theta, y, X, prior):
    theta = np.asarray(theta, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    beta = theta[:p]
    log_sigma2 = float(theta[p])
    kappa = float(theta[p + 1])
    if not np.all(np.isfinite(theta)) or log_sigma2 < -20.0 or log_sigma2 > 20.0 or (abs(kappa) > 8.0):
        return (1e+300, np.zeros_like(theta), np.eye(len(theta)))
    sigma2 = math.exp(log_sigma2)
    sigma = math.sqrt(sigma2)
    delta = math.tanh(kappa)
    delta_p = 1.0 - delta * delta
    delta_pp = -2.0 * delta * delta_p
    alpha = math.sinh(kappa)
    alpha_p = math.cosh(kappa)
    alpha_pp = alpha
    mean_z = _SQRT_2_OVER_PI * delta
    mean_z_p = _SQRT_2_OVER_PI * delta_p
    mean_z_pp = _SQRT_2_OVER_PI * delta_pp
    var_z = 1.0 - _TWO_OVER_PI * delta * delta
    sd_z = math.sqrt(max(var_z, 1e-14))
    var_z_p = -2.0 * _TWO_OVER_PI * delta * delta_p
    var_z_pp = -2.0 * _TWO_OVER_PI * (delta_p * delta_p + delta * delta_pp)
    sd_z_p = var_z_p / (2.0 * sd_z)
    sd_z_pp = var_z_pp / (2.0 * sd_z) - var_z_p * var_z_p / (4.0 * sd_z ** 3)
    u = (y - X @ beta) / sigma
    z = sd_z * u + mean_z
    t = alpha * z
    logcdf = log_ndtr(t)
    if not np.all(np.isfinite(logcdf)):
        return (1e+300, np.zeros_like(theta), np.eye(len(theta)))
    mills = np.exp(-0.5 * (_LOG_2PI + t * t) - logcdf)
    d2logcdf = -mills * (t + mills)
    log_likelihood = float(np.sum(_LOG2 + math.log(sd_z) - 0.5 * (_LOG_2PI + z * z) + logcdf - math.log(sigma)))
    h_u = sd_z * (-z + alpha * mills)
    h_uu = sd_z * sd_z * (-1.0 + alpha * alpha * d2logcdf)
    z_k = sd_z_p * u + mean_z_p
    t_k = alpha_p * z + alpha * z_k
    h_uk = sd_z_p * (-z + alpha * mills) + sd_z * (-z_k + alpha_p * mills + alpha * d2logcdf * t_k)
    z_kk = sd_z_pp * u + mean_z_pp
    t_kk = alpha_pp * z + 2.0 * alpha_p * z_k + alpha * z_kk
    h_kk = sd_z_pp / sd_z - (sd_z_p / sd_z) ** 2 - (z_k * z_k + z * z_kk) + d2logcdf * t_k * t_k + mills * t_kk
    grad_beta = -(X.T @ h_u) / sigma
    grad_log_sigma2 = -0.5 * (float(h_u @ u) + len(y))
    grad_kappa = float(np.sum(sd_z_p / sd_z - z * z_k + mills * t_k))
    prior_vars = np.full(p, float(prior.coeff_var), dtype=float)
    prior_vars[0] = float(prior.intercept_var)
    inv_prior_vars = 1.0 / prior_vars
    quad_beta = float(np.sum(beta * beta * inv_prior_vars))
    logdet_v0 = float(np.sum(np.log(prior_vars)))
    log_beta_prior = -0.5 * p * _LOG_2PI - 0.5 * p * log_sigma2 - 0.5 * logdet_v0 - 0.5 * quad_beta / sigma2
    a0 = float(prior.ig_shape)
    b0 = float(prior.ig_scale)
    log_sigma2_prior = a0 * math.log(b0) - math.lgamma(a0) - (a0 + 1.0) * log_sigma2 - b0 / sigma2
    abs_kappa = abs(kappa)
    stable_log_cosh = abs_kappa + math.log1p(math.exp(-2.0 * abs_kappa)) - _LOG2
    log_delta_prior_and_jacobian = -_LOG2 - 2.0 * stable_log_cosh
    value = float(log_likelihood + log_beta_prior + log_sigma2_prior + log_delta_prior_and_jacobian + log_sigma2)
    grad_beta += -beta * inv_prior_vars / sigma2
    grad_log_sigma2 += -0.5 * p + 0.5 * quad_beta / sigma2 - a0 + b0 / sigma2
    grad_kappa += -2.0 * delta
    gradient = np.concatenate([grad_beta, [grad_log_sigma2, grad_kappa]])
    q = 1.0 / sigma
    H = np.zeros((p + 2, p + 2), dtype=float)
    H[:p, :p] = -(q * q) * (X.T @ (h_uu[:, None] * X))
    cross = -0.5 * q * (X.T @ (h_u + h_uu * u))
    H[:p, p] = cross
    H[p, :p] = cross
    cross = q * (X.T @ h_uk)
    H[:p, p + 1] = cross
    H[p + 1, :p] = cross
    H[p, p] = -0.25 * float(np.sum(h_uu * u * u + h_u * u))
    H[p, p + 1] = 0.5 * float(np.sum(h_uk * u))
    H[p + 1, p] = H[p, p + 1]
    H[p + 1, p + 1] = -float(np.sum(h_kk))
    H[:p, :p] += np.diag(inv_prior_vars / sigma2)
    cross = -beta * inv_prior_vars / sigma2
    H[:p, p] += cross
    H[p, :p] += cross
    H[p, p] += 0.5 * quad_beta / sigma2 + b0 / sigma2
    H[p + 1, p + 1] += 2.0 * delta_p
    return (-value, -gradient, H)

def _newton_fast_mode(theta0, y, X, prior, max_iter=30):
    x = np.asarray(theta0, dtype=float).copy()
    p = X.shape[1]
    for iteration in range(int(max_iter)):
        value, grad, H = _negative_log_joint_gradient_hessian(x, y, X, prior)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)) or (not np.all(np.isfinite(H))):
            return None
        if float(np.linalg.norm(grad, ord=np.inf)) <= 1e-06:
            return (x, value, iteration)
        try:
            min_eig = float(np.min(np.linalg.eigvalsh(H)))
            shift = max(0.0, 1e-05 - min_eig)
            step = np.linalg.solve(H + shift * np.eye(len(x)), -grad)
        except Exception:
            return None
        norm = float(np.linalg.norm(step))
        if norm > 2.0:
            step *= 2.0 / norm
        slope = float(grad @ step)
        if slope >= 0.0:
            return None
        alpha_ls = 1.0
        accepted = False
        for _ in range(20):
            candidate = x + alpha_ls * step
            if LOG_SIGMA2_BOUNDS[0] <= candidate[p] <= LOG_SIGMA2_BOUNDS[1] and KAPPA_BOUNDS[0] <= candidate[p + 1] <= KAPPA_BOUNDS[1]:
                candidate_value, _ = _negative_log_joint_and_gradient(candidate, y, X, prior)
                if np.isfinite(candidate_value) and candidate_value <= value + 0.0001 * alpha_ls * slope:
                    x = candidate
                    accepted = True
                    break
            alpha_ls *= 0.5
        if not accepted:
            return None
    return None

def _optimise_joint_mode_analytic(*, y: np.ndarray, X: np.ndarray, prior, start_theta: np.ndarray, max_iter: int) -> Dict[str, object]:
    started = time.perf_counter()
    p = X.shape[1]
    start_theta = np.asarray(start_theta, dtype=float)
    start_delta = abs(math.tanh(float(start_theta[p + 1])))
    if 1e-12 < start_delta < NEWTON_FAST_DELTA_MAX:
        newton = _newton_fast_mode(start_theta, y, X, prior, max_iter=min(30, int(max_iter)))
        if newton is not None:
            theta_hat, fun, nit = newton
            _, grad = _negative_log_joint_and_gradient(theta_hat, y, X, prior)
            return {'theta_hat': np.asarray(theta_hat, dtype=float), 'log_joint_hat': -float(fun), 'runtime_s': float(time.perf_counter() - started), 'optimizer_success': True, 'optimizer_status': 0, 'optimizer_message': 'safeguarded_exact_newton', 'iterations': int(nit), 'gradient_norm': float(np.linalg.norm(grad)), 'newton_used': True}
    eps = np.finfo(float).eps
    factr = 1e-10 / eps
    theta_hat, fun, info = fmin_l_bfgs_b(_negative_log_joint_and_gradient, start_theta, args=(y, X, prior), bounds=[(None, None)] * p + [LOG_SIGMA2_BOUNDS, KAPPA_BOUNDS], m=10, factr=factr, pgtol=1e-06, maxiter=int(max_iter), maxfun=max(15000, 20 * int(max_iter)), maxls=30)
    if not np.isfinite(fun):
        raise FloatingPointError('Hybrid Newton/L-BFGS-B MULTI-S optimisation returned non-finite objective.')
    grad = np.asarray(info.get('grad', np.full_like(theta_hat, np.nan)), dtype=float)
    warnflag = int(info.get('warnflag', 2))
    return {'theta_hat': np.asarray(theta_hat, dtype=float), 'log_joint_hat': -float(fun), 'runtime_s': float(time.perf_counter() - started), 'optimizer_success': bool(warnflag == 0), 'optimizer_status': warnflag, 'optimizer_message': str(info.get('task', '')), 'iterations': int(info.get('nit', -1)), 'gradient_norm': float(np.linalg.norm(grad) if np.all(np.isfinite(grad)) else np.nan), 'newton_used': False}

def _gradient_difference_hessian(theta_hat: np.ndarray, y: np.ndarray, X: np.ndarray, prior, relative_step: float=0.0002) -> np.ndarray:






    x = np.asarray(theta_hat, dtype=float)
    d = len(x)
    H = np.empty((d, d), dtype=float)
    steps = float(relative_step) * (1.0 + np.abs(x))
    for j, h in enumerate(steps):
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        _, gp = _negative_log_joint_and_gradient(xp, y, X, prior)
        _, gm = _negative_log_joint_and_gradient(xm, y, X, prior)
        H[:, j] = (gp - gm) / (2.0 * h)
    return 0.5 * (H + H.T)

def _exact_zero_mode(*, y: np.ndarray, X: np.ndarray, prior, anchor) -> Dict[str, object]:













    started = time.perf_counter()
    p = X.shape[1]
    a_joint = float(anchor.an) + 0.5 * p
    sigma2 = float(anchor.bn) / a_joint
    theta = np.concatenate([np.asarray(anchor.beta_mean, dtype=float), [math.log(max(sigma2, 1e-300)), 0.0]])
    neg_value, gradient = _negative_log_joint_and_gradient(theta, y, X, prior)
    if not np.isfinite(neg_value) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError('Closed-form ZERO mode produced a non-finite joint value.')
    return {'theta_hat': theta, 'log_joint_hat': -float(neg_value), 'runtime_s': float(time.perf_counter() - started), 'optimizer_success': True, 'optimizer_status': 0, 'optimizer_message': 'closed_form_zero_stationary_point', 'iterations': 0, 'gradient_norm': float(np.linalg.norm(gradient, ord=np.inf)), 'boundary_hit': False, 'valid_for_laplace': True, 'retry_used': False, 'rescue_start_used': False, 'total_optim_runtime_s': 0.0, 'exact_zero_mode': True}

def _exact_zero_laplace(*, y: np.ndarray, X: np.ndarray, prior, mode: Dict[str, object]) -> Dict[str, object]:

    started = time.perf_counter()
    theta_hat = np.asarray(mode['theta_hat'], dtype=float)
    p = X.shape[1]
    sigma2 = math.exp(float(theta_hat[p]))
    prior_vars = np.full(p, float(prior.coeff_var), dtype=float)
    prior_vars[0] = float(prior.intercept_var)
    posterior_precision = X.T @ X + np.diag(1.0 / prior_vars)
    a_joint = float(prior.ig_shape) + 0.5 * len(y) + 0.5 * p
    hessian = np.zeros((p + 2, p + 2), dtype=float)
    hessian[:p, :p] = posterior_precision / sigma2
    hessian[p, p] = a_joint
    hessian[p + 1, p + 1] = 2.0
    eigenvalues = np.linalg.eigvalsh(hessian)
    min_eig = float(np.min(eigenvalues))
    if min_eig <= 1e-08 or not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError('Closed-form ZERO Hessian is not safely positive definite.')
    sign, logdet = np.linalg.slogdet(hessian)
    if sign <= 0 or not np.isfinite(logdet):
        raise FloatingPointError('Closed-form ZERO Hessian determinant invalid.')
    covariance = np.linalg.inv(hessian)
    dimension = len(theta_hat)
    logz = float(mode['log_joint_hat']) + 0.5 * dimension * _LOG_2PI - 0.5 * float(logdet)
    beta_hat = theta_hat[:p]
    sigma_hat = math.sqrt(sigma2)
    return {'theta_hat': theta_hat.copy(), 'log_joint_hat': float(mode['log_joint_hat']), 'logz': float(logz), 'hessian': hessian, 'covariance': covariance, 'hessian_logdet': float(logdet), 'hessian_min_eigenvalue': min_eig, 'hessian_runtime_s': float(time.perf_counter() - started), 'beta_hat': np.asarray(beta_hat, dtype=float), 'sigma_hat': float(sigma_hat), 'delta_hat': 0.0, 'alpha_hat': 0.0, 'kappa_hat': 0.0}

def _mode_laplace_analytic_hessian(*, y: np.ndarray, X: np.ndarray, prior, mode: Dict[str, object]) -> Dict[str, object]:

    if bool(mode.get('exact_zero_mode', False)):
        return _exact_zero_laplace(y=y, X=X, prior=prior, mode=mode)
    started = time.perf_counter()
    theta_hat = np.asarray(mode['theta_hat'], dtype=float)
    p = X.shape[1]
    hessian = _gradient_difference_hessian(theta_hat, y, X, prior)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError('MULTI-S Hessian has non-finite eigenvalues.')
    min_eig = float(np.min(eigenvalues))
    if min_eig <= 1e-08:
        raise FloatingPointError(f'MULTI-S Hessian is not safely positive definite; min eigenvalue={min_eig:.3e}.')
    sign, logdet = np.linalg.slogdet(hessian)
    if sign <= 0 or not np.isfinite(logdet):
        raise FloatingPointError('MULTI-S Hessian determinant invalid.')
    covariance = np.linalg.inv(hessian)
    dimension = len(theta_hat)
    logz = float(mode['log_joint_hat']) + 0.5 * dimension * _LOG_2PI - 0.5 * float(logdet)
    beta_hat = theta_hat[:p]
    log_sigma2_hat = float(theta_hat[p])
    kappa_hat = float(theta_hat[p + 1])
    sigma_hat = math.exp(0.5 * log_sigma2_hat)
    delta_hat = math.tanh(kappa_hat)
    alpha_hat = math.sinh(kappa_hat)
    return {'theta_hat': theta_hat.copy(), 'log_joint_hat': float(mode['log_joint_hat']), 'logz': float(logz), 'hessian': hessian, 'covariance': covariance, 'hessian_logdet': float(logdet), 'hessian_min_eigenvalue': min_eig, 'hessian_runtime_s': float(time.perf_counter() - started), 'beta_hat': np.asarray(beta_hat, dtype=float), 'sigma_hat': float(sigma_hat), 'delta_hat': float(delta_hat), 'alpha_hat': float(alpha_hat), 'kappa_hat': float(kappa_hat)}

def _optimise_hardened_candidate(*, y: np.ndarray, X: np.ndarray, prior, start_theta: np.ndarray, max_iter: int) -> Dict[str, object]:







    first = _optimise_joint_mode_analytic(y=y, X=X, prior=prior, start_theta=np.asarray(start_theta, dtype=float), max_iter=int(max_iter))
    attempts = [dict(first)]
    if not bool(first['optimizer_success']):
        second = _optimise_joint_mode_analytic(y=y, X=X, prior=prior, start_theta=np.asarray(first['theta_hat'], dtype=float), max_iter=max(2 * int(max_iter), int(max_iter) + 100))
        attempts.append(dict(second))
    total_runtime = float(sum((float(item['runtime_s']) for item in attempts)))

    def mark(candidate: Dict[str, object]) -> Dict[str, object]:
        out = dict(candidate)
        out['boundary_hit'] = _mode_hits_artificial_boundary(np.asarray(out['theta_hat'], dtype=float), X.shape[1])
        out['valid_for_laplace'] = bool(out['optimizer_success'] and (not out['boundary_hit']))
        return out
    marked = [mark(item) for item in attempts]
    valid = [item for item in marked if item['valid_for_laplace']]
    if valid:
        best = max(valid, key=lambda item: float(item['log_joint_hat']))
        best = dict(best)
        best['retry_used'] = len(attempts) > 1
        best['rescue_start_used'] = False
        best['total_optim_runtime_s'] = total_runtime
        return best
    base = max(marked, key=lambda item: float(item['log_joint_hat']))
    base_theta = np.asarray(base['theta_hat'], dtype=float).copy()
    p_design = X.shape[1]
    for kappa in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
        start = base_theta.copy()
        start[p_design + 1] = float(kappa)
        trial = _optimise_joint_mode_analytic(y=y, X=X, prior=prior, start_theta=start, max_iter=max(1500, 3 * int(max_iter)))
        total_runtime += float(trial['runtime_s'])
        trial_marked = mark(dict(trial))
        marked.append(trial_marked)
        if trial_marked['valid_for_laplace']:
            out = dict(trial_marked)
            out['retry_used'] = True
            out['rescue_start_used'] = True
            out['total_optim_runtime_s'] = total_runtime
            return out
    best = max(marked, key=lambda item: float(item['log_joint_hat']))
    best = dict(best)
    best['retry_used'] = True
    best['rescue_start_used'] = True
    best['total_optim_runtime_s'] = total_runtime
    return best

def _hardened_multimode_joint_laplace(*, y: np.ndarray, X: np.ndarray, anchor, fast: Dict[str, object], prior, max_iter: int, skew_c, skew_e) -> Dict[str, object]:
    started = time.perf_counter()
    zero_start_theta = np.asarray(skew_c.zero_start(anchor), dtype=float)
    fast_start_theta = np.asarray(skew_c.fast_start(anchor, fast), dtype=float)
    zero = _exact_zero_mode(y=y, X=X, prior=prior, anchor=anchor)
    if np.array_equal(zero_start_theta, fast_start_theta):
        fast_mode = dict(zero)
        fast_mode['runtime_s'] = 0.0
        fast_mode['total_optim_runtime_s'] = 0.0
    else:
        fast_mode = _optimise_hardened_candidate(y=y, X=X, prior=prior, start_theta=fast_start_theta, max_iter=max_iter)
    candidates = [('ZERO', zero), ('FAST', fast_mode)]
    valid = [(name, mode) for name, mode in candidates if mode['valid_for_laplace']]
    if not valid:
        raise FloatingPointError('Both hardened MULTI-S candidate optimisations were invalid after deterministic rescue starts.')
    lap_results: List[Tuple[str, Dict[str, object], Dict[str, object]]] = []
    if len(valid) == 2 and skew_e.strict_same_mode(valid[0][1], valid[1][1]):
        name, mode = max(valid, key=lambda item: float(item[1]['log_joint_hat']))
        lap = _mode_laplace_analytic_hessian(y=y, X=X, prior=prior, mode=mode)
        lap_results = [(name, mode, lap)]
        classification = 'SAME'
        separation = 0.0
    else:
        for name, mode in valid:
            try:
                lap = _mode_laplace_analytic_hessian(y=y, X=X, prior=prior, mode=mode)
                lap_results.append((name, mode, lap))
            except Exception:
                continue
        if not lap_results:
            raise FloatingPointError('No hardened MULTI-S candidate had a usable Laplace Hessian.')
        if len(lap_results) == 1:
            classification = 'SINGLE_VALID'
            separation = 0.0
        else:
            separation = float(skew_e.pooled_mahalanobis_distance(lap_results[0][2], lap_results[1][2]))
            classification = 'DISTINCT' if separation >= MULTI_SEPARATION_THRESHOLD else 'OVERLAP'
    if classification == 'DISTINCT' and len(lap_results) == 2:
        logzs = np.asarray([float(item[2]['logz']) for item in lap_results], dtype=float)
        combined_logz = float(logsumexp(logzs))
        predictive_log_weights = logzs - combined_logz
        predictive_modes = [item[2] for item in lap_results]
        dominant_idx = int(np.argmax(logzs))
        dominant_name = lap_results[dominant_idx][0]
        second_mass = float(np.exp(predictive_log_weights[1 - dominant_idx]))
    else:
        winner = max(lap_results, key=lambda item: float(item[2]['logz']))
        dominant_name = winner[0]
        combined_logz = float(winner[2]['logz'])
        predictive_modes = [winner[2]]
        predictive_log_weights = np.asarray([0.0], dtype=float)
        second_mass = 0.0
    return {'classification': classification, 'logz': combined_logz, 'runtime_s': float(time.perf_counter() - started), 'separation': float(separation), 'dominant_mode': dominant_name, 'second_mode_posterior_mass': second_mass, 'predictive_modes': predictive_modes, 'predictive_log_weights': predictive_log_weights, 'zero_optimizer_success': bool(zero['optimizer_success']), 'fast_optimizer_success': bool(fast_mode['optimizer_success']), 'zero_retry_used': bool(zero.get('retry_used', False)), 'fast_retry_used': bool(fast_mode.get('retry_used', False)), 'zero_rescue_start_used': bool(zero.get('rescue_start_used', False)), 'fast_rescue_start_used': bool(fast_mode.get('rescue_start_used', False)), 'zero_boundary_hit': bool(zero['boundary_hit']), 'fast_boundary_hit': bool(fast_mode['boundary_hit']), 'valid_candidate_count': int(len(valid)), 'usable_hessian_count': int(len(lap_results))}

@dataclass(frozen=True)
class CombinedLocalScoreDiagnostic:
    child: str
    parents: Tuple[str, ...]
    parent_count: int
    method: str
    score: float
    stage1_score: float
    stage1_method: str
    action: str
    stage1_state_count: int
    inspected_state_count: int
    inspected_stage1_posterior_mass: float
    escalated_state_count: int
    escalated_stage1_posterior_mass: float
    delta0: float
    tau: float
    mass_cutoff: float
    gaussian_confidence_cutoff: float
    fast_logbf_guard: float
    fast_calls: int
    multi_calls: int
    stage1_runtime_seconds: float
    fast_runtime_seconds: float
    multi_runtime_seconds: float
    runtime_seconds: float
    optimizer_retry_count: int
    optimizer_rescue_count: int
    boundary_hit_count: int

@dataclass
class _CombinedLocalResult:
    child: str
    parents: Tuple[str, ...]
    family: object
    states: List[Tuple[int, ...]]
    logw_g: np.ndarray
    logjoint_g: np.ndarray
    combined_state_logjoint: np.ndarray
    retained_indices: np.ndarray
    escalated_indices: Tuple[int, ...]
    fast_cache: Dict[int, Dict[str, object]]
    multi_cache: Dict[int, Dict[str, object]]
    combined_logz: float

class _UnimodalCScoringEngine:


    def __init__(self, data: pd.DataFrame, *, coefficient_prior_variance: float=1.0, intercept_prior_variance: float=4.0, inverse_gamma_shape: float=1.5, inverse_gamma_scale: float=0.5, beam_width: int=32, beam_max_sweeps: int=4, gibbs_chains: int=4, gibbs_burn_sweeps: int=30, gibbs_sample_sweeps: int=100) -> None:
        self._mean_module, self._skew_a, self._skew_c, self._skew_d, self._skew_e, self.dependency_provenance = _load_dependencies()
        self.mean_engine = self._mean_module.V3CScoringEngine(data, coefficient_prior_variance=coefficient_prior_variance, intercept_prior_variance=intercept_prior_variance, inverse_gamma_shape=inverse_gamma_shape, inverse_gamma_scale=inverse_gamma_scale, beam_width=beam_width, beam_max_sweeps=beam_max_sweeps, gibbs_chains=gibbs_chains, gibbs_burn_sweeps=gibbs_burn_sweeps, gibbs_sample_sweeps=gibbs_sample_sweeps)
        self.nodes = self.mean_engine.nodes
        self.raw_data = self.mean_engine.raw_data
        self.data = self.mean_engine.data
        self.n = self.mean_engine.n
        self.prior = self.mean_engine.prior
        self.means = self.mean_engine.means
        self.stds = self.mean_engine.stds
        self.feature_bank = self.mean_engine.feature_bank
        self.data_fingerprint = self.mean_engine.data_fingerprint
        self.beam_cfg = self.mean_engine.beam_cfg
        self._training_y_standardized = {str(node): self.data[str(node)].to_numpy(dtype=float) for node in self.nodes}
        self._skew_prior = _make_skew_prior(self.mean_engine, self._skew_a)
        self.delta0, self.kl_at_delta0 = _practical_delta_boundary(self.n, TAU_PREDICTIVE_REGRET, self._skew_a)
        self.cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}
        self.diagnostics: Dict[Tuple[str, Tuple[str, ...]], CombinedLocalScoreDiagnostic] = {}
        self.combined_results: Dict[Tuple[str, Tuple[str, ...]], _CombinedLocalResult] = {}

    def __getattr__(self, name: str):
        if name.startswith('__'):
            raise AttributeError(name)
        mean_engine = self.__dict__.get('mean_engine')
        if mean_engine is not None and hasattr(mean_engine, name):
            return getattr(mean_engine, name)
        raise AttributeError(name)

    def prior_hyperparameters(self) -> Dict[str, float]:
        values = dict(self.mean_engine.prior_hyperparameters())
        values.update({'skew_screen_stage1_posterior_mass_cutoff': MEAN_POSTERIOR_MASS_CUTOFF, 'skew_screen_gaussian_confidence_cutoff': GAUSSIAN_CONFIDENCE_CUTOFF, 'skew_screen_fast_logbf_guard': FAST_LOG_BF_GUARD, 'skew_screen_tau_predictive_regret': TAU_PREDICTIVE_REGRET, 'skew_branch_prior_G': G_BRANCH_PRIOR, 'skew_branch_prior_S': S_BRANCH_PRIOR, 'skew_fast_quad_order': FAST_QUAD_ORDER, 'skew_multi_separation_threshold': MULTI_SEPARATION_THRESHOLD, 'skew_multi_max_iter': MULTI_MAX_ITER})
        return values

    def preprocessing_metadata(self) -> Dict[str, object]:
        values = dict(self.mean_engine.preprocessing_metadata())
        values.update({'combined_continuous_v3_version': SCORER_VERSION, 'sample_size': int(self.n), 'practical_gaussian_delta0': float(self.delta0), 'practical_gaussian_kl_at_delta0': float(self.kl_at_delta0), 'dependency_provenance': self.dependency_provenance, 'implementation_optimizations': {'reuse_stage1_training_feature_bank': True, 'cached_fast_quadrature': True, 'batched_fast_across_retained_states': True, 'vectorized_fast_across_quadrature_blocks': True, 'direct_scipy_lbfgsb_wrapper': True, 'safeguarded_exact_newton_for_moderate_fast_modes': True, 'newton_fast_delta_max': NEWTON_FAST_DELTA_MAX, 'fast_logcdf_erfc_tail20': True, 'fast_logcdf_tail_switch': _FAST_LOGCDF_TAIL_SWITCH, 'analytic_lbfgsb_gradient': True, 'gradient_difference_hessian': True, 'skip_identical_zero_fast_start': True}})
        return values

    def _validate_family(self, child: str, parents: Sequence[str]) -> Tuple[str, ...]:
        return self.mean_engine._validate_family(child, parents)

    def _mean_states_and_logweights(self, child: str, parents: Tuple[str, ...]):
        family = self.mean_engine._family(child, parents)
        p = len(parents)
        if p == 0:
            return (family, [()], np.asarray([0.0], dtype=float))
        if p <= 2:
            states, logweights = self.mean_engine._exact_posterior_states(family)
            return (family, list(states), np.asarray(logweights, dtype=float))
        states, logweights = self.mean_engine.posterior_state_logweights[child, parents]
        return (family, list(states), np.asarray(logweights, dtype=float))

    def _training_design_for_fit(self, family, fit) -> np.ndarray:







        columns = tuple((int(value) for value in fit.columns))
        X = np.empty((self.n, len(columns)), dtype=float)
        bank = self.feature_bank
        if len(family.parents) == 0:
            X[:, 0] = 1.0
            return X
        bank._ensure_global()
        global_features = bank._global_features
        if global_features is None:
            raise RuntimeError('Frozen Stage-1 global feature bank was not built.')
        block_size = int(bank.BLOCK_SIZE)
        for out_col, local_col in enumerate(columns):
            if local_col == 0:
                X[:, out_col] = 1.0
                continue
            parent_index = (local_col - 1) // block_size
            block_offset = (local_col - 1) % block_size
            parent = family.parents[parent_index]
            global_col = int(bank.block_slice(parent).start) + block_offset
            X[:, out_col] = global_features[:, global_col]
        return X

    def score_local(self, child: str, parents: Sequence[str]) -> float:
        child = str(child)
        parents = self._validate_family(child, parents)
        key = (child, parents)
        if key in self.cache:
            return self.cache[key]
        total_started = time.perf_counter()
        stage1_started = time.perf_counter()
        stage1_logz = float(self.mean_engine.score_local(child, parents))
        stage1_runtime = float(time.perf_counter() - stage1_started)
        stage1_diag = self.mean_engine.get_score_diagnostic(child, parents)
        family, states, logw_g = self._mean_states_and_logweights(child, parents)
        weights_g = np.exp(logw_g)
        logjoint_g = stage1_logz + logw_g
        retained = _retained_indices(logw_g, MEAN_POSTERIOR_MASS_CUTOFF)
        inspected_mass = float(np.sum(weights_g[retained]))
        y_train_std = self._training_y_standardized[child]
        skew_prior = self._skew_prior
        fast_cache: Dict[int, Dict[str, object]] = {}
        state_training_cache: Dict[int, Tuple[np.ndarray, object]] = {}
        escalated: List[int] = []
        retained_indices = [int(value) for value in retained]
        retained_designs: List[np.ndarray] = []
        retained_anchors: List[object] = []
        fast_started = time.perf_counter()
        for idx in retained_indices:
            state = tuple(states[idx])
            fit = family.fit_state(state)
            X_state = self._training_design_for_fit(family, fit)
            anchor = _make_gaussian_anchor(fit, self.n, skew_prior, self._skew_a)
            retained_designs.append(X_state)
            retained_anchors.append(anchor)
            state_training_cache[idx] = (X_state, anchor)
        batched_fast = _fast_skew_correction_batch(y_train_std, retained_designs, retained_anchors)
        fast_runtime = float(time.perf_counter() - fast_started)
        for idx, fast_raw in zip(retained_indices, batched_fast):
            fast = dict(fast_raw)
            nodes = np.asarray(fast['quad_nodes'], dtype=float)
            postw = np.asarray(fast['posterior_weights'], dtype=float)
            gaussian_mass = float(np.sum(postw[np.abs(nodes) <= self.delta0]))
            fast['gaussian_mass'] = gaussian_mass
            fast['scout_total_runtime_s'] = float(fast.get('runtime_s', 0.0))
            fast_cache[idx] = fast
            clearly_gaussian = bool(gaussian_mass >= GAUSSIAN_CONFIDENCE_CUTOFF and float(fast['log_bf_vs_gaussian']) <= FAST_LOG_BF_GUARD)
            if not clearly_gaussian:
                escalated.append(idx)
        if not escalated:
            score = stage1_logz
            combined_state_logjoint = np.array(logjoint_g, copy=True)
            action = 'STAGE1_ONLY'
            multi_cache: Dict[int, Dict[str, object]] = {}
            multi_runtime = 0.0
        else:
            multi_cache = {}
            multi_runtime = 0.0
            combined_state_logjoint = np.array(logjoint_g, copy=True)
            for idx in escalated:
                X_state, anchor = state_training_cache[idx]
                multi = _hardened_multimode_joint_laplace(y=y_train_std, X=X_state, anchor=anchor, fast=fast_cache[idx], prior=skew_prior, max_iter=MULTI_MAX_ITER, skew_c=self._skew_c, skew_e=self._skew_e)
                multi_cache[idx] = multi
                multi_runtime += float(multi['runtime_s'])
                state = tuple(states[idx])
                logprior_state = 0.0 if len(parents) == 0 else float(self._mean_module.joint_state_log_prior(state))
                logjoint_s = logprior_state + float(multi['logz'])
                combined_state_logjoint[idx] = float(logsumexp(np.asarray([LOG_HALF + float(logjoint_g[idx]), LOG_HALF + logjoint_s], dtype=float)))
            score = float(logsumexp(combined_state_logjoint))
            action = 'SELECTIVE_MULTI'
        if not np.isfinite(score):
            raise FloatingPointError(f'Non-finite combined V3 local score for {child!r} | {parents}.')
        escalated_array = np.asarray(escalated, dtype=int)
        escalated_mass = float(np.sum(weights_g[escalated_array]) if len(escalated_array) else 0.0)
        retry_count = int(sum((int(bool(m.get('zero_retry_used', False))) + int(bool(m.get('fast_retry_used', False))) for m in multi_cache.values())))
        rescue_count = int(sum((int(bool(m.get('zero_rescue_start_used', False))) + int(bool(m.get('fast_rescue_start_used', False))) for m in multi_cache.values())))
        boundary_count = int(sum((int(bool(m.get('zero_boundary_hit', False))) + int(bool(m.get('fast_boundary_hit', False))) for m in multi_cache.values())))
        diagnostic = CombinedLocalScoreDiagnostic(child=child, parents=parents, parent_count=len(parents), method='stage1_BMA_FAST080_090_BF0_selective_MULTI_S', score=float(score), stage1_score=float(stage1_logz), stage1_method=str(stage1_diag.method), action=action, stage1_state_count=int(len(states)), inspected_state_count=int(len(retained)), inspected_stage1_posterior_mass=inspected_mass, escalated_state_count=int(len(escalated)), escalated_stage1_posterior_mass=escalated_mass, delta0=float(self.delta0), tau=TAU_PREDICTIVE_REGRET, mass_cutoff=MEAN_POSTERIOR_MASS_CUTOFF, gaussian_confidence_cutoff=GAUSSIAN_CONFIDENCE_CUTOFF, fast_logbf_guard=FAST_LOG_BF_GUARD, fast_calls=int(len(retained)), multi_calls=int(len(escalated)), stage1_runtime_seconds=stage1_runtime, fast_runtime_seconds=float(fast_runtime), multi_runtime_seconds=float(multi_runtime), runtime_seconds=float(time.perf_counter() - total_started), optimizer_retry_count=retry_count, optimizer_rescue_count=rescue_count, boundary_hit_count=boundary_count)
        result = _CombinedLocalResult(child=child, parents=parents, family=family, states=states, logw_g=np.asarray(logw_g, dtype=float), logjoint_g=np.asarray(logjoint_g, dtype=float), combined_state_logjoint=np.asarray(combined_state_logjoint, dtype=float), retained_indices=np.asarray(retained, dtype=int), escalated_indices=tuple((int(i) for i in escalated)), fast_cache=fast_cache, multi_cache=multi_cache, combined_logz=float(score))
        self.cache[key] = float(score)
        self.diagnostics[key] = diagnostic
        self.combined_results[key] = result
        return float(score)

    def get_score_diagnostic(self, child: str, parents: Sequence[str]) -> CombinedLocalScoreDiagnostic:
        parents = self._validate_family(child, parents)
        key = (str(child), parents)
        if key not in self.diagnostics:
            self.score_local(child, parents)
        return self.diagnostics[key]

    def diagnostics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame((asdict(value) for value in self.diagnostics.values()))

    def true_or_learned_dag_score(self, graph: nx.DiGraph) -> float:
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError('The supplied graph is not a DAG.')
        if set(graph.nodes) != set(self.nodes):
            raise ValueError('Graph nodes do not match scorer columns.')
        return float(sum((self.score_local(node, list(graph.predecessors(node))) for node in self.nodes)))

    def _standardize_eval(self, raw_eval: pd.DataFrame) -> pd.DataFrame:
        return self.mean_engine._standardize_eval(raw_eval)

    def local_predictive_logpdf(self, child: str, parents: Sequence[str], raw_eval: pd.DataFrame) -> np.ndarray:
        child = str(child)
        parents = self._validate_family(child, parents)
        self.score_local(child, parents)
        result = self.combined_results[child, parents]
        if not result.escalated_indices:
            return self.mean_engine.local_predictive_logpdf(child, parents, raw_eval)
        standardized_eval = self._standardize_eval(raw_eval)
        family = result.family
        master_eval = family.evaluation_master(standardized_eval)
        y_eval_std = standardized_eval[child].to_numpy(dtype=float)
        child_log_sd = math.log(float(self.stds[child]))
        escalated_set = set(result.escalated_indices)
        log_num = np.full(len(standardized_eval), -np.inf, dtype=float)
        for idx, state in enumerate(result.states):
            logq_g = np.asarray(family.state_logpdf_standardized(tuple(state), standardized_eval, master_eval), dtype=float) - child_log_sd
            if idx not in escalated_set:
                log_num = np.logaddexp(log_num, float(result.logjoint_g[idx]) + logq_g)
                continue
            log_num = np.logaddexp(log_num, LOG_HALF + float(result.logjoint_g[idx]) + logq_g)
            fit = family.fit_state(tuple(state))
            cols = np.asarray(fit.columns, dtype=int)
            X_eval_state = master_eval[:, cols]
            logq_s = np.asarray(self._skew_e.multimode_predictive_logpdf(y=y_eval_std, X=X_eval_state, result=result.multi_cache[idx]), dtype=float) - child_log_sd
            logprior_state = 0.0 if len(parents) == 0 else float(self._mean_module.joint_state_log_prior(tuple(state)))
            logjoint_s = logprior_state + float(result.multi_cache[idx]['logz'])
            log_num = np.logaddexp(log_num, LOG_HALF + logjoint_s + logq_s)
        return log_num - float(result.combined_logz)

    def joint_logpdf(self, graph: nx.DiGraph, raw_eval: pd.DataFrame, *, return_nodewise: bool=False):
        if set(graph.nodes) != set(self.nodes):
            raise ValueError('Graph nodes do not match scorer columns.')
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError('Evaluation graph is not a DAG.')
        total = np.zeros(len(raw_eval), dtype=float)
        nodewise: Dict[str, np.ndarray] = {}
        for child in nx.topological_sort(graph):
            values = self.local_predictive_logpdf(str(child), list(graph.predecessors(child)), raw_eval)
            total += values
            if return_nodewise:
                nodewise[str(child)] = values
        if return_nodewise:
            return (total, nodewise)
        return total
ROOT_MIXTURE_MIXTURE_PRIOR_PROFILE = 'BALANCED'
ROOT_MIXTURE_DIRICHLET_ALPHA = 2.0
ROOT_MIXTURE_SEPARATION_MEDIAN_SIGMA = 2.5
ROOT_MIXTURE_LOG_SEPARATION_SD = 1.0
ROOT_MIXTURE_MIXTURE_MAX_ITER = 500
ROOT_MIXTURE_HESSIAN_RELATIVE_STEP = 0.0002
ROOT_MIXTURE_HESSIAN_MIN_EIGENVALUE = 1e-08
ROOT_MIXTURE_PARAMETER_DUPLICATE_TOL = 0.0001
ROOT_MIXTURE_POSTERIOR_MODE_SEPARATION = 3.0
ROOT_MIXTURE_SCOUT_KDE_PROFILE = 'BALANCED'
ROOT_MIXTURE_SCOUT_A_DIP_STOP = 0.45
ROOT_MIXTURE_SCOUT_A_P2_STOP = 1.0 / 6.0
ROOT_MIXTURE_SCOUT_B_ROOT_P3_STOP = 0.0
ROOT_MIXTURE_SCOUT_B_COMPONENT_P2_STOP = 0.5
ROOT_MIXTURE_SCOUT_B_COMPONENT_DIP_STOP = 0.45
ROOT_MIXTURE_RESPONSIBILITY_CORE_CUTOFF = 0.65
ROOT_MIXTURE_MIN_COMPONENT_DIP_COUNT = 20
ROOT_MIXTURE_KDE_GRID_SIZE = 512
ROOT_MIXTURE_KDE_BANDWIDTH_MULTIPLIERS = (0.6, 0.8, 1.0, 1.25, 1.6, 2.0)
ROOT_MIXTURE_KDE_RELATIVE_PROMINENCE = 0.04
ROOT_MIXTURE_KDE_MINIMUM_MODE_MASS = 0.05
_STAGE3_MU_BOUNDS = (-6.0, 6.0)
_STAGE3_LOG_SIGMA_BOUNDS = (-5.0, 1.5)
_STAGE3_KAPPA_BOUNDS = (-6.0, 6.0)
_STAGE3_LOGIT_BOUNDS = (-7.0, 7.0)
_STAGE3_LOG_SEPARATION_RATIO_BOUNDS = (math.log(0.1), math.log(25.0))

@dataclass(frozen=True)
class _RootMixtureMixturePrior:
    intercept_var: float = 4.0
    ig_shape: float = 1.5
    ig_scale: float = 0.5
    dirichlet_alpha: float = ROOT_MIXTURE_DIRICHLET_ALPHA
    log_separation_mean: float = math.log(ROOT_MIXTURE_SEPARATION_MEDIAN_SIGMA)
    log_separation_sd: float = ROOT_MIXTURE_LOG_SEPARATION_SD

@dataclass
class _RootMixtureOptimizedMode:
    start_name: str
    theta: np.ndarray
    log_joint: float
    iterations: int
    gradient_inf_norm: float
    runtime_s: float
    optimizer_message: str

@dataclass
class _RootMixtureLaplaceMode:
    optimized: _RootMixtureOptimizedMode
    hessian: np.ndarray
    covariance: np.ndarray
    min_eigenvalue: float
    logdet_hessian: float
    logz: float
    hessian_runtime_s: float

@dataclass
class _RootMixtureRootResult:
    child: str
    u_logz: float
    scout_a_stop: bool
    root_scaled_dip: float
    root_kde_p2: float
    root_kde_p3: float
    root_kde_mode_path: Tuple[int, ...]
    m2_fit: object | None
    scout_b_stop: bool | None
    component_p2_max: float
    component_dip_max: float
    component_core_counts: Tuple[int, int]
    m3_fit: object | None
    available_models: Tuple[str, ...]
    selected_model: str
    selected_logz: float

@dataclass(frozen=True)
class RootMixtureRootScoreDiagnostic(CombinedLocalScoreDiagnostic):
    u_score: float
    u_action: str
    stage3_selected_model: str
    stage3_available_models: Tuple[str, ...]
    stage3_scout_a_stop: bool
    root_mixture_scaled_dip: float
    root_mixture_kde_p2: float
    root_mixture_kde_p3: float
    root_mixture_kde_mode_path: Tuple[int, ...]
    stage3_scout_b_stop: bool | None
    stage3_component_p2_max: float
    stage3_component_dip_max: float
    stage3_component_core_counts: Tuple[int, int]
    stage3_logz_m2: float
    stage3_logz_m3: float
    stage3_m2_calls: int
    stage3_m3_calls: int
    stage3_scout_a_runtime_seconds: float
    stage3_m2_runtime_seconds: float
    stage3_scout_b_runtime_seconds: float
    stage3_m3_runtime_seconds: float

def _root_mixture_stable_log_cosh(x: float) -> float:
    ax = abs(float(x))
    return ax + math.log1p(math.exp(-2.0 * ax)) - _LOG2

def _root_mixture_delta_to_alpha(delta: float) -> float:
    d = float(np.clip(delta, -1.0 + 1e-14, 1.0 - 1e-14))
    return d / math.sqrt(1.0 - d * d)

def _root_mixture_centered_skew_logpdf_standard(u: np.ndarray, delta: float) -> np.ndarray:
    d = float(np.clip(delta, -1.0 + 1e-13, 1.0 - 1e-13))
    alpha = _root_mixture_delta_to_alpha(d)
    mean_z = _SQRT_2_OVER_PI * d
    sd_z = math.sqrt(max(1.0 - _TWO_OVER_PI * d * d, 1e-14))
    z = sd_z * np.asarray(u, dtype=float) + mean_z
    return _LOG2 + math.log(sd_z) - 0.5 * (_LOG_2PI + z * z) + log_ndtr(alpha * z)

def _root_mixture_centered_skew_standard_logpdf_and_grad(u: np.ndarray, kappa: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    u = np.asarray(u, dtype=float)
    kappa = float(kappa)
    delta = math.tanh(kappa)
    delta_prime = 1.0 - delta * delta
    alpha = math.sinh(kappa)
    alpha_prime = math.cosh(kappa)
    mean_z = _SQRT_2_OVER_PI * delta
    sd_z = math.sqrt(max(1.0 - _TWO_OVER_PI * delta * delta, 1e-14))
    sd_prime = -(_TWO_OVER_PI * delta * delta_prime) / sd_z
    mean_prime = _SQRT_2_OVER_PI * delta_prime
    z = sd_z * u + mean_z
    z_prime = sd_prime * u + mean_prime
    t = alpha * z
    log_phi_t = -0.5 * (_LOG_2PI + t * t)
    log_Phi_t = log_ndtr(t)
    inv_mills = np.exp(log_phi_t - log_Phi_t)
    logf = _LOG2 + math.log(sd_z) - 0.5 * (_LOG_2PI + z * z) + log_Phi_t
    dlogf_du = sd_z * (-z + alpha * inv_mills)
    t_prime = alpha_prime * z + alpha * z_prime
    dlogf_dkappa = sd_prime / sd_z - z * z_prime + inv_mills * t_prime
    return (logf, dlogf_du, dlogf_dkappa)

def _root_mixture_softmax_baseline(z: np.ndarray) -> np.ndarray:
    logits = np.concatenate([np.asarray(z, dtype=float), [0.0]])
    logits -= np.max(logits)
    values = np.exp(logits)
    return values / values.sum()

def _root_mixture_decode_theta(theta: np.ndarray, k: int) -> Dict[str, Any]:
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2 * int(k) + 1,):
        raise ValueError(f'K={k} requires theta length {2 * int(k) + 1}.')
    mu = float(theta[0])
    log_sigma = float(theta[1])
    kappa = float(theta[2])
    logits = theta[3:3 + k - 1]
    eta = theta[3 + k - 1:]
    sigma = math.exp(log_sigma)
    weights = _root_mixture_softmax_baseline(logits)
    ratios = np.exp(eta)
    b = np.concatenate([[0.0], np.cumsum(ratios)])
    bbar = float(np.dot(weights, b))
    offsets = b - bbar
    locations = mu + sigma * offsets
    return {'mu': mu, 'log_sigma': log_sigma, 'sigma': sigma, 'kappa': kappa, 'delta': math.tanh(kappa), 'alpha': math.sinh(kappa), 'weights': weights, 'eta': eta, 'ratios': ratios, 'b': b, 'bbar': bbar, 'offsets': offsets, 'locations': locations}

def _root_mixture_mk_logpdf_standardized(y: np.ndarray, theta: np.ndarray, k: int) -> np.ndarray:
    decoded = _root_mixture_decode_theta(theta, k)
    y = np.asarray(y, dtype=float)
    t = (y - float(decoded['mu'])) / float(decoded['sigma'])
    terms = []
    for weight, offset in zip(decoded['weights'], decoded['offsets']):
        u = t - float(offset)
        terms.append(math.log(float(weight)) + _root_mixture_centered_skew_logpdf_standard(u, float(decoded['delta'])) - math.log(float(decoded['sigma'])))
    return logsumexp(np.vstack(terms), axis=0)
_STAGE3_PRIOR_CONSTANT_CACHE: Dict[Tuple[int, float, float, float, float, float, float], Dict[str, float]] = {}

def _root_mixture_prior_constants(k: int, prior: _RootMixtureMixturePrior) -> Dict[str, float]:

    key = (int(k), float(prior.intercept_var), float(prior.ig_shape), float(prior.ig_scale), float(prior.dirichlet_alpha), float(prior.log_separation_mean), float(prior.log_separation_sd))
    cached = _STAGE3_PRIOR_CONSTANT_CACHE.get(key)
    if cached is not None:
        return cached
    V = float(prior.intercept_var)
    a0 = float(prior.ig_shape)
    b0 = float(prior.ig_scale)
    alpha_w = float(prior.dirichlet_alpha)
    eta_mean = float(prior.log_separation_mean)
    eta_sd = float(prior.log_separation_sd)
    cached = {'V': V, 'a0': a0, 'b0': b0, 'alpha_w': alpha_w, 'eta_mean': eta_mean, 'eta_sd': eta_sd, 'log_V': math.log(V), 'ig_constant': a0 * math.log(b0) - gammaln(a0), 'dirichlet_constant': gammaln(int(k) * alpha_w) - int(k) * gammaln(alpha_w), 'log_eta_sd': math.log(eta_sd)}
    _STAGE3_PRIOR_CONSTANT_CACHE[key] = cached
    return cached

def _root_mixture_log_prior_and_gradient_from_decoded(theta: np.ndarray, k: int, prior: _RootMixtureMixturePrior, decoded: Dict[str, Any]) -> Tuple[float, np.ndarray]:

    mu = float(decoded['mu'])
    log_sigma = float(decoded['log_sigma'])
    sigma2 = float(decoded['sigma']) ** 2
    kappa = float(decoded['kappa'])
    delta = float(decoded['delta'])
    weights = np.asarray(decoded['weights'], dtype=float)
    eta = np.asarray(decoded['eta'], dtype=float)
    constants = _root_mixture_prior_constants(k, prior)
    V = constants['V']
    a0 = constants['a0']
    b0 = constants['b0']
    alpha_w = constants['alpha_w']
    eta_mean = constants['eta_mean']
    eta_sd = constants['eta_sd']
    log_mu = -0.5 * (_LOG_2PI + constants['log_V'] + 2.0 * log_sigma) - 0.5 * mu * mu / (V * sigma2)
    log_sigma2 = constants['ig_constant'] - (a0 + 1.0) * (2.0 * log_sigma) - b0 / sigma2
    log_sigma2_jacobian = _LOG2 + 2.0 * log_sigma
    log_delta_jacobian = -_LOG2 - 2.0 * _root_mixture_stable_log_cosh(kappa)
    log_weight_prior_jacobian = constants['dirichlet_constant'] + alpha_w * float(np.sum(np.log(weights)))
    eta_z = (eta - eta_mean) / eta_sd
    log_eta_prior = float(np.sum(-0.5 * (_LOG_2PI + 2.0 * constants['log_eta_sd'] + eta_z * eta_z)))
    value = float(log_mu + log_sigma2 + log_sigma2_jacobian + log_delta_jacobian + log_weight_prior_jacobian + log_eta_prior)
    gradient = np.zeros_like(theta, dtype=float)
    gradient[0] = -mu / (V * sigma2)
    gradient[1] = -2.0 * a0 - 1.0 + mu * mu / (V * sigma2) + 2.0 * b0 / sigma2
    gradient[2] = -2.0 * delta
    gradient[3:3 + k - 1] = alpha_w * (1.0 - int(k) * weights[:-1])
    gradient[3 + k - 1:] = -(eta - eta_mean) / (eta_sd * eta_sd)
    return (value, gradient)

def _root_mixture_log_prior_and_gradient(theta: np.ndarray, k: int, prior: _RootMixtureMixturePrior) -> Tuple[float, np.ndarray]:
    decoded = _root_mixture_decode_theta(theta, k)
    return _root_mixture_log_prior_and_gradient_from_decoded(theta, k, prior, decoded)

def _root_mixture_log_joint_and_gradient_generic_opt2(theta: np.ndarray, y: np.ndarray, k: int, prior: _RootMixtureMixturePrior) -> Tuple[float, np.ndarray]:
    y = np.asarray(y, dtype=float)
    decoded = _root_mixture_decode_theta(theta, k)
    sigma = float(decoded['sigma'])
    mu = float(decoded['mu'])
    kappa = float(decoded['kappa'])
    weights = np.asarray(decoded['weights'], dtype=float)
    b = np.asarray(decoded['b'], dtype=float)
    bbar = float(decoded['bbar'])
    offsets = np.asarray(decoded['offsets'], dtype=float)
    ratios = np.asarray(decoded['ratios'], dtype=float)
    t = (y - mu) / sigma
    u = t[:, None] - offsets[None, :]
    logf, dlog_du, dlog_dkappa = _root_mixture_centered_skew_standard_logpdf_and_grad(u, kappa)
    log_components = np.log(weights)[None, :] + logf - math.log(sigma)
    log_mix = logsumexp(log_components, axis=1)
    if not np.all(np.isfinite(log_mix)):
        return (-np.inf, np.full_like(theta, np.nan))
    responsibilities = np.exp(log_components - log_mix[:, None])
    response_sums = responsibilities.sum(axis=0)
    kappa_gradient = float(np.sum(responsibilities * dlog_dkappa))
    responsibilities *= dlog_du
    weighted_dlog_du = responsibilities.sum(axis=1)
    weighted_grad = responsibilities
    gradient = np.zeros_like(theta, dtype=float)
    gradient[0] = float(np.sum(-weighted_dlog_du / sigma))
    gradient[1] = float(np.sum(-1.0 - t * weighted_dlog_du))
    gradient[2] = kappa_gradient
    total = float(np.sum(weighted_dlog_du))
    for j in range(k - 1):
        gradient[3 + j] = float(response_sums[j] - len(y) * weights[j] + weights[j] * (b[j] - bbar) * total)
    eta_start = 3 + k - 1
    for ell in range(k - 1):
        indicator = np.zeros(k, dtype=float)
        indicator[ell + 1:] = 1.0
        tail = float(np.sum(weights[ell + 1:]))
        du = -ratios[ell] * (indicator - tail)
        gradient[eta_start + ell] = float(np.sum(weighted_grad * du[None, :]))
    lp, pg = _root_mixture_log_prior_and_gradient_from_decoded(theta, k, prior, decoded)
    return (float(np.sum(log_mix) + lp), gradient + pg)

def _root_mixture_log_joint_and_gradient_k2(theta, y, prior):
    theta = np.asarray(theta, dtype=float)
    y = np.asarray(y, dtype=float)
    if theta.shape != (5,):
        raise ValueError('K=2 requires theta length 5.')
    d = _root_mixture_decode_theta(theta, 2)
    sigma = float(d['sigma'])
    mu = float(d['mu'])
    kappa = float(d['kappa'])
    w = np.asarray(d['weights'])
    off = np.asarray(d['offsets'])
    bbar = float(d['bbar'])
    r = float(d['ratios'][0])
    t = (y - mu) / sigma
    u = t[:, None] - off[None, :]
    logf, du, dk = _root_mixture_centered_skew_standard_logpdf_and_grad(u, kappa)
    ls = math.log(sigma)
    c0 = math.log(float(w[0])) + logf[:, 0] - ls
    c1 = math.log(float(w[1])) + logf[:, 1] - ls
    mx = np.maximum(c0, c1)
    e0 = np.exp(c0 - mx)
    e1 = np.exp(c1 - mx)
    den = e0 + e1
    logmix = mx + np.log(den)
    if not np.all(np.isfinite(logmix)):
        return (-np.inf, np.full_like(theta, np.nan))
    q0 = e0 / den
    q1 = e1 / den
    kgrad = float(np.sum(q0 * dk[:, 0] + q1 * dk[:, 1]))
    wg0 = q0 * du[:, 0]
    wg1 = q1 * du[:, 1]
    wd = wg0 + wg1
    total = float(np.sum(wd))
    g = np.empty(5, dtype=float)
    g[0] = float(np.sum(-wd / sigma))
    g[1] = float(np.sum(-1.0 - t * wd))
    g[2] = kgrad
    g[3] = float(np.sum(q0) - len(y) * w[0] + w[0] * -bbar * total)
    g[4] = float(np.sum(wg0 * (r * w[1]) + wg1 * (-r * w[0])))
    lp, pg = _root_mixture_log_prior_and_gradient_from_decoded(theta, 2, prior, d)
    return (float(np.sum(logmix) + lp), g + pg)

def _root_mixture_log_joint_and_gradient_k3(theta, y, prior):
    theta = np.asarray(theta, dtype=float)
    y = np.asarray(y, dtype=float)
    if theta.shape != (7,):
        raise ValueError('K=3 requires theta length 7.')
    d = _root_mixture_decode_theta(theta, 3)
    sigma = float(d['sigma'])
    mu = float(d['mu'])
    kappa = float(d['kappa'])
    w = np.asarray(d['weights'])
    off = np.asarray(d['offsets'])
    b = np.asarray(d['b'])
    bbar = float(d['bbar'])
    r = np.asarray(d['ratios'])
    t = (y - mu) / sigma
    u = t[:, None] - off[None, :]
    logf, du, dk = _root_mixture_centered_skew_standard_logpdf_and_grad(u, kappa)
    ls = math.log(sigma)
    c0 = math.log(float(w[0])) + logf[:, 0] - ls
    c1 = math.log(float(w[1])) + logf[:, 1] - ls
    c2 = math.log(float(w[2])) + logf[:, 2] - ls
    mx = np.maximum(np.maximum(c0, c1), c2)
    e0 = np.exp(c0 - mx)
    e1 = np.exp(c1 - mx)
    e2 = np.exp(c2 - mx)
    den = e0 + e1 + e2
    logmix = mx + np.log(den)
    if not np.all(np.isfinite(logmix)):
        return (-np.inf, np.full_like(theta, np.nan))
    q0 = e0 / den
    q1 = e1 / den
    q2 = e2 / den
    kgrad = float(np.sum(q0 * dk[:, 0] + q1 * dk[:, 1] + q2 * dk[:, 2]))
    wg0 = q0 * du[:, 0]
    wg1 = q1 * du[:, 1]
    wg2 = q2 * du[:, 2]
    wd = wg0 + wg1 + wg2
    total = float(np.sum(wd))
    g = np.empty(7, dtype=float)
    g[0] = float(np.sum(-wd / sigma))
    g[1] = float(np.sum(-1.0 - t * wd))
    g[2] = kgrad
    g[3] = float(np.sum(q0) - len(y) * w[0] + w[0] * (b[0] - bbar) * total)
    g[4] = float(np.sum(q1) - len(y) * w[1] + w[1] * (b[1] - bbar) * total)
    g[5] = float(np.sum(wg0 * (r[0] * (1 - w[0])) + wg1 * (-r[0] * w[0]) + wg2 * (-r[0] * w[0])))
    g[6] = float(np.sum(wg0 * (r[1] * w[2]) + wg1 * (r[1] * w[2]) + wg2 * (-r[1] * (w[0] + w[1]))))
    lp, pg = _root_mixture_log_prior_and_gradient_from_decoded(theta, 3, prior, d)
    return (float(np.sum(logmix) + lp), g + pg)

def _root_mixture_log_joint_and_gradient(theta, y, k, prior):
    if int(k) == 2:
        return _root_mixture_log_joint_and_gradient_k2(theta, y, prior)
    if int(k) == 3:
        return _root_mixture_log_joint_and_gradient_k3(theta, y, prior)
    return _root_mixture_log_joint_and_gradient_generic_opt2(theta, y, k, prior)

def _root_mixture_centered_skew_standard_logpdf_and_grad_batch(u, kappa):
    u = np.asarray(u, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    delta = np.tanh(kappa)
    dp = 1 - delta * delta
    alpha = np.sinh(kappa)
    ap = np.cosh(kappa)
    mz = _SQRT_2_OVER_PI * delta
    sd = np.sqrt(np.maximum(1 - _TWO_OVER_PI * delta * delta, 1e-14))
    sdp = -(_TWO_OVER_PI * delta * dp) / sd
    mzp = _SQRT_2_OVER_PI * dp
    z = sd[:, None, None] * u + mz[:, None, None]
    zp = sdp[:, None, None] * u + mzp[:, None, None]
    tt = alpha[:, None, None] * z
    lp = -0.5 * (_LOG_2PI + tt * tt)
    lP = log_ndtr(tt)
    im = np.exp(lp - lP)
    logf = _LOG2 + np.log(sd)[:, None, None] - 0.5 * (_LOG_2PI + z * z) + lP
    du = sd[:, None, None] * (-z + alpha[:, None, None] * im)
    tp = ap[:, None, None] * z + alpha[:, None, None] * zp
    dk = (sdp / sd)[:, None, None] - z * zp + im * tp
    return (logf, du, dk)

def _root_mixture_gradient_batch(theta, y, k, prior):
    theta = np.asarray(theta, dtype=float)
    y = np.asarray(y, dtype=float)
    k = int(k)
    if k not in (2, 3):
        return np.vstack([_root_mixture_log_joint_and_gradient(row, y, k, prior)[1] for row in theta])
    mu = theta[:, 0]
    sigma = np.exp(theta[:, 1])
    kap = theta[:, 2]
    delta = np.tanh(kap)
    if k == 2:
        l = theta[:, 3]
        mx = np.maximum(l, 0)
        e0 = np.exp(l - mx)
        e1 = np.exp(-mx)
        den = e0 + e1
        w = np.column_stack((e0 / den, e1 / den))
        r = np.exp(theta[:, 4])
        bbar = w[:, 1] * r
        off = np.column_stack((-bbar, r - bbar))
    else:
        l0 = theta[:, 3]
        l1 = theta[:, 4]
        mx = np.maximum(np.maximum(l0, l1), 0)
        e0 = np.exp(l0 - mx)
        e1 = np.exp(l1 - mx)
        e2 = np.exp(-mx)
        den = e0 + e1 + e2
        w = np.column_stack((e0 / den, e1 / den, e2 / den))
        rr = np.exp(theta[:, 5:7])
        r0 = rr[:, 0]
        r1 = rr[:, 1]
        b1 = r0
        b2 = r0 + r1
        bbar = w[:, 1] * b1 + w[:, 2] * b2
        off = np.column_stack((-bbar, b1 - bbar, b2 - bbar))
    t = (y[None, :] - mu[:, None]) / sigma[:, None]
    u = t[:, :, None] - off[:, None, :]
    logf, du, dk = _root_mixture_centered_skew_standard_logpdf_and_grad_batch(u, kap)
    lc = np.log(w)[:, None, :] + logf - np.log(sigma)[:, None, None]
    lm = logsumexp(lc, axis=2)
    q = np.exp(lc - lm[:, :, None])
    rs = q.sum(axis=1)
    kg = np.sum(q * dk, axis=(1, 2))
    q *= du
    wd = q.sum(axis=2)
    tot = wd.sum(axis=1)
    g = np.zeros_like(theta)
    g[:, 0] = np.sum(-wd / sigma[:, None], axis=1)
    g[:, 1] = np.sum(-1 - t * wd, axis=1)
    g[:, 2] = kg
    const = _root_mixture_prior_constants(k, prior)
    V = const['V']
    a0 = const['a0']
    b0 = const['b0']
    aw = const['alpha_w']
    em = const['eta_mean']
    es = const['eta_sd']
    s2 = sigma * sigma
    if k == 2:
        g[:, 3] = rs[:, 0] - len(y) * w[:, 0] + w[:, 0] * -bbar * tot
        g[:, 4] = np.sum(q[:, :, 0] * (r * w[:, 1])[:, None] + q[:, :, 1] * (-r * w[:, 0])[:, None], axis=1)
        pg = np.zeros_like(theta)
        pg[:, 0] = -mu / (V * s2)
        pg[:, 1] = -2 * a0 - 1 + mu * mu / (V * s2) + 2 * b0 / s2
        pg[:, 2] = -2 * delta
        pg[:, 3] = aw * (1 - 2 * w[:, 0])
        pg[:, 4] = -(theta[:, 4] - em) / (es * es)
    else:
        g[:, 3] = rs[:, 0] - len(y) * w[:, 0] + w[:, 0] * -bbar * tot
        g[:, 4] = rs[:, 1] - len(y) * w[:, 1] + w[:, 1] * (b1 - bbar) * tot
        du0 = np.stack((r0 * (1 - w[:, 0]), -r0 * w[:, 0], -r0 * w[:, 0]), axis=1)
        du1 = np.stack((r1 * w[:, 2], r1 * w[:, 2], -r1 * (w[:, 0] + w[:, 1])), axis=1)
        g[:, 5] = np.sum(q * du0[:, None, :], axis=(1, 2))
        g[:, 6] = np.sum(q * du1[:, None, :], axis=(1, 2))
        pg = np.zeros_like(theta)
        pg[:, 0] = -mu / (V * s2)
        pg[:, 1] = -2 * a0 - 1 + mu * mu / (V * s2) + 2 * b0 / s2
        pg[:, 2] = -2 * delta
        pg[:, 3:5] = aw * (1 - 3 * w[:, :2])
        pg[:, 5:7] = -(theta[:, 5:7] - em) / (es * es)
    return g + pg

def _root_mixture_kmeans_1d(y: np.ndarray, k: int, max_iter: int=100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

















    y = np.asarray(y, dtype=float)
    centers = np.quantile(y, (np.arange(k) + 0.5) / k).astype(float)
    centers.sort()

    def _lloyd(current_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        current_centers = np.asarray(current_centers, dtype=float).copy()
        current_centers.sort()
        for _ in range(int(max_iter)):
            current_labels = np.argmin(np.abs(y[:, None] - current_centers[None, :]), axis=1)
            updated = current_centers.copy()
            for j in range(k):
                if np.any(current_labels == j):
                    updated[j] = float(np.mean(y[current_labels == j]))
            updated.sort()
            if np.allclose(updated, current_centers, atol=1e-10, rtol=0.0):
                current_centers = updated
                break
            current_centers = updated
        current_labels = np.argmin(np.abs(y[:, None] - current_centers[None, :]), axis=1)
        current_counts = np.bincount(current_labels, minlength=k).astype(float)
        return (current_centers, current_counts, current_labels)
    centers, counts, labels = _lloyd(centers)
    if np.all(counts > 0.0):
        return (centers, counts / counts.sum(), labels)
    repaired = centers.copy()
    for _ in range(k):
        empty_clusters = np.flatnonzero(np.isclose(counts, 0.0, rtol=1e-09, atol=1e-09))
        if empty_clusters.size == 0:
            return (repaired, counts / counts.sum(), labels)
        sse = np.full(k, -np.inf, dtype=float)
        for j in range(k):
            members = y[labels == j]
            if members.size >= 2:
                residuals = members - repaired[j]
                sse[j] = float(residuals @ residuals)
        donor = int(np.argmax(sse))
        if not np.isfinite(sse[donor]):
            break
        members = np.sort(y[labels == donor])
        split = members.size // 2
        if split <= 0 or split >= members.size:
            break
        lower_center = float(np.mean(members[:split]))
        upper_center = float(np.mean(members[split:]))
        empty = int(empty_clusters[0])
        repaired[donor] = lower_center
        repaired[empty] = upper_center
        repaired.sort()
        repaired, counts, labels = _lloyd(repaired)
    if np.any(np.isclose(counts, 0.0, rtol=1e-09, atol=1e-09)):
        raise FloatingPointError(f'root-mixture M{k} deterministic 1D k-means produced an empty cluster even after deterministic SSE-split repair.')
    return (repaired, counts / counts.sum(), labels)

def _root_mixture_initial_theta(y: np.ndarray, k: int, kappa_start: float) -> np.ndarray:
    centers, weights, labels = _root_mixture_kmeans_1d(y, k)
    mu = float(np.dot(weights, centers))
    sigma = math.sqrt(max(float(np.mean((y - centers[labels]) ** 2)), 0.0001))
    ratios = np.clip(np.diff(centers) / sigma, math.exp(_STAGE3_LOG_SEPARATION_RATIO_BOUNDS[0]), math.exp(_STAGE3_LOG_SEPARATION_RATIO_BOUNDS[1]))
    logits = np.log(weights[:-1] / weights[-1])
    return np.concatenate([np.asarray([np.clip(mu, *_STAGE3_MU_BOUNDS), np.clip(math.log(sigma), *_STAGE3_LOG_SIGMA_BOUNDS), np.clip(kappa_start, *_STAGE3_KAPPA_BOUNDS)], dtype=float), np.clip(logits, *_STAGE3_LOGIT_BOUNDS), np.log(ratios)])

def _root_mixture_bounds_for_k(k: int):
    return [_STAGE3_MU_BOUNDS, _STAGE3_LOG_SIGMA_BOUNDS, _STAGE3_KAPPA_BOUNDS] + [_STAGE3_LOGIT_BOUNDS] * (k - 1) + [_STAGE3_LOG_SEPARATION_RATIO_BOUNDS] * (k - 1)

def _root_mixture_optimize_one_mode(y: np.ndarray, k: int, prior: _RootMixtureMixturePrior, *, start_name: str, kappa_start: float, max_iter: int, initial_theta: np.ndarray | None=None) -> _RootMixtureOptimizedMode:
    y = np.asarray(y, dtype=float)
    if initial_theta is None:
        x0 = _root_mixture_initial_theta(y, k, kappa_start)
    else:
        x0 = np.asarray(initial_theta, dtype=float).copy()
        x0[2] = np.clip(float(kappa_start), *_STAGE3_KAPPA_BOUNDS)

    def fg(x: np.ndarray):
        value, gradient = _root_mixture_log_joint_and_gradient(x, y, k, prior)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return (1e+300, np.zeros_like(x))
        return (-float(value), -np.asarray(gradient, dtype=float))
    started = time.perf_counter()
    result = minimize(fun=fg, x0=x0, jac=True, method='L-BFGS-B', bounds=_root_mixture_bounds_for_k(k), options={'maxiter': int(max_iter), 'ftol': 1e-10, 'gtol': 1e-06, 'maxls': 30})
    if not bool(result.success) or not np.isfinite(result.fun):
        result = minimize(fun=fg, x0=np.asarray(result.x, dtype=float), jac=True, method='L-BFGS-B', bounds=_root_mixture_bounds_for_k(k), options={'maxiter': int(max_iter) * 2, 'ftol': 1e-11, 'gtol': 5e-07, 'maxls': 40})
    if not bool(result.success) or not np.isfinite(result.fun):
        raise FloatingPointError(f'root-mixture M{k} {start_name} optimization failed: {result.message}')
    theta = np.asarray(result.x, dtype=float)
    log_joint, gradient = _root_mixture_log_joint_and_gradient(theta, y, k, prior)
    if not np.isfinite(log_joint) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError(f'root-mixture M{k} {start_name} converged to an invalid posterior point.')
    return _RootMixtureOptimizedMode(start_name=start_name, theta=theta, log_joint=float(log_joint), iterations=int(getattr(result, 'nit', -1)), gradient_inf_norm=float(np.linalg.norm(gradient, ord=np.inf)), runtime_s=float(time.perf_counter() - started), optimizer_message=str(result.message))

def _root_mixture_candidate_modes(y: np.ndarray, k: int, prior: _RootMixtureMixturePrior, *, max_iter: int) -> List[_RootMixtureOptimizedMode]:
    starts = (('ZERO', 0.0), ('POS', math.asinh(5.0)), ('NEG', -math.asinh(5.0)))
    modes: List[_RootMixtureOptimizedMode] = []
    errors: List[str] = []
    shared_initial_theta = _root_mixture_initial_theta(y, k, 0.0)
    for name, kappa in starts:
        try:
            modes.append(_root_mixture_optimize_one_mode(y, k, prior, start_name=name, kappa_start=kappa, max_iter=max_iter, initial_theta=shared_initial_theta))
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    if not modes:
        raise FloatingPointError(f'All root-mixture M{k} candidate optimizations failed: ' + ' | '.join(errors))
    return modes

def _root_mixture_deduplicate_modes(modes: Sequence[_RootMixtureOptimizedMode]) -> List[_RootMixtureOptimizedMode]:
    ordered = sorted(modes, key=lambda mode: mode.log_joint, reverse=True)
    kept: List[_RootMixtureOptimizedMode] = []
    for mode in ordered:
        duplicate = any((np.max(np.abs(mode.theta - existing.theta)) <= ROOT_MIXTURE_PARAMETER_DUPLICATE_TOL for existing in kept))
        if not duplicate:
            kept.append(mode)
    return kept

def _root_mixture_gradient_difference_hessian(theta_hat, y, k, prior, *, relative_step=ROOT_MIXTURE_HESSIAN_RELATIVE_STEP) -> np.ndarray:

    x = np.asarray(theta_hat, dtype=float)
    d = len(x)
    steps = float(relative_step) * (1 + np.abs(x))
    pts = []
    for j, h in enumerate(steps):
        xp = x.copy()
        xp[j] += h
        pts.append(xp)
    for j, h in enumerate(steps):
        xm = x.copy()
        xm[j] -= h
        pts.append(xm)
    pts = np.asarray(pts)
    gr = []
    for start in range(0, len(pts), 2):
        gr.append(_root_mixture_gradient_batch(pts[start:start + 2], y, k, prior))
    G = np.vstack(gr)
    gp = G[:d]
    gm = G[d:]
    H = -((gp - gm) / (2 * steps[:, None])).T
    return 0.5 * (H + H.T)

def _root_mixture_laplace_at_mode(mode: _RootMixtureOptimizedMode, y: np.ndarray, k: int, prior: _RootMixtureMixturePrior) -> _RootMixtureLaplaceMode:
    started = time.perf_counter()
    H = _root_mixture_gradient_difference_hessian(mode.theta, y, k, prior)
    eigenvalues = np.linalg.eigvalsh(H)
    if not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError(f'root-mixture M{k} Hessian has non-finite eigenvalues.')
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue <= ROOT_MIXTURE_HESSIAN_MIN_EIGENVALUE:
        raise FloatingPointError(f'root-mixture M{k} Hessian not safely positive definite; min eigenvalue={min_eigenvalue:.3e}.')
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0 or not np.isfinite(logdet):
        raise FloatingPointError(f'root-mixture M{k} Hessian determinant invalid.')
    covariance = np.linalg.inv(H)
    dimension = len(mode.theta)
    logz = float(mode.log_joint) + 0.5 * dimension * _LOG_2PI - 0.5 * float(logdet)
    return _RootMixtureLaplaceMode(optimized=mode, hessian=H, covariance=covariance, min_eigenvalue=min_eigenvalue, logdet_hessian=float(logdet), logz=float(logz), hessian_runtime_s=float(time.perf_counter() - started))

def _root_mixture_pooled_mahalanobis(first: _RootMixtureLaplaceMode, second: _RootMixtureLaplaceMode) -> float:
    difference = np.asarray(first.optimized.theta, dtype=float) - np.asarray(second.optimized.theta, dtype=float)
    pooled = 0.5 * (np.asarray(first.covariance, dtype=float) + np.asarray(second.covariance, dtype=float))
    try:
        value = float(difference @ np.linalg.solve(pooled, difference))
    except np.linalg.LinAlgError:
        return float('inf')
    return math.sqrt(max(value, 0.0))

def _root_mixture_select_nonoverlapping_modes(modes: Sequence[_RootMixtureLaplaceMode]) -> Tuple[List[_RootMixtureLaplaceMode], Dict[str, float]]:
    ordered = sorted(modes, key=lambda mode: mode.logz, reverse=True)
    selected: List[_RootMixtureLaplaceMode] = []
    minimum_distances: Dict[str, float] = {}
    for candidate in ordered:
        if not selected:
            selected.append(candidate)
            minimum_distances[candidate.optimized.start_name] = float('inf')
            continue
        distances = [_root_mixture_pooled_mahalanobis(candidate, incumbent) for incumbent in selected]
        minimum = float(min(distances))
        minimum_distances[candidate.optimized.start_name] = minimum
        if minimum >= ROOT_MIXTURE_POSTERIOR_MODE_SEPARATION:
            selected.append(candidate)
    return (selected, minimum_distances)

def _root_mixture_model_laplace(y: np.ndarray, k: int, prior: _RootMixtureMixturePrior, *, max_iter: int=ROOT_MIXTURE_MIXTURE_MAX_ITER) -> Dict[str, Any]:
    started = time.perf_counter()
    candidates = _root_mixture_candidate_modes(y, k, prior, max_iter=max_iter)
    unique = _root_mixture_deduplicate_modes(candidates)
    laplace_modes: List[_RootMixtureLaplaceMode] = []
    hessian_errors: List[str] = []
    for candidate in unique:
        try:
            laplace_modes.append(_root_mixture_laplace_at_mode(candidate, y, k, prior))
        except Exception as exc:
            hessian_errors.append(f'{candidate.start_name}: {type(exc).__name__}: {exc}')
    if not laplace_modes:
        raise FloatingPointError(f'root-mixture M{k} has no valid Laplace mode. ' + ' | '.join(hessian_errors))
    selected, distances = _root_mixture_select_nonoverlapping_modes(laplace_modes)
    model_logz = float(logsumexp([mode.logz for mode in selected]))
    posterior_mode_weights = np.exp(np.asarray([mode.logz for mode in selected], dtype=float) - model_logz)
    best = max(selected, key=lambda mode: mode.logz)
    return {'k': int(k), 'logz': model_logz, 'runtime_s': float(time.perf_counter() - started), 'candidate_count': int(len(candidates)), 'candidate_start_names': tuple((mode.start_name for mode in candidates)), 'unique_candidate_count': int(len(unique)), 'unique_start_names': tuple((mode.start_name for mode in unique)), 'valid_laplace_count': int(len(laplace_modes)), 'selected_mode_count': int(len(selected)), 'selected_start_names': tuple((mode.optimized.start_name for mode in selected)), 'selected_modes': selected, 'posterior_mode_weights': posterior_mode_weights, 'best_mode': best, 'best_decoded': _root_mixture_decode_theta(best.optimized.theta, k), 'min_hessian_eigenvalue': float(min((mode.min_eigenvalue for mode in selected))), 'max_gradient_inf_norm': float(max((mode.optimized.gradient_inf_norm for mode in selected))), 'hessian_errors': hessian_errors, 'mode_min_distances': distances}

def _root_mixture_model_predictive_logpdf(fit: Dict[str, Any], eval_standardized: np.ndarray, *, raw_log_sd: float) -> np.ndarray:
    terms = []
    for mode, weight in zip(fit['selected_modes'], fit['posterior_mode_weights']):
        linear_weight = float(weight)
        if linear_weight > 0.0:
            log_weight = math.log(linear_weight)
        else:
            log_weight = float(mode.logz) - float(fit['logz'])
            if not np.isfinite(log_weight):
                raise FloatingPointError('root-mixture predictive posterior-mode log weight is non-finite.')
        terms.append(log_weight + _root_mixture_mk_logpdf_standardized(eval_standardized, mode.optimized.theta, int(fit['k'])))
    return logsumexp(np.vstack(terms), axis=0) - float(raw_log_sd)

def _root_mixture_m2_responsibilities(y_standardized: np.ndarray, fit: Dict[str, Any]) -> np.ndarray:
    theta = np.asarray(fit['best_mode'].optimized.theta, dtype=float)
    decoded = _root_mixture_decode_theta(theta, 2)
    sigma = float(decoded['sigma'])
    mu = float(decoded['mu'])
    delta = float(decoded['delta'])
    weights = np.asarray(decoded['weights'], dtype=float)
    offsets = np.asarray(decoded['offsets'], dtype=float)
    t = (np.asarray(y_standardized, dtype=float) - mu) / sigma
    terms = []
    for weight, offset in zip(weights, offsets):
        u = t - float(offset)
        terms.append(math.log(float(weight)) + _root_mixture_centered_skew_logpdf_standard(u, delta) - math.log(sigma))
    stacked = np.vstack(terms).T
    normalizer = logsumexp(stacked, axis=1)
    return np.exp(stacked - normalizer[:, None])

def _root_mixture_hartigan_dip_statistic(values: np.ndarray) -> float:

    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4 or x[0] == x[-1]:
        return 0.0
    low = 0
    high = n - 1
    dip_value = 0.0
    mn = np.zeros(n, dtype=np.int32)
    mn[0] = 0
    for j in range(1, n):
        mn[j] = j - 1
        while True:
            mnj = int(mn[j])
            mnmnj = int(mn[mnj])
            if mnj == 0 or (x[j] - x[mnj]) * (mnj - mnmnj) < (x[mnj] - x[mnmnj]) * (j - mnj):
                break
            mn[j] = mnmnj
    mj = np.zeros(n, dtype=np.int32)
    mj[n - 1] = n - 1
    for k in range(n - 2, -1, -1):
        mj[k] = k + 1
        while True:
            mjk = int(mj[k])
            mjmjk = int(mj[mjk])
            if mjk == n - 1 or (x[k] - x[mjk]) * (mjk - mjmjk) < (x[mjk] - x[mjmjk]) * (k - mjk):
                break
            mj[k] = mjmjk
    gcm = np.zeros(n, dtype=np.int32)
    lcm = np.zeros(n, dtype=np.int32)
    while True:
        gcm[0] = high
        i = 0
        while int(gcm[i]) > low:
            gcm[i + 1] = mn[int(gcm[i])]
            i += 1
        ig = i
        l_gcm = i
        ix = ig - 1
        lcm[0] = low
        i = 0
        while int(lcm[i]) < high:
            lcm[i + 1] = mj[int(lcm[i])]
            i += 1
        ih = i
        l_lcm = i
        iv = 1
        d = 0.0
        if l_gcm != 1 or l_lcm != 1:
            while True:
                gcmix = int(gcm[ix])
                lcmiv = int(lcm[iv])
                if gcmix > lcmiv:
                    gcmi1 = int(gcm[ix + 1])
                    denom = x[gcmix] - x[gcmi1]
                    dx = 0.0 if np.isclose(denom, 0.0, rtol=1e-09, atol=1e-09) else lcmiv - gcmi1 + 1 - (x[lcmiv] - x[gcmi1]) * (gcmix - gcmi1) / denom
                    iv += 1
                    if dx >= d:
                        d = float(dx)
                        ig = ix + 1
                        ih = iv - 1
                else:
                    lcmiv1 = int(lcm[iv - 1])
                    denom = x[lcmiv] - x[lcmiv1]
                    dx = 0.0 if np.isclose(denom, 0.0, rtol=1e-09, atol=1e-09) else (x[gcmix] - x[lcmiv1]) * (lcmiv - lcmiv1) / denom - (gcmix - lcmiv1 - 1)
                    ix -= 1
                    if dx >= d:
                        d = float(dx)
                        ig = ix + 1
                        ih = iv
                ix = max(ix, 0)
                iv = min(iv, l_lcm)
                if int(gcm[ix]) == int(lcm[iv]):
                    break
        if d < dip_value:
            break
        dip_l = 0.0
        for j in range(ig, l_gcm):
            max_t = 1.0
            jb = int(gcm[j + 1])
            je = int(gcm[j])
            if je - jb > 1 and x[je] != x[jb]:
                c = (je - jb) / (x[je] - x[jb])
                jj = np.arange(jb, je + 1, dtype=float)
                values_t = jj - jb + 1.0 - (x[jb:je + 1] - x[jb]) * c
                max_t = max(max_t, float(np.max(values_t)))
            dip_l = max(dip_l, max_t)
        dip_u = 0.0
        for j in range(ih, l_lcm):
            max_t = 1.0
            jb = int(lcm[j])
            je = int(lcm[j + 1])
            if je - jb > 1 and x[je] != x[jb]:
                c = (je - jb) / (x[je] - x[jb])
                jj = np.arange(jb, je + 1, dtype=float)
                values_t = (x[jb:je + 1] - x[jb]) * c - (jj - jb - 1.0)
                max_t = max(max_t, float(np.max(values_t)))
            dip_u = max(dip_u, max_t)
        dip_value = max(dip_value, dip_l, dip_u)
        if low == int(gcm[ig]) and high == int(lcm[ih]):
            break
        low = int(gcm[ig])
        high = int(lcm[ih])
    return float(dip_value / (2.0 * n))

def _root_mixture_scaled_dip(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.0
    return float(math.sqrt(len(x)) * _root_mixture_hartigan_dip_statistic(x))

def _root_mixture_normalize_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError('root-mixture scout weights shape does not match data.')
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError('root-mixture scout weights contain no positive finite mass.')
    return w / total

def _root_mixture_weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cumulative = np.cumsum(w)
    cumulative /= cumulative[-1]
    return float(np.interp(float(q), cumulative, x, left=x[0], right=x[-1]))

def _root_mixture_weighted_mean_sd(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = float(np.sum(w * x))
    variance = float(np.sum(w * (x - mean) ** 2))
    return (mean, math.sqrt(max(variance, 1e-14)))

def _root_mixture_weighted_effective_n(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    return float(1.0 / np.sum(w * w))

def _root_mixture_robust_standardize(values: np.ndarray, weights: np.ndarray | None=None) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    x = x[finite]
    if len(x) < 4:
        raise ValueError('root-mixture KDE needs at least four finite observations.')
    raw_weights = None if weights is None else np.asarray(weights, dtype=float)[finite]
    w = _root_mixture_normalize_weights(raw_weights, len(x))
    q25 = _root_mixture_weighted_quantile(x, w, 0.25)
    q50 = _root_mixture_weighted_quantile(x, w, 0.5)
    q75 = _root_mixture_weighted_quantile(x, w, 0.75)
    iqr_scale = (q75 - q25) / 1.349
    _, sd = _root_mixture_weighted_mean_sd(x, w)
    scale = float(iqr_scale)
    if not np.isfinite(scale) or scale <= 1e-08:
        scale = float(sd)
    if not np.isfinite(scale) or scale <= 1e-08:
        scale = 1.0
    return ((x - q50) / scale, w)

def _root_mixture_meaningful_peak_count(smoothed_density: np.ndarray, empirical_bin_mass: np.ndarray, *, minimum_peak_distance_bins: int) -> int:
    density = np.asarray(smoothed_density, dtype=float)
    mass = np.asarray(empirical_bin_mass, dtype=float)
    if len(density) < 3 or not np.any(density > 0.0):
        return 1
    max_density = float(np.max(density))
    prominence = max(ROOT_MIXTURE_KDE_RELATIVE_PROMINENCE * max_density, 1e-14)
    peaks, _ = find_peaks(density, prominence=prominence, distance=max(1, int(minimum_peak_distance_bins)))
    boundaries = []
    if density[0] > density[1] and density[0] >= prominence:
        boundaries.append(0)
    if density[-1] > density[-2] and density[-1] >= prominence:
        boundaries.append(len(density) - 1)
    if boundaries:
        peaks = np.unique(np.concatenate([peaks, np.asarray(boundaries, dtype=int)]))
    if len(peaks) <= 1:
        return 1
    peaks = np.sort(peaks)
    valleys = [0]
    for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
        segment = density[left_peak:right_peak + 1]
        valleys.append(left_peak + int(np.argmin(segment)))
    valleys.append(len(density) - 1)
    meaningful = 0
    for i in range(len(peaks)):
        lo = int(valleys[i]) + (1 if i > 0 else 0)
        hi = int(valleys[i + 1])
        if hi < lo:
            continue
        if float(np.sum(mass[lo:hi + 1])) >= ROOT_MIXTURE_KDE_MINIMUM_MODE_MASS:
            meaningful += 1
    return max(1, int(meaningful))

def _root_mixture_kde_mode_tree(values: np.ndarray, *, weights: np.ndarray | None=None) -> Dict[str, Any]:
    started = time.perf_counter()
    z, w = _root_mixture_robust_standardize(values, weights)
    n_eff = _root_mixture_weighted_effective_n(w)
    if n_eff < 8.0:
        return {'p2': 0.0, 'p3': 0.0, 'max_modes': 1, 'mode_counts': [1] * len(ROOT_MIXTURE_KDE_BANDWIDTH_MULTIPLIERS), 'n_eff': float(n_eff), 'runtime_s': float(time.perf_counter() - started)}
    q_low = _root_mixture_weighted_quantile(z, w, 0.0025)
    q_high = _root_mixture_weighted_quantile(z, w, 0.9975)
    low = max(float(np.min(z)), q_low - 1.0)
    high = min(float(np.max(z)), q_high + 1.0)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(z)) - 1.0
        high = float(np.max(z)) + 1.0
    counts, edges = np.histogram(z, bins=ROOT_MIXTURE_KDE_GRID_SIZE, range=(low, high), weights=w)
    counts = np.asarray(counts, dtype=float)
    total_mass = float(np.sum(counts))
    if total_mass <= 0.0:
        raise FloatingPointError('root-mixture KDE histogram has zero mass.')
    counts /= total_mass
    dx = float(edges[1] - edges[0])
    q25 = _root_mixture_weighted_quantile(z, w, 0.25)
    q75 = _root_mixture_weighted_quantile(z, w, 0.75)
    _, sd_z = _root_mixture_weighted_mean_sd(z, w)
    robust_sd = min(sd_z, (q75 - q25) / 1.349 if q75 > q25 else sd_z)
    if not np.isfinite(robust_sd) or robust_sd <= 1e-06:
        robust_sd = max(sd_z, 1.0)
    base_bandwidth = max(0.9 * robust_sd * float(n_eff) ** (-0.2), 0.35 * dx)
    mode_counts: List[int] = []
    for multiplier in ROOT_MIXTURE_KDE_BANDWIDTH_MULTIPLIERS:
        bandwidth = float(multiplier) * base_bandwidth
        sigma_bins = max(bandwidth / dx, 0.35)
        smoothed = gaussian_filter1d(counts, sigma=sigma_bins, mode='nearest')
        min_distance = max(1, int(round(0.5 * bandwidth / dx)))
        mode_counts.append(_root_mixture_meaningful_peak_count(smoothed, counts, minimum_peak_distance_bins=min_distance))
    mode_array = np.asarray(mode_counts, dtype=int)
    return {'p2': float(np.mean(mode_array >= 2)), 'p3': float(np.mean(mode_array >= 3)), 'max_modes': int(np.max(mode_array)), 'mode_counts': mode_counts, 'n_eff': float(n_eff), 'runtime_s': float(time.perf_counter() - started)}

def _root_mixture_scout_a(raw_y: np.ndarray) -> Dict[str, Any]:
    started = time.perf_counter()
    scaled_dip = _root_mixture_scaled_dip(raw_y)
    kde = _root_mixture_kde_mode_tree(raw_y)
    stop = bool(scaled_dip <= ROOT_MIXTURE_SCOUT_A_DIP_STOP and float(kde['p2']) <= ROOT_MIXTURE_SCOUT_A_P2_STOP)
    return {'stop': stop, 'scaled_dip': float(scaled_dip), 'p2': float(kde['p2']), 'p3': float(kde['p3']), 'mode_counts': tuple((int(x) for x in kde['mode_counts'])), 'runtime_s': float(time.perf_counter() - started)}

def _root_mixture_scout_b(raw_y: np.ndarray, responsibilities: np.ndarray, *, root_kde_p3: float) -> Dict[str, Any]:
    started = time.perf_counter()
    y = np.asarray(raw_y, dtype=float)
    responsibilities = np.asarray(responsibilities, dtype=float)
    if responsibilities.shape != (len(y), 2):
        raise ValueError('root-mixture Scout B expects n x 2 M2 responsibilities.')
    component_p2: List[float] = []
    component_dips: List[float] = []
    core_counts: List[int] = []
    for component in range(2):
        weights = responsibilities[:, component]
        usable_weights = np.where(
            np.isfinite(weights) & (weights > 0.0),
            weights,
            0.0,
        )

        if float(np.sum(usable_weights)) <= 0.0:
            component_p2.append(0.0)
            component_dips.append(0.0)
            core_counts.append(0)
            continue

        kde = _root_mixture_kde_mode_tree(
            y,
            weights=usable_weights,
        )
        component_p2.append(float(kde["p2"]))

        core = y[usable_weights >= ROOT_MIXTURE_RESPONSIBILITY_CORE_CUTOFF]
        core_counts.append(int(len(core)))

        component_dips.append(
            _root_mixture_scaled_dip(core)
            if len(core) >= ROOT_MIXTURE_MIN_COMPONENT_DIP_COUNT
            else 0.0
        )
    p2_max = float(max(component_p2))
    dip_max = float(max(component_dips))
    stop = bool(float(root_kde_p3) <= ROOT_MIXTURE_SCOUT_B_ROOT_P3_STOP and p2_max <= ROOT_MIXTURE_SCOUT_B_COMPONENT_P2_STOP and (dip_max <= ROOT_MIXTURE_SCOUT_B_COMPONENT_DIP_STOP))
    return {'stop': stop, 'component_p2': tuple(component_p2), 'component_p2_max': p2_max, 'component_dips': tuple(component_dips), 'component_dip_max': dip_max, 'component_core_counts': tuple(core_counts), 'runtime_s': float(time.perf_counter() - started)}

class _RootAwareCScoringEngine(_UnimodalCScoringEngine):








    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._root_mixture_prior = _RootMixtureMixturePrior(intercept_var=float(self.prior.intercept_var), ig_shape=float(self.prior.ig_shape), ig_scale=float(self.prior.ig_scale), dirichlet_alpha=ROOT_MIXTURE_DIRICHLET_ALPHA, log_separation_mean=math.log(ROOT_MIXTURE_SEPARATION_MEDIAN_SIGMA), log_separation_sd=ROOT_MIXTURE_LOG_SEPARATION_SD)
        self.root_mixture_scores: Dict[str, float] = {}
        self.root_mixture_results: Dict[str, _RootMixtureRootResult] = {}
        self.root_mixture_diagnostics: Dict[str, RootMixtureRootScoreDiagnostic] = {}

    def prior_hyperparameters(self) -> Dict[str, float]:
        values = dict(super().prior_hyperparameters())
        values.update({'root_mixture_mixture_dirichlet_alpha': ROOT_MIXTURE_DIRICHLET_ALPHA, 'root_mixture_separation_median_sigma': ROOT_MIXTURE_SEPARATION_MEDIAN_SIGMA, 'root_mixture_log_separation_sd': ROOT_MIXTURE_LOG_SEPARATION_SD, 'root_mixture_mixture_max_iter': ROOT_MIXTURE_MIXTURE_MAX_ITER, 'root_mixture_scout_a_dip_stop': ROOT_MIXTURE_SCOUT_A_DIP_STOP, 'root_mixture_scout_a_p2_stop': ROOT_MIXTURE_SCOUT_A_P2_STOP, 'root_mixture_scout_b_root_p3_stop': ROOT_MIXTURE_SCOUT_B_ROOT_P3_STOP, 'root_mixture_scout_b_component_p2_stop': ROOT_MIXTURE_SCOUT_B_COMPONENT_P2_STOP, 'root_mixture_scout_b_component_dip_stop': ROOT_MIXTURE_SCOUT_B_COMPONENT_DIP_STOP})
        return values

    def preprocessing_metadata(self) -> Dict[str, object]:
        values = dict(super().preprocessing_metadata())
        optimizations = dict(values.get('implementation_optimizations', {}))
        optimizations.update({'root_mixture_root_only': True, 'root_mixture_reuse_scout_a_root_kde_p3_in_scout_b': True, 'root_mixture_reuse_m2_fit_for_scout_b_responsibilities': True, 'root_mixture_no_m2_or_m3_fit_when_scout_a_stops': True, 'root_mixture_no_m3_fit_when_scout_b_stops': True, 'root_mixture_lbfgsb_joint_fun_jac': True, 'root_mixture_reuse_kmeans_initialization_across_skew_starts': True})
        values.update({'combined_continuous_v3_version': SCORER_VERSION, 'root_mixture_enabled': True, 'root_mixture_scope': 'continuous roots only', 'root_mixture_hierarchy': 'U -> ScoutA -> M2 -> ScoutB -> M3', 'root_mixture_model_selection': 'argmax marginal log evidence among available branches', 'root_mixture_model_index_prior': 'none introduced; Part-B evidence-selection convention', 'root_mixture_mixture_prior_profile': ROOT_MIXTURE_MIXTURE_PRIOR_PROFILE, 'root_mixture_scout_kde_profile': ROOT_MIXTURE_SCOUT_KDE_PROFILE, 'implementation_optimizations': optimizations})
        return values

    def score_local(self, child: str, parents: Sequence[str]) -> float:
        child = str(child)
        parents = self._validate_family(child, parents)
        if len(parents) != 0:
            return float(super().score_local(child, parents))
        if child in self.root_mixture_scores:
            return float(self.root_mixture_scores[child])
        total_started = time.perf_counter()
        u_logz = float(super().score_local(child, ()))
        base_diag = self.diagnostics[child, ()]
        raw_y = self.raw_data[child].to_numpy(dtype=float)
        y_standardized = self._training_y_standardized[child]
        scout_a = _root_mixture_scout_a(raw_y)
        m2_fit = None
        m3_fit = None
        scout_b = None
        m2_runtime = 0.0
        m3_runtime = 0.0
        scout_b_runtime = 0.0
        logz_m2 = float('nan')
        logz_m3 = float('nan')
        if bool(scout_a['stop']):
            available = ('U',)
            selected = 'U'
            selected_logz = u_logz
        else:
            m2_fit = _root_mixture_model_laplace(y_standardized, 2, self._root_mixture_prior, max_iter=ROOT_MIXTURE_MIXTURE_MAX_ITER)
            m2_runtime = float(m2_fit['runtime_s'])
            logz_m2 = float(m2_fit['logz'])
            responsibilities = _root_mixture_m2_responsibilities(y_standardized, m2_fit)
            scout_b = _root_mixture_scout_b(raw_y, responsibilities, root_kde_p3=float(scout_a['p3']))
            scout_b_runtime = float(scout_b['runtime_s'])
            if bool(scout_b['stop']):
                available = ('U', 'M2')
                model_logz = {'U': u_logz, 'M2': logz_m2}
            else:
                m3_fit = _root_mixture_model_laplace(y_standardized, 3, self._root_mixture_prior, max_iter=ROOT_MIXTURE_MIXTURE_MAX_ITER)
                m3_runtime = float(m3_fit['runtime_s'])
                logz_m3 = float(m3_fit['logz'])
                available = ('U', 'M2', 'M3')
                model_logz = {'U': u_logz, 'M2': logz_m2, 'M3': logz_m3}
            selected = max(model_logz, key=model_logz.get)
            selected_logz = float(model_logz[selected])
        if not np.isfinite(selected_logz):
            raise FloatingPointError(f'Non-finite RootMixture root score for {child!r}.')
        component_p2_max = float(scout_b['component_p2_max']) if scout_b is not None else float('nan')
        component_dip_max = float(scout_b['component_dip_max']) if scout_b is not None else float('nan')
        component_core_counts = tuple((int(x) for x in scout_b['component_core_counts'])) if scout_b is not None else (0, 0)
        root_result = _RootMixtureRootResult(child=child, u_logz=u_logz, scout_a_stop=bool(scout_a['stop']), root_scaled_dip=float(scout_a['scaled_dip']), root_kde_p2=float(scout_a['p2']), root_kde_p3=float(scout_a['p3']), root_kde_mode_path=tuple((int(x) for x in scout_a['mode_counts'])), m2_fit=m2_fit, scout_b_stop=bool(scout_b['stop']) if scout_b is not None else None, component_p2_max=component_p2_max, component_dip_max=component_dip_max, component_core_counts=component_core_counts, m3_fit=m3_fit, available_models=tuple(available), selected_model=str(selected), selected_logz=float(selected_logz))
        if bool(scout_a['stop']):
            action = 'ROOT_MIXTURE_SCOUT_A_STOP_U'
        elif scout_b is not None and bool(scout_b['stop']):
            action = f'ROOT_MIXTURE_SCOUT_B_STOP_SELECT_{selected}'
        else:
            action = f'ROOT_MIXTURE_M3_AVAILABLE_SELECT_{selected}'
        diagnostic_values = asdict(base_diag)
        diagnostic_values.update({'method': 'combined_v3_root_mixture_balanced_hierarchical_root', 'score': float(selected_logz), 'action': action, 'runtime_seconds': float(time.perf_counter() - total_started)})
        diagnostic = RootMixtureRootScoreDiagnostic(**diagnostic_values, u_score=u_logz, u_action=str(base_diag.action), stage3_selected_model=str(selected), stage3_available_models=tuple(available), stage3_scout_a_stop=bool(scout_a['stop']), root_mixture_scaled_dip=float(scout_a['scaled_dip']), root_mixture_kde_p2=float(scout_a['p2']), root_mixture_kde_p3=float(scout_a['p3']), root_mixture_kde_mode_path=tuple((int(x) for x in scout_a['mode_counts'])), stage3_scout_b_stop=bool(scout_b['stop']) if scout_b is not None else None, stage3_component_p2_max=component_p2_max, stage3_component_dip_max=component_dip_max, stage3_component_core_counts=component_core_counts, stage3_logz_m2=logz_m2, stage3_logz_m3=logz_m3, stage3_m2_calls=int(m2_fit is not None), stage3_m3_calls=int(m3_fit is not None), stage3_scout_a_runtime_seconds=float(scout_a['runtime_s']), stage3_m2_runtime_seconds=m2_runtime, stage3_scout_b_runtime_seconds=scout_b_runtime, stage3_m3_runtime_seconds=m3_runtime)
        self.root_mixture_scores[child] = float(selected_logz)
        self.root_mixture_results[child] = root_result
        self.root_mixture_diagnostics[child] = diagnostic
        return float(selected_logz)

    def get_score_diagnostic(self, child: str, parents: Sequence[str]):
        child = str(child)
        parents = self._validate_family(child, parents)
        if len(parents) == 0:
            if child not in self.root_mixture_diagnostics:
                self.score_local(child, parents)
            return self.root_mixture_diagnostics[child]
        return super().get_score_diagnostic(child, parents)

    def diagnostics_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for key, base_diag in self.diagnostics.items():
            child, parents = key
            if len(parents) == 0 and child in self.root_mixture_diagnostics:
                rows.append(asdict(self.root_mixture_diagnostics[child]))
            else:
                rows.append(asdict(base_diag))
        return pd.DataFrame(rows)

    def local_predictive_logpdf(self, child: str, parents: Sequence[str], raw_eval: pd.DataFrame) -> np.ndarray:
        child = str(child)
        parents = self._validate_family(child, parents)
        if len(parents) != 0:
            return super().local_predictive_logpdf(child, parents, raw_eval)
        self.score_local(child, parents)
        root_result = self.root_mixture_results[child]
        if root_result.selected_model == 'U':
            return super().local_predictive_logpdf(child, parents, raw_eval)
        standardized_eval = self._standardize_eval(raw_eval)
        y_eval_standardized = standardized_eval[child].to_numpy(dtype=float)
        raw_log_sd = math.log(float(self.stds[child]))
        if root_result.selected_model == 'M2':
            if root_result.m2_fit is None:
                raise RuntimeError('RootMixture selected M2 without a cached M2 fit.')
            return _root_mixture_model_predictive_logpdf(root_result.m2_fit, y_eval_standardized, raw_log_sd=raw_log_sd)
        if root_result.selected_model == 'M3':
            if root_result.m3_fit is None:
                raise RuntimeError('RootMixture selected M3 without a cached M3 fit.')
            return _root_mixture_model_predictive_logpdf(root_result.m3_fit, y_eval_standardized, raw_log_sd=raw_log_sd)
        raise RuntimeError(f'Unknown RootMixture selected root model {root_result.selected_model!r}.')
V3CombinedScoringEngine = _RootAwareCScoringEngine
V3CombinedRootMixtureScoringEngine = _RootAwareCScoringEngine

_FlatBaseV3CScoringEngine = _RootAwareCScoringEngine

                                                                         
class _A2API:
    _loglik_and_grad_skew = _a2__loglik_and_grad_skew
    _repairing_kmeans_1d = _a2__repairing_kmeans_1d
    centered_skew_logpdf_and_grad_standard = _a2_centered_skew_logpdf_and_grad_standard
    centered_skew_logpdf_standard = _a2_centered_skew_logpdf_standard

class _PBAPI:
    KAPPA_BOUNDS = _pb_KAPPA_BOUNDS
    LaplaceMode = _pb_LaplaceMode
    OptimizedMode = _pb_OptimizedMode
    PriorProfile = _pb_PriorProfile
    _LOG_2PI = _pb__LOG_2PI
    _combine_laplace_modes = _pb__combine_laplace_modes
    _decode_mixture = _pb__decode_mixture
    _deduplicate_modes = _pb__deduplicate_modes
    _fit_skew_evidence = _pb__fit_skew_evidence
    _gaussian_logz_exact = _pb__gaussian_logz_exact
    _helmert_basis = _pb__helmert_basis
    _laplace_mode = _pb__laplace_mode
    _log_delta_prior_kappa = _pb__log_delta_prior_kappa
    _log_ref_scale_prior = _pb__log_ref_scale_prior
    _mixture_bounds = _pb__mixture_bounds
    _mixture_initial_theta = _pb__mixture_initial_theta
    _mixture_log_joint_and_gradient = _pb__mixture_log_joint_and_gradient
    _mixture_loglik_and_gradient = _pb__mixture_loglik_and_gradient
    _optimize_mode = _pb__optimize_mode
    gammaln = gammaln

class _PDAPI:
    scaled_dip = _pd_scaled_dip
    kde_mode_tree = _pd_kde_mode_tree

class _PTAPI:
    PyTensorScaleTemplate = _pt_PyTensorScaleTemplate

class _BaseAPI:
    V3CScoringEngine = _FlatBaseV3CScoringEngine

_a2 = _A2API
_pb = _PBAPI
_pd = _PDAPI
_pt = _PTAPI
_base_mod = _BaseAPI
_mean = _MeanAPI

                                                    
SCORER_VERSION = "v3-cc-monolithic-partH-candidate-v1"
SCORER_PATCH = "corrected-U-plus-partG-frozen-plus-embedded-namespace-v2+m2-hand-rescue-v1"
CONDITIONAL_STATE_MASS = 0.99
CONDITIONAL_STATE_CAP = 128

_A_PROM, _A_MASS, _A_DIP, _A_P2 = 0.060, 0.070, 0.45, 1.0 / 3.0
_B_PROM, _B_MASS, _B_CORE = 0.025, 0.030, 0.75
_B_P3, _B_P2, _B_DIP, _B_MIN_CORE = 0.0, 1.0 / 6.0, 0.25, 20
_LOG_HALF = _math.log(0.5)
_LOG_2PI = _math.log(2.0 * _math.pi)




                                                                        
_original_helmert = _pb._helmert_basis
_HELMERT = {
    2: np.asarray(_original_helmert(2), dtype=float),
    3: np.asarray(_original_helmert(3), dtype=float),
}
def _cached_helmert(k):
    k = int(k)
    if k not in _HELMERT:
        _HELMERT[k] = np.asarray(_original_helmert(k), dtype=float)
    return _HELMERT[k]
_pb._helmert_basis = _cached_helmert
                                                                             
_pb__helmert_basis = _cached_helmert


@_dataclass
class _ResidualFit:
    model: str
    logz: float
    dominant_theta: np.ndarray
    retained_modes: tuple
    classification: str
    separation: float
    candidate_modes: int
    min_effective_count: float
    runtime_s: float


@_dataclass
class _Decision:
    idx: int
    rank: int
    state: tuple
    model: str
    uplift: float
    beta: np.ndarray
    logz_u: float
    logz_g: float
    logz_s: float
    logz_m2: float
    logz_m3: float
    scout_a: dict
    scout_b: dict | None
    fit: object | None


@_dataclass
class _FamilyResult:
    child: str
    parents: tuple
    base_score: float
    score: float
    base_result: object | None
    state_logjoint: np.ndarray | None
    chosen: tuple
    chosen_mass: float
    cap_hit: bool
    decisions: dict
    root: bool
    runtime_s: float


def _select_states(logjoint, mass=0.99, cap=128):
    x = np.asarray(logjoint, dtype=float)
    p = np.exp(x - float(_logsumexp(x)))
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    needed = int(np.searchsorted(cum, float(mass), side="left") + 1)
    count = min(needed, int(cap), len(x))
    chosen = np.asarray(order[:count], dtype=int)
    return chosen, float(np.sum(p[chosen])), bool(needed > int(cap))


def _state_beta(result, idx):
    idx = int(idx)
    state = tuple(result.states[idx])
    gfit = result.family.fit_state(state)
    beta_g = np.asarray(gfit.beta_mean, dtype=float)

    if idx not in set(int(x) for x in result.escalated_indices):
        return beta_g.copy()

    multi = result.multi_cache[idx]
    modes = list(multi.get("predictive_modes", []))
    lw = np.asarray(multi.get("predictive_log_weights", [0.0]), dtype=float)
    if not modes:
        return beta_g.copy()

    w = np.exp(lw - _logsumexp(lw))
    beta_s = np.zeros_like(beta_g)
    for wi, mode in zip(w, modes):
        beta_s += float(wi) * np.asarray(mode["beta_hat"], dtype=float)

    logs = np.asarray(
        [
            _LOG_HALF + float(gfit.log_evidence),
            _LOG_HALF + float(multi["logz"]),
        ]
    )
    bw = np.exp(logs - _logsumexp(logs))
    return float(bw[0]) * beta_g + float(bw[1]) * beta_s


def _starts(y, k):
    y = np.asarray(y, dtype=float)
    k = int(k)
    base = np.asarray(_pb._mixture_initial_theta(y, k, 0.0, _a2), dtype=float)
    lo, hi = _pb.KAPPA_BOUNDS
    out = []
    for name, kapp in (
        ("ZERO", 0.0),
        ("POS", _math.asinh(5.0)),
        ("NEG", -_math.asinh(5.0)),
    ):
        t = base.copy()
        t[k] = np.clip(float(kapp), lo, hi)
        out.append((name, t))
    return tuple(out)


def _m3_joint(theta, y, k, prior, a2):
                                                                                  
    if int(k) != 3:
        return _pb._mixture_log_joint_and_gradient(theta, y, k, prior, a2)

    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    d = _pb._decode_mixture(theta, 3)
    sigmas = np.asarray(d["sigmas"], dtype=float)
    weights = np.asarray(d["weights"], dtype=float)
    loc = np.asarray(d["locations"], dtype=float)
    kappa = float(d["kappa"])
    ref_sigma = float(d["sigma_ref"])
    b = np.asarray(d["b"], dtype=float)
    bbar = float(d["bbar"])
    ratios = np.asarray(d["ratios"], dtype=float)

    u = (y[:, None] - loc[None, :]) / sigmas[None, :]
    logf, ddu, ddk = a2.centered_skew_logpdf_and_grad_standard(u, kappa)
    terms = np.log(weights)[None, :] + logf - np.log(sigmas)[None, :]
    logmix = _logsumexp(terms, axis=1)
    if not np.all(np.isfinite(logmix)):
        return -np.inf, np.full_like(theta, np.nan)

    resp = np.exp(terms - logmix[:, None])
    G = resp * ddu
    rsum = np.sum(resp, axis=0)

    grad_ref = float(
        -len(y)
        - np.sum(G * u)
        - np.sum(G * (loc / sigmas)[None, :])
    )
    expanded_width_grad = ( -rsum - np.sum(G*u, axis=0))
    grad_gamma = (np.asarray(d["H"], dtype=float).T @ expanded_width_grad)
    grad_kappa = float(np.sum(resp * ddk))
    weighted_G_scale = float(np.sum(G / sigmas[None, :]))

    grad_logits = np.empty(2)
    for ell in range(2):
        location_shift = ref_sigma * weights[ell] * (b[ell] - bbar)
        grad_logits[ell] = (
            rsum[ell] - len(y) * weights[ell]
            + location_shift * weighted_G_scale
        )

    grad_eta = np.empty(2)
    for ell in range(2):
        indicator = np.zeros(3)
        indicator[ell + 1 :] = 1.0
        tail = float(np.sum(weights[ell + 1 :]))
        dloc = ref_sigma * ratios[ell] * (indicator - tail)
        du = -dloc/sigmas
        grad_eta[ell] = float(np.sum(G * du[None, :]))

    gl = np.concatenate(
        [
            np.asarray([grad_ref]),
            np.asarray(grad_gamma),
            np.asarray([grad_kappa]),
            grad_logits,
            grad_eta,
        ]
    )
    ll = float(np.sum(logmix))
    if not np.isfinite(ll) or not np.all(np.isfinite(gl)):
        return -np.inf, np.full_like(theta, np.nan)

    gp = np.zeros_like(theta)
    cursor = 0
    lp_scale, g_scale = _pb._log_ref_scale_prior(
        d["log_sigma_ref"], prior.ig_shape, prior.ig_scale
    )
    gp[cursor] = g_scale
    cursor += 1

    gamma = np.asarray(d["gamma"])
    tau = float(prior.scale_contrast_sd)
    lp_gamma = float(
        np.sum(-0.5 * (_pb._LOG_2PI + 2.0 * _math.log(tau) + (gamma / tau) ** 2))
    )
    gp[cursor : cursor + 2] = -gamma / (tau * tau)
    cursor += 2

    lp_delta, g_delta = _pb._log_delta_prior_kappa(float(d["kappa"]))
    gp[cursor] = g_delta
    cursor += 1

    alpha = float(prior.dirichlet_alpha)
    lp_weights = float(
        _pb.gammaln(3.0 * alpha)
        - 3.0 * _pb.gammaln(alpha)
        + alpha * np.sum(np.log(weights))
    )
    gp[cursor : cursor + 2] = alpha * (1.0 - 3.0 * weights[:-1])
    cursor += 2

    eta = np.asarray(d["eta"])
    sd = float(prior.log_separation_sd)
    z = (eta - float(prior.log_separation_mean)) / sd
    lp_eta = float(
        np.sum(-0.5 * (_pb._LOG_2PI + 2.0 * _math.log(sd) + z * z))
    )
    gp[cursor : cursor + 2] = -(
        eta - float(prior.log_separation_mean)
    ) / (sd * sd)

    lp = float(lp_scale + lp_gamma + lp_delta + lp_weights + lp_eta)
    return (float(ll+lp), np.asarray(gl+gp, dtype=float))


def _fit_m3(y, prior):
    started = _time.perf_counter()
    y = np.asarray(y, dtype=float)
    fg = lambda x: _m3_joint(x, y, 3, prior, _a2)
    modes, errors = [], []
    bounds = _pb._mixture_bounds(3)
    for name, theta0 in _starts(y, 3):
        try:
            modes.append(_pb._optimize_mode("M3", name, theta0, y, fg, bounds))
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    modes = _pb._deduplicate_modes(modes)
    if not modes:
        raise FloatingPointError("All M3 starts failed: " + " | ".join(errors))

    lap = []
    for mode in modes:
        try:
            lap.append(_pb._laplace_mode(mode, fg))
        except Exception:
            pass
    logz, classification, separation, kept = _pb._combine_laplace_modes(lap)
    dom = max(kept, key=lambda m: m.logz)
    _, _, resp = _pb._mixture_loglik_and_gradient(dom.optimized.theta, y, 3, _a2)
    return _ResidualFit(
        "M3", float(logz), np.asarray(dom.optimized.theta),
        tuple(kept), str(classification), float(separation), len(modes),
        float(np.min(resp.sum(axis=0))),
        float(_time.perf_counter() - started),
    )


def _pt_mode(template, y, theta0, name, bounds):
    started = _time.perf_counter()
    result = _minimize(
        fun=lambda x: template.objective_gradient(x, y),
        x0=np.asarray(theta0, dtype=float),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6, "maxls": 30},
    )
    if (not result.success) or (not np.isfinite(result.fun)):
        result = _minimize(
            fun=lambda x: template.objective_gradient(x, y),
            x0=np.asarray(result.x, dtype=float),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-11, "gtol": 5e-7, "maxls": 40},
        )
    theta = np.asarray(result.x, dtype=float)
    neg, grad = template.objective_gradient(theta, y)
    gi = float(np.linalg.norm(grad, ord=np.inf))
    if not np.isfinite(neg) or not np.all(np.isfinite(grad)):
        raise FloatingPointError(f"Invalid PyTensor M2 {name} MAP")
    if (not result.success) and gi > 2e-3:
        raise FloatingPointError(f"PyTensor M2 {name} did not converge")
    return _pb.OptimizedMode(
        "M2", str(name), theta, -float(neg),
        int(getattr(result, "nit", -1)), gi,
        float(_time.perf_counter() - started),
        bool(result.success), str(result.message),
    )


def _pt_laplace(template, mode, y):
    started = _time.perf_counter()
    theta = np.asarray(mode.theta, dtype=float)
    try:
        H, min_eig, logdet = template.validate_hessian(
            template.exact_hessian(theta, y)
        )
    except Exception:
        H, min_eig, logdet = template.validate_hessian(
            template.fd_hessian(theta, y, relative_step=2e-4)
        )
    neg, _ = template.objective_gradient(theta, y)
    lj = -float(neg)
    logz = float(lj + 0.5 * len(theta) * _LOG_2PI - 0.5 * float(logdet))
    om = _pb.OptimizedMode(
        "M2", mode.start_name, theta, lj, mode.iterations,
        mode.gradient_inf_norm, mode.runtime_s, mode.success, mode.message
    )
    return _pb.LaplaceMode(
        om, np.asarray(H), np.linalg.inv(H), float(min_eig),
        float(logdet), logz, float(_time.perf_counter() - started)
    )


def _fit_m2(y, template):
    started = _time.perf_counter()
    y = np.asarray(y, dtype=float)
    modes, errors = [], []
    bounds = list(_pb._mixture_bounds(2))
    for name, theta0 in _starts(y, 2):
        try:
            modes.append(_pt_mode(template, y, theta0, name, bounds))
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    modes = _pb._deduplicate_modes(modes)
    if not modes:
        raise FloatingPointError("All M2 starts failed: " + " | ".join(errors))

    lap = []
    for mode in modes:
        try:
            lap.append(_pt_laplace(template, mode, y))
        except Exception:
            pass
    logz, classification, separation, kept = _pb._combine_laplace_modes(lap)
    dom = max(kept, key=lambda m: m.logz)
    _, _, resp = _pb._mixture_loglik_and_gradient(dom.optimized.theta, y, 2, _a2)
    return _ResidualFit(
        "M2", float(logz), np.asarray(dom.optimized.theta),
        tuple(kept), str(classification), float(separation), len(modes),
        float(np.min(resp.sum(axis=0))),
        float(_time.perf_counter() - started),
    )


def _scout_a(y):
    dip, _ = _pd.scaled_dip(np.asarray(y, dtype=float))
    kde = _pd.kde_mode_tree(
        np.asarray(y, dtype=float), weights=None,
        relative_prominence=_A_PROM, minimum_mode_mass=_A_MASS
    )
    return {
        "stop": bool(float(dip) <= _A_DIP and float(kde["p2"]) <= _A_P2),
        "scaled_dip": float(dip),
        "p2": float(kde["p2"]),
        "p3": float(kde["p3"]),
        "mode_counts": tuple(kde.get("mode_counts", ())),
    }


def _scout_b(y, r, p3):
    y, r = np.asarray(y, dtype=float), np.asarray(r, dtype=float)
    p2s, dips, cores = [], [], []
    for k in range(2):
        w = r[:, k]
        usable_w = np.where(np.isfinite(w) & (w>0.0), w, 0.0)
        if float(np.sum(usable_w)) <= 0.0:
            p2s.append(0.0)
            dips.append(0.0)
            cores.append(0.0)
            continue
        kde = _pd.kde_mode_tree(
            y, weights=usable_w, relative_prominence=_B_PROM,
            minimum_mode_mass=_B_MASS
        )
        p2s.append(float(kde["p2"]))
        core = y[usable_w >= _B_CORE]
        cores.append(int(len(core)))
        if len(core) >= _B_MIN_CORE:
            d, _ = _pd.scaled_dip(core)
            dips.append(float(d))
        else:
            dips.append(0.0)
    p2m, dipm = max(p2s), max(dips)
    return {
        "stop": bool(float(p3) <= _B_P3 and p2m <= _B_P2 and dipm <= _B_DIP),
        "component_p2_max": float(p2m),
        "component_dip_max": float(dipm),
        "component_core_counts": tuple(cores),
    }


def _residual_logpdf(values, fit, k):
    values = np.asarray(values, dtype=float)
    rows = []
    for lm in fit.retained_modes:
        d = _pb._decode_mixture(np.asarray(lm.optimized.theta), int(k))
        sig = np.asarray(d["sigmas"])
        w = np.asarray(d["weights"])
        loc = np.asarray(d["locations"])
        u = (values[:, None] - loc[None, :]) / sig[None, :]
        lf = _a2.centered_skew_logpdf_standard(u, float(d["delta"]))
        lq = _logsumexp(
            np.log(w)[None, :] + lf - np.log(sig)[None, :], axis=1
        )
        rows.append(float(lm.logz) - float(fit.logz) + lq)
    if not rows:
        raise FloatingPointError(f"{fit.model} has no retained modes")
    return _logsumexp(np.vstack(rows), axis=0)



                        
def _fit_m2_hand_rescue(residual, prior):

    y = np.asarray(residual, dtype=float)
    started = _time.perf_counter()
    starts = (
        ("ZERO", 0.0),
        ("POS", _math.asinh(5.0)),
        ("NEG", -_math.asinh(5.0)),
    )
    modes = []
    start_errors = []
    fg = lambda x: _pb._mixture_log_joint_and_gradient(x, y, 2, prior, _a2)
    bounds = _pb._mixture_bounds(2)

    for name, kappa in starts:
        try:
            theta0 = _pb._mixture_initial_theta(y, 2, kappa, _a2)
            modes.append(_pb._optimize_mode("M2", name, theta0, y, fg, bounds))
        except Exception as exc:
            start_errors.append(f"{name}:{type(exc).__name__}:{exc}")

    modes = _pb._deduplicate_modes(modes)
    if not modes:
        raise FloatingPointError(
            "Hand M2 rescue: all posterior starts failed: " + " | ".join(start_errors)
        )

    lap = []
    lap_errors = []
    for mode in modes:
        try:
            lap.append(_pb._laplace_mode(mode, fg))
        except Exception as exc:
            lap_errors.append(f"{mode.start_name}:{type(exc).__name__}:{exc}")

    if not lap:
        raise FloatingPointError(
            "Hand M2 rescue: no valid Laplace modes: " + " | ".join(lap_errors)
        )

    logz, classification, separation, kept = _pb._combine_laplace_modes(lap)
    if not kept:
        raise FloatingPointError("Hand M2 rescue: no retained Laplace modes.")

    dominant = max(kept, key=lambda m: float(m.logz))
    _, _, resp = _pb._mixture_loglik_and_gradient(
        np.asarray(dominant.optimized.theta, dtype=float), y, 2, _a2
    )
    min_eff = float(np.min(np.asarray(resp, dtype=float).sum(axis=0)))

    return _ResidualFit(
        model="M2",
        logz=float(logz),
        dominant_theta=np.asarray(dominant.optimized.theta, dtype=float),
        retained_modes=tuple(kept),
        classification=str(classification),
        separation=float(separation),
        candidate_modes=int(len(modes)),
        min_effective_count=min_eff,
        runtime_s=float(_time.perf_counter() - started),
    )


class V3CScoringEngine:


    def __init__(
        self, data, *,
        conditional_state_mass=CONDITIONAL_STATE_MASS,
        conditional_state_cap=CONDITIONAL_STATE_CAP,
        **base_kwargs
    ):
        self._base = _base_mod.V3CScoringEngine(data, **base_kwargs)
        self.nodes = tuple(self._base.nodes)
        self.n = int(self._base.n)
        self.means = self._base.means
        self.stds = self._base.stds
        self.feature_bank = self._base.feature_bank
        self.conditional_state_mass = float(conditional_state_mass)
        self.conditional_state_cap = int(conditional_state_cap)

        self._prior = _pb.PriorProfile(name="BALANCED", scale_contrast_sd=0.6)
        self._pt_template = None
        self._pt_compile_seconds = 0.0

        self.cache = {}
        self.results = {}
        self.diagnostics = {}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._base, name)

    def _validate_family(self, child, parents):
        return tuple(self._base._validate_family(
            str(child), tuple(str(x) for x in parents)
        ))

    def _m2_template(self):
        if self._pt_template is None:
            self._pt_template = _pt.PyTensorScaleTemplate(
                2, ig_shape=1.5, ig_scale=0.5,
                dirichlet_alpha=2.0,
                log_separation_mean=_math.log(2.5),
                log_separation_sd=1.0,
                scale_contrast_sd=0.6,
            )
            self._pt_compile_seconds = float(self._pt_template.compile_seconds)
        return self._pt_template

    def _u(self, residual):
        residual = np.asarray(residual, dtype=float)
        lg = float(_pb._gaussian_logz_exact(residual))
        ctx = {
            "generator": "final", "case": "statewise",
            "case_name": "statewise", "sample_size": len(residual),
            "n": len(residual), "replicate": 0,
            "prior_profile": "BALANCED", "profile": "BALANCED",
            "role": "production",
        }
        sfit = _pb._fit_skew_evidence(residual, _a2, [], ctx)
        ls = float(sfit.logz)
        lu = float(_logsumexp([_LOG_HALF + lg, _LOG_HALF + ls]))
        return lg, ls, lu

    def _one_state(self, residual, idx, rank, state, beta):
        sa = _scout_a(residual)
        if sa["stop"]:
            return _Decision(
                idx, rank, tuple(state), "U", 0.0, np.asarray(beta),
                np.nan, np.nan, np.nan, np.nan, np.nan,
                sa, None, None
            )

        lg, ls, lu = self._u(residual)
        try:
            m2 = _fit_m2(residual, self._m2_template())
        except FloatingPointError as _pt_m2_exc:
            try:
                m2 = _fit_m2_hand_rescue(residual, self._prior)
            except FloatingPointError as _hand_m2_exc:
                                                                             
                                                                           
                                                                      
                sa = dict(sa)
                sa["m2_numerical_rescue"] = "U_FALLBACK"
                sa["m2_primary_failure"] = (
                    f"{type(_pt_m2_exc).__name__}: {_pt_m2_exc}"
                )
                sa["m2_hand_failure"] = (
                    f"{type(_hand_m2_exc).__name__}: {_hand_m2_exc}"
                )
                return _Decision(
                    idx, rank, tuple(state), "U", 0.0, np.asarray(beta),
                    float(lu), float(lg), float(ls), np.nan, np.nan,
                    sa, None, None
                )
        _, _, resp = _pb._mixture_loglik_and_gradient(
            m2.dominant_theta, np.asarray(residual), 2, _a2
        )
        sb = _scout_b(residual, resp, sa["p3"])
                                
        if sb["stop"]:
            m3 = None
        else:
            try:
                m3 = _fit_m3(residual, self._prior)
            except FloatingPointError as _m3_exc:
                sb = dict(sb)
                sb["m3_numerical_rescue"] = "M3_UNAVAILABLE"
                sb["m3_failure"] = f"{type(_m3_exc).__name__}: {_m3_exc}"
                m3 = None

        available = {"U": lu, "M2": float(m2.logz)}
        if m3 is not None:
            available["M3"] = float(m3.logz)
        model = max(available, key=available.get)
        fit = m2 if model == "M2" else (m3 if model == "M3" else None)
        return _Decision(
            idx, rank, tuple(state), model,
            float(available[model] - lu), np.asarray(beta),
            float(lu), float(lg), float(ls), float(m2.logz),
            float(m3.logz) if m3 is not None else np.nan,
            sa, sb, fit
        )

    def score_local(self, child, parents):
        child = str(child)
        parents = self._validate_family(child, parents)
        key = (child, parents)
        if key in self.cache:
            return float(self.cache[key])

        started = _time.perf_counter()
        base_score = float(self._base.score_local(child, parents))

        if not parents:
            r = _FamilyResult(
                child, parents, base_score, base_score, None, None,
                (), 0.0, False, {}, True,
                float(_time.perf_counter() - started)
            )
            self.cache[key] = base_score
            self.results[key] = r
            self.diagnostics[key] = {
                "child": child, "parents": parents, "parent_count": 0,
                "score": base_score, "base_score": base_score,
                "conditional_uplift": 0.0, "root_preserved": True,
                "runtime_seconds": r.runtime_s,
            }
            return base_score

        br = self._base.combined_results[key]
        old = np.asarray(br.combined_state_logjoint, dtype=float)
        chosen, mass, cap_hit = _select_states(
            old, self.conditional_state_mass, self.conditional_state_cap
        )
        new = old.copy()
        ystd = np.asarray(self._base._training_y_standardized[child], dtype=float)
        decisions = {}

        for rank, idx in enumerate(chosen):
            idx = int(idx)
            state = tuple(br.states[idx])
            sf = br.family.fit_state(state)
            X = self._base._training_design_for_fit(br.family, sf)
            beta = _state_beta(br, idx)
            d = self._one_state(ystd - X @ beta, idx, rank, state, beta)
            new[idx] += float(d.uplift)
            decisions[idx] = d

        score = float(_logsumexp(new))
        r = _FamilyResult(
            child, parents, base_score, score, br, new,
            tuple(int(x) for x in chosen), float(mass), bool(cap_hit),
            decisions, False, float(_time.perf_counter() - started)
        )
        self.cache[key] = score
        self.results[key] = r
        self.diagnostics[key] = {
            "child": child, "parents": parents, "parent_count": len(parents),
            "score": score, "base_score": base_score,
            "conditional_uplift": float(score - base_score),
            "selected_state_count": len(chosen),
            "selected_state_mass": float(mass),
            "state_cap_hit": bool(cap_hit),
            "state_count": len(old),
            "scout_a_stop_count": sum(d.scout_a["stop"] for d in decisions.values()),
            "m2_state_count": sum(not d.scout_a["stop"] for d in decisions.values()),
            "m3_state_count": sum(
                np.isfinite(d.logz_m3) for d in decisions.values()
            ),
            "root_preserved": False,
            "runtime_seconds": r.runtime_s,
        }
        return score

    def get_score_diagnostic(self, child, parents):
        parents = self._validate_family(child, parents)
        key = (str(child), parents)
        if key not in self.diagnostics:
            self.score_local(child, parents)
        return dict(self.diagnostics[key])

    def diagnostics_dataframe(self):
        return pd.DataFrame(list(self.diagnostics.values()))

    def prior_hyperparameters(self):
        out = dict(self._base.prior_hyperparameters())
        out.update({
            "conditional_residual_U": "0.5G+0.5S",
            "conditional_scale_contrast_sd": 0.6,
            "conditional_state_mass": self.conditional_state_mass,
            "conditional_state_cap": self.conditional_state_cap,
        })
        return out

    def preprocessing_metadata(self):
        out = dict(self._base.preprocessing_metadata())
        out.update({
            "final_cc_version": SCORER_VERSION,
            "final_cc_patch": SCORER_PATCH,
            "single_file": True,
            "corrected_U": True,
            "partG": {
                "shared_initialization": True,
                "pytensor_m2_exact": True,
                "cached_helmert": True,
                "single_decode_hand_m3": True,
                "pytensor_m3": False,
                "threaded_starts": False,
                "fused_forward": False,
            },
            "pytensor_m2_compile_seconds": self._pt_compile_seconds
        })
        return out

    def _base_state_logpdf(self, child, br, idx, standardized, master):
        state = tuple(br.states[idx])
        fam = br.family
        child_log_sd = _math.log(float(self.stds[child]))
        lqg = np.asarray(
            fam.state_logpdf_standardized(state, standardized, master)
        ) - child_log_sd
        if idx not in set(int(x) for x in br.escalated_indices):
            return lqg

        sf = fam.fit_state(state)
        cols = np.asarray(sf.columns, dtype=int)
        Xeval = master[:, cols]
        ystd = standardized[child].to_numpy(dtype=float)
        lqs = np.asarray(
            self._base._skew_e.multimode_predictive_logpdf(
                y=ystd, X=Xeval, result=br.multi_cache[idx]
            )
        ) - child_log_sd
        lg, ls = float(sf.log_evidence), float(br.multi_cache[idx]["logz"])
        lu = float(_logsumexp([_LOG_HALF + lg, _LOG_HALF + ls]))
        return _logsumexp(
            np.vstack([
                _LOG_HALF + lg + lqg,
                _LOG_HALF + ls + lqs,
            ]), axis=0
        ) - lu

    def local_predictive_logpdf(self, child, parents, raw_eval):
        child = str(child)
        parents = self._validate_family(child, parents)
        self.score_local(child, parents)
        fr = self.results[(child, parents)]
        if fr.root:
            return self._base.local_predictive_logpdf(child, parents, raw_eval)

        standardized = self._base._standardize_eval(raw_eval)
        br = fr.base_result
        fam = br.family
        master = fam.evaluation_master(standardized)
        ystd = standardized[child].to_numpy(dtype=float)
        child_log_sd = _math.log(float(self.stds[child]))
        lognum = np.full(len(standardized), -np.inf)

        for idx, state in enumerate(br.states):
            d = fr.decisions.get(idx)
            if d is None or d.model == "U":
                lq = self._base_state_logpdf(child, br, idx, standardized, master)
            else:
                sf = fam.fit_state(tuple(state))
                cols = np.asarray(sf.columns, dtype=int)
                Xeval = master[:, cols]
                residual = ystd - Xeval @ np.asarray(d.beta)
                k = 2 if d.model == "M2" else 3
                lq = _residual_logpdf(residual, d.fit, k) - child_log_sd
            lognum = np.logaddexp(lognum, float(fr.state_logjoint[idx]) + lq)
        return lognum - float(fr.score)

    def true_or_learned_dag_score(self, graph):
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("The supplied graph is not a DAG.")
        if set(graph.nodes) != set(self.nodes):
            raise ValueError("Graph nodes do not match scorer columns.")
        return float(sum(
            self.score_local(node, list(graph.predecessors(node)))
            for node in self.nodes
        ))

    def joint_logpdf(self, graph, raw_eval, *, return_nodewise=False):
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Evaluation graph is not a DAG.")
        if set(graph.nodes) != set(self.nodes):
            raise ValueError("Graph nodes do not match scorer columns.")
        total = np.zeros(len(raw_eval))
        nodewise = {}
        for node in nx.topological_sort(graph):
            v = self.local_predictive_logpdf(
                str(node), list(graph.predecessors(node)), raw_eval
            )
            if not np.all(np.isfinite(v)):
                raise FloatingPointError(f"Invalid predictive density for {node}")
            total += v
            nodewise[str(node)] = v
        return (total, nodewise) if return_nodewise else total


NIGPrior = _mean.NIGPrior
LocalScoreDiagnostic = _mean.LocalScoreDiagnostic
T_GRID = _mean.T_GRID
STATE_L = _mean.STATE_L
STATE_C = _mean.STATE_C

__all__ = [
    "V3CScoringEngine", "NIGPrior", "LocalScoreDiagnostic",
    "T_GRID", "STATE_L", "STATE_C", "SCORER_VERSION", "SCORER_PATCH",
]


import sys
from types import SimpleNamespace
_FROZEN_PRODUCTION_API = SimpleNamespace(**globals())

CANDIDATE_VERSION = "v3-cc-root-aligned-contextual-threshold-refine-candidate-v2"
CANDIDATE_PATCH = "aligned-root+joint-state-contextual-empirical-T-refinement-v2"

                                                                
REFINE_STATE_POSTERIOR_MASS = 0.995
REFINE_STATE_CAP = 10
REFINE_MIN_ANY_T_POSTERIOR_MASS = 0.0
REFINE_PER_PARENT_EXTRA = 2
REFINE_COORDINATE_SWEEPS = 2
REFINE_REGION_HALF_WIDTH = 0.005
REFINE_MAX_EMPIRICAL_SPLITS = 32
REFINE_MIN_EMPIRICAL_SPLITS = 2
REFINE_BEAM_EXTRA_SWEEPS = 2
_EPS = 1e-12


_PROD = _FROZEN_PRODUCTION_API
PRODUCTION_SCORER_PATH = str(__file__)
PRODUCTION_SCORER_SHA256 = "consolidated-no-runtime-hash-check"

@dataclass(frozen=True)
class _ThresholdCoordinate:
    parent_index: int
    parent: str
    state_id: int
    q_index: int
    coarse_quantile: float
    production_knot: float
    selected_knot: float
    candidate_count: int
    feature_mean: float
    feature_sd: float
    conditional_region_logz: float
    conditional_correction: float


@dataclass
class _RefinedState:
    state: Tuple[int, ...]
    production_logevidence: float
    map_logevidence: float
    integrated_logevidence: float
    evidence_correction: float
    X_map: np.ndarray
    beta_mean: np.ndarray
    sigma2_mean: float
    condition_number: float
    jitter: float
    coordinates: Tuple[_ThresholdCoordinate, ...]


def _indicator_matrix(x: np.ndarray, knots: np.ndarray):
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    raw = (x[:, None] > knots[None, :]).astype(float)
    means = raw.mean(axis=0)
    sds = raw.std(axis=0, ddof=0)
    valid = np.isfinite(sds) & (sds > _EPS)
    if not np.any(valid):
        return np.empty((len(x), 0)), np.empty(0), np.empty(0), np.empty(0)
    return (
        (raw[:, valid] - means[valid][None, :]) / sds[valid][None, :],
        knots[valid],
        means[valid],
        sds[valid],
    )


def _empirical_split_candidates(x: np.ndarray, q_center: float, production_knot: float):
    x = np.asarray(x, dtype=float)
    values, counts = np.unique(x, return_counts=True)
    if len(values) < 2:
        return np.asarray([float(production_knot)], dtype=float)
    cum = np.cumsum(counts)[:-1] / float(len(x))
    mids = 0.5 * (values[:-1] + values[1:])
    lo = max(0.0, float(q_center) - REFINE_REGION_HALF_WIDTH)
    hi = min(1.0, float(q_center) + REFINE_REGION_HALF_WIDTH)
    chosen = list(map(float, mids[(cum >= lo - 1e-15) & (cum <= hi + 1e-15)]))
    chosen.append(float(production_knot))

    ordered = np.sort(x)
    by_rank: Dict[int, float] = {}
    for knot in chosen:
        rank = int(np.searchsorted(ordered, knot, side="right"))
        if 0 < rank < len(ordered):
            by_rank.setdefault(rank, float(knot))
    knots = np.asarray([by_rank[r] for r in sorted(by_rank)], dtype=float)
    if len(knots) > REFINE_MAX_EMPIRICAL_SPLITS:
        idx = np.linspace(0, len(knots) - 1, REFINE_MAX_EMPIRICAL_SPLITS)
        idx = np.unique(np.round(idx).astype(int))
        knots = knots[idx]
    return knots


def _prior_variances(d: int, prior) -> np.ndarray:
    v = np.full(int(d), float(prior.coeff_var), dtype=float)
    v[0] = float(prior.intercept_var)
    return v


def _nig_fit_explicit(X: np.ndarray, y: np.ndarray, prior, *, logevidence_override=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    variances = _prior_variances(d, prior)
    XtX = X.T @ X
    Xty = X.T @ y
    precision = XtX + np.diag(1.0 / variances)
    jitter = 0.0
    try:
        np.linalg.cholesky(precision)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * max(1.0, float(np.max(np.diag(precision))))
        precision = precision + jitter * np.eye(d)
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0 or not np.isfinite(logdet):
        raise FloatingPointError("Invalid contextual Stage1 posterior precision.")
    beta = np.linalg.solve(precision, Xty)
    quad = float(y @ y - beta @ Xty)
    a0 = float(prior.ig_shape)
    b0 = float(prior.ig_scale)
    an = a0 + 0.5 * n
    bn = b0 + 0.5 * quad
    if not np.isfinite(bn) or bn <= 0.0:
        if bn > -1e-10:
            bn = 1e-12
        else:
            raise FloatingPointError(f"Invalid contextual Stage1 IG scale: {bn}")
    logev = float(
        -0.5 * n * math.log(2.0 * math.pi)
        + 0.5 * (-float(logdet) - float(np.sum(np.log(variances))))
        + a0 * math.log(b0)
        - an * math.log(bn)
        + float(gammaln(an) - gammaln(a0))
    )
    sigma2 = float(bn / (an - 1.0))
    return {
        "logevidence": float(logev if logevidence_override is None else logevidence_override),
        "raw_logevidence": logev,
        "beta": np.asarray(beta, dtype=float),
        "sigma2": sigma2,
        "condition_number": float(np.linalg.cond(precision)),
        "jitter": float(jitter),
    }


def _varying_column_evidences(A: np.ndarray, Z: np.ndarray, y: np.ndarray, prior):





    A = np.asarray(A, dtype=float)
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    K = int(Z.shape[1])
    if K == 0:
        return np.empty(0, dtype=float)
    n, d0 = A.shape
    AtA = A.T @ A
    Aty = A.T @ y
    cross = Z.T @ A
    zty = Z.T @ y
    ztz = np.sum(Z * Z, axis=0)
    d = d0 + 1
    M = np.zeros((K, d, d), dtype=float)
    M[:, :d0, :d0] = AtA[None, :, :]
    M[:, :d0, d0] = cross
    M[:, d0, :d0] = cross
    M[:, d0, d0] = ztz
    variances = np.full(d, float(prior.coeff_var), dtype=float)
    variances[:d0] = _prior_variances(d0, prior)
    diag = np.arange(d)
    M[:, diag, diag] += 1.0 / variances[None, :]
    Xty = np.column_stack([np.tile(Aty, (K, 1)), zty])
    sign, logdet = np.linalg.slogdet(M)
    good = (sign > 0) & np.isfinite(logdet)
    out = np.empty(K, dtype=float)
    if np.any(good):
        beta = np.linalg.solve(M[good], Xty[good, :, None])[:, :, 0]
        quad = float(y @ y) - np.einsum("bi,bi->b", beta, Xty[good])
        a0 = float(prior.ig_shape)
        b0 = float(prior.ig_scale)
        an = a0 + 0.5 * n
        bn = b0 + 0.5 * quad
        bn = np.maximum(bn, 1e-12)
        logdet_v0 = float(np.sum(np.log(variances)))
        out[good] = (
            -0.5 * n * math.log(2.0 * math.pi)
            + 0.5 * (-logdet[good] - logdet_v0)
            + a0 * math.log(b0)
            - an * np.log(bn)
            + float(gammaln(an) - gammaln(a0))
        )
    for k in np.nonzero(~good)[0]:
        X = np.column_stack([A, Z[:, k]])
        out[k] = _nig_fit_explicit(X, y, prior)["raw_logevidence"]
    return out


class _ContextualLocalFamily(_PROD._mean__LocalFamily):
    def __init__(self, *, engine, child, parents, feature_bank, prior):
        self._engine = engine
        super().__init__(child=child, parents=parents, feature_bank=feature_bank, prior=prior)
        self.production_log_evidence_cache: Dict[Tuple[int, ...], float] = {}
        self.refined_states: Dict[Tuple[int, ...], _RefinedState] = {}
        self.refine_enabled_states: set[Tuple[int, ...]] = set()
        self._production_master_train = None

    def production_log_evidence_many(self, states):
        normalized = [tuple(int(v) for v in s) for s in states]
        if not normalized:
            return np.empty(0, dtype=float)
        out = np.empty(len(normalized), dtype=float)
        missing: Dict[int, list] = {}
        for i, state in enumerate(normalized):
            if state in self.production_log_evidence_cache:
                out[i] = self.production_log_evidence_cache[state]
            else:
                cols = self.columns_for_state(state)
                missing.setdefault(len(cols), []).append((i, state, cols))
        for items in missing.values():
            matrix = np.asarray([c for _, _, c in items], dtype=np.int64)
            vals = self._batch_log_evidence_for_columns(matrix)
            for (i, state, _), value in zip(items, vals):
                value = float(value)
                self.production_log_evidence_cache[state] = value
                out[i] = value
        return out

    def production_log_evidence(self, state):
        return float(self.production_log_evidence_many([state])[0])

    def _training_master(self):
        if self._production_master_train is None:
            if not self.parents:
                self._production_master_train = np.ones((self.n, 1), dtype=float)
            else:
                self.bank._ensure_global()
                gf = self.bank._global_features
                if gf is None:
                    raise RuntimeError("Production feature bank did not build.")
                blocks = [gf[:, self.bank.block_slice(p)] for p in self.parents]
                self._production_master_train = np.column_stack(
                    [np.ones(self.n, dtype=float), *blocks]
                )
        return self._production_master_train

    def production_design_for_state(self, state):
        cols = np.asarray(self.columns_for_state(tuple(state)), dtype=int)
        return np.asarray(self._training_master()[:, cols], dtype=float).copy()

    def _threshold_output_positions(self, state):
        columns = tuple(int(v) for v in self.columns_for_state(state))
        rows = []
        for j, state_id in enumerate(state):
            if int(state_id) < 2:
                continue
            q_index = int(state_id) - 2
            local_col = int(self.t_columns[j][q_index])
            out_pos = int(columns.index(local_col))
            metadata = self.bank.metadata(self.parents[j])
            production_knot = float(np.asarray(metadata["t_knots"], dtype=float)[q_index])
            rows.append((j, q_index, out_pos, production_knot))
        return rows

    def _candidate_bank(self, parent_index: int, q_index: int, production_knot: float, X_context=None, out_pos=None, y=None):








        parent = self.parents[parent_index]
        x = self.bank.data[parent].to_numpy(dtype=float)
        q0 = float(_PROD._mean_T_GRID[q_index])
        lo = max(0.001, q0 - REFINE_REGION_HALF_WIDTH)
        hi = min(0.999, q0 + REFINE_REGION_HALF_WIDTH)
        qs = np.arange(lo, hi + 0.0005001, 0.001, dtype=float)
        scout_knots = np.asarray([float(np.quantile(x, q)) for q in qs], dtype=float)
        scout_knots = np.append(scout_knots, float(production_knot))
        Z0, k0, m0, s0 = _indicator_matrix(x, scout_knots)
        if Z0.shape[1] == 0:
            return Z0, k0, m0, s0

        if X_context is not None and out_pos is not None and y is not None:
            A = np.delete(np.asarray(X_context, dtype=float), int(out_pos), axis=1)
            ev0 = _varying_column_evidences(A, Z0, np.asarray(y, dtype=float), self.prior)
            best_knot = float(k0[int(np.argmax(ev0))])
        else:
            best_knot = float(k0[int(np.argmin(np.abs(k0 - production_knot)))])

        ordered = np.sort(x)
        rank = int(np.searchsorted(ordered, best_knot, side="right"))
        qbest = rank / float(len(x))
        values, counts = np.unique(x, return_counts=True)
        cum = np.cumsum(counts)[:-1] / float(len(x))
        mids = 0.5 * (values[:-1] + values[1:])
        local = mids[(cum >= qbest - 0.0015) & (cum <= qbest + 0.0015)]
        final_knots = np.concatenate([k0, np.asarray(local, dtype=float), np.asarray([production_knot])])
                                           
        by_rank = {}
        for knot in final_knots:
            r = int(np.searchsorted(ordered, float(knot), side="right"))
            if 0 < r < len(ordered):
                by_rank.setdefault(r, float(knot))
        final_knots = np.asarray([by_rank[r] for r in sorted(by_rank)], dtype=float)
        if len(final_knots) > REFINE_MAX_EMPIRICAL_SPLITS:
                                                                            
            ranks = np.asarray([np.searchsorted(ordered, k, side="right") for k in final_knots])
            target = int(round(qbest * len(x)))
            order = np.argsort(np.abs(ranks - target))
            keep = set(map(int, order[: max(1, REFINE_MAX_EMPIRICAL_SPLITS - 2)]))
            keep.update([0, len(final_knots) - 1])
            final_knots = final_knots[sorted(keep)]
        return _indicator_matrix(x, final_knots)

    def refine_state(self, state):
        state = tuple(int(v) for v in state)
        cached = self.refined_states.get(state)
        if cached is not None:
            return cached
        tcoords = self._threshold_output_positions(state)
        prod_logev = self.production_log_evidence(state)
        if not tcoords:
            return None

        X = self.production_design_for_state(state)
        y = self.bank.data[self.child].to_numpy(dtype=float)
        coord_banks = {}
        selected = {}
        for j, q_index, out_pos, production_knot in tcoords:
            Z, knots, means, sds = self._candidate_bank(j, q_index, production_knot, X_context=X, out_pos=out_pos, y=y)
            if Z.shape[1] < REFINE_MIN_EMPIRICAL_SPLITS:
                continue
            coord_banks[j] = (q_index, out_pos, production_knot, Z, knots, means, sds)
                                                                                        
            k0 = int(np.argmin(np.abs(knots - production_knot)))
            selected[j] = k0
            X[:, out_pos] = Z[:, k0]

        if not coord_banks:
            return None

                                                                                 
                                                                          
                                        
        for _ in range(REFINE_COORDINATE_SWEEPS):
            changed = False
            for j in sorted(coord_banks):
                q_index, out_pos, production_knot, Z, knots, means, sds = coord_banks[j]
                A = np.delete(X, out_pos, axis=1)
                ev = _varying_column_evidences(A, Z, y, self.prior)
                k = int(np.argmax(ev))
                if k != selected[j]:
                    changed = True
                    selected[j] = k
                X[:, out_pos] = Z[:, k]
            if not changed:
                break

        map_fit = _nig_fit_explicit(X, y, self.prior)
        map_logev = float(map_fit["raw_logevidence"])
        correction = 0.0
        coordinates = []
        for j in sorted(coord_banks):
            q_index, out_pos, production_knot, Z, knots, means, sds = coord_banks[j]
            A = np.delete(X, out_pos, axis=1)
            ev = _varying_column_evidences(A, Z, y, self.prior)
            region_logz = float(logsumexp(ev) - math.log(len(ev)))
            c = float(region_logz - map_logev)
            correction += c
            k = int(selected[j])
            coordinates.append(
                _ThresholdCoordinate(
                    parent_index=int(j),
                    parent=str(self.parents[j]),
                    state_id=int(state[j]),
                    q_index=int(q_index),
                    coarse_quantile=float(_PROD._mean_T_GRID[q_index]),
                    production_knot=float(production_knot),
                    selected_knot=float(knots[k]),
                    candidate_count=int(len(knots)),
                    feature_mean=float(means[k]),
                    feature_sd=float(sds[k]),
                    conditional_region_logz=float(region_logz),
                    conditional_correction=float(c),
                )
            )

        integrated = float(map_logev + correction)
        fit = _nig_fit_explicit(X, y, self.prior, logevidence_override=integrated)
        rs = _RefinedState(
            state=state,
            production_logevidence=float(prod_logev),
            map_logevidence=float(map_logev),
            integrated_logevidence=float(integrated),
            evidence_correction=float(integrated - map_logev),
            X_map=np.asarray(X, dtype=float),
            beta_mean=np.asarray(fit["beta"], dtype=float),
            sigma2_mean=float(fit["sigma2"]),
            condition_number=float(fit["condition_number"]),
            jitter=float(fit["jitter"]),
            coordinates=tuple(coordinates),
        )
        self.refined_states[state] = rs
        return rs

    def _state_evidence_correction(self, state):
        rs = self.refined_states.get(tuple(state))
        return 0.0 if rs is None else float(rs.evidence_correction)

    def log_evidence_many(self, states):
        normalized = [tuple(int(v) for v in s) for s in states]
        out = self.production_log_evidence_many(normalized)
        out = np.asarray(out, dtype=float).copy()
        for i, state in enumerate(normalized):
            if state in self.refine_enabled_states:
                rs = self.refine_state(state)
                if rs is not None:
                    out[i] = float(rs.integrated_logevidence)
        return out

    def log_evidence(self, state):
        return float(self.log_evidence_many([state])[0])

    def fit_state(self, state):
        state = tuple(int(v) for v in state)
        if state in self.refine_enabled_states:
            rs = self.refine_state(state)
            if rs is not None:
                columns = tuple(int(v) for v in self.columns_for_state(state))
                return _PROD._mean_StateFit(
                    state=state,
                    columns=columns,
                    beta_mean=np.asarray(rs.beta_mean, dtype=float),
                    sigma2_mean=float(rs.sigma2_mean),
                    log_evidence=float(rs.integrated_logevidence),
                    condition_number=float(rs.condition_number),
                    jitter=float(rs.jitter),
                )
                                                                                
                                                             
        columns = tuple(int(v) for v in self.columns_for_state(state))
        X = self.production_design_for_state(state)
        y = self.bank.data[self.child].to_numpy(dtype=float)
        f = _nig_fit_explicit(X, y, self.prior, logevidence_override=self.production_log_evidence(state))
        return _PROD._mean_StateFit(
            state=state,
            columns=columns,
            beta_mean=np.asarray(f["beta"], dtype=float),
            sigma2_mean=float(f["sigma2"]),
            log_evidence=float(f["logevidence"]),
            condition_number=float(f["condition_number"]),
            jitter=float(f["jitter"]),
        )

    def training_design_for_fit(self, fit):
        state = tuple(int(v) for v in fit.state)
        rs = self.refined_states.get(state)
        if state in self.refine_enabled_states and rs is not None:
            return np.asarray(rs.X_map, dtype=float)
        return self.production_design_for_state(state)

    def state_logpdf_standardized(self, state, standardized_eval, master_eval=None):
        state = tuple(int(v) for v in state)
        fit = self.fit_state(state)
        if master_eval is None:
            master_eval = self.evaluation_master(standardized_eval)
        cols = np.asarray(fit.columns, dtype=int)
        X = np.asarray(master_eval[:, cols], dtype=float).copy()
        rs = self.refined_states.get(state)
        if state in self.refine_enabled_states and rs is not None:
            columns = tuple(int(v) for v in fit.columns)
            for coord in rs.coordinates:
                local_col = int(self.t_columns[coord.parent_index][coord.q_index])
                out_pos = int(columns.index(local_col))
                x = standardized_eval[coord.parent].to_numpy(dtype=float)
                raw = (x > float(coord.selected_knot)).astype(float)
                X[:, out_pos] = (raw - float(coord.feature_mean)) / float(coord.feature_sd)
        mu = X @ np.asarray(fit.beta_mean, dtype=float)
        y = standardized_eval[self.child].to_numpy(dtype=float)
        sigma2 = max(float(fit.sigma2_mean), 1e-15)
        return -0.5 * (math.log(2.0 * math.pi * sigma2) + (y - mu) ** 2 / sigma2)


class _ContextualMeanEngine(_PROD._mean_V3CScoringEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contextual_refinement_diagnostics: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}

    def _family(self, child, parents):
        key = (child, parents)
        family = self.local_families.get(key)
        if family is None:
            family = _ContextualLocalFamily(
                engine=self,
                child=child,
                parents=parents,
                feature_bank=self.feature_bank,
                prior=self.prior,
            )
            self.local_families[key] = family
        return family

    @staticmethod
    def _has_T(state):
        return any(int(v) >= 2 for v in state)

    def _choose_refine_states(self, states, production_logjoint, family):
        if family.p == 0 or not states:
            return []
        logjoint = np.asarray(production_logjoint, dtype=float)
        total = float(logsumexp(logjoint))
        weights = np.exp(logjoint - total)
        eligible = [i for i, s in enumerate(states) if self._has_T(s)]
        if not eligible:
            return []
        t_mass = float(np.sum(weights[np.asarray(eligible, dtype=int)]))

        selected = set()
        order = sorted(eligible, key=lambda i: float(logjoint[i]), reverse=True)
        target = REFINE_STATE_POSTERIOR_MASS * t_mass
        acc = 0.0
        for i in order:
            selected.add(int(i))
            acc += float(weights[i])
            if acc >= target or len(selected) >= REFINE_STATE_CAP:
                break

                                                                              
                                                                           
                                                                 
        for j in range(family.p):
            js = [i for i in eligible if int(states[i][j]) >= 2]
            js.sort(key=lambda i: float(logjoint[i]), reverse=True)
            for i in js[:REFINE_PER_PARENT_EXTRA]:
                selected.add(int(i))

        ordered = sorted(selected, key=lambda i: float(logjoint[i]), reverse=True)
        return ordered[:REFINE_STATE_CAP]

    def _production_beam(self, child, parents, family):
        width = self.beam_cfg["width"]
        max_sweeps = self.beam_cfg["max_sweeps"]
                                                                        
        p = family.p
        t50 = 2 + int(np.argmin(np.abs(np.asarray(_PROD._mean_T_GRID) - 0.5)))
        best_t = []
        for j in range(p):
            candidate_states = []
            tids = []
            for q in range(_PROD._mean_K_T):
                s = [_PROD._mean_STATE_L] * p
                s[j] = 2 + q
                candidate_states.append(tuple(s)); tids.append(2 + q)
            ev = family.production_log_evidence_many(candidate_states)
            pri = np.asarray([_PROD._mean_joint_state_log_prior(s) for s in candidate_states])
            best_t.append(int(tids[int(np.argmax(ev + pri))]))
        seeds = {
            tuple([_PROD._mean_STATE_L] * p),
            tuple([_PROD._mean_STATE_C] * p),
            tuple([t50] * p),
            tuple(best_t),
        }
        for j in range(p):
            s = [_PROD._mean_STATE_L] * p; s[j] = _PROD._mean_STATE_C; seeds.add(tuple(s))
            s = [_PROD._mean_STATE_L] * p; s[j] = best_t[j]; seeds.add(tuple(s))
        seeds = sorted(seeds)

        def prod_joint(ss):
            ev = family.production_log_evidence_many(ss)
            pri = np.asarray([_PROD._mean_joint_state_log_prior(s) for s in ss], dtype=float)
            return ev + pri

        seed_values = prod_joint(seeds)
        visited = {s: float(v) for s, v in zip(seeds, seed_values)}
        beam = _PROD._mean_V3CScoringEngine._top_beam(seeds, seed_values, width)
        for _ in range(max_sweeps):
            candidates = _PROD._mean_V3CScoringEngine._expand_beam(beam, p)
            unseen = [s for s in candidates if s not in visited]
            if unseen:
                vals = prod_joint(unseen)
                visited.update({s: float(v) for s, v in zip(unseen, vals)})
            cv = np.asarray([visited[s] for s in candidates], dtype=float)
            new_beam = _PROD._mean_V3CScoringEngine._top_beam(candidates, cv, width)
            if set(new_beam) == set(beam):
                beam = new_beam
                break
            beam = new_beam
        return visited, beam

    def score_local(self, child, parents):
        child = str(child)
        parents = self._validate_family(child, parents)
        key = (child, parents)
        if key in self.cache:
            return float(self.cache[key])
        started = time.perf_counter()
        family = self._family(child, parents)
        p = len(parents)
        if p == 0:
            state = ()
            score = float(family.production_log_evidence(state))
            self.posterior_state_logweights[key] = ([()], np.asarray([0.0]))
            diag = _PROD._mean_LocalScoreDiagnostic(
                child=child, parents=parents, parent_count=0,
                method="exact_nig_root_contextual_T_none", score=score,
                state_space_size=1, scored_state_count=1,
                runtime_seconds=float(time.perf_counter() - started),
            )
            self.cache[key] = score; self.diagnostics[key] = diag
            return score

        if p <= 2:
            states = [tuple(int(v) for v in s) for s in product(range(_PROD._mean_N_STATES), repeat=p)]
            prod_ev = family.production_log_evidence_many(states)
            pri = np.asarray([_PROD._mean_joint_state_log_prior(s) for s in states], dtype=float)
            prod_joint = prod_ev + pri
            selected_idx = self._choose_refine_states(states, prod_joint, family)
            family.refine_enabled_states.update(states[i] for i in selected_idx)
            ev = np.asarray(prod_ev, dtype=float).copy()
            for i in selected_idx:
                rs = family.refine_state(states[i])
                if rs is not None:
                    ev[i] = rs.integrated_logevidence
            joint = ev + pri
            score = float(logsumexp(joint))
            logw = joint - score
            self.posterior_state_logweights[key] = (states, logw)
            map_i = int(np.argmax(joint))
            diag = _PROD._mean_LocalScoreDiagnostic(
                child=child, parents=parents, parent_count=p,
                method=f"exact_bma_{_PROD._mean_N_STATES ** p}_states+contextual_T",
                score=score, state_space_size=int(_PROD._mean_N_STATES ** p),
                scored_state_count=len(states),
                runtime_seconds=float(time.perf_counter() - started),
                reference_state=_PROD._mean_state_tuple_label(states[map_i]),
                reference_sample_weight=float(math.exp(logw[map_i])),
            )
            self.contextual_refinement_diagnostics[key] = {
                "production_score": float(logsumexp(prod_joint)),
                "corrected_score": score,
                "selected_refined_state_count": len(selected_idx),
                "refined_state_count": len(family.refined_states),
            }
        else:
            visited, beam = self._production_beam(child, parents, family)
            states = list(visited.keys())
            prod_joint = np.asarray([visited[s] for s in states], dtype=float)
            selected_idx = self._choose_refine_states(states, prod_joint, family)
            family.refine_enabled_states.update(states[i] for i in selected_idx)

                                     
            corrected = dict(visited)
            for i in selected_idx:
                s = states[i]
                rs = family.refine_state(s)
                if rs is not None:
                    corrected[s] = float(rs.integrated_logevidence + _PROD._mean_joint_state_log_prior(s))

            width = self.beam_cfg["width"]
            beam_vals = np.asarray([corrected[s] for s in beam], dtype=float)
            beam = _PROD._mean_V3CScoringEngine._top_beam(beam, beam_vals, width)
                                                                                 
                                                                 
            for _ in range(REFINE_BEAM_EXTRA_SWEEPS):
                candidates = _PROD._mean_V3CScoringEngine._expand_beam(beam, p)
                unseen = [s for s in candidates if s not in corrected]
                if unseen:
                    pev = family.production_log_evidence_many(unseen)
                    for s, e in zip(unseen, pev):
                        corrected[s] = float(e + _PROD._mean_joint_state_log_prior(s))
                cstates = list(candidates)
                cvals = np.asarray([corrected[s] for s in cstates], dtype=float)
                extra_idx = self._choose_refine_states(cstates, cvals, family)
                for i in extra_idx:
                    s = cstates[i]
                    if s not in family.refine_enabled_states:
                        family.refine_enabled_states.add(s)
                        rs = family.refine_state(s)
                        if rs is not None:
                            corrected[s] = float(rs.integrated_logevidence + _PROD._mean_joint_state_log_prior(s))
                cvals = np.asarray([corrected[s] for s in cstates], dtype=float)
                new_beam = _PROD._mean_V3CScoringEngine._top_beam(cstates, cvals, width)
                if set(new_beam) == set(beam):
                    break
                beam = new_beam

            states = list(corrected.keys())
            joint = np.asarray([corrected[s] for s in states], dtype=float)
            score = float(logsumexp(joint))
            logw = joint - score
            self.posterior_state_logweights[key] = (states, logw)
            map_i = int(np.argmax(joint))
            diag = _PROD._mean_LocalScoreDiagnostic(
                child=child, parents=parents, parent_count=p,
                method=f"joint_beam_B{self.beam_cfg['width']}_S{self.beam_cfg['max_sweeps']}+contextual_T",
                score=score, state_space_size=int(_PROD._mean_N_STATES ** p),
                scored_state_count=len(states), runtime_seconds=float(time.perf_counter() - started),
                reference_state=_PROD._mean_state_tuple_label(states[map_i]),
                reference_sample_weight=float(math.exp(logw[map_i])),
                beam_width=int(self.beam_cfg["width"]),
                beam_max_sweeps=int(self.beam_cfg["max_sweeps"]),
                beam_sweeps_completed=int(self.beam_cfg["max_sweeps"] + REFINE_BEAM_EXTRA_SWEEPS),
                beam_converged=False,
                beam_visited_state_count=len(states),
                beam_visited_state_fraction=float(len(states) / (_PROD._mean_N_STATES ** p)),
                beam_final_logz_increment=float("nan"),
            )
            self.contextual_refinement_diagnostics[key] = {
                "production_score": float(logsumexp(prod_joint)),
                "corrected_score": score,
                "selected_refined_state_count": len(family.refine_enabled_states),
                "refined_state_count": len(family.refined_states),
            }

        if not np.isfinite(score):
            raise FloatingPointError(f"Non-finite contextual Stage1 score for {child!r} | {parents}.")
        self.cache[key] = score
        self.diagnostics[key] = diag
        return score

    def _exact_posterior_states(self, family):
        p = family.p
        states = [tuple(int(v) for v in s) for s in product(range(_PROD._mean_N_STATES), repeat=p)]
        key = (family.child, family.parents)
        if key not in self.cache:
            self.score_local(family.child, family.parents)
                                                                                    
        ev = family.log_evidence_many(states)
        joint = ev + np.asarray([_PROD._mean_joint_state_log_prior(s) for s in states], dtype=float)
        return states, joint - float(logsumexp(joint))

    def preprocessing_metadata(self):
        out = dict(super().preprocessing_metadata())
        out.update({
            "contextual_threshold_refinement": True,
            "refinement_scope": "joint Stage1 state / local family",
            "refine_state_posterior_mass": REFINE_STATE_POSTERIOR_MASS,
            "refine_state_cap": REFINE_STATE_CAP,
            "refine_region_half_width": REFINE_REGION_HALF_WIDTH,
            "refine_max_empirical_splits": REFINE_MAX_EMPIRICAL_SPLITS,
            "multi_T_region_integration": "factorized conditional logmeanexp around joint MAP",
        })
        return out


class _CorrectedRootAwareBase(_PROD._RootAwareCScoringEngine):
    def __init__(self, *args, **kwargs):
        original = _PROD._MeanAPI.V3CScoringEngine
        _PROD._MeanAPI.V3CScoringEngine = _ContextualMeanEngine
        try:
            super().__init__(*args, **kwargs)
        finally:
            _PROD._MeanAPI.V3CScoringEngine = original

    def _training_design_for_fit(self, family, fit):
        if isinstance(family, _ContextualLocalFamily):
            return family.training_design_for_fit(fit)
        return super()._training_design_for_fit(family, fit)

    def _score_stage12_local(self, child, parents):







        child = str(child)
        parents = self._validate_family(child, parents)
        key = (child, parents)
        if key in self.cache:
            return float(self.cache[key])

        total_started = time.perf_counter()
        stage1_started = time.perf_counter()
        stage1_logz = float(self.mean_engine.score_local(child, parents))
        stage1_runtime = float(time.perf_counter() - stage1_started)
        stage1_diag = self.mean_engine.get_score_diagnostic(child, parents)
        family, states, logw_g = self._mean_states_and_logweights(child, parents)
        weights_g = np.exp(logw_g)
        logjoint_g = stage1_logz + logw_g
        retained = _PROD._retained_indices(logw_g, _PROD.MEAN_POSTERIOR_MASS_CUTOFF)
        inspected_mass = float(np.sum(weights_g[retained]))
        y_train_std = self._training_y_standardized[child]
        skew_prior = self._skew_prior
        fast_cache = {}
        state_training_cache = {}
        escalated = []
        retained_indices = [int(v) for v in retained]
        retained_designs = []
        retained_anchors = []

        fast_started = time.perf_counter()
        for idx in retained_indices:
            state = tuple(states[idx])
            fit = family.fit_state(state)
            X_state = self._training_design_for_fit(family, fit)
            anchor = _PROD._make_gaussian_anchor(fit, self.n, skew_prior, self._skew_a)
            retained_designs.append(X_state)
            retained_anchors.append(anchor)
            state_training_cache[idx] = (X_state, anchor)
        batched_fast = _PROD._fast_skew_correction_batch(y_train_std, retained_designs, retained_anchors)
        fast_runtime = float(time.perf_counter() - fast_started)
        for idx, fast_raw in zip(retained_indices, batched_fast):
            fast = dict(fast_raw)
            nodes = np.asarray(fast["quad_nodes"], dtype=float)
            postw = np.asarray(fast["posterior_weights"], dtype=float)
            gaussian_mass = float(np.sum(postw[np.abs(nodes) <= self.delta0]))
            fast["gaussian_mass"] = gaussian_mass
            fast["scout_total_runtime_s"] = float(fast.get("runtime_s", 0.0))
            fast_cache[idx] = fast
            clearly_gaussian = bool(
                gaussian_mass >= _PROD.GAUSSIAN_CONFIDENCE_CUTOFF
                and float(fast["log_bf_vs_gaussian"]) <= _PROD.FAST_LOG_BF_GUARD
            )
            if not clearly_gaussian:
                escalated.append(idx)

        if not escalated:
            score = stage1_logz
            combined_state_logjoint = np.array(logjoint_g, copy=True)
            action = "STAGE1_ONLY"
            multi_cache = {}
            multi_runtime = 0.0
        else:
            multi_cache = {}
            multi_runtime = 0.0
            combined_state_logjoint = np.array(logjoint_g, copy=True)
            for idx in escalated:
                X_state, anchor = state_training_cache[idx]
                multi = _PROD._hardened_multimode_joint_laplace(
                    y=y_train_std, X=X_state, anchor=anchor,
                    fast=fast_cache[idx], prior=skew_prior,
                    max_iter=_PROD.MULTI_MAX_ITER,
                    skew_c=self._skew_c, skew_e=self._skew_e,
                )
                multi = dict(multi)
                state = tuple(states[idx])
                correction = family._state_evidence_correction(state) if isinstance(family, _ContextualLocalFamily) else 0.0
                multi["logz_raw_before_threshold_refinement"] = float(multi["logz"])
                multi["threshold_refinement_evidence_correction"] = float(correction)
                multi["logz"] = float(multi["logz"] + correction)
                multi_cache[idx] = multi
                multi_runtime += float(multi["runtime_s"])
                logprior_state = 0.0 if len(parents) == 0 else float(self._mean_module.joint_state_log_prior(state))
                logjoint_s = logprior_state + float(multi["logz"])
                combined_state_logjoint[idx] = float(logsumexp(np.asarray([
                    _PROD.LOG_HALF + float(logjoint_g[idx]),
                    _PROD.LOG_HALF + logjoint_s,
                ])))
            score = float(logsumexp(combined_state_logjoint))
            action = "SELECTIVE_MULTI"

        if not np.isfinite(score):
            raise FloatingPointError(f"Non-finite corrected Stage1+2 score for {child!r} | {parents}.")

        escalated_array = np.asarray(escalated, dtype=int)
        escalated_mass = float(np.sum(weights_g[escalated_array]) if len(escalated_array) else 0.0)
        retry_count = int(sum(int(bool(m.get("zero_retry_used", False))) + int(bool(m.get("fast_retry_used", False))) for m in multi_cache.values()))
        rescue_count = int(sum(int(bool(m.get("zero_rescue_start_used", False))) + int(bool(m.get("fast_rescue_start_used", False))) for m in multi_cache.values()))
        boundary_count = int(sum(int(bool(m.get("zero_boundary_hit", False))) + int(bool(m.get("fast_boundary_hit", False))) for m in multi_cache.values()))
        diagnostic = _PROD.CombinedLocalScoreDiagnostic(
            child=child, parents=parents, parent_count=len(parents),
            method="stage1_BMA_FAST080_090_BF0_selective_MULTI_S+contextual_T",
            score=float(score), stage1_score=float(stage1_logz),
            stage1_method=str(stage1_diag.method), action=action,
            stage1_state_count=int(len(states)), inspected_state_count=int(len(retained)),
            inspected_stage1_posterior_mass=inspected_mass,
            escalated_state_count=int(len(escalated)), escalated_stage1_posterior_mass=escalated_mass,
            delta0=float(self.delta0), tau=_PROD.TAU_PREDICTIVE_REGRET,
            mass_cutoff=_PROD.MEAN_POSTERIOR_MASS_CUTOFF,
            gaussian_confidence_cutoff=_PROD.GAUSSIAN_CONFIDENCE_CUTOFF,
            fast_logbf_guard=_PROD.FAST_LOG_BF_GUARD,
            fast_calls=int(len(retained)), multi_calls=int(len(escalated)),
            stage1_runtime_seconds=stage1_runtime, fast_runtime_seconds=float(fast_runtime),
            multi_runtime_seconds=float(multi_runtime), runtime_seconds=float(time.perf_counter() - total_started),
            optimizer_retry_count=retry_count, optimizer_rescue_count=rescue_count,
            boundary_hit_count=boundary_count,
        )
        result = _PROD._CombinedLocalResult(
            child=child, parents=parents, family=family, states=states,
            logw_g=np.asarray(logw_g, dtype=float), logjoint_g=np.asarray(logjoint_g, dtype=float),
            combined_state_logjoint=np.asarray(combined_state_logjoint, dtype=float),
            retained_indices=np.asarray(retained, dtype=int),
            escalated_indices=tuple(int(i) for i in escalated),
            fast_cache=fast_cache, multi_cache=multi_cache, combined_logz=float(score),
        )
        self.cache[key] = float(score)
        self.diagnostics[key] = diagnostic
        self.combined_results[key] = result
        return float(score)

    def score_local(self, child, parents):
        child = str(child)
        parents = self._validate_family(child, parents)
        if parents:
            return self._score_stage12_local(child, parents)
        return super().score_local(child, parents)


class V3CScoringEngine(_PROD.V3CScoringEngine):

    def __init__(self, data, *args, **kwargs):
        original = _PROD._BaseAPI.V3CScoringEngine
        _PROD._BaseAPI.V3CScoringEngine = _CorrectedRootAwareBase
        try:
            super().__init__(data, *args, **kwargs)
        finally:
            _PROD._BaseAPI.V3CScoringEngine = original

    def score_local(self, child, parents):
        child = str(child)
        parents = self._validate_family(child, parents)
        key = (child, parents)
        if key in self.cache:
            return float(self.cache[key])
        started = time.perf_counter()

                                                                              
                                                              
        if parents:
            base_score = float(self._base.score_local(child, parents))
        else:
            base_score = float(self._base._score_stage12_local(child, ()))

        br = self._base.combined_results[key]
        old = np.asarray(br.combined_state_logjoint, dtype=float)
        chosen, mass, cap_hit = _PROD._select_states(old, self.conditional_state_mass, self.conditional_state_cap)
        new = old.copy()
        ystd = np.asarray(self._base._training_y_standardized[child], dtype=float)
        decisions = {}
        for rank, idx in enumerate(chosen):
            idx = int(idx)
            state = tuple(br.states[idx])
            sf = br.family.fit_state(state)
            X = self._base._training_design_for_fit(br.family, sf)
            beta = _PROD._state_beta(br, idx)
            d = self._one_state(ystd - X @ beta, idx, rank, state, beta)
            new[idx] += float(d.uplift)
            decisions[idx] = d
        score = float(logsumexp(new))
        if not np.isfinite(score):
            raise FloatingPointError(f"Non-finite corrected V3 score for {child!r} | {parents}.")
        r = _PROD._FamilyResult(
            child, parents, base_score, score, br, new,
            tuple(int(x) for x in chosen), float(mass), bool(cap_hit),
            decisions, False, float(time.perf_counter() - started),
        )
        self.cache[key] = score
        self.results[key] = r
        self.diagnostics[key] = {
            "child": child, "parents": parents, "parent_count": len(parents),
            "score": score, "base_score": base_score,
            "conditional_uplift": float(score - base_score),
            "selected_state_count": len(chosen), "selected_state_mass": float(mass),
            "state_cap_hit": bool(cap_hit), "state_count": len(old),
            "scout_a_stop_count": sum(d.scout_a["stop"] for d in decisions.values()),
            "m2_state_count": sum(not d.scout_a["stop"] for d in decisions.values()),
            "m3_state_count": sum(np.isfinite(d.logz_m3) for d in decisions.values()),
            "root_aligned": bool(not parents), "root_preserved": False,
            "runtime_seconds": r.runtime_s,
        }
        return score

    def prior_hyperparameters(self):
        out = dict(super().prior_hyperparameters())
        out.update({
            "candidate_version": CANDIDATE_VERSION,
            "candidate_patch": CANDIDATE_PATCH,
            "aligned_zero_parent_factorization": True,
            "standalone_stage3a_root_score_used_for_structure": False,
            "contextual_stage1_threshold_refinement": True,
        })
        return out

    def preprocessing_metadata(self):
        out = dict(super().preprocessing_metadata())
        out.update({
            "candidate_version": CANDIDATE_VERSION,
            "candidate_patch": CANDIDATE_PATCH,
            "production_scorer_path": PRODUCTION_SCORER_PATH,
            "production_scorer_sha256": PRODUCTION_SCORER_SHA256,
            "root_score_alignment": "Stage1+Stage2+same Stage3B statewise residual uplift",
            "stage1_threshold_refinement": {
                "scope": "joint state / local family",
                "top_level_states": 93,
                "coarse_regions": 91,
                "region_half_width": REFINE_REGION_HALF_WIDTH,
                "max_empirical_splits": REFINE_MAX_EMPIRICAL_SPLITS,
                "state_posterior_mass": REFINE_STATE_POSTERIOR_MASS,
                "state_cap": REFINE_STATE_CAP,
                "coordinate_sweeps": REFINE_COORDINATE_SWEEPS,
                "representative": "joint-state contextual MAP empirical split",
                "within_region_evidence": "conditional logmeanexp with original region prior mass",
            },
        })
        return out

    def contextual_refinement_dataframe(self):
        rows = []
        mean_engine = getattr(self._base, "mean_engine", None)
        if mean_engine is None:
            return pd.DataFrame()
        for (child, parents), family in mean_engine.local_families.items():
            if not isinstance(family, _ContextualLocalFamily):
                continue
            for state, rs in family.refined_states.items():
                for coord in rs.coordinates:
                    rows.append({
                        "child": child,
                        "parents": "|".join(parents),
                        "state": _PROD._mean_state_tuple_label(state),
                        "parent": coord.parent,
                        "coarse_quantile": coord.coarse_quantile,
                        "production_knot": coord.production_knot,
                        "selected_knot": coord.selected_knot,
                        "candidate_count": coord.candidate_count,
                        "production_state_logevidence": rs.production_logevidence,
                        "map_state_logevidence": rs.map_logevidence,
                        "integrated_state_logevidence": rs.integrated_logevidence,
                        "state_evidence_correction": rs.evidence_correction,
                    })
        return pd.DataFrame(rows)


V3CombinedScoringEngine = V3CScoringEngine
V3CombinedStage3AScoringEngine = V3CScoringEngine
SCORER_VERSION = CANDIDATE_VERSION
SCORER_PATCH = CANDIDATE_PATCH

__all__ = [
    "V3CScoringEngine", "V3CombinedScoringEngine",
    "V3CombinedStage3AScoringEngine", "SCORER_VERSION", "SCORER_PATCH",
    "CANDIDATE_VERSION", "CANDIDATE_PATCH", "PRODUCTION_SCORER_PATH",
    "PRODUCTION_SCORER_SHA256", "_ContextualMeanEngine",
]


from pathlib import Path
_MODEL_DIR = Path(__file__).resolve().parent
_MODEL_DIR_STR = str(_MODEL_DIR)
if _MODEL_DIR_STR not in sys.path:
    sys.path.insert(0, _MODEL_DIR_STR)
from HybridV3_CDv3Candidate import (
    AtomicAdd2TabuGreedyHillClimbing, HybridV3ScoringEngine,
    LaplaceDiagnostic, LocalScoreInfo, MixtureDiagnostic,
    QuantileDiscretizer, TabuGreedyHillClimbing,
)
HybridScoringEngine = HybridV3ScoringEngine
__all__ = [
    "HybridScoringEngine", "HybridV3ScoringEngine", "V3CScoringEngine",
    "QuantileDiscretizer", "AtomicAdd2TabuGreedyHillClimbing",
    "TabuGreedyHillClimbing", "LaplaceDiagnostic", "MixtureDiagnostic",
    "LocalScoreInfo", "SCORER_VERSION", "SCORER_PATCH",
]

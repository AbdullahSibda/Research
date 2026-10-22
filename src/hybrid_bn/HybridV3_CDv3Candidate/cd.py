"""cd.py — Discrete child with continuous parents: softmax logistic regression, nonlinear feature candidates, MAP optimisation, Laplace evidence and predictive probabilities.

All executable definitions are retained from the research implementation;
only explanatory comments and internal docstrings are consolidated here.
"""





























from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import logsumexp

try:
    from sklearn.preprocessing import SplineTransformer
except Exception:                                                        
    SplineTransformer = None

from .common import LaplaceDiagnostic

_MIN_PROB = 1e-300
_DEFAULT_MODELS = ("linear", "quadratic", "cubic", "spline5")


@dataclass
class _FeatureMap:
    kind: str
    c_parents: tuple[str, ...]
    spline: Any | None = None

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[list(self.c_parents)].to_numpy(dtype=float)
        if self.kind == "linear":
            features = x
        elif self.kind == "quadratic":
            features = np.hstack([x, x ** 2])
        elif self.kind == "cubic":
            features = np.hstack([x, x ** 2, x ** 3])
        elif self.kind == "spline5":
            if self.spline is None:
                raise RuntimeError("Spline feature map was not fitted.")
            features = np.asarray(self.spline.transform(x), dtype=float)
        else:
            raise ValueError(f"Unknown CD V3 feature family: {self.kind}")
        return np.column_stack([np.ones(len(frame)), features])


@dataclass
class _CandidateFit:
    kind: str
    feature_map: _FeatureMap
    theta: np.ndarray
    evidence: float
    diagnostic: LaplaceDiagnostic
    n_classes: int
    prior_variances: np.ndarray

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        X = self.feature_map.matrix(frame)
        H = helmert(self.n_classes, full=False).T
        gamma = self.theta.reshape(self.n_classes - 1, X.shape[1])
        coefficients = H @ gamma
        logits = X @ coefficients.T
        logits = logits - logsumexp(logits, axis=1, keepdims=True)
        return np.exp(logits)


@dataclass
class _RegimePredictive:
    candidates: tuple[_CandidateFit, ...]
    posterior_model_weights: np.ndarray

    def log_prob(self, frame: pd.DataFrame, child: str) -> np.ndarray:
        if not self.candidates:
            raise RuntimeError("Empty CD V3 predictive regime.")
        y = frame[child].to_numpy(dtype=np.int64)
        log_terms = []
        for weight, fit in zip(self.posterior_model_weights, self.candidates):
            if float(weight) <= 0.0:
                continue
            probs = fit.probabilities(frame)
            chosen = np.clip(probs[np.arange(len(frame)), y], _MIN_PROB, 1.0)
            log_terms.append(np.log(float(weight)) + np.log(chosen))
        return logsumexp(np.vstack(log_terms), axis=0)


@dataclass
class CDV3PredictiveFamily:
    child: str
    c_parents: tuple[str, ...]
    d_parents: tuple[str, ...]
    regimes: dict[tuple[int, ...], _RegimePredictive]
    fallback: _RegimePredictive

    @staticmethod
    def _key(values: Any) -> tuple[int, ...]:
        if isinstance(values, tuple):
            return tuple(int(x) for x in values)
        if isinstance(values, (list, np.ndarray)):
            return tuple(int(x) for x in values)
        return (int(values),)

    def log_prob(self, frame: pd.DataFrame) -> np.ndarray:
        frame = frame.reset_index(drop=True)
        result = np.empty(len(frame), dtype=float)
        if not self.d_parents:
            return self.fallback.log_prob(frame, self.child)

        matrix = frame[list(self.d_parents)].to_numpy(dtype=np.int64)
        grouped: dict[tuple[int, ...], list[int]] = {}
        for i, row in enumerate(matrix):
            grouped.setdefault(tuple(int(x) for x in row), []).append(i)
        for key, indices in grouped.items():
            idx = np.asarray(indices, dtype=int)
            model = self.regimes.get(key, self.fallback)
            result[idx] = model.log_prob(frame.iloc[idx], self.child)
        return result


class CDScorer:
    family_name = "softmax_laplace_bma_v3"

    def __init__(
        self,
        data: pd.DataFrame,
        cardinalities: Mapping[str, int],
        *,
        coefficient_prior_variance: float = 2.0,
        intercept_prior_variance: float = 8.0,
        max_iter: int = 300,
        candidate_models: Sequence[str] = _DEFAULT_MODELS,
    ) -> None:
        self.data = data
        self.cardinality = {str(k): int(v) for k, v in cardinalities.items()}
        self.coefficient_prior_variance = float(coefficient_prior_variance)
        self.intercept_prior_variance = float(intercept_prior_variance)
        self.max_iter = int(max_iter)
        self.candidate_models = tuple(dict.fromkeys(map(str, candidate_models)))
        self.diagnostics: list[LaplaceDiagnostic] = []
        self.model_diagnostics: list[dict[str, Any]] = []

        if self.coefficient_prior_variance <= 0 or self.intercept_prior_variance <= 0:
            raise ValueError("Softmax prior variances must be positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        allowed = set(_DEFAULT_MODELS)
        if not self.candidate_models or not set(self.candidate_models) <= allowed:
            raise ValueError(
                f"candidate_models must be a non-empty subset of {sorted(allowed)}"
            )
        if "spline5" in self.candidate_models and SplineTransformer is None:
            raise ImportError(
                "CD V3 spline5 requires scikit-learn's SplineTransformer."
            )

    def score(
        self,
        child: str,
        c_parents: Sequence[str],
        d_parents: Sequence[str],
        all_parents: Tuple[str, ...],
    ) -> Tuple[float, int, int, int]:
        c_parents = tuple(sorted(map(str, c_parents)))
        d_parents = tuple(sorted(map(str, d_parents)))
        if not c_parents:
            raise ValueError("CDScorer requires at least one continuous parent.")

        total = 0.0
        observed = 0
        selected_parameter_counts = []
        log_model_prior = -math.log(len(self.candidate_models))

        for configuration, group in self._iter_groups(d_parents):
            observed += 1
            fits = self._fit_all_candidates(
                group=group,
                child=child,
                c_parents=c_parents,
                n_classes=self.cardinality[child],
                parents=all_parents,
                configuration=configuration,
            )
            evidences = np.asarray([fit.evidence for fit in fits], dtype=float)
            finite = np.isfinite(evidences)
            if not finite.any():
                raise FloatingPointError(
                    f"All CD V3 candidates failed for {child!r} | {all_parents} "
                    f"configuration={configuration!r}."
                )
            regime_score = float(logsumexp(evidences[finite] + log_model_prior))
            total += regime_score

            best_index = int(np.nanargmax(evidences))
            best = fits[best_index]
            selected_parameter_counts.append(len(best.theta))
            posterior_logw = evidences + log_model_prior - regime_score
            posterior_w = np.where(np.isfinite(posterior_logw), np.exp(posterior_logw), 0.0)
            if posterior_w.sum() > 0:
                posterior_w /= posterior_w.sum()

            for i, fit in enumerate(fits):
                self.diagnostics.append(fit.diagnostic)
                self.model_diagnostics.append({
                    "child": child,
                    "parents": tuple(all_parents),
                    "discrete_configuration": configuration,
                    "candidate": fit.kind,
                    "evidence": float(fit.evidence),
                    "posterior_model_weight": float(posterior_w[i]),
                    "selected_by_max_evidence": bool(i == best_index),
                    "parameter_count": int(len(fit.theta)),
                    "converged": bool(fit.diagnostic.converged),
                    "gradient_norm": float(fit.diagnostic.gradient_norm),
                })

        q = self._configuration_count(d_parents)
                                                                              
                                                                         
                                                                          
        k = int(sum(selected_parameter_counts))
        return float(total), int(observed), int(k), int(q)

    def fit_predictive_family(
        self,
        child: str,
        c_parents: Sequence[str],
        d_parents: Sequence[str],
        all_parents: Sequence[str] | None = None,
    ) -> CDV3PredictiveFamily:






        c_parents = tuple(sorted(map(str, c_parents)))
        d_parents = tuple(sorted(map(str, d_parents)))
        parents = tuple(sorted(map(str, all_parents or (*c_parents, *d_parents))))
        if not c_parents:
            raise ValueError("CD predictive family requires C parents.")

                                                                              
        fallback = self._fit_predictive_regime(
            self.data,
            child,
            c_parents,
            parents,
            configuration="__fallback__",
        )
        regimes: dict[tuple[int, ...], _RegimePredictive] = {}
        if d_parents:
            for configuration, group in self._iter_groups(d_parents):
                key = self._normalize_configuration(configuration)
                regimes[key] = self._fit_predictive_regime(
                    group, child, c_parents, parents, configuration
                )
        return CDV3PredictiveFamily(
            child=child,
            c_parents=c_parents,
            d_parents=d_parents,
            regimes=regimes,
            fallback=fallback,
        )

    def _fit_predictive_regime(
        self,
        group: pd.DataFrame,
        child: str,
        c_parents: tuple[str, ...],
        parents: tuple[str, ...],
        configuration: Any,
    ) -> _RegimePredictive:
        fits = self._fit_all_candidates(
            group=group,
            child=child,
            c_parents=c_parents,
            n_classes=self.cardinality[child],
            parents=parents,
            configuration=configuration,
        )
        evidences = np.asarray([f.evidence for f in fits], dtype=float)
        finite = np.isfinite(evidences)
        if not finite.any():
            raise FloatingPointError(
                f"No predictive CD V3 candidate fitted for {child!r}."
            )
        log_prior = -math.log(len(self.candidate_models))
        norm = float(logsumexp(evidences[finite] + log_prior))
        weights = np.where(
            finite,
            np.exp(evidences + log_prior - norm),
            0.0,
        )
        weights /= weights.sum()
        return _RegimePredictive(tuple(fits), weights)

    def _fit_all_candidates(
        self,
        *,
        group: pd.DataFrame,
        child: str,
        c_parents: tuple[str, ...],
        n_classes: int,
        parents: Sequence[str],
        configuration: Any,
    ) -> list[_CandidateFit]:
        y = group[child].to_numpy(np.int64)
        fits = []
        for kind in self.candidate_models:
            try:
                fmap = self._fit_feature_map(kind, group, c_parents)
                X = fmap.matrix(group)
                fit = self._softmax_laplace_fit(
                    X=X,
                    y=y,
                    n_classes=n_classes,
                    child=child,
                    parents=tuple(parents),
                    configuration=configuration,
                    candidate=kind,
                    feature_map=fmap,
                )
            except Exception as exc:
                warnings.warn(
                    f"CD V3 candidate {kind} failed for {child!r} | "
                    f"{tuple(parents)} config={configuration!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                                                                        
                                                              
                dummy = LaplaceDiagnostic(
                    child=child,
                    parents=tuple(parents),
                    discrete_configuration=configuration,
                    converged=False,
                    iterations=0,
                    gradient_norm=float("inf"),
                    parameter_norm=float("nan"),
                    hessian_log_determinant=float("nan"),
                )
                fits.append(_CandidateFit(
                    kind=kind,
                    feature_map=_FeatureMap("linear", c_parents),
                    theta=np.zeros((n_classes - 1) * (len(c_parents) + 1)),
                    evidence=float("-inf"),
                    diagnostic=dummy,
                    n_classes=n_classes,
                    prior_variances=np.empty(0),
                ))
                continue
            fits.append(fit)
        return fits

    def _fit_feature_map(
        self,
        kind: str,
        group: pd.DataFrame,
        c_parents: tuple[str, ...],
    ) -> _FeatureMap:
        if kind != "spline5":
            return _FeatureMap(kind=kind, c_parents=c_parents)
        if SplineTransformer is None:
            raise ImportError("SplineTransformer unavailable.")
        x = group[list(c_parents)].to_numpy(dtype=float)
                                                                              
                                                                           
        spline = SplineTransformer(
            n_knots=5,
            degree=3,
            include_bias=False,
            knots="quantile",
            extrapolation="linear",
        )
        spline.fit(x)
        return _FeatureMap(kind=kind, c_parents=c_parents, spline=spline)

    def _softmax_laplace_fit(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        child: str,
        parents: tuple[str, ...],
        configuration: Any,
        candidate: str,
        feature_map: _FeatureMap,
    ) -> _CandidateFit:
        H, _, variances, objective, gradient, probabilities = self._softmax_problem(
            X, y, n_classes, use_prior=True
        )
        dimension = (n_classes - 1) * X.shape[1]
        result = minimize(
            objective,
            np.zeros(dimension),
            method="L-BFGS-B",
            jac=gradient,
            options={
                "maxiter": self.max_iter,
                "ftol": 1e-10,
                "gtol": 1e-7,
                "maxls": 50,
            },
        )
        theta = np.asarray(result.x, dtype=float)
        if not np.all(np.isfinite(theta)):
            raise FloatingPointError("Non-finite softmax MAP.")
        probs, _ = probabilities(theta)
        hessian = self._softmax_hessian(X, probs, H)
        hessian += np.diag(1.0 / variances.ravel())
        sign, logdet = np.linalg.slogdet(hessian)
        if sign <= 0 or not np.isfinite(logdet):
            identity, jitter = np.eye(dimension), 1e-10
            for _ in range(8):
                sign, logdet = np.linalg.slogdet(hessian + jitter * identity)
                if sign > 0 and np.isfinite(logdet):
                    break
                jitter *= 10.0
            else:
                raise FloatingPointError("Invalid posterior Hessian.")
        evidence = (
            -objective(theta)
            + 0.5 * dimension * np.log(2.0 * np.pi)
            - 0.5 * logdet
        )
        grad_norm = float(np.linalg.norm(gradient(theta), ord=np.inf))
        if not result.success and grad_norm > 1e-4:
            warnings.warn(
                f"CD V3 {candidate} MAP did not fully converge for {child!r} | "
                f"{parents}; gradient infinity norm={grad_norm:.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )
        diagnostic = LaplaceDiagnostic(
            child=child,
            parents=parents,
            discrete_configuration=(configuration, candidate),
            converged=bool(result.success),
            iterations=int(getattr(result, "nit", 0)),
            gradient_norm=grad_norm,
            parameter_norm=float(np.linalg.norm(theta)),
            hessian_log_determinant=float(logdet),
        )
        return _CandidateFit(
            kind=candidate,
            feature_map=feature_map,
            theta=theta,
            evidence=float(evidence),
            diagnostic=diagnostic,
            n_classes=n_classes,
            prior_variances=np.asarray(variances, dtype=float),
        )

    def _softmax_problem(self, X, y, n_classes, *, use_prior: bool):
        n, p = X.shape
        H = helmert(n_classes, full=False).T
        free_classes = n_classes - 1
        one_hot = np.eye(n_classes)[y]
        variances = np.full((free_classes, p), self.coefficient_prior_variance)
        variances[:, 0] = self.intercept_prior_variance

        def probabilities(theta: np.ndarray):
            gamma = theta.reshape(free_classes, p)
            coefficients = H @ gamma
            logits = X @ coefficients.T
            log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
            return np.exp(log_probs), log_probs

        def objective(theta: np.ndarray) -> float:
            _, log_probs = probabilities(theta)
            value = -float(np.sum(log_probs[np.arange(n), y]))
            if use_prior:
                shaped = theta.reshape(free_classes, p)
                value += 0.5 * float(np.sum(shaped ** 2 / variances))
                value += 0.5 * float(np.sum(np.log(2.0 * np.pi * variances)))
            return value

        def gradient(theta: np.ndarray) -> np.ndarray:
            probs, _ = probabilities(theta)
            grad = H.T @ ((probs - one_hot).T @ X)
            if use_prior:
                grad += theta.reshape(free_classes, p) / variances
            return grad.ravel()

        return H, one_hot, variances, objective, gradient, probabilities

    @staticmethod
    def _softmax_hessian(X: np.ndarray, probs: np.ndarray, H: np.ndarray) -> np.ndarray:
        free_classes, p = H.shape[1], X.shape[1]
        dimension = free_classes * p
        hessian = np.zeros((dimension, dimension))
        transformed = np.empty((len(X), free_classes, free_classes))
        for row, probability in enumerate(probs):
            covariance = np.diag(probability) - np.outer(probability, probability)
            transformed[row] = H.T @ covariance @ H
        for a in range(free_classes):
            a_slice = slice(a * p, (a + 1) * p)
            for b in range(free_classes):
                b_slice = slice(b * p, (b + 1) * p)
                hessian[a_slice, b_slice] = X.T @ (
                    X * transformed[:, a, b, None]
                )
        return hessian

    @staticmethod
    def _normalize_configuration(configuration: Any) -> tuple[int, ...]:
        if configuration is None:
            return ()
        if isinstance(configuration, tuple):
            return tuple(int(x) for x in configuration)
        if isinstance(configuration, (list, np.ndarray)):
            return tuple(int(x) for x in configuration)
        return (int(configuration),)

    def _configuration_count(self, d_parents: Sequence[str]) -> int:
        return int(np.prod([self.cardinality[p] for p in d_parents])) if d_parents else 1

    def _iter_groups(self, d_parents: Sequence[str]):
        if not d_parents:
            yield None, self.data
            return
        key = d_parents[0] if len(d_parents) == 1 else list(d_parents)
        yield from self.data.groupby(key, sort=False, observed=True)

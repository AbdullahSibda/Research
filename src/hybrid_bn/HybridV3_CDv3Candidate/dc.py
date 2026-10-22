"""dc.py — Continuous child with discrete parents: configuration-specific regression and Gaussian mixture-of-experts alternatives, model selection and conditional predictive densities.

All executable definitions are retained from the research implementation;
only explanatory comments and internal docstrings are consolidated here.
"""







































from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from .common import MixtureDiagnostic


                                                                             
                                     
                                                                             
DC_BMA_ENABLED = False


class DCScorer:
    CONDITIONAL_GAUSSIAN = "conditional_gaussian"
    CONDITIONAL_MIXTURE = "conditional_mixture_regression"
    GAUSSIAN_REGRESSION = "gaussian_regression"
    MIXTURE_REGRESSION = "mixture_regression"

    def __init__(
        self,
        data: pd.DataFrame,
        cardinalities: Mapping[str, int],
        *,
        clg_coefficient_prior_variance: float = 9.0,
        clg_intercept_prior_variance: float = 40.0,
        clg_inverse_gamma_shape: float = 1.5,
        clg_inverse_gamma_scale: float = 0.5,
        continuous_coefficient_prior_variance: float = 1.0,
        continuous_intercept_prior_variance: float = 4.0,
        continuous_inverse_gamma_shape: float = 1.5,
        continuous_inverse_gamma_scale: float = 0.5,
        mixture_gate_coefficient_prior_variance: float = 0.5,
        mixture_gate_intercept_prior_variance: float = 0.5,
        mixture_max_components: int = 3,
        mixture_max_iter: int = 300,
        mixture_restarts: int = 3,
        mixture_hessian_step: float = 1e-4,
        mixture_min_effective_count: float = 5.0,
    ) -> None:
        self.data = data
        self.cardinality = {str(k): int(v) for k, v in cardinalities.items()}
        self.clg_coefficient_prior_variance = float(clg_coefficient_prior_variance)
        self.clg_intercept_prior_variance = float(clg_intercept_prior_variance)
        self.clg_inverse_gamma_shape = float(clg_inverse_gamma_shape)
        self.clg_inverse_gamma_scale = float(clg_inverse_gamma_scale)
        self.continuous_coefficient_prior_variance = float(continuous_coefficient_prior_variance)
        self.continuous_intercept_prior_variance = float(continuous_intercept_prior_variance)
        self.continuous_inverse_gamma_shape = float(continuous_inverse_gamma_shape)
        self.continuous_inverse_gamma_scale = float(continuous_inverse_gamma_scale)
        self.mixture_gate_coefficient_prior_variance = float(mixture_gate_coefficient_prior_variance)
        self.mixture_gate_intercept_prior_variance = float(mixture_gate_intercept_prior_variance)
        self.mixture_max_components = int(mixture_max_components)
        self.mixture_max_iter = int(mixture_max_iter)
        self.mixture_restarts = int(mixture_restarts)
        self.mixture_hessian_step = float(mixture_hessian_step)
        self.mixture_min_effective_count = float(mixture_min_effective_count)
        self.mixture_diagnostics: List[MixtureDiagnostic] = []

                                                                             
                                                                                        
        self.bma_diagnostics: List[dict] = []

    def score(
        self,
        child: str,
        c_parents: Sequence[str],
        d_parents: Sequence[str],
        all_parents: Tuple[str, ...],
    ) -> Tuple[float, str, int, int, int, int]:
        c_parents = tuple(sorted(map(str, c_parents)))
        d_parents = tuple(sorted(map(str, d_parents)))
        if not d_parents:
            raise ValueError("DCScorer requires at least one discrete parent.")
        score, family, observed, parameter_count, selected_components = self._score_continuous_child(
            child, c_parents, d_parents, all_parents
        )
        q = self._configuration_count(d_parents)
        return (
            float(score), str(family), int(observed), int(parameter_count),
            int(selected_components), int(q)
        )

    @staticmethod
    def _design_matrix(group: pd.DataFrame, c_parents: Sequence[str]) -> np.ndarray:
        if not c_parents:
            return np.ones((len(group), 1))
        return np.column_stack([np.ones(len(group)), group[list(c_parents)].to_numpy(float)])

    def _configuration_count(self, d_parents: Sequence[str]) -> int:
        return int(np.prod([self.cardinality[p] for p in d_parents])) if d_parents else 1

    def _iter_groups(self, d_parents: Sequence[str]):
        if not d_parents:
            yield None, self.data
            return
        key = d_parents[0] if len(d_parents) == 1 else list(d_parents)
        yield from self.data.groupby(key, sort=False, observed=True)

    def _score_continuous_child(
        self,
        child: str,
        c_parents: Sequence[str],
        d_parents: Sequence[str],
        all_parents: Tuple[str, ...],
    ) -> Tuple[float, str, int, int, int]:












        if d_parents:
            total_score = 0.0
            observed = 0
            selected_parameter_count = 0
            selected_components_by_group: List[int] = []

            for configuration, group in self._iter_groups(d_parents):
                observed += 1
                y = group[child].to_numpy(float)
                X = self._design_matrix(group, c_parents)
                (
                    group_score,
                    group_parameter_count,
                    group_components,
                ) = self._score_continuous_group(
                    child=child,
                    all_parents=all_parents,
                    configuration=configuration,
                    X=X,
                    y=y,
                    coefficient_prior_variance=self.clg_coefficient_prior_variance,
                    intercept_prior_variance=self.clg_intercept_prior_variance,
                    inverse_gamma_shape=self.clg_inverse_gamma_shape,
                    inverse_gamma_scale=self.clg_inverse_gamma_scale,
                )
                total_score += group_score
                selected_parameter_count += group_parameter_count
                selected_components_by_group.append(group_components)

                                                                             
                                                                            
                                                                             
                                                                               
            q = self._configuration_count(d_parents)
            p = len(c_parents) + 1
            selected_parameter_count += max(q - observed, 0) * (p + 1)

            max_selected_components = (
                max(selected_components_by_group)
                if selected_components_by_group
                else 1
            )
            family = (
                self.CONDITIONAL_GAUSSIAN
                if max_selected_components == 1
                else self.CONDITIONAL_MIXTURE
            )
            return (
                float(total_score),
                family,
                observed,
                int(selected_parameter_count),
                int(max_selected_components),
            )

        y = self.data[child].to_numpy(float)
        X = self._design_matrix(self.data, c_parents)
        (
            selected_score,
            parameter_count,
            selected_components,
        ) = self._score_continuous_group(
            child=child,
            all_parents=all_parents,
            configuration=None,
            X=X,
            y=y,
            coefficient_prior_variance=self.continuous_coefficient_prior_variance,
            intercept_prior_variance=self.continuous_intercept_prior_variance,
            inverse_gamma_shape=self.continuous_inverse_gamma_shape,
            inverse_gamma_scale=self.continuous_inverse_gamma_scale,
        )
        family = (
            self.GAUSSIAN_REGRESSION
            if selected_components == 1
            else self.MIXTURE_REGRESSION
        )
        return (
            float(selected_score),
            family,
            1,
            int(parameter_count),
            int(selected_components),
        )

    def _score_continuous_group(
        self,
        *,
        child: str,
        all_parents: Tuple[str, ...],
        configuration: object,
        X: np.ndarray,
        y: np.ndarray,
        coefficient_prior_variance: float,
        intercept_prior_variance: float,
        inverse_gamma_shape: float,
        inverse_gamma_scale: float,
    ) -> Tuple[float, int, int]:






        p = X.shape[1]
        candidates: List[
            Tuple[int, float, int, bool, int, float, float, float, str]
        ] = []

        score_k1 = self._nig_log_marginal(
            y,
            X,
            coefficient_prior_variance=coefficient_prior_variance,
            intercept_prior_variance=intercept_prior_variance,
            inverse_gamma_shape=inverse_gamma_shape,
            inverse_gamma_scale=inverse_gamma_scale,
        )
        candidates.append(
            (1, float(score_k1), p + 1, True, 0, np.nan, np.nan, float(len(y)), "")
        )

        for n_components in range(2, self.mixture_max_components + 1):
            try:
                (
                    score,
                    converged,
                    iterations,
                    gradient_norm,
                    logdet,
                    min_effective_count,
                    parameter_count,
                ) = self._mixture_laplace(
                    X,
                    y,
                    n_components=n_components,
                    coefficient_prior_variance=coefficient_prior_variance,
                    intercept_prior_variance=intercept_prior_variance,
                    inverse_gamma_shape=inverse_gamma_shape,
                    inverse_gamma_scale=inverse_gamma_scale,
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                score = -np.inf
                converged = False
                iterations = 0
                gradient_norm = np.inf
                logdet = np.nan
                min_effective_count = 0.0
                parameter_count = self._mixture_parameter_count(p, n_components)
                failure_reason = str(exc)
            else:
                failure_reason = ""

            candidates.append(
                (
                    n_components,
                    float(score),
                    int(parameter_count),
                    bool(converged),
                    int(iterations),
                    float(gradient_norm),
                    float(logdet),
                    float(min_effective_count),
                    failure_reason,
                )
            )

        finite_candidates = [
            candidate for candidate in candidates if np.isfinite(candidate[1])
        ]
        if not finite_candidates:
            raise FloatingPointError(
                f"No finite continuous-child score for {child!r} | {all_parents} "
                f"in configuration {configuration!r}."
            )
                                                                
        selected = max(finite_candidates, key=lambda item: item[1])
        selected_components, map_score, parameter_count = selected[:3]

                                                                               
                                                                               
                                                                               
                           
        log_model_prior = -np.log(float(self.mixture_max_components))
        finite_log_joint = np.asarray(
            [
                float(candidate[1]) + log_model_prior
                for candidate in candidates
                if np.isfinite(candidate[1])
            ],
            dtype=float,
        )

        if DC_BMA_ENABLED:
            returned_score = float(logsumexp(finite_log_joint))
        else:
            returned_score = float(map_score)

                                                                     
                                                                            
                                                                      
        finite_norm = float(logsumexp(finite_log_joint))
        posterior_weights = {}
        for candidate in candidates:
            k = int(candidate[0])
            score_k = float(candidate[1])
            posterior_weights[k] = (
                float(np.exp(score_k + log_model_prior - finite_norm))
                if np.isfinite(score_k)
                else 0.0
            )

        self.bma_diagnostics.append(
            {
                "child": str(child),
                "parents": tuple(all_parents),
                "discrete_configuration": configuration,
                "bma_enabled": bool(DC_BMA_ENABLED),
                "map_components": int(selected_components),
                "map_score": float(map_score),
                "returned_score": float(returned_score),
                "bma_minus_map": float(returned_score - map_score),
                "posterior_weights": dict(posterior_weights),
                "posterior_weight_k1": float(posterior_weights.get(1, 0.0)),
                "posterior_weight_k2": float(posterior_weights.get(2, 0.0)),
                "posterior_weight_k3": float(posterior_weights.get(3, 0.0)),
            }
        )

        for (
            n_components,
            score,
            candidate_parameter_count,
            converged,
            iterations,
            gradient_norm,
            logdet,
            min_effective_count,
            failure_reason,
        ) in candidates:
            self.mixture_diagnostics.append(
                MixtureDiagnostic(
                    child=child,
                    parents=all_parents,
                    discrete_configuration=configuration,
                    n_components=int(n_components),
                    method="exact_nig" if n_components == 1 else "laplace_moe",
                    score=float(score),
                    selected=bool(n_components == selected_components),
                    converged=bool(converged),
                    iterations=int(iterations),
                    gradient_norm=float(gradient_norm),
                    parameter_count=int(candidate_parameter_count),
                    min_effective_component_size=float(min_effective_count),
                    hessian_log_determinant=float(logdet),
                    failure_reason=str(failure_reason),
                )
            )

        return (
            float(returned_score),
            int(parameter_count),
            int(selected_components),
        )

    @staticmethod
    def _mixture_parameter_count(p: int, n_components: int) -> int:
                                                               
                                                                           
        return int(n_components * (p + 1) + (n_components - 1) * p)

    def _mixture_initializations(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_components: int,
    ) -> List[np.ndarray]:
        n, p = X.shape
        beta_global, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ beta_global
        residual_variance = max(float(np.var(residual)), 0.05)
        response_scale = max(float(np.std(y, ddof=0)), 0.25)

        def pack(betas: np.ndarray, log_variances: np.ndarray, gamma: np.ndarray) -> np.ndarray:
            return np.concatenate([betas.ravel(), log_variances.ravel(), gamma.ravel()])

        starts: List[np.ndarray] = []

                                                                             
        offsets = np.linspace(-0.75, 0.75, n_components) * response_scale
        betas = np.tile(beta_global, (n_components, 1))
        betas[:, 0] += offsets
        log_variances = np.full(n_components, np.log(residual_variance))
        gamma = np.zeros((n_components - 1, p))
        starts.append(pack(betas, log_variances, gamma))

                                                               
        if n >= max(10, n_components * (p + 2)):
            boundaries = np.quantile(
                residual,
                np.linspace(0.0, 1.0, n_components + 1)[1:-1],
            )
            labels = np.digitize(residual, boundaries, right=True)
            q_betas = np.tile(beta_global, (n_components, 1))
            q_log_variances = np.full(n_components, np.log(residual_variance))
            for component in range(n_components):
                mask = labels == component
                if int(mask.sum()) >= p + 2:
                    beta_component, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
                    q_betas[component] = beta_component
                    component_residual = y[mask] - X[mask] @ beta_component
                    q_log_variances[component] = np.log(
                        max(float(np.var(component_residual)), 0.03)
                    )
            starts.append(
                pack(
                    q_betas,
                    q_log_variances,
                    np.zeros((n_components - 1, p)),
                )
            )

                                                                                
        base = starts[0]
        while len(starts) < self.mixture_restarts:
            index = len(starts)
            perturbation = 0.04 * np.sin(
                np.arange(base.size, dtype=float) + 1.7 * index
            )
            starts.append(base + perturbation)

        return starts[: self.mixture_restarts]

    def _mixture_problem(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_components: int,
        *,
        coefficient_prior_variance: float,
        intercept_prior_variance: float,
        inverse_gamma_shape: float,
        inverse_gamma_scale: float,
    ):
        _, p = X.shape
        H = helmert(n_components, full=False).T
        free_components = n_components - 1

        expert_variances = np.full(p, coefficient_prior_variance)
        expert_variances[0] = intercept_prior_variance
        expert_precision = 1.0 / expert_variances
        expert_logdet = float(np.sum(np.log(expert_variances)))

        gate_variances = np.full(
            (free_components, p),
            self.mixture_gate_coefficient_prior_variance,
        )
        gate_variances[:, 0] = self.mixture_gate_intercept_prior_variance

        a0 = inverse_gamma_shape
        b0 = inverse_gamma_scale

        beta_size = n_components * p
        variance_size = n_components

        def unpack(theta: np.ndarray):
            theta = np.asarray(theta, dtype=float)
            betas = theta[:beta_size].reshape(n_components, p)
            start = beta_size
            log_variances = theta[start : start + variance_size]
            gamma = theta[start + variance_size :].reshape(free_components, p)
            return betas, log_variances, gamma

                                                                               
                                                                               
                                                                              
                                                                             
                                                                             
                                                                             
        cached_theta: np.ndarray | None = None
        cached_state = None

        def state(theta: np.ndarray):
            nonlocal cached_theta, cached_state
            theta = np.asarray(theta, dtype=float)
            if cached_theta is not None and np.array_equal(theta, cached_theta):
                return cached_state

            betas, log_variances, gamma = unpack(theta)
            if np.any(np.abs(log_variances) > 30.0):
                raise FloatingPointError("Mixture log-variance left the stable range.")
            variances = np.exp(log_variances)
            gate_coefficients = H @ gamma
            logits = X @ gate_coefficients.T
            log_gate = logits - logsumexp(logits, axis=1, keepdims=True)
            means = X @ betas.T
            residuals = y[:, None] - means
            log_normal = -0.5 * (
                np.log(2.0 * np.pi)
                + log_variances[None, :]
                + residuals**2 / variances[None, :]
            )
            component_log_joint = log_gate + log_normal
            log_density = logsumexp(component_log_joint, axis=1)
            responsibilities = np.exp(component_log_joint - log_density[:, None])
            gate_probabilities = np.exp(log_gate)
            result = (
                betas,
                log_variances,
                gamma,
                variances,
                residuals,
                log_density,
                responsibilities,
                gate_probabilities,
            )
            cached_theta = theta.copy()
            cached_state = result
            return result

        def log_joint(theta: np.ndarray) -> float:
            (
                betas,
                log_variances,
                gamma,
                variances,
                _,
                log_density,
                _,
                _,
            ) = state(theta)

            value = float(np.sum(log_density))

                                                     
            beta_quadratic = np.sum(betas**2 * expert_precision[None, :], axis=1)
            value += float(
                np.sum(
                    -0.5 * (
                        p * np.log(2.0 * np.pi)
                        + expert_logdet
                        + p * log_variances
                        + beta_quadratic / variances
                    )
                )
            )

                                                                        
            value += float(
                np.sum(
                    a0 * np.log(b0)
                    - gammaln(a0)
                    - (a0 + 1.0) * log_variances
                    - b0 / variances
                    + log_variances
                )
            )

                                                                              
            value += float(
                np.sum(
                    -0.5 * (
                        gamma**2 / gate_variances
                        + np.log(2.0 * np.pi * gate_variances)
                    )
                )
            )
            return value

        def objective(theta: np.ndarray) -> float:
            try:
                value = -log_joint(theta)
            except FloatingPointError:
                return 1e100
            return float(value) if np.isfinite(value) else 1e100

        def gradient(theta: np.ndarray) -> np.ndarray:
            (
                betas,
                log_variances,
                gamma,
                variances,
                residuals,
                _,
                responsibilities,
                gate_probabilities,
            ) = state(theta)

            beta_gradient = np.empty_like(betas)
            log_variance_gradient = np.empty_like(log_variances)

            for component in range(n_components):
                weights = responsibilities[:, component]
                beta_gradient[component] = (
                    -X.T @ (weights * residuals[:, component]) / variances[component]
                    + betas[component] * expert_precision / variances[component]
                )

                beta_quadratic = float(
                    np.sum(betas[component] ** 2 * expert_precision)
                )
                log_variance_gradient[component] = (
                    float(
                        np.sum(
                            weights
                            * (
                                0.5
                                - 0.5
                                * residuals[:, component] ** 2
                                / variances[component]
                            )
                        )
                    )
                    + 0.5 * p
                    + a0
                    - 0.5 * beta_quadratic / variances[component]
                    - b0 / variances[component]
                )

            gate_gradient = H.T @ (
                (gate_probabilities - responsibilities).T @ X
            )
            gate_gradient += gamma / gate_variances

            result = np.concatenate(
                [
                    beta_gradient.ravel(),
                    log_variance_gradient.ravel(),
                    gate_gradient.ravel(),
                ]
            )
            if not np.all(np.isfinite(result)):
                raise FloatingPointError("Non-finite mixture gradient.")
            return result

        def responsibilities(theta: np.ndarray) -> np.ndarray:
            return state(theta)[6]

        return objective, gradient, log_joint, responsibilities

    def _numerical_hessian_from_gradient(
        self,
        gradient,
        theta: np.ndarray,
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        dimension = len(theta)
        hessian = np.empty((dimension, dimension), dtype=float)

        for column in range(dimension):
            step = self.mixture_hessian_step * max(1.0, abs(theta[column]))
            plus = theta.copy()
            minus = theta.copy()
            plus[column] += step
            minus[column] -= step
            hessian[:, column] = (gradient(plus) - gradient(minus)) / (2.0 * step)

        return 0.5 * (hessian + hessian.T)

    def _mixture_laplace(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_components: int,
        coefficient_prior_variance: float,
        intercept_prior_variance: float,
        inverse_gamma_shape: float,
        inverse_gamma_scale: float,
    ) -> Tuple[float, bool, int, float, float, float, int]:
        if n_components < 2:
            raise ValueError("Mixture Laplace scoring requires K >= 2.")

        objective, gradient, log_joint, responsibilities = self._mixture_problem(
            X,
            y,
            n_components,
            coefficient_prior_variance=coefficient_prior_variance,
            intercept_prior_variance=intercept_prior_variance,
            inverse_gamma_shape=inverse_gamma_shape,
            inverse_gamma_scale=inverse_gamma_scale,
        )
        starts = self._mixture_initializations(X, y, n_components)

        best_result = None
        best_objective = np.inf
        beta_size = n_components * X.shape[1]
        variance_size = n_components
        bounds = (
            [(None, None)] * beta_size
            + [(-8.0, 4.0)] * variance_size
            + [(None, None)] * ((n_components - 1) * X.shape[1])
        )

        for initial in starts:
            result = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                jac=gradient,
                bounds=bounds,
                options={
                    "maxiter": self.mixture_max_iter,
                    "ftol": 1e-10,
                    "gtol": 1e-7,
                    "maxls": 50,
                },
            )
            if np.isfinite(result.fun) and float(result.fun) < best_objective:
                best_objective = float(result.fun)
                best_result = result

        if best_result is None:
            raise FloatingPointError("All mixture MAP restarts failed.")

        theta = np.asarray(best_result.x, dtype=float)
        grad_norm = float(np.linalg.norm(gradient(theta), ord=np.inf))
        if not np.all(np.isfinite(theta)) or not np.isfinite(grad_norm):
            raise FloatingPointError("Non-finite mixture MAP solution.")

        hessian = self._numerical_hessian_from_gradient(gradient, theta)
        eigenvalues = np.linalg.eigvalsh(hessian)
        min_eigenvalue = float(eigenvalues.min())
        if min_eigenvalue < -1e-4:
            raise FloatingPointError(
                f"Mixture MAP Hessian is materially indefinite (min eigenvalue={min_eigenvalue:.3e})."
            )
        if min_eigenvalue <= 0.0:
            hessian = hessian + np.eye(len(theta)) * (-min_eigenvalue + 1e-8)

        sign, logdet = np.linalg.slogdet(hessian)
        if sign <= 0 or not np.isfinite(logdet):
            raise FloatingPointError("Invalid mixture posterior Hessian determinant.")

        dimension = len(theta)
                                                                             
                                                                                
                                                                     
        score = (
            log_joint(theta)
            + 0.5 * dimension * np.log(2.0 * np.pi)
            - 0.5 * logdet
            + gammaln(n_components + 1.0)
        )

        effective_counts = responsibilities(theta).sum(axis=0)
        min_effective_count = float(np.min(effective_counts))

                                                                           
                                                                              
                                                                             
                                                                        
        if min_effective_count < self.mixture_min_effective_count:
            raise FloatingPointError(
                "Mixture candidate has a near-empty component "
                f"(minimum effective count={min_effective_count:.3f})."
            )

        if not best_result.success and grad_norm > 1e-3:
            raise FloatingPointError(
                "Mixture MAP did not converge adequately "
                f"(gradient infinity norm={grad_norm:.3e})."
            )

        parameter_count = self._mixture_parameter_count(X.shape[1], n_components)
        return (
            float(score),
            bool(best_result.success),
            int(getattr(best_result, "nit", 0)),
            grad_norm,
            float(logdet),
            min_effective_count,
            int(parameter_count),
        )

    @staticmethod
    def _nig_log_marginal(
        y: np.ndarray,
        X: np.ndarray,
        *,
        coefficient_prior_variance: float,
        intercept_prior_variance: float,
        inverse_gamma_shape: float,
        inverse_gamma_scale: float,
    ) -> float:
        n, p = X.shape
        prior_variances = np.full(
            p,
            coefficient_prior_variance,
            dtype=float,
        )
        prior_variances[0] = intercept_prior_variance

        prior_precision = np.diag(1.0 / prior_variances)
        posterior_precision = prior_precision + X.T @ X
        posterior_mean = np.linalg.solve(
            posterior_precision,
            X.T @ y,
        )

        a0 = inverse_gamma_shape
        b0 = inverse_gamma_scale
        an = a0 + 0.5 * n
        bn = b0 + 0.5 * max(
            float(
                y @ y
                - posterior_mean
                @ posterior_precision
                @ posterior_mean
            ),
            0.0,
        )

        _, prior_logdet = np.linalg.slogdet(prior_precision)
        _, posterior_logdet = np.linalg.slogdet(
            posterior_precision
        )

        return float(
            -0.5 * n * np.log(2.0 * np.pi)
            + 0.5 * (prior_logdet - posterior_logdet)
            + a0 * np.log(b0)
            - an * np.log(bn)
            + gammaln(an)
            - gammaln(a0)
        )

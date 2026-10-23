from __future__ import annotations

"""
02_small_categorical_imputer.py

Reusable stochastic imputation component for the four small-missingness variables:

    - Lower Mobility
    - Managing Meds
    - Hearing
    - Eyesight

Designed for the RQ2 pipeline.

Key design rule:
    FIT on TRAINING data only.
    APPLY the fitted training distributions to BOTH training and test data.

For final full-wave discovery (after CV), fit on the full wave and generate repeated
completed datasets with different seeds.

This module does not save files by itself.
"""

from dataclasses import dataclass
from typing import Any, Iterable

import math
import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue


# ======================================================================================
# VARIABLE DEFINITIONS
# ======================================================================================

VARIABLES = {
    "Lower_Mobility": {
        "column": lambda w: f"r{w}lowermob",
        "valid_values": (0, 1, 2, 3, 4),
    },
    "Managing_Meds": {
        "column": lambda w: f"r{w}medsa",
        "valid_values": (0, 1),
    },
    "Hearing": {
        "column": lambda w: f"r{w}hearing",
        "valid_values": (1, 2, 3, 4, 5),
    },
    "Eyesight": {
        "column": lambda w: f"r{w}sight",
        "valid_values": (1, 2, 3, 4, 5, 6),
    },
}


# ======================================================================================
# HELPERS
# ======================================================================================

def _to_number(value: Any) -> float | None:
    """Convert ordinary numeric values to float; treat Stata/NaN as missing."""
    if isinstance(value, StataMissingValue):
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x):
        return None

    return x


def _as_numeric_series(series: pd.Series) -> pd.Series:
    return series.map(_to_number)


def _valid_mask(series: pd.Series, valid_values: tuple[int, ...]) -> pd.Series:
    numeric = _as_numeric_series(series)
    return numeric.isin(valid_values)


# ======================================================================================
# FITTED DISTRIBUTION
# ======================================================================================

@dataclass(frozen=True)
class FittedCategoricalDistribution:
    variable: str
    column: str
    values: tuple[int, ...]
    probabilities: tuple[float, ...]
    n_observed: int

    def sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> np.ndarray:
        if size <= 0:
            return np.empty(0, dtype=int)

        return rng.choice(
            np.asarray(self.values, dtype=int),
            size=size,
            replace=True,
            p=np.asarray(self.probabilities, dtype=float),
        )


# ======================================================================================
# IMPUTER
# ======================================================================================

class SmallCategoricalStochasticImputer:
    """
    Fit/apply stochastic categorical imputer for the four small-gap ELSA variables.

    Example
    -------
    imputer = SmallCategoricalStochasticImputer()

    # IMPORTANT: fit ONLY on training fold
    imputer.fit(train_df, wave=2)

    train_complete, train_audit = imputer.transform(
        train_df,
        seed=2001,
        return_audit=True,
    )

    test_complete, test_audit = imputer.transform(
        test_df,
        seed=2002,
        return_audit=True,
    )

    Notes
    -----
    - Observed valid values are never changed.
    - Any value outside the known valid substantive categories is treated as missing.
      This safely catches .d, .r, ordinary Stata missing '.', NaN, etc.
    - Sampling probabilities are learned only in fit().
    - No smoothing is used: probabilities are the empirical training-fold frequencies.
    """

    def __init__(self) -> None:
        self.wave: int | None = None
        self.fitted_: dict[str, FittedCategoricalDistribution] = {}

    # ----------------------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        wave: int,
    ) -> "SmallCategoricalStochasticImputer":

        if wave not in {2, 4, 6, 8}:
            raise ValueError(
                f"wave must be one of 2, 4, 6, 8; got {wave}"
            )

        self.wave = wave
        self.fitted_ = {}

        for variable, spec in VARIABLES.items():
            column = spec["column"](wave)
            valid_values = tuple(spec["valid_values"])

            if column not in train_df.columns:
                raise KeyError(
                    f"Training dataframe is missing required column '{column}' "
                    f"for {variable}."
                )

            numeric = _as_numeric_series(train_df[column])
            valid = numeric[numeric.isin(valid_values)]

            if valid.empty:
                raise ValueError(
                    f"No valid observed training values found for {variable} "
                    f"({column}) in wave {wave}."
                )

            counts = (
                valid.astype(int)
                .value_counts()
                .reindex(valid_values, fill_value=0)
                .astype(int)
            )

            n_observed = int(counts.sum())

            probabilities = (
                counts / n_observed
            ).to_numpy(dtype=float)

            # Numerical safety.
            probabilities = probabilities / probabilities.sum()

            self.fitted_[variable] = FittedCategoricalDistribution(
                variable=variable,
                column=column,
                values=valid_values,
                probabilities=tuple(float(x) for x in probabilities),
                n_observed=n_observed,
            )

        return self

    # ----------------------------------------------------------------------------------
    # APPLY
    # ----------------------------------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        seed: int,
        *,
        copy: bool = True,
        return_audit: bool = False,
        dataset_label: str | None = None,
    ):
        """
        Fill missing/invalid cells using the distributions learned in fit().

        Parameters
        ----------
        df:
            Training fold, test fold, or full-wave dataframe to complete.
        seed:
            Explicit random seed. Different completion -> different seed.
        copy:
            If True, return a copy and leave input untouched.
        return_audit:
            If True, also return a dataframe containing exactly which cells were filled.
        dataset_label:
            Optional label such as 'fold1_train', 'fold1_test', 'full_wave'.

        Returns
        -------
        completed_df
        or
        (completed_df, audit_df)
        """
        self._check_fitted()

        if not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer for reproducibility.")

        out = df.copy(deep=True) if copy else df
        rng = np.random.default_rng(int(seed))

        audit_rows: list[dict[str, Any]] = []

        for variable, fitted in self.fitted_.items():
            column = fitted.column

            if column not in out.columns:
                raise KeyError(
                    f"Dataframe passed to transform() is missing '{column}'."
                )

            valid_values = tuple(fitted.values)
            valid = _valid_mask(out[column], valid_values)
            missing_idx = out.index[~valid]

            n_missing = len(missing_idx)

            if n_missing == 0:
                continue

            draws = fitted.sample(
                rng=rng,
                size=n_missing,
            )

            originals = out.loc[missing_idx, column].copy()

            # Object dtype is safest if the source still contains StataMissingValue objects.
            if not pd.api.types.is_numeric_dtype(out[column]):
                out[column] = out[column].astype(object)

            out.loc[missing_idx, column] = draws

            for idx, original, imputed in zip(
                missing_idx,
                originals.tolist(),
                draws.tolist(),
            ):
                audit_rows.append(
                    {
                        "row_index": idx,
                        "dataset_label": dataset_label,
                        "wave": self.wave,
                        "seed": int(seed),
                        "variable": variable,
                        "column": column,
                        "original_value": str(original),
                        "imputed_value": int(imputed),
                    }
                )

        self._validate_completed(out)

        if return_audit:
            audit = pd.DataFrame(
                audit_rows,
                columns=[
                    "row_index",
                    "dataset_label",
                    "wave",
                    "seed",
                    "variable",
                    "column",
                    "original_value",
                    "imputed_value",
                ],
            )
            return out, audit

        return out

    # ----------------------------------------------------------------------------------
    # CONVENIENCE METHODS
    # ----------------------------------------------------------------------------------

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        wave: int,
        seed: int,
        *,
        copy: bool = True,
        return_audit: bool = False,
        dataset_label: str | None = None,
    ):
        """
        Convenience wrapper.

        In CV, this is appropriate for the TRAINING fold only.
        For the test fold, call transform() on the already-fitted object.
        """
        self.fit(
            train_df=train_df,
            wave=wave,
        )

        return self.transform(
            train_df,
            seed=seed,
            copy=copy,
            return_audit=return_audit,
            dataset_label=dataset_label,
        )

    def transform_many(
        self,
        df: pd.DataFrame,
        seeds: Iterable[int],
        *,
        copy: bool = True,
        dataset_label: str | None = None,
    ):
        """
        Generator yielding repeated stochastic completions.

        Yields
        ------
        seed, completed_df, audit_df

        This is useful for final repeated full-wave graph learning, or for repeated
        completions inside one CV fold.
        """
        self._check_fitted()

        for seed in seeds:
            completed, audit = self.transform(
                df=df,
                seed=int(seed),
                copy=copy,
                return_audit=True,
                dataset_label=dataset_label,
            )

            yield int(seed), completed, audit

    def probability_table(self) -> pd.DataFrame:
        """Return the fitted training-fold empirical category probabilities."""
        self._check_fitted()

        rows = []

        for variable, fitted in self.fitted_.items():
            for value, probability in zip(
                fitted.values,
                fitted.probabilities,
            ):
                rows.append(
                    {
                        "wave": self.wave,
                        "variable": variable,
                        "column": fitted.column,
                        "value": value,
                        "probability": probability,
                        "n_observed_training": fitted.n_observed,
                    }
                )

        return pd.DataFrame(rows)

    # ----------------------------------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.wave is None or not self.fitted_:
            raise RuntimeError(
                "Imputer has not been fitted. Call fit(train_df, wave) first."
            )

    def _validate_completed(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Confirm that all four target variables now contain only valid substantive values.
        """
        for variable, fitted in self.fitted_.items():
            valid = _valid_mask(
                df[fitted.column],
                fitted.values,
            )

            if not bool(valid.all()):
                n_bad = int((~valid).sum())

                raise RuntimeError(
                    f"Imputation validation failed for {variable} "
                    f"({fitted.column}): {n_bad} invalid/missing values remain."
                )


# ======================================================================================
# RECOMMENDED SEED HELPER
# ======================================================================================

def make_completion_seed(
    *,
    base_seed: int,
    wave: int,
    fold: int,
    completion: int,
    split: str,
) -> int:
    """
    Deterministically derive a seed for one fold/completion/split.

    This is only a convenience helper. It makes it easy to reproduce every stochastic
    completion exactly.

    Example
    -------
    train_seed = make_completion_seed(
        base_seed=2026,
        wave=2,
        fold=1,
        completion=7,
        split="train",
    )

    test_seed = make_completion_seed(
        base_seed=2026,
        wave=2,
        fold=1,
        completion=7,
        split="test",
    )
    """
    split = split.lower().strip()

    if split not in {"train", "test", "full"}:
        raise ValueError(
            "split must be 'train', 'test', or 'full'."
        )

    split_code = {
        "train": 11,
        "test": 23,
        "full": 37,
    }[split]

    # SeedSequence avoids fragile string hashing and gives deterministic independent streams.
    ss = np.random.SeedSequence(
        [
            int(base_seed),
            int(wave),
            int(fold),
            int(completion),
            int(split_code),
        ]
    )

    return int(
        ss.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


# ======================================================================================
# MINIMAL USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    print(
        """
02_small_categorical_imputer.py is a reusable module.

K-FOLD USAGE
------------

from 02_small_categorical_imputer import (
    SmallCategoricalStochasticImputer,
    make_completion_seed,
)

imputer = SmallCategoricalStochasticImputer()
imputer.fit(train_df, wave=2)        # FIT ON TRAIN ONLY

train_seed = make_completion_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="train",
)

test_seed = make_completion_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="test",
)

train_complete = imputer.transform(
    train_df,
    seed=train_seed,
)

test_complete = imputer.transform(
    test_df,
    seed=test_seed,
)

Both train and test are therefore filled from TRAINING-FOLD category probabilities.

FINAL FULL-WAVE USAGE
---------------------

imputer = SmallCategoricalStochasticImputer()
imputer.fit(full_wave_df, wave=2)

for m in range(1, 21):
    seed = make_completion_seed(
        base_seed=2026,
        wave=2,
        fold=0,
        completion=m,
        split="full",
    )

    completed = imputer.transform(
        full_wave_df,
        seed=seed,
    )

    # learn graph from `completed`
"""
    )

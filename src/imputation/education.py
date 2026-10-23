from __future__ import annotations

"""
07_education_conditional_imputer.py

Reusable stochastic imputer for harmonized Education (`raeduc_e`).

Final agreed recovery hierarchy
-------------------------------
1) If `raeduc_e` is observed:
       keep it unchanged.

2) If `raeduc_e` is missing AND `raedyrs_e` is observed:
       sample from the TRAINING-FOLD conditional distribution

           P(raeduc_e | raedyrs_e)

3) If BOTH `raeduc_e` and `raedyrs_e` are missing:
       sample from the TRAINING-FOLD marginal distribution

           P(raeduc_e)

Why no deterministic mapping?
-----------------------------
The mapping audit showed that `raedyrs_e` is informative but far from deterministic.
So we preserve that uncertainty by stochastic conditional sampling.

Why no `raminscage`?
--------------------
Adding minimum school-leaving age did not improve held-out mapping performance.

CV rule
-------
FIT on TRAINING data only.
APPLY the fitted training distributions to BOTH training and test data.

Final full-wave discovery
-------------------------
Fit once on the full wave and generate repeated completions using different seeds.

This module does not save files by itself.
"""

from dataclasses import dataclass
from typing import Any, Iterable

import math
import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue


# ======================================================================================
# DEFINITIONS
# ======================================================================================

EDUCATION_COL = "raeduc_e"
EDYRS_COL = "raedyrs_e"

VALID_EDUCATION = (-15, 1, 2, 3, 4, 5, 6)
VALID_EDYRS = (0, 1, 2, 3, 4, 5, 6)


# ======================================================================================
# VALUE HELPERS
# ======================================================================================

def _to_number(value: Any) -> float | None:
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


def _numeric_series(series: pd.Series) -> pd.Series:
    return series.map(_to_number)


def _education_valid_mask(series: pd.Series) -> pd.Series:
    return _numeric_series(series).isin(VALID_EDUCATION)


def _edyrs_valid_mask(series: pd.Series) -> pd.Series:
    return _numeric_series(series).isin(VALID_EDYRS)


# ======================================================================================
# FITTED DISTRIBUTIONS
# ======================================================================================

@dataclass(frozen=True)
class CategoricalDistribution:
    values: tuple[int, ...]
    probabilities: tuple[float, ...]
    n_observed: int

    def sample(
        self,
        rng: np.random.Generator,
        size: int = 1,
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

class EducationConditionalStochasticImputer:
    """
    Stochastic imputer for raeduc_e using training-derived distributions.

    Recovery hierarchy:
        observed raeduc_e
            -> unchanged

        missing raeduc_e + observed raedyrs_e
            -> P(raeduc_e | raedyrs_e)

        missing raeduc_e + missing/invalid raedyrs_e
            -> P(raeduc_e)

    Notes
    -----
    - `-15` is a VALID substantive Education category ("Other").
    - No deterministic backfilling is performed.
    - No use is made of spouse education or `raminscage`.
    - If a valid EdYrs category is absent from the training fold, the imputer
      automatically falls back to the marginal Education distribution.
    """

    def __init__(self) -> None:
        self.marginal_: CategoricalDistribution | None = None
        self.conditional_: dict[int, CategoricalDistribution] = {}
        self.fitted_: bool = False

    # ----------------------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
    ) -> "EducationConditionalStochasticImputer":

        for col in (EDUCATION_COL, EDYRS_COL):
            if col not in train_df.columns:
                raise KeyError(
                    f"Training dataframe is missing required column '{col}'."
                )

        educ = _numeric_series(
            train_df[EDUCATION_COL]
        )

        edyrs = _numeric_series(
            train_df[EDYRS_COL]
        )

        educ_valid = educ.isin(
            VALID_EDUCATION
        )

        edyrs_valid = edyrs.isin(
            VALID_EDYRS
        )

        observed_educ = educ[
            educ_valid
        ].astype(int)

        if observed_educ.empty:
            raise ValueError(
                "No valid observed raeduc_e values were found in the training data."
            )

        # --------------------------------------------------------------
        # Marginal P(Education)
        # --------------------------------------------------------------

        marginal_counts = (
            observed_educ
            .value_counts()
            .reindex(
                VALID_EDUCATION,
                fill_value=0,
            )
            .astype(int)
        )

        marginal_n = int(
            marginal_counts.sum()
        )

        marginal_probs = (
            marginal_counts
            / marginal_n
        ).to_numpy(
            dtype=float
        )

        marginal_probs = (
            marginal_probs
            / marginal_probs.sum()
        )

        self.marginal_ = CategoricalDistribution(
            values=tuple(
                int(x)
                for x in VALID_EDUCATION
            ),
            probabilities=tuple(
                float(x)
                for x in marginal_probs
            ),
            n_observed=marginal_n,
        )

        # --------------------------------------------------------------
        # Conditional P(Education | EdYrs)
        # --------------------------------------------------------------

        self.conditional_ = {}

        for edyrs_value in VALID_EDYRS:
            mask = (
                educ_valid
                & edyrs_valid
                & edyrs.eq(
                    edyrs_value
                )
            )

            group = educ[
                mask
            ].astype(int)

            if group.empty:
                continue

            counts = (
                group
                .value_counts()
                .reindex(
                    VALID_EDUCATION,
                    fill_value=0,
                )
                .astype(int)
            )

            n = int(
                counts.sum()
            )

            probs = (
                counts
                / n
            ).to_numpy(
                dtype=float
            )

            probs = (
                probs
                / probs.sum()
            )

            self.conditional_[
                int(edyrs_value)
            ] = CategoricalDistribution(
                values=tuple(
                    int(x)
                    for x in VALID_EDUCATION
                ),
                probabilities=tuple(
                    float(x)
                    for x in probs
                ),
                n_observed=n,
            )

        self.fitted_ = True

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
        Stochastically fill missing raeduc_e values.

        Parameters
        ----------
        df:
            Training split, test split, or full-wave dataframe.
        seed:
            Explicit integer random seed.
        copy:
            If True, leave input untouched.
        return_audit:
            If True, also return one row per filled Education cell.
        dataset_label:
            Optional human-readable identifier, e.g. "W4_fold2_test".
        """
        self._check_fitted()

        if not isinstance(
            seed,
            (int, np.integer),
        ):
            raise TypeError(
                "seed must be an integer."
            )

        if EDUCATION_COL not in df.columns:
            raise KeyError(
                f"Dataframe is missing '{EDUCATION_COL}'."
            )

        if EDYRS_COL not in df.columns:
            raise KeyError(
                f"Dataframe is missing '{EDYRS_COL}'."
            )

        out = (
            df.copy(deep=True)
            if copy
            else df
        )

        rng = np.random.default_rng(
            int(seed)
        )

        educ = _numeric_series(
            out[EDUCATION_COL]
        )

        edyrs = _numeric_series(
            out[EDYRS_COL]
        )

        educ_valid = educ.isin(
            VALID_EDUCATION
        )

        missing_idx = out.index[
            ~educ_valid
        ]

        audit_rows: list[dict[str, Any]] = []

        if len(missing_idx) == 0:
            if return_audit:
                return out, self._empty_audit()
            return out

        # Object dtype is safest when the source still contains StataMissingValue.
        if not pd.api.types.is_numeric_dtype(
            out[EDUCATION_COL]
        ):
            out[EDUCATION_COL] = (
                out[EDUCATION_COL]
                .astype(object)
            )

        for idx in missing_idx:
            edyrs_value = edyrs.loc[
                idx
            ]

            original = out.loc[
                idx,
                EDUCATION_COL,
            ]

            if (
                pd.notna(
                    edyrs_value
                )
                and int(
                    edyrs_value
                ) in self.conditional_
                and int(
                    edyrs_value
                ) in VALID_EDYRS
            ):
                edyrs_int = int(
                    edyrs_value
                )

                distribution = (
                    self.conditional_[
                        edyrs_int
                    ]
                )

                source = (
                    "conditional_P(Education|EdYrs)"
                )
                source_group = (
                    edyrs_int
                )

            else:
                distribution = (
                    self.marginal_
                )

                source = (
                    "marginal_P(Education)"
                )
                source_group = np.nan

            draw = int(
                distribution.sample(
                    rng=rng,
                    size=1,
                )[0]
            )

            out.loc[
                idx,
                EDUCATION_COL,
            ] = draw

            # Probability of the actual draw under the source distribution.
            value_to_prob = dict(
                zip(
                    distribution.values,
                    distribution.probabilities,
                )
            )

            audit_rows.append(
                {
                    "row_index": idx,
                    "dataset_label": dataset_label,
                    "seed": int(seed),
                    "original_education": str(
                        original
                    ),
                    "edyrs_value": (
                        int(
                            edyrs_value
                        )
                        if (
                            pd.notna(
                                edyrs_value
                            )
                            and int(
                                edyrs_value
                            ) in VALID_EDYRS
                        )
                        else np.nan
                    ),
                    "imputation_source": source,
                    "source_group_edyrs": source_group,
                    "source_n_observed_training": (
                        distribution.n_observed
                    ),
                    "imputed_education": draw,
                    "draw_probability": float(
                        value_to_prob[
                            draw
                        ]
                    ),
                }
            )

        self._validate_completed(
            out
        )

        if return_audit:
            return (
                out,
                pd.DataFrame(
                    audit_rows,
                    columns=[
                        "row_index",
                        "dataset_label",
                        "seed",
                        "original_education",
                        "edyrs_value",
                        "imputation_source",
                        "source_group_edyrs",
                        "source_n_observed_training",
                        "imputed_education",
                        "draw_probability",
                    ],
                ),
            )

        return out

    # ----------------------------------------------------------------------------------
    # CONVENIENCE
    # ----------------------------------------------------------------------------------

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        seed: int,
        *,
        copy: bool = True,
        return_audit: bool = False,
        dataset_label: str | None = None,
    ):
        self.fit(
            train_df
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
        Yield repeated stochastic completions.

        Yields
        ------
        seed, completed_df, audit_df
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

            yield (
                int(seed),
                completed,
                audit,
            )

    def probability_table(
        self,
    ) -> pd.DataFrame:
        """
        Return all fitted training-derived distributions.
        """
        self._check_fitted()

        rows = []

        # Marginal
        assert self.marginal_ is not None

        for value, probability in zip(
            self.marginal_.values,
            self.marginal_.probabilities,
        ):
            rows.append(
                {
                    "distribution": "marginal",
                    "edyrs_group": np.nan,
                    "education_value": value,
                    "probability": probability,
                    "n_observed_training": (
                        self.marginal_.n_observed
                    ),
                }
            )

        # Conditionals
        for edyrs_value in sorted(
            self.conditional_
        ):
            dist = self.conditional_[
                edyrs_value
            ]

            for value, probability in zip(
                dist.values,
                dist.probabilities,
            ):
                rows.append(
                    {
                        "distribution": "conditional",
                        "edyrs_group": edyrs_value,
                        "education_value": value,
                        "probability": probability,
                        "n_observed_training": (
                            dist.n_observed
                        ),
                    }
                )

        return pd.DataFrame(
            rows
        )

    # ----------------------------------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------------------------------

    def _check_fitted(
        self,
    ) -> None:
        if (
            not self.fitted_
            or self.marginal_ is None
        ):
            raise RuntimeError(
                "Imputer has not been fitted. "
                "Call fit(train_df) first."
            )

    def _validate_completed(
        self,
        df: pd.DataFrame,
    ) -> None:
        valid = _education_valid_mask(
            df[EDUCATION_COL]
        )

        if not bool(
            valid.all()
        ):
            n_bad = int(
                (~valid).sum()
            )

            raise RuntimeError(
                f"Education imputation validation failed: "
                f"{n_bad} missing/invalid raeduc_e values remain."
            )

    @staticmethod
    def _empty_audit(
    ) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "row_index",
                "dataset_label",
                "seed",
                "original_education",
                "edyrs_value",
                "imputation_source",
                "source_group_edyrs",
                "source_n_observed_training",
                "imputed_education",
                "draw_probability",
            ]
        )


# ======================================================================================
# REPRODUCIBLE SEED HELPER
# ======================================================================================

def make_education_seed(
    *,
    base_seed: int,
    wave: int,
    fold: int,
    completion: int,
    split: str,
) -> int:
    """
    Deterministically derive a reproducible seed.

    split:
        "train", "test", or "full"
    """
    split = split.lower().strip()

    if split not in {
        "train",
        "test",
        "full",
    }:
        raise ValueError(
            "split must be 'train', 'test', or 'full'."
        )

    split_code = {
        "train": 101,
        "test": 211,
        "full": 307,
    }[
        split
    ]

    sequence = np.random.SeedSequence(
        [
            int(base_seed),
            int(wave),
            int(fold),
            int(completion),
            int(split_code),
        ]
    )

    return int(
        sequence.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    print(
        """
07_education_conditional_imputer.py is a reusable module.

K-FOLD USAGE
------------

imputer = EducationConditionalStochasticImputer()

# CRITICAL: fit on TRAIN only
imputer.fit(train_df)

train_seed = make_education_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="train",
)

test_seed = make_education_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="test",
)

train_complete, train_audit = imputer.transform(
    train_df,
    seed=train_seed,
    return_audit=True,
    dataset_label="W2_F1_train",
)

test_complete, test_audit = imputer.transform(
    test_df,
    seed=test_seed,
    return_audit=True,
    dataset_label="W2_F1_test",
)

For missing Education:
    observed raedyrs_e  -> sample from P(Education | EdYrs)
    missing raedyrs_e   -> sample from P(Education)

Both distributions come from TRAINING DATA ONLY.


FINAL FULL-WAVE USAGE
---------------------

imputer = EducationConditionalStochasticImputer()
imputer.fit(full_wave_df)

for m in range(1, 21):

    seed = make_education_seed(
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

from __future__ import annotations

"""
10_wealth_education_conditional_imputer.py

Reusable stochastic imputer for Total Family Wealth (`h{wave}atotb`).

Final agreed rule
-----------------
1) If Total Family Wealth is observed:
       keep it unchanged.

2) If Total Family Wealth is missing and Education is valid:
       sample from the TRAINING-FOLD empirical donor distribution

           P(Wealth | Education)

3) If Education is unavailable/invalid, or the Education donor cell contains
   fewer than MIN_DONORS observed wealth values:
       sample from the TRAINING-FOLD marginal empirical donor distribution

           P(Wealth)

Pipeline order
--------------
Education should be completed BEFORE Wealth.

CV:
    fit this imputer on TRAIN only;
    apply those training-derived donor pools to both train and test.

Final discovery:
    fit on the full wave;
    generate repeated stochastic completions with different seeds.

Important
---------
- Ordinary negative wealth values are VALID and remain in donor pools.
- `raeduc_e = -15` ("Other") is a VALID education category.
- This module does not save files by itself.
"""

from dataclasses import dataclass
from typing import Any, Iterable

import math
import re
import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue


# ======================================================================================
# SETTINGS
# ======================================================================================

EDUCATION_COL = "raeduc_e"
VALID_EDUCATION = (-15, 1, 2, 3, 4, 5, 6)
MIN_DONORS = 30

WEALTH_COL = lambda wave: f"h{wave}atotb"


# ======================================================================================
# VALUE / MISSING HELPERS
# ======================================================================================

def _is_stata_missing(value: Any) -> bool:
    return isinstance(value, StataMissingValue)


def _is_nan(value: Any) -> bool:
    if _is_stata_missing(value):
        return False

    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _to_number(value: Any) -> float | None:
    if _is_stata_missing(value) or _is_nan(value):
        return None

    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    return x if math.isfinite(x) else None


def _special_tag(label: str) -> str | None:
    match = re.match(
        r"^\s*(\.[A-Za-z])(?:\s*[:=-]|\b)",
        str(label),
    )
    return match.group(1).lower() if match else None


def _wealth_missing(value: Any) -> bool:
    """
    Handles direct Stata missing objects / NaN.

    Ordinary negative numeric wealth is VALID.
    In the pipeline, source Stata special missings are expected to remain
    StataMissingValue objects because the DTA is read with convert_missing=True.
    """
    if _is_stata_missing(value):
        return True

    if _is_nan(value):
        return True

    return _to_number(value) is None


def _education_value(value: Any) -> int | None:
    x = _to_number(value)

    if x is None:
        return None

    if x in VALID_EDUCATION:
        return int(x)

    return None


# ======================================================================================
# FITTED DONOR POOL
# ======================================================================================

@dataclass(frozen=True)
class WealthDonorPool:
    values: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.values))

    def sample(
        self,
        rng: np.random.Generator,
        size: int = 1,
    ) -> np.ndarray:
        if size <= 0:
            return np.empty(
                0,
                dtype=float,
            )

        idx = rng.integers(
            0,
            len(self.values),
            size=size,
        )

        return self.values[
            idx
        ]


# ======================================================================================
# IMPUTER
# ======================================================================================

class WealthEducationConditionalImputer:
    """
    Empirical donor imputer for Total Family Wealth.

    Fit:
        learns observed TRAINING wealth donor pools:
            P(Wealth)
            P(Wealth | Education)

    Transform:
        fills only missing wealth cells.

    Notes
    -----
    - Education should already have been completed by the Education imputer.
    - If Education is still invalid, transform() safely falls back to marginal.
    - Small conditional cells also fall back to marginal.
    """

    def __init__(
        self,
        min_donors: int = MIN_DONORS,
    ) -> None:
        if min_donors < 1:
            raise ValueError(
                "min_donors must be >= 1."
            )

        self.min_donors = int(
            min_donors
        )

        self.wave: int | None = None
        self.wealth_column: str | None = None

        self.marginal_: WealthDonorPool | None = None
        self.conditional_: dict[int, WealthDonorPool] = {}

        self.fitted_: bool = False

    # ----------------------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        wave: int,
    ) -> "WealthEducationConditionalImputer":

        if wave not in {
            2,
            4,
            6,
            8,
        }:
            raise ValueError(
                f"wave must be one of 2, 4, 6, 8; got {wave}"
            )

        wealth_col = WEALTH_COL(
            wave
        )

        for col in (
            wealth_col,
            EDUCATION_COL,
        ):
            if col not in train_df.columns:
                raise KeyError(
                    f"Training dataframe is missing required column '{col}'."
                )

        self.wave = int(
            wave
        )

        self.wealth_column = (
            wealth_col
        )

        wealth_numeric = train_df[
            wealth_col
        ].map(
            _to_number
        )

        wealth_valid = ~train_df[
            wealth_col
        ].map(
            _wealth_missing
        )

        educ = train_df[
            EDUCATION_COL
        ].map(
            _education_value
        )

        observed_wealth = wealth_numeric[
            wealth_valid
        ].dropna().to_numpy(
            dtype=float
        )

        if len(
            observed_wealth
        ) == 0:
            raise ValueError(
                f"No observed valid wealth values found in training data for W{wave}."
            )

        # --------------------------------------------------------------
        # Marginal donor pool
        # --------------------------------------------------------------

        self.marginal_ = WealthDonorPool(
            values=np.asarray(
                observed_wealth,
                dtype=float,
            )
        )

        # --------------------------------------------------------------
        # Conditional donor pools by Education
        # --------------------------------------------------------------

        self.conditional_ = {}

        for education_value in VALID_EDUCATION:
            mask = (
                wealth_valid
                & educ.eq(
                    education_value
                )
            )

            values = wealth_numeric[
                mask
            ].dropna().to_numpy(
                dtype=float
            )

            if len(
                values
            ) == 0:
                continue

            self.conditional_[
                int(
                    education_value
                )
            ] = WealthDonorPool(
                values=np.asarray(
                    values,
                    dtype=float,
                )
            )

        self.fitted_ = True

        return self

    # ----------------------------------------------------------------------------------
    # TRANSFORM
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
        Fill missing Total Family Wealth values using training-derived donor pools.
        """
        self._check_fitted()

        if not isinstance(
            seed,
            (int, np.integer),
        ):
            raise TypeError(
                "seed must be an integer."
            )

        assert (
            self.wealth_column
            is not None
        )

        assert (
            self.marginal_
            is not None
        )

        wealth_col = (
            self.wealth_column
        )

        for col in (
            wealth_col,
            EDUCATION_COL,
        ):
            if col not in df.columns:
                raise KeyError(
                    f"Dataframe passed to transform() is missing '{col}'."
                )

        out = (
            df.copy(
                deep=True
            )
            if copy
            else df
        )

        rng = np.random.default_rng(
            int(
                seed
            )
        )

        missing = out[
            wealth_col
        ].map(
            _wealth_missing
        )

        missing_idx = out.index[
            missing
        ]

        audit_rows: list[
            dict[str, Any]
        ] = []

        if len(
            missing_idx
        ) == 0:
            if return_audit:
                return (
                    out,
                    self._empty_audit(),
                )

            return out

        # Object dtype is safest when original StataMissingValue objects remain.
        if not pd.api.types.is_numeric_dtype(
            out[
                wealth_col
            ]
        ):
            out[
                wealth_col
            ] = out[
                wealth_col
            ].astype(
                object
            )

        for idx in missing_idx:
            original = out.loc[
                idx,
                wealth_col,
            ]

            education = _education_value(
                out.loc[
                    idx,
                    EDUCATION_COL,
                ]
            )

            use_conditional = False

            if (
                education is not None
                and education in self.conditional_
                and self.conditional_[
                    education
                ].n
                >= self.min_donors
            ):
                pool = self.conditional_[
                    education
                ]

                source = (
                    "conditional_P(Wealth|Education)"
                )

                use_conditional = True

            else:
                pool = self.marginal_

                source = (
                    "marginal_P(Wealth)"
                )

            draw = float(
                pool.sample(
                    rng=rng,
                    size=1,
                )[0]
            )

            out.loc[
                idx,
                wealth_col,
            ] = draw

            audit_rows.append(
                {
                    "row_index": idx,
                    "dataset_label": dataset_label,
                    "wave": self.wave,
                    "seed": int(
                        seed
                    ),
                    "wealth_column": wealth_col,
                    "original_wealth": str(
                        original
                    ),
                    "education_value": (
                        education
                        if education is not None
                        else np.nan
                    ),
                    "imputation_source": source,
                    "used_conditional": use_conditional,
                    "donor_pool_n": pool.n,
                    "imputed_wealth": draw,
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
                        "wave",
                        "seed",
                        "wealth_column",
                        "original_wealth",
                        "education_value",
                        "imputation_source",
                        "used_conditional",
                        "donor_pool_n",
                        "imputed_wealth",
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
        wave: int,
        seed: int,
        *,
        copy: bool = True,
        return_audit: bool = False,
        dataset_label: str | None = None,
    ):
        self.fit(
            train_df=train_df,
            wave=wave,
        )

        return self.transform(
            df=train_df,
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
        Yield repeated stochastic wealth completions.

        Yields
        ------
        seed, completed_df, audit_df
        """
        self._check_fitted()

        for seed in seeds:
            completed, audit = self.transform(
                df=df,
                seed=int(
                    seed
                ),
                copy=copy,
                return_audit=True,
                dataset_label=dataset_label,
            )

            yield (
                int(
                    seed
                ),
                completed,
                audit,
            )

    def donor_summary(
        self,
    ) -> pd.DataFrame:
        """
        Return fitted donor pool sizes and distribution summaries.
        """
        self._check_fitted()

        assert (
            self.marginal_
            is not None
        )

        rows = []

        def describe(
            name: str,
            education: int | None,
            pool: WealthDonorPool,
        ) -> None:
            values = pool.values

            rows.append(
                {
                    "distribution": name,
                    "education_value": (
                        education
                        if education is not None
                        else np.nan
                    ),
                    "n_donors": pool.n,
                    "n_negative": int(
                        (
                            values
                            < 0
                        ).sum()
                    ),
                    "mean": float(
                        np.mean(
                            values
                        )
                    ),
                    "median": float(
                        np.median(
                            values
                        )
                    ),
                    "p05": float(
                        np.quantile(
                            values,
                            0.05,
                        )
                    ),
                    "p95": float(
                        np.quantile(
                            values,
                            0.95,
                        )
                    ),
                }
            )

        describe(
            "marginal",
            None,
            self.marginal_,
        )

        for education in sorted(
            self.conditional_
        ):
            describe(
                "conditional",
                education,
                self.conditional_[
                    education
                ],
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
            or self.marginal_
            is None
            or self.wealth_column
            is None
        ):
            raise RuntimeError(
                "Imputer has not been fitted. "
                "Call fit(train_df, wave) first."
            )

    def _validate_completed(
        self,
        df: pd.DataFrame,
    ) -> None:
        assert (
            self.wealth_column
            is not None
        )

        missing = df[
            self.wealth_column
        ].map(
            _wealth_missing
        )

        if bool(
            missing.any()
        ):
            n_bad = int(
                missing.sum()
            )

            raise RuntimeError(
                f"Wealth imputation validation failed: "
                f"{n_bad} missing/invalid values remain in {self.wealth_column}."
            )

    @staticmethod
    def _empty_audit(
    ) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "row_index",
                "dataset_label",
                "wave",
                "seed",
                "wealth_column",
                "original_wealth",
                "education_value",
                "imputation_source",
                "used_conditional",
                "donor_pool_n",
                "imputed_wealth",
            ]
        )


# ======================================================================================
# REPRODUCIBLE SEED HELPER
# ======================================================================================

def make_wealth_seed(
    *,
    base_seed: int,
    wave: int,
    fold: int,
    completion: int,
    split: str,
) -> int:
    """
    Deterministically derive one wealth-imputation seed.

    split must be:
        train
        test
        full
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
        "train": 401,
        "test": 503,
        "full": 607,
    }[
        split
    ]

    sequence = np.random.SeedSequence(
        [
            int(
                base_seed
            ),
            int(
                wave
            ),
            int(
                fold
            ),
            int(
                completion
            ),
            int(
                split_code
            ),
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
10_wealth_education_conditional_imputer.py is a reusable module.

PIPELINE ORDER
--------------

1. Complete Education first.
2. Fit/apply Wealth second.


K-FOLD EXAMPLE
--------------

wealth_imputer = WealthEducationConditionalImputer()

# TRAIN ONLY
wealth_imputer.fit(
    train_df,
    wave=2,
)

train_seed = make_wealth_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="train",
)

test_seed = make_wealth_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="test",
)

train_complete, train_audit = wealth_imputer.transform(
    train_df,
    seed=train_seed,
    return_audit=True,
    dataset_label="W2_F1_train",
)

test_complete, test_audit = wealth_imputer.transform(
    test_df,
    seed=test_seed,
    return_audit=True,
    dataset_label="W2_F1_test",
)

Both use TRAINING-FOLD donor distributions.


FINAL FULL-WAVE EXAMPLE
-----------------------

wealth_imputer = WealthEducationConditionalImputer()
wealth_imputer.fit(
    full_wave_df,
    wave=2,
)

for m in range(1, 21):

    seed = make_wealth_seed(
        base_seed=2026,
        wave=2,
        fold=0,
        completion=m,
        split="full",
    )

    completed = wealth_imputer.transform(
        full_wave_df,
        seed=seed,
    )

    # Learn graph from completed dataset.
"""
    )

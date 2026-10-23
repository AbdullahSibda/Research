from __future__ import annotations

"""
16_orientation_stochastic_imputer.py

Reusable stochastic imputer for Harmonized ELSA Orientation.

FINAL AGREED RULE
=================

Orientation summary:
    r{wave}orient = DayOfMonth + Month + Year + DayOfWeek

Components:
    r{wave}dy  = Day of month correct (0/1)
    r{wave}mo  = Month correct       (0/1)
    r{wave}yr  = Year correct        (0/1)
    r{wave}dw  = Day of week correct (0/1)

Imputation hierarchy
--------------------

1) Orientation observed
       -> keep unchanged.

2) Orientation missing, all four components observed
       -> reconstruct EXACTLY:

          Orientation = DY + MO + YR + DW

3) Orientation missing, SOME components observed
       -> jointly sample the vector of missing components from TRAINING donors
          matched on the observed component pattern only.

          Example:
              observed: Year=1, DayOfWeek=1
              missing:  DayOfMonth, Month

          draw jointly from:
              P(DayOfMonth, Month | Year=1, DayOfWeek=1)

          then recompute Orientation exactly as the sum of all four components.

          If the matching donor pool has fewer than MIN_DONORS rows, fall back
          to the TRAINING marginal joint distribution of those missing components.

4) Orientation missing, ALL FOUR components missing
       -> sample Total Orientation directly from:

          P(Orientation | AgeBand, Education)

          Age bands:
              50-59
              60-69
              70-79
              80+

          If the exact AgeBand x Education donor cell has fewer than MIN_DONORS
          training rows, fall back to:

              P(Orientation)

HARD METHODOLOGICAL RULE
========================
DEMENTIA IS NEVER USED TO IMPUTE ANY PREDICTOR.

This module never reads or uses dementia.

We also do NOT use:
    - Total Word Recall
    - Proxy
    - SelfCompletion
    - Nurse

as Orientation conditioning variables.

PIPELINE ORDER
==============
Education must already have been completed before Orientation.

CV:
    - split original incomplete data first
    - fit this imputer on TRAIN only
    - apply training-derived donor pools to TRAIN and TEST

Final discovery:
    - fit on full wave
    - generate repeated stochastic completions with different seeds

IMPORTANT
=========
- Observed Orientation is NEVER changed.
- Partial component fills are JOINT, not sequential.
- When all four components are missing, only the Orientation summary is needed
  for the BN and is filled directly; the four raw component fields remain missing.
- This module does not save files by itself.
"""

from dataclasses import dataclass
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue


# ======================================================================================
# SETTINGS
# ======================================================================================

VALID_WAVES = (2, 4, 6, 8)

EDUCATION_COL = "raeduc_e"
VALID_EDUCATION = (-15, 1, 2, 3, 4, 5, 6)

MIN_DONORS = 30

ORIENT = lambda wave: f"r{wave}orient"
DAY_OF_MONTH = lambda wave: f"r{wave}dy"
MONTH = lambda wave: f"r{wave}mo"
YEAR = lambda wave: f"r{wave}yr"
DAY_OF_WEEK = lambda wave: f"r{wave}dw"
AGE = lambda wave: f"r{wave}agey"

COMPONENTS = (
    "DayOfMonth",
    "Month",
    "Year",
    "DayOfWeek",
)

COMPONENT_BUILDERS = {
    "DayOfMonth": DAY_OF_MONTH,
    "Month": MONTH,
    "Year": YEAR,
    "DayOfWeek": DAY_OF_WEEK,
}


# ======================================================================================
# BASIC VALUE HELPERS
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


def _valid_integer(
    value: Any,
    *,
    low: int,
    high: int,
) -> int | None:
    x = _to_number(value)

    if x is None:
        return None

    if not float(x).is_integer():
        return None

    x_int = int(x)

    if low <= x_int <= high:
        return x_int

    return None


def _education_value(value: Any) -> int | None:
    x = _to_number(value)

    if x is None:
        return None

    if not float(x).is_integer():
        return None

    x_int = int(x)

    if x_int in VALID_EDUCATION:
        return x_int

    return None


def _age_band(value: Any) -> str | None:
    x = _to_number(value)

    if x is None:
        return None

    if 50 <= x < 60:
        return "50-59"

    if 60 <= x < 70:
        return "60-69"

    if 70 <= x < 80:
        return "70-79"

    if x >= 80:
        return "80+"

    return None


# ======================================================================================
# DONOR POOLS
# ======================================================================================

@dataclass(frozen=True)
class ScalarDonorPool:
    values: np.ndarray

    @property
    def n(self) -> int:
        return int(
            len(
                self.values
            )
        )

    def sample(
        self,
        rng: np.random.Generator,
    ) -> int:
        index = int(
            rng.integers(
                0,
                len(
                    self.values
                ),
            )
        )

        return int(
            self.values[
                index
            ]
        )


@dataclass(frozen=True)
class VectorDonorPool:
    values: np.ndarray

    @property
    def n(self) -> int:
        return int(
            len(
                self.values
            )
        )

    def sample(
        self,
        rng: np.random.Generator,
    ) -> np.ndarray:
        index = int(
            rng.integers(
                0,
                len(
                    self.values
                ),
            )
        )

        return self.values[
            index
        ].astype(
            int,
            copy=True,
        )


# ======================================================================================
# IMPUTER
# ======================================================================================

class OrientationStochasticImputer:
    """
    Stochastic Orientation imputer.

    FIT learns from TRAINING DATA ONLY:

        P(Orientation)
        P(Orientation | AgeBand, Education)

    and, from rows with all four binary components observed:

        joint component donor distributions conditioned on any observed
        subset/pattern required at transform time.
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

        self.orientation_col: str | None = None
        self.age_col: str | None = None
        self.component_cols: dict[
            str,
            str,
        ] = {}

        self.orientation_marginal_: ScalarDonorPool | None = None

        self.orientation_conditional_: dict[
            tuple[str, int],
            ScalarDonorPool,
        ] = {}

        self.complete_component_training_: pd.DataFrame | None = None

        self.fitted_: bool = False

    # ----------------------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        wave: int,
    ) -> "OrientationStochasticImputer":

        if wave not in VALID_WAVES:
            raise ValueError(
                f"wave must be one of {VALID_WAVES}; got {wave}"
            )

        orientation_col = ORIENT(
            wave
        )

        age_col = AGE(
            wave
        )

        component_cols = {
            name: builder(
                wave
            )
            for name, builder in COMPONENT_BUILDERS.items()
        }

        required = [
            orientation_col,
            age_col,
            EDUCATION_COL,
            *component_cols.values(),
        ]

        for col in required:
            if col not in train_df.columns:
                raise KeyError(
                    f"Training dataframe is missing required column '{col}'."
                )

        self.wave = int(
            wave
        )

        self.orientation_col = orientation_col

        self.age_col = age_col

        self.component_cols = component_cols

        orientation = train_df[
            orientation_col
        ].map(
            lambda value: _valid_integer(
                value,
                low=0,
                high=4,
            )
        )

        ages = train_df[
            age_col
        ].map(
            _age_band
        )

        education = train_df[
            EDUCATION_COL
        ].map(
            _education_value
        )

        component_frame = pd.DataFrame(
            index=train_df.index
        )

        for name, col in component_cols.items():
            component_frame[
                name
            ] = train_df[
                col
            ].map(
                lambda value: _valid_integer(
                    value,
                    low=0,
                    high=1,
                )
            )

        # --------------------------------------------------------------------------
        # Validate observed identity wherever all five values exist
        # --------------------------------------------------------------------------

        complete_components = component_frame.notna().all(
            axis=1
        )

        identity_mask = (
            orientation.notna()
            & complete_components
        )

        if bool(
            identity_mask.any()
        ):
            component_sum = component_frame.loc[
                identity_mask
            ].sum(
                axis=1
            ).astype(
                int
            )

            observed_orientation = orientation.loc[
                identity_mask
            ].astype(
                int
            )

            if not observed_orientation.equals(
                component_sum
            ):
                raise RuntimeError(
                    "Observed Orientation identity check failed in training data: "
                    "Orientation != sum of four orientation components."
                )

        # --------------------------------------------------------------------------
        # Total Orientation donor pools
        # --------------------------------------------------------------------------

        observed_orientation = orientation.dropna().astype(
            int
        )

        if observed_orientation.empty:
            raise ValueError(
                f"No observed Orientation values found in W{wave} training data."
            )

        self.orientation_marginal_ = ScalarDonorPool(
            values=observed_orientation.to_numpy(
                dtype=int
            )
        )

        total_donor_frame = pd.DataFrame(
            {
                "Orientation": orientation,
                "AgeBand": ages,
                "Education": education,
            },
            index=train_df.index,
        )

        total_donor_frame = total_donor_frame[
            total_donor_frame[
                "Orientation"
            ].notna()
            & total_donor_frame[
                "AgeBand"
            ].notna()
            & total_donor_frame[
                "Education"
            ].notna()
        ].copy()

        self.orientation_conditional_ = {}

        for (
            age_group,
            education_value,
        ), sub in total_donor_frame.groupby(
            [
                "AgeBand",
                "Education",
            ],
            observed=True,
            dropna=True,
        ):
            values = sub[
                "Orientation"
            ].astype(
                int
            ).to_numpy(
                dtype=int
            )

            self.orientation_conditional_[
                (
                    str(
                        age_group
                    ),
                    int(
                        education_value
                    ),
                )
            ] = ScalarDonorPool(
                values=values
            )

        # --------------------------------------------------------------------------
        # Complete component rows for dynamic joint donor matching
        # --------------------------------------------------------------------------

        component_training = component_frame[
            complete_components
        ].copy()

        if component_training.empty:
            raise ValueError(
                f"No complete Orientation component rows found in W{wave} training data."
            )

        for component in COMPONENTS:
            component_training[
                component
            ] = component_training[
                component
            ].astype(
                int
            )

        self.complete_component_training_ = component_training

        self.fitted_ = True

        return self

    # ----------------------------------------------------------------------------------
    # PARTIAL COMPONENT DONOR POOL
    # ----------------------------------------------------------------------------------

    def _component_donor_pool(
        self,
        *,
        observed_values: dict[str, int],
        missing_components: tuple[str, ...],
    ) -> tuple[VectorDonorPool, bool]:
        """
        Return donor vectors for the missing components.

        Primary:
            joint donor distribution conditional on ALL observed component values.

        Fallback:
            marginal joint distribution of the same missing component vector.
        """
        assert (
            self.complete_component_training_
            is not None
        )

        training = self.complete_component_training_

        match = pd.Series(
            True,
            index=training.index,
        )

        for component, value in observed_values.items():
            match &= training[
                component
            ].eq(
                value
            )

        conditional_values = training.loc[
            match,
            list(
                missing_components
            ),
        ].to_numpy(
            dtype=int
        )

        if len(
            conditional_values
        ) >= self.min_donors:
            return (
                VectorDonorPool(
                    values=conditional_values
                ),
                False,
            )

        marginal_values = training[
            list(
                missing_components
            )
        ].to_numpy(
            dtype=int
        )

        return (
            VectorDonorPool(
                values=marginal_values
            ),
            True,
        )

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
        Fill missing Orientation values.

        Observed Orientation is never changed.
        """
        self._check_fitted()

        if not isinstance(
            seed,
            (
                int,
                np.integer,
            ),
        ):
            raise TypeError(
                "seed must be an integer."
            )

        assert (
            self.orientation_col
            is not None
        )

        assert (
            self.age_col
            is not None
        )

        assert (
            self.orientation_marginal_
            is not None
        )

        required = [
            self.orientation_col,
            self.age_col,
            EDUCATION_COL,
            *self.component_cols.values(),
        ]

        for col in required:
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

        # Separate RNG streams for partial-component and all-four-missing paths.
        root_sequence = np.random.SeedSequence(
            int(
                seed
            )
        )

        partial_sequence, total_sequence = root_sequence.spawn(
            2
        )

        rng_partial = np.random.default_rng(
            partial_sequence
        )

        rng_total = np.random.default_rng(
            total_sequence
        )

        orientation_before = out[
            self.orientation_col
        ].copy(
            deep=True
        )

        orientation_valid_before = orientation_before.map(
            lambda value: _valid_integer(
                value,
                low=0,
                high=4,
            )
        )

        # Preserve Stata missing objects safely.
        for col in [
            self.orientation_col,
            *self.component_cols.values(),
        ]:
            if not pd.api.types.is_numeric_dtype(
                out[
                    col
                ]
            ):
                out[
                    col
                ] = out[
                    col
                ].astype(
                    object
                )

        audit_rows: list[
            dict[str, Any]
        ] = []

        for idx in out.index:

            original_orientation = _valid_integer(
                out.loc[
                    idx,
                    self.orientation_col,
                ],
                low=0,
                high=4,
            )

            # Observed Orientation is sacred.
            if original_orientation is not None:
                continue

            component_values = {}

            for component, col in self.component_cols.items():
                component_values[
                    component
                ] = _valid_integer(
                    out.loc[
                        idx,
                        col,
                    ],
                    low=0,
                    high=1,
                )

            observed_components = {
                component: value
                for component, value in component_values.items()
                if value is not None
            }

            missing_components = tuple(
                component
                for component in COMPONENTS
                if component_values[
                    component
                ]
                is None
            )

            # ----------------------------------------------------------------------
            # Case 1: all four components observed -> exact reconstruction
            # ----------------------------------------------------------------------

            if len(
                missing_components
            ) == 0:

                completed_orientation = int(
                    sum(
                        component_values.values()
                    )
                )

                out.loc[
                    idx,
                    self.orientation_col,
                ] = completed_orientation

                audit_rows.append(
                    {
                        "row_index": idx,
                        "dataset_label": dataset_label,
                        "wave": self.wave,
                        "seed": int(
                            seed
                        ),
                        "pathway": "exact_component_reconstruction",
                        "missing_components": "",
                        "observed_component_pattern": "|".join(
                            f"{name}={value}"
                            for name, value in observed_components.items()
                        ),
                        "conditioning_value": "",
                        "donor_pool_n": np.nan,
                        "used_fallback": False,
                        "imputed_component_values": "",
                        "imputed_orientation": completed_orientation,
                    }
                )

                continue

            # ----------------------------------------------------------------------
            # Case 2: partial component missingness -> joint donor recovery
            # ----------------------------------------------------------------------

            if 0 < len(
                missing_components
            ) < 4:

                pool, fallback = self._component_donor_pool(
                    observed_values=observed_components,
                    missing_components=missing_components,
                )

                draw = pool.sample(
                    rng_partial
                )

                imputed_pairs = []

                for component, value in zip(
                    missing_components,
                    draw,
                ):
                    col = self.component_cols[
                        component
                    ]

                    value_int = int(
                        value
                    )

                    out.loc[
                        idx,
                        col,
                    ] = value_int

                    component_values[
                        component
                    ] = value_int

                    imputed_pairs.append(
                        f"{component}={value_int}"
                    )

                completed_orientation = int(
                    sum(
                        int(
                            component_values[
                                component
                            ]
                        )
                        for component in COMPONENTS
                    )
                )

                out.loc[
                    idx,
                    self.orientation_col,
                ] = completed_orientation

                audit_rows.append(
                    {
                        "row_index": idx,
                        "dataset_label": dataset_label,
                        "wave": self.wave,
                        "seed": int(
                            seed
                        ),
                        "pathway": "joint_P(missing_components|observed_components)",
                        "missing_components": "|".join(
                            missing_components
                        ),
                        "observed_component_pattern": "|".join(
                            f"{name}={value}"
                            for name, value in observed_components.items()
                        ),
                        "conditioning_value": "ObservedComponentsOnly",
                        "donor_pool_n": pool.n,
                        "used_fallback": fallback,
                        "imputed_component_values": "|".join(
                            imputed_pairs
                        ),
                        "imputed_orientation": completed_orientation,
                    }
                )

                continue

            # ----------------------------------------------------------------------
            # Case 3: all four missing -> Orientation | AgeBand, Education
            # ----------------------------------------------------------------------

            age_group = _age_band(
                out.loc[
                    idx,
                    self.age_col,
                ]
            )

            education = _education_value(
                out.loc[
                    idx,
                    EDUCATION_COL,
                ]
            )

            conditional_pool = (
                self.orientation_conditional_.get(
                    (
                        str(
                            age_group
                        ),
                        int(
                            education
                        ),
                    )
                )
                if (
                    age_group is not None
                    and education is not None
                )
                else None
            )

            fallback = (
                conditional_pool is None
                or conditional_pool.n
                < self.min_donors
            )

            pool = (
                self.orientation_marginal_
                if fallback
                else conditional_pool
            )

            draw_orientation = int(
                pool.sample(
                    rng_total
                )
            )

            out.loc[
                idx,
                self.orientation_col,
            ] = draw_orientation

            audit_rows.append(
                {
                    "row_index": idx,
                    "dataset_label": dataset_label,
                    "wave": self.wave,
                    "seed": int(
                        seed
                    ),
                    "pathway": "P(Orientation|AgeBand,Education)",
                    "missing_components": "|".join(
                        COMPONENTS
                    ),
                    "observed_component_pattern": "",
                    "conditioning_value": (
                        f"{age_group}|{education}"
                        if (
                            age_group is not None
                            and education is not None
                        )
                        else "<MISSING_CONDITION>"
                    ),
                    "donor_pool_n": pool.n,
                    "used_fallback": fallback,
                    "imputed_component_values": "",
                    "imputed_orientation": draw_orientation,
                }
            )

        self._validate(
            out,
            orientation_valid_before=orientation_valid_before,
        )

        audit = pd.DataFrame(
            audit_rows,
            columns=[
                "row_index",
                "dataset_label",
                "wave",
                "seed",
                "pathway",
                "missing_components",
                "observed_component_pattern",
                "conditioning_value",
                "donor_pool_n",
                "used_fallback",
                "imputed_component_values",
                "imputed_orientation",
            ],
        )

        if return_audit:
            return (
                out,
                audit,
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
        Yield repeated stochastic Orientation completions.

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
        Inspect the fitted total-Orientation donor pools.
        """
        self._check_fitted()

        assert (
            self.orientation_marginal_
            is not None
        )

        rows = [
            {
                "distribution": "P(Orientation)",
                "condition": "<ALL>",
                "n_donors": self.orientation_marginal_.n,
            }
        ]

        for (
            age_group,
            education,
        ), pool in sorted(
            self.orientation_conditional_.items()
        ):
            rows.append(
                {
                    "distribution": "P(Orientation|AgeBand,Education)",
                    "condition": f"{age_group}|{education}",
                    "n_donors": pool.n,
                }
            )

        if (
            self.complete_component_training_
            is not None
        ):
            rows.append(
                {
                    "distribution": "CompleteComponentRows",
                    "condition": "<ALL>",
                    "n_donors": len(
                        self.complete_component_training_
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
            or self.orientation_col is None
            or self.age_col is None
            or self.orientation_marginal_ is None
            or self.complete_component_training_ is None
        ):
            raise RuntimeError(
                "Imputer has not been fitted. Call fit(train_df, wave) first."
            )

    def _validate(
        self,
        out: pd.DataFrame,
        *,
        orientation_valid_before: pd.Series,
    ) -> None:

        assert (
            self.orientation_col
            is not None
        )

        orientation_after = out[
            self.orientation_col
        ].map(
            lambda value: _valid_integer(
                value,
                low=0,
                high=4,
            )
        )

        if bool(
            orientation_after.isna().any()
        ):
            n_bad = int(
                orientation_after.isna().sum()
            )

            raise RuntimeError(
                f"Orientation imputation validation failed: "
                f"{n_bad} missing/invalid Orientation values remain."
            )

        # Originally observed Orientation must not change.
        originally_observed = orientation_valid_before.notna()

        before = orientation_valid_before[
            originally_observed
        ].astype(
            int
        )

        after = orientation_after[
            originally_observed
        ].astype(
            int
        )

        if not before.equals(
            after
        ):
            raise RuntimeError(
                "Orientation imputation validation failed: "
                "at least one originally observed Orientation value changed."
            )

        # Wherever all four components are observed after transform,
        # a newly imputed Orientation must agree with their exact sum.
        component_after = pd.DataFrame(
            index=out.index
        )

        for component, col in self.component_cols.items():
            component_after[
                component
            ] = out[
                col
            ].map(
                lambda value: _valid_integer(
                    value,
                    low=0,
                    high=1,
                )
            )

        all_components_present = component_after.notna().all(
            axis=1
        )

        newly_imputed = orientation_valid_before.isna()

        check = (
            all_components_present
            & newly_imputed
        )

        if bool(
            check.any()
        ):
            component_sum = component_after.loc[
                check
            ].sum(
                axis=1
            ).astype(
                int
            )

            completed_orientation = orientation_after.loc[
                check
            ].astype(
                int
            )

            if not component_sum.equals(
                completed_orientation
            ):
                raise RuntimeError(
                    "Orientation imputation validation failed: "
                    "a component-derived Orientation does not equal the four-item sum."
                )


# ======================================================================================
# REPRODUCIBLE SEED HELPER
# ======================================================================================

def make_orientation_seed(
    *,
    base_seed: int,
    wave: int,
    fold: int,
    completion: int,
    split: str,
) -> int:
    """
    Deterministically derive one Orientation-imputation seed.

    split:
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
        "train": 1009,
        "test": 1103,
        "full": 1201,
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
16_orientation_stochastic_imputer.py is a reusable module.

HARD RULE
---------
Dementia is never used by this imputer.

PIPELINE ORDER
--------------
Education must already be completed.

K-FOLD
------

orientation_imputer = OrientationStochasticImputer()

# FIT ON TRAIN ONLY
orientation_imputer.fit(
    train_df,
    wave=2,
)

train_seed = make_orientation_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="train",
)

test_seed = make_orientation_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="test",
)

train_complete, train_audit = orientation_imputer.transform(
    train_df,
    seed=train_seed,
    return_audit=True,
    dataset_label="W2_F1_train",
)

test_complete, test_audit = orientation_imputer.transform(
    test_df,
    seed=test_seed,
    return_audit=True,
    dataset_label="W2_F1_test",
)


FINAL FULL-WAVE DISCOVERY
-------------------------

orientation_imputer = OrientationStochasticImputer()

orientation_imputer.fit(
    full_wave_df,
    wave=2,
)

for m in range(1, 21):

    seed = make_orientation_seed(
        base_seed=2026,
        wave=2,
        fold=0,
        completion=m,
        split="full",
    )

    completed = orientation_imputer.transform(
        full_wave_df,
        seed=seed,
    )

    # Learn BN from completed data.
"""
    )

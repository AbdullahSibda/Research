from __future__ import annotations

"""
13_recall_stochastic_imputer.py

Reusable stochastic imputer for Total Word Recall.

FINAL AGREED RECALL RULE
========================

For a missing Total Recall (`r{wave}tr20`):

1) Both Immediate and Delayed observed
       -> recompute Total EXACTLY:

          Total = Immediate + Delayed

2) Immediate missing, Delayed observed
       -> sample Immediate from TRAINING:

          P(Immediate | Delayed)

       then recompute:

          Total = Immediate + Delayed

3) Delayed missing, Immediate observed
       -> sample Delayed from TRAINING:

          P(Delayed | Immediate)

       then recompute:

          Total = Immediate + Delayed

4) Both Immediate and Delayed missing
       -> sample Total directly from TRAINING:

          P(Total | AgeBand, Education)

       Age bands:
          50-59, 60-69, 70-79, 80+

       If the exact AgeBand x Education donor cell has fewer than MIN_DONORS
       training observations, fall back to the marginal TRAINING distribution:

          P(Total)


HARD METHODOLOGICAL RULE
========================
DEMENTIA IS NEVER USED TO IMPUTE ANY PREDICTOR.

This module never reads, conditions on, matches on, or otherwise uses dementia.


PIPELINE ORDER
==============
Education should already have been completed before Recall.

CV:
    - split original incomplete data first
    - fit this imputer on TRAIN only
    - apply fitted TRAIN donor distributions to TRAIN and TEST

Final full-wave discovery:
    - fit on full wave
    - generate repeated stochastic completions with different seeds


IMPORTANT
=========
- Observed Total Recall is NEVER changed.
- For partial component cases, the missing component is filled and Total is
  recomputed exactly.
- For both-components-missing cases, only Total is required for the BN predictor.
  The two unavailable raw component fields are left untouched.
- All stochastic donor distributions preserve observed empirical score shapes.
- No process variable (Proxy / SelfCompletion / Nurse) is used as a conditioner.

This module does not save files by itself.
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

TOTAL = lambda wave: f"r{wave}tr20"
IMMEDIATE = lambda wave: f"r{wave}imrc"
DELAYED = lambda wave: f"r{wave}dlrc"
AGE = lambda wave: f"r{wave}agey"


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
# DONOR POOL
# ======================================================================================

@dataclass(frozen=True)
class IntegerDonorPool:
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
        idx = int(
            rng.integers(
                0,
                len(
                    self.values
                ),
            )
        )

        return int(
            self.values[
                idx
            ]
        )


# ======================================================================================
# IMPUTER
# ======================================================================================

class RecallStochasticImputer:
    """
    Stochastic Total Word Recall imputer.

    Fit learns, from TRAINING DATA ONLY:

        P(Total)
        P(Total | AgeBand, Education)

        P(Immediate)
        P(Immediate | Delayed)

        P(Delayed)
        P(Delayed | Immediate)
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

        self.total_col: str | None = None
        self.immediate_col: str | None = None
        self.delayed_col: str | None = None
        self.age_col: str | None = None

        self.total_marginal_: IntegerDonorPool | None = None
        self.total_conditional_: dict[
            tuple[str, int],
            IntegerDonorPool,
        ] = {}

        self.immediate_marginal_: IntegerDonorPool | None = None
        self.immediate_given_delayed_: dict[
            int,
            IntegerDonorPool,
        ] = {}

        self.delayed_marginal_: IntegerDonorPool | None = None
        self.delayed_given_immediate_: dict[
            int,
            IntegerDonorPool,
        ] = {}

        self.fitted_: bool = False

    # ----------------------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        wave: int,
    ) -> "RecallStochasticImputer":

        if wave not in VALID_WAVES:
            raise ValueError(
                f"wave must be one of {VALID_WAVES}; got {wave}"
            )

        total_col = TOTAL(
            wave
        )

        immediate_col = IMMEDIATE(
            wave
        )

        delayed_col = DELAYED(
            wave
        )

        age_col = AGE(
            wave
        )

        required = (
            total_col,
            immediate_col,
            delayed_col,
            age_col,
            EDUCATION_COL,
        )

        for col in required:
            if col not in train_df.columns:
                raise KeyError(
                    f"Training dataframe is missing required column '{col}'."
                )

        self.wave = int(
            wave
        )

        self.total_col = total_col
        self.immediate_col = immediate_col
        self.delayed_col = delayed_col
        self.age_col = age_col

        total = train_df[
            total_col
        ].map(
            lambda value: _valid_integer(
                value,
                low=0,
                high=20,
            )
        )

        immediate = train_df[
            immediate_col
        ].map(
            lambda value: _valid_integer(
                value,
                low=0,
                high=10,
            )
        )

        delayed = train_df[
            delayed_col
        ].map(
            lambda value: _valid_integer(
                value,
                low=0,
                high=10,
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

        # --------------------------------------------------------------------------
        # Sanity check observed component identity
        # --------------------------------------------------------------------------

        identity_mask = (
            total.notna()
            & immediate.notna()
            & delayed.notna()
        )

        if bool(
            identity_mask.any()
        ):
            mismatch = (
                total[
                    identity_mask
                ].astype(
                    int
                )
                != (
                    immediate[
                        identity_mask
                    ].astype(
                        int
                    )
                    + delayed[
                        identity_mask
                    ].astype(
                        int
                    )
                )
            )

            if bool(
                mismatch.any()
            ):
                raise RuntimeError(
                    "Observed Recall identity check failed in training data: "
                    "Total != Immediate + Delayed for at least one case."
                )

        # --------------------------------------------------------------------------
        # Total Recall donor pools
        # --------------------------------------------------------------------------

        observed_total = total.dropna().astype(
            int
        )

        if observed_total.empty:
            raise ValueError(
                f"No observed Total Recall values found in W{wave} training data."
            )

        self.total_marginal_ = IntegerDonorPool(
            values=observed_total.to_numpy(
                dtype=int
            )
        )

        self.total_conditional_ = {}

        donor_frame = pd.DataFrame(
            {
                "Total": total,
                "AgeBand": ages,
                "Education": education,
            },
            index=train_df.index,
        )

        donor_frame = donor_frame[
            donor_frame[
                "Total"
            ].notna()
            & donor_frame[
                "AgeBand"
            ].notna()
            & donor_frame[
                "Education"
            ].notna()
        ].copy()

        for (
            age_group,
            education_value,
        ), sub in donor_frame.groupby(
            [
                "AgeBand",
                "Education",
            ],
            observed=True,
            dropna=True,
        ):
            values = sub[
                "Total"
            ].astype(
                int
            ).to_numpy(
                dtype=int
            )

            self.total_conditional_[
                (
                    str(
                        age_group
                    ),
                    int(
                        education_value
                    ),
                )
            ] = IntegerDonorPool(
                values=values
            )

        # --------------------------------------------------------------------------
        # Component donor pools
        # --------------------------------------------------------------------------

        component_complete = (
            immediate.notna()
            & delayed.notna()
        )

        component_frame = pd.DataFrame(
            {
                "Immediate": immediate[
                    component_complete
                ].astype(
                    int
                ),
                "Delayed": delayed[
                    component_complete
                ].astype(
                    int
                ),
            }
        )

        if component_frame.empty:
            raise ValueError(
                f"No complete Immediate/Delayed Recall pairs found in W{wave} training data."
            )

        self.immediate_marginal_ = IntegerDonorPool(
            values=component_frame[
                "Immediate"
            ].to_numpy(
                dtype=int
            )
        )

        self.delayed_marginal_ = IntegerDonorPool(
            values=component_frame[
                "Delayed"
            ].to_numpy(
                dtype=int
            )
        )

        self.immediate_given_delayed_ = {}

        for delayed_value, sub in component_frame.groupby(
            "Delayed",
            observed=True,
        ):
            self.immediate_given_delayed_[
                int(
                    delayed_value
                )
            ] = IntegerDonorPool(
                values=sub[
                    "Immediate"
                ].to_numpy(
                    dtype=int
                )
            )

        self.delayed_given_immediate_ = {}

        for immediate_value, sub in component_frame.groupby(
            "Immediate",
            observed=True,
        ):
            self.delayed_given_immediate_[
                int(
                    immediate_value
                )
            ] = IntegerDonorPool(
                values=sub[
                    "Delayed"
                ].to_numpy(
                    dtype=int
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
        Fill missing TOTAL WORD RECALL values.

        Observed Total Recall is never changed.

        In partial cases, the missing component is also filled because the final
        total must be recomputed as Immediate + Delayed.
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

        assert self.total_col is not None
        assert self.immediate_col is not None
        assert self.delayed_col is not None
        assert self.age_col is not None

        assert self.total_marginal_ is not None
        assert self.immediate_marginal_ is not None
        assert self.delayed_marginal_ is not None

        required = (
            self.total_col,
            self.immediate_col,
            self.delayed_col,
            self.age_col,
            EDUCATION_COL,
        )

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

        # Independent substreams keep the three pathways reproducible even if the
        # number of missing cases in another pathway changes.
        root_sequence = np.random.SeedSequence(
            int(
                seed
            )
        )

        child_sequences = root_sequence.spawn(
            3
        )

        rng_immediate = np.random.default_rng(
            child_sequences[
                0
            ]
        )

        rng_delayed = np.random.default_rng(
            child_sequences[
                1
            ]
        )

        rng_total = np.random.default_rng(
            child_sequences[
                2
            ]
        )

        total_before = out[
            self.total_col
        ].copy(
            deep=True
        )

        total_valid_before = total_before.map(
            lambda value: _valid_integer(
                value,
                low=0,
                high=20,
            )
        )

        audit_rows: list[
            dict[str, Any]
        ] = []

        # Use object dtype if StataMissingValue objects are still present.
        for col in (
            self.total_col,
            self.immediate_col,
            self.delayed_col,
        ):
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

        for idx in out.index:
            original_total = _valid_integer(
                out.loc[
                    idx,
                    self.total_col,
                ],
                low=0,
                high=20,
            )

            # Observed Total is sacred: leave it unchanged.
            if original_total is not None:
                continue

            immediate = _valid_integer(
                out.loc[
                    idx,
                    self.immediate_col,
                ],
                low=0,
                high=10,
            )

            delayed = _valid_integer(
                out.loc[
                    idx,
                    self.delayed_col,
                ],
                low=0,
                high=10,
            )

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

            # ----------------------------------------------------------------------
            # Case 1: both components already observed -> exact reconstruction
            # ----------------------------------------------------------------------

            if (
                immediate is not None
                and delayed is not None
            ):
                completed_total = int(
                    immediate
                    + delayed
                )

                out.loc[
                    idx,
                    self.total_col,
                ] = completed_total

                audit_rows.append(
                    {
                        "row_index": idx,
                        "dataset_label": dataset_label,
                        "wave": self.wave,
                        "seed": int(
                            seed
                        ),
                        "pathway": "exact_component_reconstruction",
                        "conditioning_value": "",
                        "donor_pool_n": np.nan,
                        "used_fallback": False,
                        "imputed_immediate": np.nan,
                        "imputed_delayed": np.nan,
                        "imputed_total": completed_total,
                    }
                )

                continue

            # ----------------------------------------------------------------------
            # Case 2: Immediate missing, Delayed observed
            # ----------------------------------------------------------------------

            if (
                immediate is None
                and delayed is not None
            ):
                pool = self.immediate_given_delayed_.get(
                    delayed
                )

                fallback = (
                    pool is None
                    or pool.n
                    < self.min_donors
                )

                if fallback:
                    pool = self.immediate_marginal_

                draw_immediate = int(
                    pool.sample(
                        rng_immediate
                    )
                )

                completed_total = int(
                    draw_immediate
                    + delayed
                )

                out.loc[
                    idx,
                    self.immediate_col,
                ] = draw_immediate

                out.loc[
                    idx,
                    self.total_col,
                ] = completed_total

                audit_rows.append(
                    {
                        "row_index": idx,
                        "dataset_label": dataset_label,
                        "wave": self.wave,
                        "seed": int(
                            seed
                        ),
                        "pathway": "P(Immediate|Delayed)",
                        "conditioning_value": delayed,
                        "donor_pool_n": pool.n,
                        "used_fallback": fallback,
                        "imputed_immediate": draw_immediate,
                        "imputed_delayed": np.nan,
                        "imputed_total": completed_total,
                    }
                )

                continue

            # ----------------------------------------------------------------------
            # Case 3: Delayed missing, Immediate observed
            # ----------------------------------------------------------------------

            if (
                immediate is not None
                and delayed is None
            ):
                pool = self.delayed_given_immediate_.get(
                    immediate
                )

                fallback = (
                    pool is None
                    or pool.n
                    < self.min_donors
                )

                if fallback:
                    pool = self.delayed_marginal_

                draw_delayed = int(
                    pool.sample(
                        rng_delayed
                    )
                )

                completed_total = int(
                    immediate
                    + draw_delayed
                )

                out.loc[
                    idx,
                    self.delayed_col,
                ] = draw_delayed

                out.loc[
                    idx,
                    self.total_col,
                ] = completed_total

                audit_rows.append(
                    {
                        "row_index": idx,
                        "dataset_label": dataset_label,
                        "wave": self.wave,
                        "seed": int(
                            seed
                        ),
                        "pathway": "P(Delayed|Immediate)",
                        "conditioning_value": immediate,
                        "donor_pool_n": pool.n,
                        "used_fallback": fallback,
                        "imputed_immediate": np.nan,
                        "imputed_delayed": draw_delayed,
                        "imputed_total": completed_total,
                    }
                )

                continue

            # ----------------------------------------------------------------------
            # Case 4: both components missing -> Total | AgeBand, Education
            # ----------------------------------------------------------------------

            (
                age_group,
                education,
            )

            conditional_pool = (
                self.total_conditional_.get(
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
                self.total_marginal_
                if fallback
                else conditional_pool
            )

            draw_total = int(
                pool.sample(
                    rng_total
                )
            )

            out.loc[
                idx,
                self.total_col,
            ] = draw_total

            audit_rows.append(
                {
                    "row_index": idx,
                    "dataset_label": dataset_label,
                    "wave": self.wave,
                    "seed": int(
                        seed
                    ),
                    "pathway": "P(Total|AgeBand,Education)",
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
                    "imputed_immediate": np.nan,
                    "imputed_delayed": np.nan,
                    "imputed_total": draw_total,
                }
            )

        self._validate(
            out,
            total_valid_before=total_valid_before,
        )

        audit = pd.DataFrame(
            audit_rows,
            columns=[
                "row_index",
                "dataset_label",
                "wave",
                "seed",
                "pathway",
                "conditioning_value",
                "donor_pool_n",
                "used_fallback",
                "imputed_immediate",
                "imputed_delayed",
                "imputed_total",
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
        Yield repeated stochastic Recall completions.

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
        Inspect fitted donor-pool sizes.
        """
        self._check_fitted()

        assert self.total_marginal_ is not None
        assert self.immediate_marginal_ is not None
        assert self.delayed_marginal_ is not None

        rows = [
            {
                "distribution": "P(Total)",
                "condition": "<ALL>",
                "n_donors": self.total_marginal_.n,
            },
            {
                "distribution": "P(Immediate)",
                "condition": "<ALL>",
                "n_donors": self.immediate_marginal_.n,
            },
            {
                "distribution": "P(Delayed)",
                "condition": "<ALL>",
                "n_donors": self.delayed_marginal_.n,
            },
        ]

        for (
            age_group,
            education,
        ), pool in sorted(
            self.total_conditional_.items()
        ):
            rows.append(
                {
                    "distribution": "P(Total|AgeBand,Education)",
                    "condition": f"{age_group}|{education}",
                    "n_donors": pool.n,
                }
            )

        for delayed, pool in sorted(
            self.immediate_given_delayed_.items()
        ):
            rows.append(
                {
                    "distribution": "P(Immediate|Delayed)",
                    "condition": delayed,
                    "n_donors": pool.n,
                }
            )

        for immediate, pool in sorted(
            self.delayed_given_immediate_.items()
        ):
            rows.append(
                {
                    "distribution": "P(Delayed|Immediate)",
                    "condition": immediate,
                    "n_donors": pool.n,
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
            or self.total_marginal_ is None
            or self.immediate_marginal_ is None
            or self.delayed_marginal_ is None
            or self.total_col is None
            or self.immediate_col is None
            or self.delayed_col is None
            or self.age_col is None
        ):
            raise RuntimeError(
                "Imputer has not been fitted. Call fit(train_df, wave) first."
            )

    def _validate(
        self,
        out: pd.DataFrame,
        *,
        total_valid_before: pd.Series,
    ) -> None:
        assert self.total_col is not None

        # Every Total Recall value must be valid after transform.
        total_after = out[
            self.total_col
        ].map(
            lambda value: _valid_integer(
                value,
                low=0,
                high=20,
            )
        )

        if bool(
            total_after.isna().any()
        ):
            n_bad = int(
                total_after.isna().sum()
            )

            raise RuntimeError(
                f"Recall imputation validation failed: "
                f"{n_bad} missing/invalid Total Recall values remain."
            )

        # Observed Total Recall must never have been altered.
        originally_observed = total_valid_before.notna()

        before_numeric = total_valid_before[
            originally_observed
        ].astype(
            int
        )

        after_numeric = total_after[
            originally_observed
        ].astype(
            int
        )

        if not before_numeric.equals(
            after_numeric
        ):
            raise RuntimeError(
                "Recall imputation validation failed: "
                "at least one originally observed Total Recall value changed."
            )


# ======================================================================================
# REPRODUCIBLE SEED HELPER
# ======================================================================================

def make_recall_seed(
    *,
    base_seed: int,
    wave: int,
    fold: int,
    completion: int,
    split: str,
) -> int:
    """
    Deterministically derive one Recall-imputation seed.

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
        "train": 701,
        "test": 809,
        "full": 907,
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
13_recall_stochastic_imputer.py is a reusable module.

HARD RULE
---------
Dementia is never used by this imputer.


K-FOLD PIPELINE
---------------

# Education must already be completed.

recall_imputer = RecallStochasticImputer()

# FIT ON TRAIN ONLY
recall_imputer.fit(
    train_df,
    wave=2,
)

train_seed = make_recall_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="train",
)

test_seed = make_recall_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="test",
)

train_complete, train_audit = recall_imputer.transform(
    train_df,
    seed=train_seed,
    return_audit=True,
    dataset_label="W2_F1_train",
)

test_complete, test_audit = recall_imputer.transform(
    test_df,
    seed=test_seed,
    return_audit=True,
    dataset_label="W2_F1_test",
)


FINAL FULL-WAVE DISCOVERY
-------------------------

recall_imputer = RecallStochasticImputer()
recall_imputer.fit(
    full_wave_df,
    wave=2,
)

for m in range(1, 21):

    seed = make_recall_seed(
        base_seed=2026,
        wave=2,
        fold=0,
        completion=m,
        split="full",
    )

    completed = recall_imputer.transform(
        full_wave_df,
        seed=seed,
    )

    # Learn BN from completed data.
"""
    )

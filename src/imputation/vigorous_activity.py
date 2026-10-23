from __future__ import annotations

"""
42_vigorous_activity_stochastic_imputer.py

Final reusable imputer for Harmonized ELSA Vigorous Activity.

TARGET
------
    r{wave}vgactx_e

Discrete ordinal-frequency categories:
    2 = >1 per week
    3 = 1 per week
    4 = 1-3 per month
    5 = hardly ever or never

FINAL LOCKED RULE
-----------------
For missing Vigorous Activity:

    VigorousActivity_missing
        ~ P(VigorousActivity | LowerMobility)

with wave-specific TRAINING donors and a wave-specific marginal fallback if
Lower Mobility is unavailable or the conditional donor pool is too small.

WHY LOWER MOBILITY ONLY?
------------------------
Script 41 showed a clear and consistent predictive gain over the marginal rule:

    W2  Brier improvement ~6.86%
    W4  ~7.12%
    W6  ~7.86%
    W8  ~8.39%

with 5/5 fold wins in every wave.

AgeBand + LowerMobility was numerically best or near-best (~8-10%), but the
extra AgeBand gain was only about 1 percentage point. Because conditional
imputation can inject BN relationships, the smaller LowerMobility-only rule is
preferred.

MISSINGNESS CONTEXT
-------------------
Actual frozen-cohort missingness is tiny:

    W2: 116 / 9,162  = 1.266%
        110/116 are proxy-coded .p
    W4: 3 / 10,738   = 0.028%
    W6: 1 / 10,365   = 0.010%
    W8: 0 / 8,352    = 0%

Process nodes remain observed BN nodes but are NOT used as imputing conditioners.

HARD RULE
---------
DEMENTIA IS NEVER USED TO IMPUTE ANY PREDICTOR.

PIPELINE / CV
-------------
Lower Mobility should already have been completed before Vigorous Activity.

For CV:
    1. split ORIGINAL incomplete data first
    2. fit on TRAIN only
    3. apply TRAIN-derived distributions to TRAIN and TEST

For final discovery:
    - fit on the full wave
    - generate repeated stochastic completions using different seeds

Observed Vigorous Activity values are NEVER overwritten.

This module does not save files by itself.
"""

from typing import Any, Iterable
import math

import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue


# ======================================================================================
# SETTINGS
# ======================================================================================

VALID_WAVES = (2, 4, 6, 8)

VIG = lambda wave: f"r{wave}vgactx_e"
LOWER_MOBILITY = lambda wave: f"r{wave}lowermob"

VALID_VIG = (2, 3, 4, 5)
VALID_LOWER_MOBILITY = (0, 1, 2, 3, 4)

MIN_DONORS = 30


# ======================================================================================
# BASIC HELPERS
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

    return x if math.isfinite(x) else None


def _valid_category(
    value: Any,
    allowed: tuple[int, ...],
) -> int | None:
    x = _to_number(value)

    if x is None or not float(x).is_integer():
        return None

    x_int = int(x)

    return x_int if x_int in allowed else None


# ======================================================================================
# IMPUTER
# ======================================================================================

class VigorousActivityStochasticImputer:
    """
    Final hierarchical imputer for rWvgactx_e.

    FIT learns from TRAINING ONLY:
        - P(VigorousActivity)
        - P(VigorousActivity | LowerMobility)
    """

    def __init__(
        self,
        *,
        min_donors: int = MIN_DONORS,
    ) -> None:
        if min_donors < 1:
            raise ValueError(
                "min_donors must be >= 1."
            )

        self.min_donors = int(min_donors)

        self.wave: int | None = None
        self.target_col: str | None = None
        self.lower_mobility_col: str | None = None

        self.marginal_: np.ndarray | None = None

        self.by_lower_mobility_: dict[
            int,
            np.ndarray,
        ] = {}

        self.fitted_ = False

    # ----------------------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        wave: int,
    ) -> "VigorousActivityStochasticImputer":

        if wave not in VALID_WAVES:
            raise ValueError(
                f"wave must be one of {VALID_WAVES}; got {wave}"
            )

        self.wave = int(wave)
        self.target_col = VIG(wave)
        self.lower_mobility_col = LOWER_MOBILITY(wave)

        for column in (
            self.target_col,
            self.lower_mobility_col,
        ):
            if column not in train_df.columns:
                raise KeyError(
                    f"Training dataframe is missing required column '{column}'."
                )

        target = train_df[
            self.target_col
        ].map(
            lambda value: _valid_category(
                value,
                VALID_VIG,
            )
        )

        lower_mobility = train_df[
            self.lower_mobility_col
        ].map(
            lambda value: _valid_category(
                value,
                VALID_LOWER_MOBILITY,
            )
        )

        marginal = target.dropna().to_numpy(
            dtype=int
        )

        if len(marginal) < self.min_donors:
            raise ValueError(
                f"W{wave}: only {len(marginal)} observed Vigorous Activity donors; "
                f"need at least {self.min_donors}."
            )

        self.marginal_ = marginal.copy()

        donor_frame = pd.DataFrame(
            {
                "Vigorous": target,
                "LowerMobility": lower_mobility,
            },
            index=train_df.index,
        )

        donor_frame = donor_frame[
            donor_frame["Vigorous"].notna()
            & donor_frame["LowerMobility"].notna()
        ].copy()

        self.by_lower_mobility_ = {}

        for mobility_value, sub in donor_frame.groupby(
            "LowerMobility",
            observed=True,
            dropna=True,
        ):
            self.by_lower_mobility_[
                int(mobility_value)
            ] = sub[
                "Vigorous"
            ].to_numpy(
                dtype=int
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

        self._check_fitted()

        if not isinstance(
            seed,
            (int, np.integer),
        ):
            raise TypeError(
                "seed must be an integer."
            )

        assert self.wave is not None
        assert self.target_col is not None
        assert self.lower_mobility_col is not None
        assert self.marginal_ is not None

        for column in (
            self.target_col,
            self.lower_mobility_col,
        ):
            if column not in df.columns:
                raise KeyError(
                    f"Dataframe passed to transform() is missing '{column}'."
                )

        out = (
            df.copy(deep=True)
            if copy
            else df
        )

        rng = np.random.default_rng(
            int(seed)
        )

        before = out[
            self.target_col
        ].copy(deep=True)

        valid_before = before.map(
            lambda value: _valid_category(
                value,
                VALID_VIG,
            )
        )

        if not pd.api.types.is_numeric_dtype(
            out[self.target_col]
        ):
            out[
                self.target_col
            ] = out[
                self.target_col
            ].astype(object)

        audit_rows: list[
            dict[str, Any]
        ] = []

        for idx in out.index:

            original = _valid_category(
                out.loc[
                    idx,
                    self.target_col,
                ],
                VALID_VIG,
            )

            # Observed values are sacred.
            if original is not None:
                continue

            lower_mobility = _valid_category(
                out.loc[
                    idx,
                    self.lower_mobility_col,
                ],
                VALID_LOWER_MOBILITY,
            )

            candidate = (
                self.by_lower_mobility_.get(
                    lower_mobility
                )
                if lower_mobility is not None
                else None
            )

            if (
                candidate is not None
                and len(candidate) >= self.min_donors
            ):
                donor_pool = candidate
                used_fallback = False
                fallback_level = "NONE"
                pathway = "VIGOROUS_GIVEN_LOWER_MOBILITY"

            else:
                donor_pool = self.marginal_
                used_fallback = True
                fallback_level = "MARGINAL_VIGOROUS"
                pathway = "MARGINAL_VIGOROUS_FALLBACK"

            imputed = int(
                rng.choice(
                    donor_pool
                )
            )

            out.loc[
                idx,
                self.target_col,
            ] = imputed

            audit_rows.append(
                {
                    "row_index": idx,
                    "dataset_label": dataset_label,
                    "wave": self.wave,
                    "seed": int(seed),
                    "pathway": pathway,
                    "lower_mobility": lower_mobility,
                    "donor_pool_n": int(
                        len(donor_pool)
                    ),
                    "used_fallback": bool(
                        used_fallback
                    ),
                    "fallback_level": fallback_level,
                    "imputed_vigorous_activity": imputed,
                }
            )

        self._validate(
            out,
            valid_before=valid_before,
        )

        audit = pd.DataFrame(
            audit_rows,
            columns=[
                "row_index",
                "dataset_label",
                "wave",
                "seed",
                "pathway",
                "lower_mobility",
                "donor_pool_n",
                "used_fallback",
                "fallback_level",
                "imputed_vigorous_activity",
            ],
        )

        if return_audit:
            return out, audit

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

    def donor_summary(
        self,
    ) -> pd.DataFrame:

        self._check_fitted()

        assert self.marginal_ is not None

        rows = [
            {
                "distribution": "P(VigorousActivity)",
                "condition": "<ALL>",
                "n_donors": len(
                    self.marginal_
                ),
            }
        ]

        for lower_mobility, donors in sorted(
            self.by_lower_mobility_.items()
        ):
            counts = {
                category: int(
                    np.sum(
                        donors == category
                    )
                )
                for category in VALID_VIG
            }

            rows.append(
                {
                    "distribution": "P(VigorousActivity|LowerMobility)",
                    "condition": int(
                        lower_mobility
                    ),
                    "n_donors": len(
                        donors
                    ),
                    **{
                        f"n_category_{category}": count
                        for category, count in counts.items()
                    },
                }
            )

        return pd.DataFrame(rows)

    # ----------------------------------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------------------------------

    def _check_fitted(
        self,
    ) -> None:

        if (
            not self.fitted_
            or self.wave is None
            or self.target_col is None
            or self.lower_mobility_col is None
            or self.marginal_ is None
        ):
            raise RuntimeError(
                "Imputer has not been fitted. Call fit(train_df, wave) first."
            )

    def _validate(
        self,
        out: pd.DataFrame,
        *,
        valid_before: pd.Series,
    ) -> None:

        assert self.target_col is not None

        after = out[
            self.target_col
        ].map(
            lambda value: _valid_category(
                value,
                VALID_VIG,
            )
        )

        if bool(
            after.isna().any()
        ):
            n_bad = int(
                after.isna().sum()
            )

            raise RuntimeError(
                f"Vigorous Activity imputation validation failed: "
                f"{n_bad} missing/invalid values remain."
            )

        originally_observed = valid_before.notna()

        before_values = valid_before[
            originally_observed
        ].astype(int)

        after_values = after[
            originally_observed
        ].astype(int)

        if not np.array_equal(
            before_values.to_numpy(),
            after_values.to_numpy(),
        ):
            raise RuntimeError(
                "Vigorous Activity imputation validation failed: "
                "at least one originally observed value changed."
            )


# ======================================================================================
# REPRODUCIBLE SEED HELPER
# ======================================================================================

def make_vigorous_seed(
    *,
    base_seed: int,
    wave: int,
    fold: int,
    completion: int,
    split: str,
) -> int:

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
        "train": 4201,
        "test": 4211,
        "full": 4219,
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
# USAGE
# ======================================================================================

if __name__ == "__main__":

    print(
        """
42_vigorous_activity_stochastic_imputer.py

Final rule:
    missing rWvgactx_e
        ~ P(VigorousActivity | LowerMobility)

with wave-specific TRAIN donors and marginal fallback.

Lower Mobility must already have been completed.
Observed Vigorous Activity is never changed.
Process nodes are not conditioners.
Dementia is never used.
"""
    )

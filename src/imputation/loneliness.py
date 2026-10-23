from __future__ import annotations

"""
25_loneliness_stochastic_imputer.py

Reusable stochastic imputer for Harmonized ELSA 3-item loneliness (`rWlnlys3`).

FINAL AGREED RULE
=================

Script 23 established that:
    - `rWlnlys3` is exactly the mean of whichever of the three component items
      are available;
    - an observed official score is therefore valid even if only 1 or 2 items
      contributed and must be preserved unchanged;
    - whenever `rWlnlys3` itself is missing, ALL THREE component items are also
      missing.

Script 24 then showed that Lower Mobility was the best conservative single
conditioner across waves.

Imputation rule
---------------

1) Loneliness observed
       -> keep unchanged.

2) Loneliness missing
       -> stochastic empirical donor draw from TRAINING:

          P(Loneliness | LowerMobility)

3) If the exact LowerMobility donor cell has fewer than MIN_DONORS rows, or
   LowerMobility itself is unavailable:
       -> fall back to marginal TRAINING:

          P(Loneliness)

The three raw loneliness component items remain missing in all-three-missing
cases. Only the BN summary variable is filled.

WHY LOWER MOBILITY?
===================
Held-out 5-fold CV across W2/W4/W6/W8 gave approximately:

    W2  ~4.01% CRPS improvement
    W4  ~2.54%
    W6  ~3.22%
    W8  ~2.59%

Weighted overall improvement: ~3.09%.

LowerMobility beat marginal filling in all 20/20 wave x fold comparisons.

AgeBand + LowerMobility was microscopically better in W2 only, but slightly
worse in W4/W6/W8. The single-variable LowerMobility rule is therefore preferred
because it gives essentially the same predictive performance while injecting
one fewer relationship into the completed data.

HARD METHODOLOGICAL RULE
========================
DEMENTIA IS NEVER USED TO IMPUTE ANY PREDICTOR.

This module never reads or uses dementia.

It also does NOT use:
    - CES-D
    - CASP-19
    - Friend Support
    - Social Participation
    - Recall
    - Orientation
    - Wealth
    - Proxy
    - SelfCompletion
    - Nurse

as Loneliness conditioners.

PIPELINE ORDER
==============
Lower Mobility should already have been completed before Loneliness.

CV:
    - split original incomplete data first
    - fit this imputer on TRAIN only
    - apply TRAIN-derived donor pools to TRAIN and TEST

Final full-wave discovery:
    - fit on the full wave
    - generate repeated stochastic completions with different seeds

IMPORTANT
=========
- Observed Loneliness is NEVER changed.
- The target remains on its original 1..3 mean-score scale, including fractional
  observed donor values.
- No rounding is performed.
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

MIN_DONORS = 30

LONELINESS = lambda wave: f"r{wave}lnlys3"
LOWER_MOBILITY = lambda wave: f"r{wave}lowermob"

VALID_LOWER_MOBILITY = (0, 1, 2, 3, 4)

LONELINESS_MIN = 1.0
LONELINESS_MAX = 3.0


# ======================================================================================
# BASIC VALUE HELPERS
# ======================================================================================

def _to_number(
    value: Any,
) -> float | None:

    if isinstance(
        value,
        StataMissingValue,
    ):
        return None

    try:
        if pd.isna(
            value
        ):
            return None
    except Exception:
        pass

    try:
        x = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        x
    ):
        return None

    return x


def _valid_loneliness(
    value: Any,
) -> float | None:

    x = _to_number(
        value
    )

    if x is None:
        return None

    if (
        x < LONELINESS_MIN
        or x > LONELINESS_MAX
    ):
        return None

    return float(
        x
    )


def _valid_lower_mobility(
    value: Any,
) -> int | None:

    x = _to_number(
        value
    )

    if x is None:
        return None

    if not float(
        x
    ).is_integer():
        return None

    x_int = int(
        x
    )

    if x_int not in VALID_LOWER_MOBILITY:
        return None

    return x_int


# ======================================================================================
# DONOR POOL
# ======================================================================================

@dataclass(frozen=True)
class FloatDonorPool:
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
    ) -> float:

        index = int(
            rng.integers(
                0,
                len(
                    self.values
                ),
            )
        )

        return float(
            self.values[
                index
            ]
        )


# ======================================================================================
# IMPUTER
# ======================================================================================

class LonelinessStochasticImputer:
    """
    Stochastic 3-item loneliness-summary imputer.

    FIT learns from TRAINING DATA ONLY:

        P(Loneliness)
        P(Loneliness | LowerMobility)
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

        self.loneliness_col: str | None = None

        self.lower_mobility_col: str | None = None

        self.marginal_: FloatDonorPool | None = None

        self.conditional_: dict[
            int,
            FloatDonorPool,
        ] = {}

        self.fitted_: bool = False

    # ----------------------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        wave: int,
    ) -> "LonelinessStochasticImputer":

        if wave not in VALID_WAVES:

            raise ValueError(
                f"wave must be one of {VALID_WAVES}; got {wave}"
            )

        loneliness_col = LONELINESS(
            wave
        )

        lower_mobility_col = LOWER_MOBILITY(
            wave
        )

        for column in (
            loneliness_col,
            lower_mobility_col,
        ):

            if column not in train_df.columns:

                raise KeyError(
                    f"Training dataframe is missing required column '{column}'."
                )

        self.wave = int(
            wave
        )

        self.loneliness_col = loneliness_col

        self.lower_mobility_col = lower_mobility_col

        loneliness = train_df[
            loneliness_col
        ].map(
            _valid_loneliness
        )

        lower_mobility = train_df[
            lower_mobility_col
        ].map(
            _valid_lower_mobility
        )

        observed_loneliness = loneliness.dropna().astype(
            float
        )

        if observed_loneliness.empty:

            raise ValueError(
                f"No observed Loneliness values found in W{wave} training data."
            )

        self.marginal_ = FloatDonorPool(
            values=observed_loneliness.to_numpy(
                dtype=float
            )
        )

        donor_frame = pd.DataFrame(
            {
                "Loneliness": loneliness,
                "LowerMobility": lower_mobility,
            },
            index=train_df.index,
        )

        donor_frame = donor_frame[
            donor_frame[
                "Loneliness"
            ].notna()
            & donor_frame[
                "LowerMobility"
            ].notna()
        ].copy()

        self.conditional_ = {}

        for mobility_value, sub in donor_frame.groupby(
            "LowerMobility",
            observed=True,
            dropna=True,
        ):

            values = sub[
                "Loneliness"
            ].astype(
                float
            ).to_numpy(
                dtype=float
            )

            self.conditional_[
                int(
                    mobility_value
                )
            ] = FloatDonorPool(
                values=values
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
            (
                int,
                np.integer,
            ),
        ):

            raise TypeError(
                "seed must be an integer."
            )

        assert self.loneliness_col is not None
        assert self.lower_mobility_col is not None
        assert self.marginal_ is not None

        for column in (
            self.loneliness_col,
            self.lower_mobility_col,
        ):

            if column not in df.columns:

                raise KeyError(
                    f"Dataframe passed to transform() is missing '{column}'."
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

        loneliness_before = out[
            self.loneliness_col
        ].copy(
            deep=True
        )

        loneliness_valid_before = loneliness_before.map(
            _valid_loneliness
        )

        if not pd.api.types.is_numeric_dtype(
            out[
                self.loneliness_col
            ]
        ):

            out[
                self.loneliness_col
            ] = out[
                self.loneliness_col
            ].astype(
                object
            )

        audit_rows: list[
            dict[str, Any]
        ] = []

        for idx in out.index:

            original_loneliness = _valid_loneliness(
                out.loc[
                    idx,
                    self.loneliness_col,
                ]
            )

            # Observed official Loneliness is sacred.
            if original_loneliness is not None:
                continue

            lower_mobility = _valid_lower_mobility(
                out.loc[
                    idx,
                    self.lower_mobility_col,
                ]
            )

            conditional_pool = (
                self.conditional_.get(
                    lower_mobility
                )
                if lower_mobility is not None
                else None
            )

            fallback = (
                conditional_pool is None
                or conditional_pool.n
                < self.min_donors
            )

            pool = (
                self.marginal_
                if fallback
                else conditional_pool
            )

            draw = float(
                pool.sample(
                    rng
                )
            )

            out.loc[
                idx,
                self.loneliness_col,
            ] = draw

            audit_rows.append(
                {
                    "row_index": idx,
                    "dataset_label": dataset_label,
                    "wave": self.wave,
                    "seed": int(
                        seed
                    ),
                    "pathway": "P(Loneliness|LowerMobility)",
                    "lower_mobility": lower_mobility,
                    "donor_pool_n": pool.n,
                    "used_fallback": fallback,
                    "fallback_level": (
                        "MARGINAL_LONELINESS"
                        if fallback
                        else "NONE"
                    ),
                    "imputed_loneliness": draw,
                }
            )

        self._validate(
            out,
            loneliness_valid_before=loneliness_valid_before,
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
                "imputed_loneliness",
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

        self._check_fitted()

        assert self.marginal_ is not None

        rows = [
            {
                "distribution": "P(Loneliness)",
                "condition": "<ALL>",
                "n_donors": self.marginal_.n,
            }
        ]

        for mobility_value, pool in sorted(
            self.conditional_.items()
        ):

            rows.append(
                {
                    "distribution": "P(Loneliness|LowerMobility)",
                    "condition": mobility_value,
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
            or self.loneliness_col is None
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
        loneliness_valid_before: pd.Series,
    ) -> None:

        assert self.loneliness_col is not None

        loneliness_after = out[
            self.loneliness_col
        ].map(
            _valid_loneliness
        )

        if bool(
            loneliness_after.isna().any()
        ):

            n_bad = int(
                loneliness_after.isna().sum()
            )

            raise RuntimeError(
                f"Loneliness imputation validation failed: "
                f"{n_bad} missing/invalid values remain."
            )

        originally_observed = loneliness_valid_before.notna()

        before = loneliness_valid_before[
            originally_observed
        ].astype(
            float
        )

        after = loneliness_after[
            originally_observed
        ].astype(
            float
        )

        if not np.allclose(
            before.to_numpy(),
            after.to_numpy(),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):

            raise RuntimeError(
                "Loneliness imputation validation failed: "
                "at least one originally observed loneliness score changed."
            )


# ======================================================================================
# REPRODUCIBLE SEED HELPER
# ======================================================================================

def make_loneliness_seed(
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
        "train": 1901,
        "test": 2003,
        "full": 2111,
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
25_loneliness_stochastic_imputer.py is a reusable module.

HARD RULE
---------
Dementia is never used by this imputer.

PIPELINE ORDER
--------------
Lower Mobility should already be completed.

K-FOLD
------

loneliness_imputer = LonelinessStochasticImputer()

loneliness_imputer.fit(
    train_df,
    wave=2,
)

train_seed = make_loneliness_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="train",
)

test_seed = make_loneliness_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="test",
)

train_complete, train_audit = loneliness_imputer.transform(
    train_df,
    seed=train_seed,
    return_audit=True,
    dataset_label="W2_F1_train",
)

test_complete, test_audit = loneliness_imputer.transform(
    test_df,
    seed=test_seed,
    return_audit=True,
    dataset_label="W2_F1_test",
)


FINAL FULL-WAVE DISCOVERY
-------------------------

loneliness_imputer = LonelinessStochasticImputer()

loneliness_imputer.fit(
    full_wave_df,
    wave=2,
)

for m in range(1, 21):

    seed = make_loneliness_seed(
        base_seed=2026,
        wave=2,
        fold=0,
        completion=m,
        split="full",
    )

    completed = loneliness_imputer.transform(
        full_wave_df,
        seed=seed,
    )

    # Learn BN from completed data.
"""
    )

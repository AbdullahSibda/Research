from __future__ import annotations

"""
19_cesd_stochastic_imputer.py

Reusable stochastic imputer for Harmonized ELSA CES-D (0..8).

FINAL AGREED RULE
=================

1) CES-D observed
       -> keep unchanged.

2) CES-D missing
       -> sample from the TRAINING empirical donor distribution:

          P(CESD | LowerMobility)

3) If the exact LowerMobility donor cell has fewer than MIN_DONORS training rows
   or LowerMobility itself is unavailable:
       -> fall back to marginal TRAINING:

          P(CESD)

WHY THIS RULE?
==============
Held-out 5-fold CV across W2/W4/W6/W8 showed that LowerMobility was the
best single conditioning variable in EVERY wave and improved CRPS by about
9% overall versus marginal filling. The improvement was positive in all 20
wave x fold comparisons.

Because conditional imputation can induce dependencies, this deliberately uses
ONE conditioner only. A later imputation-injection sensitivity audit should
compare the observed LowerMobility--CESD relationship with the completed-data
relationship.

HARD METHODOLOGICAL RULE
========================
DEMENTIA IS NEVER USED TO IMPUTE ANY PREDICTOR.

This module never reads or uses dementia.

It also does NOT use:
    - Recall
    - Orientation
    - Loneliness
    - CASP19
    - Friend Support
    - Social Participation
    - Proxy
    - SelfCompletion
    - Nurse

as CES-D conditioners.

PIPELINE ORDER
==============
Lower Mobility should already have been completed before CES-D. Its own
missingness is tiny, and this keeps the selected conditioning variable available.

CV:
    - split original incomplete data first
    - fit this imputer on TRAIN only
    - apply TRAIN-derived donor pools to TRAIN and TEST

Final full-wave discovery:
    - fit on full wave
    - generate repeated stochastic completions with different seeds

IMPORTANT
=========
- Observed CES-D is NEVER changed.
- No CES-D component item is imputed.
- Missing CES-D totals are filled directly.
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

CESD = lambda wave: f"r{wave}cesd"
LOWER_MOBILITY = lambda wave: f"r{wave}lowermob"

VALID_CESD = tuple(range(9))
VALID_LOWER_MOBILITY = (0, 1, 2, 3, 4)


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
    allowed: tuple[int, ...],
) -> int | None:
    x = _to_number(value)

    if x is None:
        return None

    if not float(x).is_integer():
        return None

    x_int = int(x)

    if x_int in allowed:
        return x_int

    return None


# ======================================================================================
# DONOR POOL
# ======================================================================================

@dataclass(frozen=True)
class IntegerDonorPool:
    values: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.values))

    def sample(
        self,
        rng: np.random.Generator,
    ) -> int:
        idx = int(
            rng.integers(
                0,
                len(self.values),
            )
        )

        return int(
            self.values[idx]
        )


# ======================================================================================
# IMPUTER
# ======================================================================================

class CESDStochasticImputer:
    """
    Stochastic CES-D imputer.

    Fit learns from TRAINING DATA ONLY:

        P(CESD)
        P(CESD | LowerMobility)
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

        self.cesd_col: str | None = None
        self.lower_mobility_col: str | None = None

        self.marginal_: IntegerDonorPool | None = None

        self.conditional_: dict[
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
    ) -> "CESDStochasticImputer":

        if wave not in VALID_WAVES:
            raise ValueError(
                f"wave must be one of {VALID_WAVES}; got {wave}"
            )

        cesd_col = CESD(
            wave
        )

        lower_mobility_col = LOWER_MOBILITY(
            wave
        )

        for col in (
            cesd_col,
            lower_mobility_col,
        ):
            if col not in train_df.columns:
                raise KeyError(
                    f"Training dataframe is missing required column '{col}'."
                )

        self.wave = int(
            wave
        )

        self.cesd_col = cesd_col
        self.lower_mobility_col = lower_mobility_col

        cesd = train_df[
            cesd_col
        ].map(
            lambda value: _valid_integer(
                value,
                allowed=VALID_CESD,
            )
        )

        lower_mobility = train_df[
            lower_mobility_col
        ].map(
            lambda value: _valid_integer(
                value,
                allowed=VALID_LOWER_MOBILITY,
            )
        )

        observed_cesd = cesd.dropna().astype(
            int
        )

        if observed_cesd.empty:
            raise ValueError(
                f"No observed CES-D values found in W{wave} training data."
            )

        self.marginal_ = IntegerDonorPool(
            values=observed_cesd.to_numpy(
                dtype=int
            )
        )

        donor_frame = pd.DataFrame(
            {
                "CESD": cesd,
                "LowerMobility": lower_mobility,
            },
            index=train_df.index,
        )

        donor_frame = donor_frame[
            donor_frame[
                "CESD"
            ].notna()
            & donor_frame[
                "LowerMobility"
            ].notna()
        ].copy()

        self.conditional_ = {}

        for lower_mobility_value, sub in donor_frame.groupby(
            "LowerMobility",
            observed=True,
            dropna=True,
        ):
            values = sub[
                "CESD"
            ].astype(
                int
            ).to_numpy(
                dtype=int
            )

            self.conditional_[
                int(
                    lower_mobility_value
                )
            ] = IntegerDonorPool(
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
        """
        Fill missing CES-D values.

        Observed CES-D is never changed.
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

        assert self.cesd_col is not None
        assert self.lower_mobility_col is not None
        assert self.marginal_ is not None

        for col in (
            self.cesd_col,
            self.lower_mobility_col,
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

        cesd_before = out[
            self.cesd_col
        ].copy(
            deep=True
        )

        cesd_valid_before = cesd_before.map(
            lambda value: _valid_integer(
                value,
                allowed=VALID_CESD,
            )
        )

        if not pd.api.types.is_numeric_dtype(
            out[
                self.cesd_col
            ]
        ):
            out[
                self.cesd_col
            ] = out[
                self.cesd_col
            ].astype(
                object
            )

        audit_rows: list[
            dict[str, Any]
        ] = []

        for idx in out.index:

            original_cesd = _valid_integer(
                out.loc[
                    idx,
                    self.cesd_col,
                ],
                allowed=VALID_CESD,
            )

            # Observed CES-D is sacred.
            if original_cesd is not None:
                continue

            lower_mobility = _valid_integer(
                out.loc[
                    idx,
                    self.lower_mobility_col,
                ],
                allowed=VALID_LOWER_MOBILITY,
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

            draw = int(
                pool.sample(
                    rng
                )
            )

            out.loc[
                idx,
                self.cesd_col,
            ] = draw

            audit_rows.append(
                {
                    "row_index": idx,
                    "dataset_label": dataset_label,
                    "wave": self.wave,
                    "seed": int(
                        seed
                    ),
                    "pathway": "P(CESD|LowerMobility)",
                    "lower_mobility": lower_mobility,
                    "donor_pool_n": pool.n,
                    "used_fallback": fallback,
                    "imputed_cesd": draw,
                }
            )

        self._validate(
            out,
            cesd_valid_before=cesd_valid_before,
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
                "imputed_cesd",
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
        Yield repeated stochastic CES-D completions.

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

        assert self.marginal_ is not None

        rows = [
            {
                "distribution": "P(CESD)",
                "condition": "<ALL>",
                "n_donors": self.marginal_.n,
            }
        ]

        for lower_mobility, pool in sorted(
            self.conditional_.items()
        ):
            rows.append(
                {
                    "distribution": "P(CESD|LowerMobility)",
                    "condition": lower_mobility,
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
            or self.cesd_col is None
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
        cesd_valid_before: pd.Series,
    ) -> None:

        assert self.cesd_col is not None

        cesd_after = out[
            self.cesd_col
        ].map(
            lambda value: _valid_integer(
                value,
                allowed=VALID_CESD,
            )
        )

        if bool(
            cesd_after.isna().any()
        ):
            n_bad = int(
                cesd_after.isna().sum()
            )

            raise RuntimeError(
                f"CES-D imputation validation failed: "
                f"{n_bad} missing/invalid CES-D values remain."
            )

        originally_observed = cesd_valid_before.notna()

        before = cesd_valid_before[
            originally_observed
        ].astype(
            int
        )

        after = cesd_after[
            originally_observed
        ].astype(
            int
        )

        if not before.equals(
            after
        ):
            raise RuntimeError(
                "CES-D imputation validation failed: "
                "at least one originally observed CES-D value changed."
            )


# ======================================================================================
# REPRODUCIBLE SEED HELPER
# ======================================================================================

def make_cesd_seed(
    *,
    base_seed: int,
    wave: int,
    fold: int,
    completion: int,
    split: str,
) -> int:
    """
    Deterministically derive one CES-D-imputation seed.

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
        "train": 1301,
        "test": 1409,
        "full": 1511,
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
19_cesd_stochastic_imputer.py is a reusable module.

HARD RULE
---------
Dementia is never used by this imputer.

PIPELINE ORDER
--------------
Lower Mobility should already be completed.

K-FOLD
------

cesd_imputer = CESDStochasticImputer()

# FIT ON TRAIN ONLY
cesd_imputer.fit(
    train_df,
    wave=2,
)

train_seed = make_cesd_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="train",
)

test_seed = make_cesd_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="test",
)

train_complete, train_audit = cesd_imputer.transform(
    train_df,
    seed=train_seed,
    return_audit=True,
    dataset_label="W2_F1_train",
)

test_complete, test_audit = cesd_imputer.transform(
    test_df,
    seed=test_seed,
    return_audit=True,
    dataset_label="W2_F1_test",
)


FINAL FULL-WAVE DISCOVERY
-------------------------

cesd_imputer = CESDStochasticImputer()

cesd_imputer.fit(
    full_wave_df,
    wave=2,
)

for m in range(1, 21):

    seed = make_cesd_seed(
        base_seed=2026,
        wave=2,
        fold=0,
        completion=m,
        split="full",
    )

    completed = cesd_imputer.transform(
        full_wave_df,
        seed=seed,
    )

    # Learn BN from completed data.
"""
    )

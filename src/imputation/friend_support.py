from __future__ import annotations

"""
28_friend_support_stochastic_imputer.py

Reusable stochastic imputer for Harmonized ELSA 6-item Friend Support Lack
(`rWfsupport6`).

FINAL AGREED RULE
=================

Script 26 established that:
    - `rWfsupport6` exactly equals the mean of whichever of the six official
      friend-support items are observed;
    - observed official scores are valid even with partial item availability and
      must be preserved unchanged;
    - whenever `rWfsupport6` itself is missing, ALL SIX component items are also
      missing.

Script 27 then established that all missing cases fall cleanly into one of three
person-level component-code patterns:

    .c|.c|.c|.c|.c|.c = no self-completion
    .n|.n|.n|.n|.n|.n = respondent reports no friends
    .m|.m|.m|.m|.m|.m = other all-item missingness

There were no mixed reason patterns.

CONDITIONING RESULT
-------------------
Hearing was the numerically best external conditioner in every wave, but its
weighted overall CRPS improvement over marginal filling was only about 0.43%.

Because this thesis explicitly aims to avoid manufacturing BN dependencies
through imputation, that gain is judged too small to justify forcing an
additional FriendSupport--Hearing relationship into thousands of completed
rows.

Therefore the primary rule is intentionally conservative:

    FriendSupport_missing ~ P(FriendSupport)

using the empirical TRAINING distribution only.

NO-FRIENDS CASES
----------------
The `.n = no friends` group is conceptually different from ordinary missingness:
the six questions are undefined because there are no friends to rate.

We therefore DO NOT:
    - map "no friends" to 4 (maximum lack of support);
    - map "no friends" to any deterministic score;
    - condition its imputation on another substantive BN variable.

For computational completion only, `.n` cases receive the SAME marginal
stochastic draw as other missing cases. The audit trail preserves their
NO_FRIENDS status so they can be examined in a later sensitivity analysis.

This is a neutral completion device, not a claim that a no-friends respondent
has the sampled support score.

HARD METHODOLOGICAL RULE
========================
DEMENTIA IS NEVER USED TO IMPUTE ANY PREDICTOR.

This module never reads or uses dementia.

It also does NOT use:
    - Loneliness
    - CES-D
    - CASP-19
    - Social Participation
    - Recall
    - Orientation
    - Wealth
    - Hearing
    - Eyesight
    - Lower Mobility
    - Education
    - Age
    - Proxy
    - SelfCompletion
    - Nurse

as Friend Support conditioners.

CV
==
    - split the original incomplete data first;
    - fit this imputer on TRAIN only;
    - apply the TRAINING marginal donor distribution to TRAIN and TEST.

Final full-wave discovery
=========================
    - fit on the full wave;
    - generate repeated stochastic completions using different deterministic
      seeds.

IMPORTANT
=========
- Observed Friend Support scores are NEVER changed.
- No raw component item is filled when all six are missing.
- No-friends status is preserved in the returned audit dataframe.
- The target remains on its original 1..4 mean-score scale; no rounding.
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

FSUPPORT6 = lambda wave: f"r{wave}fsupport6"

COMPONENT_BUILDERS = {
    "FriendsUnderstand": lambda wave: f"r{wave}fustdfe",
    "CanRelyOnFriends": lambda wave: f"r{wave}frely",
    "CanOpenUpToFriends": lambda wave: f"r{wave}fopenup",
    "FriendsCriticize": lambda wave: f"r{wave}fcritze",
    "FriendsLetDown": lambda wave: f"r{wave}fletdow",
    "FriendsGetOnNerves": lambda wave: f"r{wave}fgetnev",
}

COMPONENTS = tuple(COMPONENT_BUILDERS)

FSUPPORT_MIN = 1.0
FSUPPORT_MAX = 4.0


# ======================================================================================
# BASIC VALUE HELPERS
# ======================================================================================

def _to_number(
    value: Any,
) -> float | None:

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


def _valid_friend_support(
    value: Any,
) -> float | None:

    x = _to_number(value)

    if x is None:
        return None

    if x < FSUPPORT_MIN or x > FSUPPORT_MAX:
        return None

    return float(x)


def _raw_missing_code(
    value: Any,
) -> str:

    if isinstance(value, StataMissingValue):
        text = str(value).strip().lower()

        if text.startswith(".") and len(text) >= 2:
            return text[:2]

        return "."

    try:
        if pd.isna(value):
            return "."
    except Exception:
        pass

    return "OBSERVED"


def _reason_class(
    row: pd.Series,
    component_columns: dict[str, str],
) -> tuple[str, str]:

    codes = tuple(
        _raw_missing_code(
            row[column]
        )
        for column in component_columns.values()
    )

    pattern = "|".join(codes)

    if all(code == ".n" for code in codes):
        return (
            "NO_FRIENDS_ALL6_N",
            pattern,
        )

    if all(code == ".c" for code in codes):
        return (
            "NO_SELF_COMPLETION_ALL6_C",
            pattern,
        )

    if all(code == ".m" for code in codes):
        return (
            "OTHER_MISSING_ALL6_M",
            pattern,
        )

    if any(code == ".n" for code in codes):
        return (
            "MIXED_PATTERN_INCLUDES_N",
            pattern,
        )

    if any(code == ".c" for code in codes):
        return (
            "MIXED_PATTERN_INCLUDES_C",
            pattern,
        )

    if any(code == ".m" for code in codes):
        return (
            "MIXED_PATTERN_INCLUDES_M",
            pattern,
        )

    return (
        "UNCLASSIFIED_MISSING",
        pattern,
    )


# ======================================================================================
# DONOR POOL
# ======================================================================================

@dataclass(frozen=True)
class FloatDonorPool:
    values: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.values))

    def sample(
        self,
        rng: np.random.Generator,
    ) -> float:

        index = int(
            rng.integers(
                0,
                len(self.values),
            )
        )

        return float(
            self.values[index]
        )


# ======================================================================================
# IMPUTER
# ======================================================================================

class FriendSupportStochasticImputer:
    """
    Stochastic Friend Support Lack imputer.

    FIT learns from TRAINING DATA ONLY:

        P(FriendSupport)
    """

    def __init__(
        self,
    ) -> None:

        self.wave: int | None = None
        self.target_col: str | None = None

        self.component_cols: dict[
            str,
            str,
        ] = {}

        self.marginal_: FloatDonorPool | None = None

        self.fitted_: bool = False

    # ----------------------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        wave: int,
    ) -> "FriendSupportStochasticImputer":

        if wave not in VALID_WAVES:
            raise ValueError(
                f"wave must be one of {VALID_WAVES}; got {wave}"
            )

        target_col = FSUPPORT6(wave)

        component_cols = {
            component: builder(wave)
            for component, builder in COMPONENT_BUILDERS.items()
        }

        required = [
            target_col,
            *component_cols.values(),
        ]

        for column in required:
            if column not in train_df.columns:
                raise KeyError(
                    f"Training dataframe is missing required column '{column}'."
                )

        self.wave = int(wave)
        self.target_col = target_col
        self.component_cols = component_cols

        friend_support = train_df[
            target_col
        ].map(
            _valid_friend_support
        )

        observed = friend_support.dropna().astype(float)

        if observed.empty:
            raise ValueError(
                f"No observed Friend Support values found in W{wave} training data."
            )

        self.marginal_ = FloatDonorPool(
            values=observed.to_numpy(
                dtype=float
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

        assert self.target_col is not None
        assert self.marginal_ is not None

        required = [
            self.target_col,
            *self.component_cols.values(),
        ]

        for column in required:
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

        target_before = out[
            self.target_col
        ].copy(deep=True)

        valid_before = target_before.map(
            _valid_friend_support
        )

        if not pd.api.types.is_numeric_dtype(
            out[
                self.target_col
            ]
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

            original = _valid_friend_support(
                out.loc[
                    idx,
                    self.target_col,
                ]
            )

            # Observed official score is sacred.
            if original is not None:
                continue

            reason_class, reason_pattern = _reason_class(
                out.loc[idx],
                self.component_cols,
            )

            draw = float(
                self.marginal_.sample(
                    rng
                )
            )

            out.loc[
                idx,
                self.target_col,
            ] = draw

            audit_rows.append(
                {
                    "row_index": idx,
                    "dataset_label": dataset_label,
                    "wave": self.wave,
                    "seed": int(seed),
                    "pathway": "P(FriendSupport)",
                    "reason_class": reason_class,
                    "component_reason_pattern": reason_pattern,
                    "is_no_friends": (
                        reason_class
                        == "NO_FRIENDS_ALL6_N"
                    ),
                    "donor_pool_n": self.marginal_.n,
                    "imputed_friend_support": draw,
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
                "reason_class",
                "component_reason_pattern",
                "is_no_friends",
                "donor_pool_n",
                "imputed_friend_support",
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

        return pd.DataFrame(
            [
                {
                    "distribution": "P(FriendSupport)",
                    "condition": "<ALL>",
                    "n_donors": self.marginal_.n,
                }
            ]
        )

    # ----------------------------------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------------------------------

    def _check_fitted(
        self,
    ) -> None:

        if (
            not self.fitted_
            or self.target_col is None
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
            _valid_friend_support
        )

        if bool(after.isna().any()):

            n_bad = int(
                after.isna().sum()
            )

            raise RuntimeError(
                f"Friend Support imputation validation failed: "
                f"{n_bad} missing/invalid values remain."
            )

        originally_observed = valid_before.notna()

        before_values = valid_before[
            originally_observed
        ].astype(float)

        after_values = after[
            originally_observed
        ].astype(float)

        if not np.allclose(
            before_values.to_numpy(),
            after_values.to_numpy(),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise RuntimeError(
                "Friend Support imputation validation failed: "
                "at least one originally observed score changed."
            )


# ======================================================================================
# REPRODUCIBLE SEED HELPER
# ======================================================================================

def make_friend_support_seed(
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
        "train": 2203,
        "test": 2309,
        "full": 2411,
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
28_friend_support_stochastic_imputer.py is a reusable module.

HARD RULE
---------
Dementia is never used by this imputer.

K-FOLD
------

imputer = FriendSupportStochasticImputer()

imputer.fit(
    train_df,
    wave=2,
)

train_seed = make_friend_support_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="train",
)

test_seed = make_friend_support_seed(
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


FINAL FULL-WAVE DISCOVERY
-------------------------

imputer = FriendSupportStochasticImputer()

imputer.fit(
    full_wave_df,
    wave=2,
)

for m in range(1, 21):

    seed = make_friend_support_seed(
        base_seed=2026,
        wave=2,
        fold=0,
        completion=m,
        split="full",
    )

    completed, audit = imputer.transform(
        full_wave_df,
        seed=seed,
        return_audit=True,
    )

    # Keep audit['is_no_friends'] for the later sensitivity analysis.
    # Learn BN from completed data.
"""
    )

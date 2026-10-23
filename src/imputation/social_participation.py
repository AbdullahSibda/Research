from __future__ import annotations

"""
32_social_participation_stochastic_imputer.py

Reusable stochastic imputer for Harmonized ELSA Social Participation
(`rWsocyr`, binary 0/1).

FINAL AGREED RULE
=================

Script 30 established that Social Participation missingness is substantial and
comes from several mechanisms:
    - .c = no self-completion
    - .r = refusal
    - .m = missing
    - .i = invalid answer (especially W8)

Observed values are clean binary:
    0 = No
    1 = Yes

Script 31 showed:

    Education alone improves held-out Brier score versus marginal filling by:

        W2  5.39%
        W4  4.62%
        W6  5.17%
        W8  5.72%

    and beats the marginal rule in all 20/20 wave x fold comparisons.

AgeBand + Education is numerically best overall (~6.01% pooled Brier
improvement versus ~5.19% for Education alone), but the additional gain from
adding AgeBand is small (~0.82 percentage points of Brier improvement overall).

Because conditional imputation can manufacture BN relationships, the simpler
single-conditioner rule is preferred:

    SocialParticipation_missing
        ~ P(SocialParticipation | Education)

Fallback:
    if Education is unavailable or its training donor cell contains fewer than
    MIN_DONORS rows:

        SocialParticipation_missing
            ~ P(SocialParticipation)

HARD METHODOLOGICAL RULE
========================
DEMENTIA IS NEVER USED TO IMPUTE ANY PREDICTOR.

This module never reads or uses dementia.

It also does NOT use:
    - Age
    - CES-D
    - CASP-19
    - Loneliness
    - Friend Support
    - Recall
    - Orientation
    - Wealth
    - Hearing
    - Eyesight
    - Lower Mobility
    - Managing Meds
    - Proxy
    - SelfCompletion
    - Nurse

as Social Participation conditioners.

PIPELINE ORDER
==============
Education should already have been completed before Social Participation.

CV:
    - split original incomplete data first
    - fit this imputer on TRAIN only
    - apply TRAIN-derived probabilities to TRAIN and TEST

Final full-wave discovery:
    - fit on the full wave
    - generate repeated stochastic completions with different seeds

IMPORTANT
=========
- Observed Social Participation is NEVER changed.
- Missing values are stochastically sampled as Bernoulli draws from empirical
  TRAINING probabilities.
- Original missing-reason class is retained in the audit output.
- This module does not save files by itself.
"""

from typing import Any, Iterable
import math
import re

import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue


# ======================================================================================
# SETTINGS
# ======================================================================================

VALID_WAVES = (2, 4, 6, 8)

MIN_DONORS = 30

EDUCATION_COL = "raeduc_e"

VALID_EDUCATION = (-15, 1, 2, 3, 4, 5, 6)

SOCIAL_PARTICIPATION = lambda wave: f"r{wave}socyr"


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


def _valid_binary(
    value: Any,
) -> int | None:

    x = _to_number(value)

    if x is None:
        return None

    if not float(x).is_integer():
        return None

    x_int = int(x)

    if x_int not in (0, 1):
        return None

    return x_int


def _valid_education(
    value: Any,
) -> int | None:

    x = _to_number(value)

    if x is None:
        return None

    if not float(x).is_integer():
        return None

    x_int = int(x)

    if x_int not in VALID_EDUCATION:
        return None

    return x_int


def _raw_missing_reason(
    value: Any,
) -> str:
    """
    Best-effort compact missing-reason classification from the raw target value.
    """
    if isinstance(value, StataMissingValue):

        text = str(value).strip().lower()

        if text.startswith(".") and len(text) >= 2:
            tag = text[:2]
        else:
            tag = "."

    else:

        try:
            if pd.isna(value):
                tag = "."
            else:
                x = _to_number(value)

                if x in (0.0, 1.0):
                    return "OBSERVED"

                return "OTHER_INVALID"
        except Exception:
            return "OTHER_INVALID"

    mapping = {
        ".c": "NO_SELF_COMPLETION",
        ".r": "REFUSAL",
        ".m": "MISSING",
        ".i": "INVALID_ANSWER",
        ".v": "SP_NR",
        ".u": "UNMARRIED",
        ".d": "DONT_KNOW",
        ".": "PLAIN_MISSING",
    }

    return mapping.get(
        tag,
        f"OTHER_SPECIAL_{tag}",
    )


# ======================================================================================
# IMPUTER
# ======================================================================================

class SocialParticipationStochasticImputer:
    """
    Stochastic binary Social Participation imputer.

    FIT learns from TRAINING DATA ONLY:

        P(SocialParticipation = 1)
        P(SocialParticipation = 1 | Education)
    """

    def __init__(
        self,
        min_donors: int = MIN_DONORS,
    ) -> None:

        if min_donors < 1:
            raise ValueError(
                "min_donors must be >= 1."
            )

        self.min_donors = int(min_donors)

        self.wave: int | None = None
        self.target_col: str | None = None

        self.marginal_p_yes_: float | None = None
        self.marginal_n_: int | None = None

        self.by_education_: dict[
            int,
            tuple[float, int],
        ] = {}

        self.fitted_: bool = False

    # ----------------------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        wave: int,
    ) -> "SocialParticipationStochasticImputer":

        if wave not in VALID_WAVES:
            raise ValueError(
                f"wave must be one of {VALID_WAVES}; got {wave}"
            )

        target_col = SOCIAL_PARTICIPATION(wave)

        for column in (
            target_col,
            EDUCATION_COL,
        ):
            if column not in train_df.columns:
                raise KeyError(
                    f"Training dataframe is missing required column '{column}'."
                )

        self.wave = int(wave)
        self.target_col = target_col

        target = train_df[
            target_col
        ].map(
            _valid_binary
        )

        education = train_df[
            EDUCATION_COL
        ].map(
            _valid_education
        )

        observed = target.dropna().astype(int)

        if observed.empty:
            raise ValueError(
                f"No observed Social Participation values found in W{wave} training data."
            )

        self.marginal_p_yes_ = float(
            observed.mean()
        )

        self.marginal_n_ = int(
            len(observed)
        )

        donor_frame = pd.DataFrame(
            {
                "Target": target,
                "Education": education,
            },
            index=train_df.index,
        )

        donor_frame = donor_frame[
            donor_frame["Target"].notna()
            & donor_frame["Education"].notna()
        ].copy()

        self.by_education_ = {}

        for education_value, sub in donor_frame.groupby(
            "Education",
            observed=True,
            dropna=True,
        ):

            target_values = sub[
                "Target"
            ].astype(int)

            self.by_education_[
                int(education_value)
            ] = (
                float(
                    target_values.mean()
                ),
                int(
                    len(target_values)
                ),
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
        assert self.marginal_p_yes_ is not None
        assert self.marginal_n_ is not None

        for column in (
            self.target_col,
            EDUCATION_COL,
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

        target_before = out[
            self.target_col
        ].copy(deep=True)

        valid_before = target_before.map(
            _valid_binary
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

            original = _valid_binary(
                out.loc[
                    idx,
                    self.target_col,
                ]
            )

            # Observed 0/1 target is sacred.
            if original is not None:
                continue

            raw_before = target_before.loc[idx]

            missing_reason = _raw_missing_reason(
                raw_before
            )

            education = _valid_education(
                out.loc[
                    idx,
                    EDUCATION_COL,
                ]
            )

            conditional = (
                self.by_education_.get(
                    education
                )
                if education is not None
                else None
            )

            fallback = (
                conditional is None
                or conditional[1] < self.min_donors
            )

            if fallback:

                p_yes = self.marginal_p_yes_
                donor_n = self.marginal_n_
                fallback_level = "MARGINAL_SOCIAL_PARTICIPATION"

            else:

                p_yes, donor_n = conditional
                fallback_level = "NONE"

            draw = int(
                rng.random()
                < p_yes
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
                    "pathway": "P(SocialParticipation|Education)",
                    "original_missing_reason": missing_reason,
                    "education": education,
                    "p_yes": float(p_yes),
                    "donor_pool_n": int(donor_n),
                    "used_fallback": bool(fallback),
                    "fallback_level": fallback_level,
                    "imputed_social_participation": draw,
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
                "original_missing_reason",
                "education",
                "p_yes",
                "donor_pool_n",
                "used_fallback",
                "fallback_level",
                "imputed_social_participation",
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

        assert self.marginal_p_yes_ is not None
        assert self.marginal_n_ is not None

        rows = [
            {
                "distribution": "P(SocialParticipation)",
                "condition": "<ALL>",
                "p_yes": self.marginal_p_yes_,
                "n_donors": self.marginal_n_,
            }
        ]

        for education, (
            p_yes,
            n_donors,
        ) in sorted(
            self.by_education_.items()
        ):

            rows.append(
                {
                    "distribution": "P(SocialParticipation|Education)",
                    "condition": education,
                    "p_yes": p_yes,
                    "n_donors": n_donors,
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
            or self.target_col is None
            or self.marginal_p_yes_ is None
            or self.marginal_n_ is None
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
            _valid_binary
        )

        if bool(
            after.isna().any()
        ):

            n_bad = int(
                after.isna().sum()
            )

            raise RuntimeError(
                f"Social Participation imputation validation failed: "
                f"{n_bad} missing/invalid target values remain."
            )

        originally_observed = valid_before.notna()

        before_values = valid_before[
            originally_observed
        ].astype(int)

        after_values = after[
            originally_observed
        ].astype(int)

        if not before_values.equals(
            after_values
        ):
            raise RuntimeError(
                "Social Participation imputation validation failed: "
                "at least one originally observed 0/1 value changed."
            )


# ======================================================================================
# REPRODUCIBLE SEED HELPER
# ======================================================================================

def make_social_participation_seed(
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
        "train": 2503,
        "test": 2609,
        "full": 2711,
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
32_social_participation_stochastic_imputer.py is a reusable module.

HARD RULE
---------
Dementia is never used by this imputer.

PIPELINE ORDER
--------------
Education should already have been completed.

K-FOLD
------

imputer = SocialParticipationStochasticImputer()

imputer.fit(
    train_df,
    wave=2,
)

train_seed = make_social_participation_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="train",
)

test_seed = make_social_participation_seed(
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

imputer = SocialParticipationStochasticImputer()

imputer.fit(
    full_wave_df,
    wave=2,
)

for m in range(1, 21):

    seed = make_social_participation_seed(
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

    # Learn BN from completed data.
"""
    )

from __future__ import annotations

"""
37_bmi_stochastic_imputer.py

Reusable final BMI imputer for Harmonized ELSA W2/W4/W6/W8.

FINAL LOCKED BMI HIERARCHY
==========================

BMI is the continuous node:
    r{wave}mbmi

The final missing-BMI rule is:

1) Current height + current weight both observed
   ------------------------------------------------
   Safety reconstruction:
       BMI = current_weight / current_height^2

   Script 33 found no actual missing-BMI cases in this state, but the branch is
   retained as a logically exact safeguard.

2) Current weight observed, current height missing, PRIOR height available
   -----------------------------------------------------------------------
   Deterministic person-specific reconstruction:
       BMI = current_weight / prior_height^2

   Use the MOST RECENT VALID EARLIER measured height only.

   Prior-wave order:
       W2: none
       W4: W2
       W6: W4, then W2
       W8: W6, then W4, then W2

   Script 34 validation:
       W4 MAE ~0.46 BMI, ~91.9% within 1, ~98.6% within 2
       W6 MAE ~0.47 BMI, ~90.2% within 1, ~98.6% within 2

   W8 receives essentially no extra rescue here because Harmonized ELSA already
   carries earlier measured height into r8mheight where available.

3) Current weight observed, no usable current/prior height
   -------------------------------------------------------
   Stochastic empirical donor rule:
       P(BMI | 100 nearest TRAINING current-weight donors)

   Script 35:
       ~41-47% CRPS improvement vs marginal in W2/W4/W6.

   Script 36:
       W8 NearestWeight_100 improves CRPS by ~45.75%, 5/5 fold wins.

4) Current height observed, current weight missing
   ------------------------------------------------
   Marginal stochastic BMI:
       BMI_missing ~ P(BMI)

   Nearest-height donor rules did not improve CRPS reliably, so no Height->BMI
   relationship is injected here.

5) Both current height and weight missing
   ---------------------------------------
   Stochastic empirical rule:
       BMI_missing ~ P(BMI | LowerMobility)

   Marginal fallback if LowerMobility is unavailable or the donor pool is too
   small.

   Script 35:
       Lower Mobility improves CRPS by roughly:
           W2 2.20%
           W4 2.66%
           W6 3.63%
           W8 3.65%
       with 5/5 fold wins in every wave.

NOT USED
========
- Prior weight is NOT carried forward.
- Prior BMI is NOT carried forward.
- Later/future-wave measurements are NEVER used.
- Age, Education, Hearing, Eyesight, psychosocial variables and process nodes
  are NOT BMI conditioners.
- DEMENTIA IS NEVER USED TO IMPUTE BMI OR ANY OTHER PREDICTOR.

WHY PRIOR HEIGHT BUT NOT PRIOR WEIGHT/BMI?
==========================================
Prior height reconstructed current BMI very accurately. Prior weight / prior BMI
were materially less stable and could erase genuine within-person weight or BMI
change, which may itself be relevant to dementia-related dependency discovery.

PIPELINE ORDER
==============
Lower Mobility should already have been completed before BMI.

For CV:
    1. split ORIGINAL incomplete data first
    2. fit this imputer on TRAIN only
    3. apply TRAIN-derived donor distributions to TRAIN and TEST
    4. prior-height reconstruction uses only the person's own earlier observed
       height and current observed weight; it is not learned from TEST data

For final full-wave discovery:
    - fit on the full current wave
    - generate repeated stochastic completions with different seeds

REQUIRED COLUMNS
================
Current wave:
    r{wave}mbmi
    r{wave}mheight
    r{wave}mweight
    r{wave}lowermob

For W4/W6/W8, keep earlier height helper columns available during imputation:
    r2mheight
    r4mheight
    r6mheight
as applicable.

These earlier height columns are helper/source variables only and should NOT be
included as BN nodes.

IMPORTANT
=========
- Originally observed BMI is NEVER overwritten.
- Deterministic reconstructions are identical across multiple completions.
- Stochastic branches are resampled across completions.
- Raw height/weight variables are NOT themselves imputed or overwritten.
- This module does not save files by itself.
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

BMI = lambda wave: f"r{wave}mbmi"
HEIGHT = lambda wave: f"r{wave}mheight"
WEIGHT = lambda wave: f"r{wave}mweight"
LOWER_MOBILITY = lambda wave: f"r{wave}lowermob"

VALID_LOWER_MOBILITY = (0, 1, 2, 3, 4)

BMI_MIN = 5.0
BMI_MAX = 100.0

HEIGHT_MIN = 0.5
HEIGHT_MAX = 2.5

WEIGHT_MIN = 10.0
WEIGHT_MAX = 300.0

WEIGHT_NEIGHBORS = 100
MIN_DONORS = 30


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


def _valid_continuous(
    value: Any,
    *,
    minimum: float,
    maximum: float,
) -> float | None:

    x = _to_number(
        value
    )

    if x is None:
        return None

    if x < minimum or x > maximum:
        return None

    return float(
        x
    )


def _valid_bmi(
    value: Any,
) -> float | None:

    return _valid_continuous(
        value,
        minimum=BMI_MIN,
        maximum=BMI_MAX,
    )


def _valid_height(
    value: Any,
) -> float | None:

    return _valid_continuous(
        value,
        minimum=HEIGHT_MIN,
        maximum=HEIGHT_MAX,
    )


def _valid_weight(
    value: Any,
) -> float | None:

    return _valid_continuous(
        value,
        minimum=WEIGHT_MIN,
        maximum=WEIGHT_MAX,
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


def _prior_waves(
    wave: int,
) -> tuple[int, ...]:

    if wave == 2:
        return ()

    if wave == 4:
        return (2,)

    if wave == 6:
        return (4, 2)

    if wave == 8:
        return (6, 4, 2)

    raise ValueError(
        f"wave must be one of {VALID_WAVES}; got {wave}"
    )


def _safe_bmi_from_weight_height(
    weight: float,
    height: float,
) -> float | None:

    if height <= 0:
        return None

    bmi = float(
        weight
        / (
            height
            ** 2
        )
    )

    if (
        not math.isfinite(
            bmi
        )
        or bmi < BMI_MIN
        or bmi > BMI_MAX
    ):
        return None

    return bmi


def _nearest_weight_indices(
    donor_weights: np.ndarray,
    query_weight: float,
    k: int,
) -> np.ndarray:

    n = len(
        donor_weights
    )

    if n == 0:
        return np.array(
            [],
            dtype=int,
        )

    k_eff = min(
        int(
            k
        ),
        n,
    )

    distances = np.abs(
        donor_weights
        - float(
            query_weight
        )
    )

    if k_eff == n:
        return np.arange(
            n,
            dtype=int,
        )

    return np.argpartition(
        distances,
        k_eff - 1,
    )[
        :k_eff
    ]


# ======================================================================================
# IMPUTER
# ======================================================================================

class BMIStochasticImputer:
    """
    Final hierarchical BMI imputer.

    FIT learns from TRAINING DATA ONLY:
        - P(BMI)
        - P(BMI | LowerMobility)
        - current Weight -> 100-nearest TRAIN BMI donor pool

    Deterministic reconstruction branches use only the person's own observed
    source measurements and do not estimate parameters from the held-out set.
    """

    def __init__(
        self,
        *,
        min_donors: int = MIN_DONORS,
        weight_neighbors: int = WEIGHT_NEIGHBORS,
    ) -> None:

        if min_donors < 1:
            raise ValueError(
                "min_donors must be >= 1."
            )

        if weight_neighbors < 1:
            raise ValueError(
                "weight_neighbors must be >= 1."
            )

        self.min_donors = int(
            min_donors
        )

        self.weight_neighbors = int(
            weight_neighbors
        )

        self.wave: int | None = None

        self.bmi_col: str | None = None
        self.height_col: str | None = None
        self.weight_col: str | None = None
        self.lower_mobility_col: str | None = None

        self.marginal_bmi_: np.ndarray | None = None

        self.by_lower_mobility_: dict[
            int,
            np.ndarray,
        ] = {}

        self.weight_donor_weights_: np.ndarray | None = None
        self.weight_donor_bmi_: np.ndarray | None = None

        self.fitted_: bool = False

    # ----------------------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        wave: int,
    ) -> "BMIStochasticImputer":

        if wave not in VALID_WAVES:
            raise ValueError(
                f"wave must be one of {VALID_WAVES}; got {wave}"
            )

        bmi_col = BMI(
            wave
        )

        height_col = HEIGHT(
            wave
        )

        weight_col = WEIGHT(
            wave
        )

        lower_mobility_col = LOWER_MOBILITY(
            wave
        )

        for column in (
            bmi_col,
            height_col,
            weight_col,
            lower_mobility_col,
        ):
            if column not in train_df.columns:
                raise KeyError(
                    f"Training dataframe is missing required column '{column}'."
                )

        self.wave = int(
            wave
        )

        self.bmi_col = bmi_col
        self.height_col = height_col
        self.weight_col = weight_col
        self.lower_mobility_col = lower_mobility_col

        bmi = train_df[
            bmi_col
        ].map(
            _valid_bmi
        )

        weight = train_df[
            weight_col
        ].map(
            _valid_weight
        )

        lower_mobility = train_df[
            lower_mobility_col
        ].map(
            _valid_lower_mobility
        )

        observed_bmi = bmi.dropna().to_numpy(
            dtype=float
        )

        if len(
            observed_bmi
        ) < self.min_donors:
            raise ValueError(
                f"W{wave}: only {len(observed_bmi)} observed BMI training donors; "
                f"need at least {self.min_donors}."
            )

        self.marginal_bmi_ = observed_bmi.copy()

        # P(BMI | LowerMobility)
        donor_frame = pd.DataFrame(
            {
                "BMI": bmi,
                "LowerMobility": lower_mobility,
            },
            index=train_df.index,
        )

        donor_frame = donor_frame[
            donor_frame[
                "BMI"
            ].notna()
            & donor_frame[
                "LowerMobility"
            ].notna()
        ].copy()

        self.by_lower_mobility_ = {}

        for mobility_value, sub in donor_frame.groupby(
            "LowerMobility",
            observed=True,
            dropna=True,
        ):

            self.by_lower_mobility_[
                int(
                    mobility_value
                )
            ] = sub[
                "BMI"
            ].to_numpy(
                dtype=float
            )

        # Weight-local BMI donor database.
        weight_frame = pd.DataFrame(
            {
                "Weight": weight,
                "BMI": bmi,
            },
            index=train_df.index,
        )

        weight_frame = weight_frame[
            weight_frame[
                "Weight"
            ].notna()
            & weight_frame[
                "BMI"
            ].notna()
        ].copy()

        if len(
            weight_frame
        ) < self.min_donors:

            self.weight_donor_weights_ = np.array(
                [],
                dtype=float,
            )

            self.weight_donor_bmi_ = np.array(
                [],
                dtype=float,
            )

        else:

            self.weight_donor_weights_ = weight_frame[
                "Weight"
            ].to_numpy(
                dtype=float
            )

            self.weight_donor_bmi_ = weight_frame[
                "BMI"
            ].to_numpy(
                dtype=float
            )

        self.fitted_ = True

        return self

    # ----------------------------------------------------------------------------------
    # PRIOR HEIGHT
    # ----------------------------------------------------------------------------------

    def _most_recent_prior_height(
        self,
        df: pd.DataFrame,
        row_index: Any,
    ) -> tuple[
        float | None,
        int | None,
    ]:

        self._check_fitted()

        assert self.wave is not None

        for prior_wave in _prior_waves(
            self.wave
        ):

            column = HEIGHT(
                prior_wave
            )

            # Missing helper columns are allowed: just continue to the next older wave.
            if column not in df.columns:
                continue

            value = _valid_height(
                df.loc[
                    row_index,
                    column,
                ]
            )

            if value is not None:
                return (
                    value,
                    prior_wave,
                )

        return (
            None,
            None,
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

        assert self.wave is not None
        assert self.bmi_col is not None
        assert self.height_col is not None
        assert self.weight_col is not None
        assert self.lower_mobility_col is not None
        assert self.marginal_bmi_ is not None
        assert self.weight_donor_weights_ is not None
        assert self.weight_donor_bmi_ is not None

        for column in (
            self.bmi_col,
            self.height_col,
            self.weight_col,
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

        bmi_before = out[
            self.bmi_col
        ].copy(
            deep=True
        )

        valid_before = bmi_before.map(
            _valid_bmi
        )

        # Object dtype safely accepts numeric fills into Stata-missing columns.
        if not pd.api.types.is_numeric_dtype(
            out[
                self.bmi_col
            ]
        ):
            out[
                self.bmi_col
            ] = out[
                self.bmi_col
            ].astype(
                object
            )

        audit_rows: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for idx in out.index:

            original_bmi = _valid_bmi(
                out.loc[
                    idx,
                    self.bmi_col,
                ]
            )

            # Observed BMI is sacred.
            if original_bmi is not None:
                continue

            current_height = _valid_height(
                out.loc[
                    idx,
                    self.height_col,
                ]
            )

            current_weight = _valid_weight(
                out.loc[
                    idx,
                    self.weight_col,
                ]
            )

            lower_mobility = _valid_lower_mobility(
                out.loc[
                    idx,
                    self.lower_mobility_col,
                ]
            )

            prior_height, prior_height_wave = self._most_recent_prior_height(
                out,
                idx,
            )

            imputed_bmi: float
            pathway: str
            stochastic: bool
            donor_n: int
            used_fallback: bool
            fallback_level: str
            selected_neighbor_k: int | None

            # ------------------------------------------------------------------
            # 1) Current exact reconstruction safety branch
            # ------------------------------------------------------------------

            if (
                current_height is not None
                and current_weight is not None
            ):

                reconstructed = _safe_bmi_from_weight_height(
                    current_weight,
                    current_height,
                )

                if reconstructed is not None:

                    imputed_bmi = reconstructed
                    pathway = "CURRENT_HEIGHT_WEIGHT_RECONSTRUCTION"
                    stochastic = False
                    donor_n = 0
                    used_fallback = False
                    fallback_level = "NONE"
                    selected_neighbor_k = None

                else:

                    imputed_bmi = float(
                        rng.choice(
                            self.marginal_bmi_
                        )
                    )
                    pathway = "CURRENT_RECONSTRUCTION_INVALID_TO_MARGINAL"
                    stochastic = True
                    donor_n = len(
                        self.marginal_bmi_
                    )
                    used_fallback = True
                    fallback_level = "MARGINAL_BMI"
                    selected_neighbor_k = None

            # ------------------------------------------------------------------
            # 2) Current weight + own prior height
            # ------------------------------------------------------------------

            elif (
                current_weight is not None
                and current_height is None
                and prior_height is not None
            ):

                reconstructed = _safe_bmi_from_weight_height(
                    current_weight,
                    prior_height,
                )

                if reconstructed is not None:

                    imputed_bmi = reconstructed
                    pathway = "CURRENT_WEIGHT_PRIOR_HEIGHT_RECONSTRUCTION"
                    stochastic = False
                    donor_n = 0
                    used_fallback = False
                    fallback_level = "NONE"
                    selected_neighbor_k = None

                else:

                    # If a bizarre out-of-range reconstruction occurs, use the
                    # validated current-weight donor branch rather than forcing it.
                    (
                        imputed_bmi,
                        donor_n,
                        used_fallback,
                        fallback_level,
                        selected_neighbor_k,
                    ) = self._draw_from_weight_neighbors(
                        current_weight,
                        rng,
                    )

                    pathway = "PRIOR_HEIGHT_RECONSTRUCTION_INVALID_TO_WEIGHT_DONORS"
                    stochastic = True

            # ------------------------------------------------------------------
            # 3) Current weight, no usable height
            # ------------------------------------------------------------------

            elif (
                current_weight is not None
                and current_height is None
            ):

                (
                    imputed_bmi,
                    donor_n,
                    used_fallback,
                    fallback_level,
                    selected_neighbor_k,
                ) = self._draw_from_weight_neighbors(
                    current_weight,
                    rng,
                )

                pathway = "NEAREST_CURRENT_WEIGHT_DONORS"
                stochastic = True

            # ------------------------------------------------------------------
            # 4) Current height only -> marginal BMI
            # ------------------------------------------------------------------

            elif (
                current_height is not None
                and current_weight is None
            ):

                imputed_bmi = float(
                    rng.choice(
                        self.marginal_bmi_
                    )
                )

                pathway = "MARGINAL_BMI_HEIGHT_ONLY"
                stochastic = True
                donor_n = len(
                    self.marginal_bmi_
                )
                used_fallback = False
                fallback_level = "NONE"
                selected_neighbor_k = None

            # ------------------------------------------------------------------
            # 5) Both current components missing -> LowerMobility-conditioned BMI
            # ------------------------------------------------------------------

            else:

                candidate = (
                    self.by_lower_mobility_.get(
                        lower_mobility
                    )
                    if lower_mobility is not None
                    else None
                )

                if (
                    candidate is not None
                    and len(
                        candidate
                    )
                    >= self.min_donors
                ):

                    donor_pool = candidate
                    used_fallback = False
                    fallback_level = "NONE"

                else:

                    donor_pool = self.marginal_bmi_
                    used_fallback = True
                    fallback_level = "MARGINAL_BMI"

                imputed_bmi = float(
                    rng.choice(
                        donor_pool
                    )
                )

                pathway = "BMI_GIVEN_LOWER_MOBILITY"
                stochastic = True
                donor_n = len(
                    donor_pool
                )
                selected_neighbor_k = None

            out.loc[
                idx,
                self.bmi_col,
            ] = imputed_bmi

            audit_rows.append(
                {
                    "row_index": idx,
                    "dataset_label": dataset_label,
                    "wave": self.wave,
                    "seed": int(
                        seed
                    ),
                    "pathway": pathway,
                    "stochastic": stochastic,
                    "current_height": current_height,
                    "current_weight": current_weight,
                    "prior_height": prior_height,
                    "prior_height_source_wave": prior_height_wave,
                    "lower_mobility": lower_mobility,
                    "selected_neighbor_k": selected_neighbor_k,
                    "donor_pool_n": int(
                        donor_n
                    ),
                    "used_fallback": bool(
                        used_fallback
                    ),
                    "fallback_level": fallback_level,
                    "imputed_bmi": float(
                        imputed_bmi
                    ),
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
                "stochastic",
                "current_height",
                "current_weight",
                "prior_height",
                "prior_height_source_wave",
                "lower_mobility",
                "selected_neighbor_k",
                "donor_pool_n",
                "used_fallback",
                "fallback_level",
                "imputed_bmi",
            ],
        )

        if return_audit:
            return (
                out,
                audit,
            )

        return out

    # ----------------------------------------------------------------------------------
    # WEIGHT-NEIGHBOR DRAW
    # ----------------------------------------------------------------------------------

    def _draw_from_weight_neighbors(
        self,
        current_weight: float,
        rng: np.random.Generator,
    ) -> tuple[
        float,
        int,
        bool,
        str,
        int | None,
    ]:

        assert self.marginal_bmi_ is not None
        assert self.weight_donor_weights_ is not None
        assert self.weight_donor_bmi_ is not None

        if len(
            self.weight_donor_bmi_
        ) < self.min_donors:

            donor_pool = self.marginal_bmi_

            return (
                float(
                    rng.choice(
                        donor_pool
                    )
                ),
                len(
                    donor_pool
                ),
                True,
                "MARGINAL_BMI",
                None,
            )

        indices = _nearest_weight_indices(
            self.weight_donor_weights_,
            current_weight,
            self.weight_neighbors,
        )

        donor_pool = self.weight_donor_bmi_[
            indices
        ]

        if len(
            donor_pool
        ) < self.min_donors:

            donor_pool = self.marginal_bmi_

            return (
                float(
                    rng.choice(
                        donor_pool
                    )
                ),
                len(
                    donor_pool
                ),
                True,
                "MARGINAL_BMI",
                None,
            )

        return (
            float(
                rng.choice(
                    donor_pool
                )
            ),
            len(
                donor_pool
            ),
            False,
            "NONE",
            len(
                donor_pool
            ),
        )

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

        assert self.marginal_bmi_ is not None
        assert self.weight_donor_bmi_ is not None

        rows = [
            {
                "distribution": "P(BMI)",
                "condition": "<ALL>",
                "n_donors": len(
                    self.marginal_bmi_
                ),
                "mean_bmi": float(
                    np.mean(
                        self.marginal_bmi_
                    )
                ),
                "median_bmi": float(
                    np.median(
                        self.marginal_bmi_
                    )
                ),
            },
            {
                "distribution": "Weight-neighbor donor database",
                "condition": f"k={self.weight_neighbors}",
                "n_donors": len(
                    self.weight_donor_bmi_
                ),
                "mean_bmi": (
                    float(
                        np.mean(
                            self.weight_donor_bmi_
                        )
                    )
                    if len(
                        self.weight_donor_bmi_
                    )
                    else np.nan
                ),
                "median_bmi": (
                    float(
                        np.median(
                            self.weight_donor_bmi_
                        )
                    )
                    if len(
                        self.weight_donor_bmi_
                    )
                    else np.nan
                ),
            },
        ]

        for lower_mobility, donors in sorted(
            self.by_lower_mobility_.items()
        ):

            rows.append(
                {
                    "distribution": "P(BMI|LowerMobility)",
                    "condition": int(
                        lower_mobility
                    ),
                    "n_donors": len(
                        donors
                    ),
                    "mean_bmi": float(
                        np.mean(
                            donors
                        )
                    ),
                    "median_bmi": float(
                        np.median(
                            donors
                        )
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
            or self.wave is None
            or self.bmi_col is None
            or self.height_col is None
            or self.weight_col is None
            or self.lower_mobility_col is None
            or self.marginal_bmi_ is None
            or self.weight_donor_weights_ is None
            or self.weight_donor_bmi_ is None
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

        assert self.bmi_col is not None

        after = out[
            self.bmi_col
        ].map(
            _valid_bmi
        )

        if bool(
            after.isna().any()
        ):

            n_bad = int(
                after.isna().sum()
            )

            raise RuntimeError(
                f"BMI imputation validation failed: "
                f"{n_bad} missing/invalid BMI values remain."
            )

        originally_observed = valid_before.notna()

        before_values = valid_before[
            originally_observed
        ].astype(
            float
        )

        after_values = after[
            originally_observed
        ].astype(
            float
        )

        if not np.allclose(
            before_values.to_numpy(),
            after_values.to_numpy(),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                "BMI imputation validation failed: "
                "at least one originally observed BMI value changed."
            )


# ======================================================================================
# REPRODUCIBLE SEED HELPER
# ======================================================================================

def make_bmi_seed(
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
        "train": 3701,
        "test": 3719,
        "full": 3733,
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
        r"""
37_bmi_stochastic_imputer.py is a reusable module.

HARD RULE
---------
Dementia is never used by this imputer.

PIPELINE ORDER
--------------
Lower Mobility should already have been completed.

Keep earlier measured-height helper columns available during BMI imputation:
    W4 needs r2mheight
    W6 can use r4mheight then r2mheight
    W8 can use r6mheight then r4mheight then r2mheight

These helper columns are NOT BN nodes and can be dropped after BMI completion.


K-FOLD EXAMPLE
--------------

imputer = BMIStochasticImputer()

imputer.fit(
    train_df,
    wave=6,
)

train_seed = make_bmi_seed(
    base_seed=2026,
    wave=6,
    fold=1,
    completion=1,
    split="train",
)

test_seed = make_bmi_seed(
    base_seed=2026,
    wave=6,
    fold=1,
    completion=1,
    split="test",
)

train_complete, train_audit = imputer.transform(
    train_df,
    seed=train_seed,
    return_audit=True,
    dataset_label="W6_F1_train",
)

test_complete, test_audit = imputer.transform(
    test_df,
    seed=test_seed,
    return_audit=True,
    dataset_label="W6_F1_test",
)


FINAL FULL-WAVE DISCOVERY
-------------------------

imputer = BMIStochasticImputer()

imputer.fit(
    full_wave_df,
    wave=6,
)

for m in range(1, 21):

    seed = make_bmi_seed(
        base_seed=2026,
        wave=6,
        fold=0,
        completion=m,
        split="full",
    )

    completed, audit = imputer.transform(
        full_wave_df,
        seed=seed,
        return_audit=True,
        dataset_label=f"W6_full_M{m}",
    )

    # Learn BN from completed data.
"""
    )

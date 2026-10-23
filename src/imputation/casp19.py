from __future__ import annotations

"""
22_casp19_stochastic_imputer.py

Reusable stochastic imputer for Harmonized ELSA CASP-19.

FINAL AGREED RULE
=================

CASP-19 is the sum of 19 scored item contributions, each 0..3.
Six negatively worded items are reverse-scored:

    AgePrevents
    OutOfControl
    LeftOut
    FamilyPrevents
    HealthStops
    MoneyShortage

For those items:
    scored = 3 - raw

For all others:
    scored = raw

CASP-19 range:
    0..57


IMPUTATION HIERARCHY
====================

1) CASP-19 observed
       -> keep unchanged.

2) CASP-19 missing, all 19 raw items observed
       -> reconstruct CASP-19 EXACTLY from the official scoring formula.

3) CASP-19 missing, SOME items observed
       -> compute the subtotal of the observed SCORED CASP items.

       Primary donor rule:
           choose TRAINING donor rows with the same observed-item subtotal.

       Then sample ONE donor row and copy the missing scored item vector jointly.
       Convert sampled scored contributions back to raw 0..3 item values and
       recompute CASP-19 exactly.

       This is effectively:

           P(missing CASP item vector | observed CASP subtotal)

       If the subtotal-matched donor pool has fewer than MIN_DONORS rows:
           fall back to the marginal joint donor distribution of the missing
           item vector from complete TRAINING CASP respondents.

       IMPORTANT:
           We deliberately do NOT add Age, Education, Hearing, Eyesight, or
           Lower Mobility to partial-case conditioning. Held-out CV showed only
           ~1% additional CRPS gain over the CASP-internal observed subtotal,
           which is not enough to justify manufacturing extra BN relationships.

4) CASP-19 missing, ALL 19 items missing
       -> sample the total directly from TRAINING:

           P(CASP19 | LowerMobility)

       If the LowerMobility cell has fewer than MIN_DONORS rows, or the
       LowerMobility value itself is unavailable:
           fall back to marginal TRAINING P(CASP19).

       WHY LOWER MOBILITY?
           Across W2/W4/W6/W8, LowerMobility improved held-out CRPS by about
           9.6% overall versus marginal filling and was the best simple rule
           in W2/W4/W8 and essentially tied with AgeBand+LowerMobility in W6.
           It is therefore the smallest defensible external conditioning rule.


HARD METHODOLOGICAL RULE
========================
DEMENTIA IS NEVER USED TO IMPUTE ANY PREDICTOR.

This module never reads or uses dementia.

It also does NOT use the following as CASP conditioners:
    - Recall
    - Orientation
    - CES-D
    - Loneliness
    - Friend Support
    - Social Participation
    - Wealth
    - Proxy
    - SelfCompletion
    - Nurse
    - CASP-12 or CASP domain indices


PIPELINE ORDER
==============
Lower Mobility should already have been completed before CASP-19.

CV:
    - split original incomplete data first
    - fit this imputer on TRAIN only
    - apply TRAIN-derived rules to TRAIN and TEST

Final full-wave discovery:
    - fit on full wave
    - generate repeated stochastic completions with different seeds


IMPORTANT
=========
- Observed CASP-19 is NEVER changed.
- Partial missing item vectors are sampled JOINTLY from one donor row.
- For all-19-missing cases, only the final CASP-19 total is imputed; the raw
  19 item fields remain missing because no item-level information survives.
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

CASP19 = lambda wave: f"r{wave}casp19"
LOWER_MOBILITY = lambda wave: f"r{wave}lowermob"

COMPONENT_BUILDERS = {
    "AgePrevents": lambda wave: f"r{wave}ageprv",
    "OutOfControl": lambda wave: f"r{wave}outcon",
    "FreePlan": lambda wave: f"r{wave}freepln",
    "LeftOut": lambda wave: f"r{wave}flftout",
    "CanDo": lambda wave: f"r{wave}cando",
    "FamilyPrevents": lambda wave: f"r{wave}famprv",
    "PleaseSelf": lambda wave: f"r{wave}plsslf",
    "HealthStops": lambda wave: f"r{wave}hlthstp",
    "MoneyShortage": lambda wave: f"r{wave}mshtge",
    "LookForward": lambda wave: f"r{wave}lookfwd",
    "LifeMeaning": lambda wave: f"r{wave}lfemean",
    "EnjoyDoing": lambda wave: f"r{wave}enjdo",
    "EnjoyCompany": lambda wave: f"r{wave}enjcmpny",
    "SenseHappy": lambda wave: f"r{wave}snshappy",
    "FullEnergy": lambda wave: f"r{wave}fenerg",
    "ChooseNewThings": lambda wave: f"r{wave}chsnth",
    "LifeTurnedOut": lambda wave: f"r{wave}lftrnout",
    "LifeOpportunities": lambda wave: f"r{wave}lifeopp",
    "FutureLooksGood": lambda wave: f"r{wave}futregd",
}

COMPONENTS = tuple(
    COMPONENT_BUILDERS
)

REVERSED_COMPONENTS = {
    "AgePrevents",
    "OutOfControl",
    "LeftOut",
    "FamilyPrevents",
    "HealthStops",
    "MoneyShortage",
}

VALID_RAW_ITEM = (0, 1, 2, 3)
VALID_CASP19 = tuple(range(58))
VALID_LOWER_MOBILITY = (0, 1, 2, 3, 4)


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


def _valid_integer(
    value: Any,
    *,
    allowed: tuple[int, ...],
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

    if x_int in allowed:
        return x_int

    return None


def _raw_to_scored(
    component: str,
    raw_value: int,
) -> int:

    if component in REVERSED_COMPONENTS:
        return int(
            3
            - raw_value
        )

    return int(
        raw_value
    )


def _scored_to_raw(
    component: str,
    scored_value: int,
) -> int:

    if component in REVERSED_COMPONENTS:
        return int(
            3
            - scored_value
        )

    return int(
        scored_value
    )


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
class RowDonorPool:
    rows: pd.DataFrame

    @property
    def n(self) -> int:
        return int(
            len(
                self.rows
            )
        )

    def sample_row(
        self,
        rng: np.random.Generator,
    ) -> pd.Series:

        position = int(
            rng.integers(
                0,
                len(
                    self.rows
                ),
            )
        )

        return self.rows.iloc[
            position
        ]


# ======================================================================================
# IMPUTER
# ======================================================================================

class CASP19StochasticImputer:
    """
    Stochastic CASP-19 imputer.

    FIT learns from TRAINING DATA ONLY:

        P(CASP19)
        P(CASP19 | LowerMobility)

    plus complete 19-item scored donor rows for partial-item recovery.
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

        self.casp_col: str | None = None
        self.lower_mobility_col: str | None = None

        self.component_cols: dict[
            str,
            str,
        ] = {}

        self.casp_marginal_: ScalarDonorPool | None = None

        self.casp_by_lower_mobility_: dict[
            int,
            ScalarDonorPool,
        ] = {}

        self.complete_scored_rows_: pd.DataFrame | None = None

        self.fitted_: bool = False

    # ----------------------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        wave: int,
    ) -> "CASP19StochasticImputer":

        if wave not in VALID_WAVES:
            raise ValueError(
                f"wave must be one of {VALID_WAVES}; got {wave}"
            )

        casp_col = CASP19(
            wave
        )

        lower_mobility_col = LOWER_MOBILITY(
            wave
        )

        component_cols = {
            component: builder(
                wave
            )
            for component, builder in COMPONENT_BUILDERS.items()
        }

        required = [
            casp_col,
            lower_mobility_col,
            *component_cols.values(),
        ]

        for column in required:

            if column not in train_df.columns:
                raise KeyError(
                    f"Training dataframe is missing required column '{column}'."
                )

        self.wave = int(
            wave
        )

        self.casp_col = casp_col
        self.lower_mobility_col = lower_mobility_col
        self.component_cols = component_cols

        casp = train_df[
            casp_col
        ].map(
            lambda value: _valid_integer(
                value,
                allowed=VALID_CASP19,
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

        raw_components = pd.DataFrame(
            index=train_df.index
        )

        scored_components = pd.DataFrame(
            index=train_df.index
        )

        for component, column in component_cols.items():

            raw = train_df[
                column
            ].map(
                lambda value: _valid_integer(
                    value,
                    allowed=VALID_RAW_ITEM,
                )
            )

            raw_components[
                component
            ] = raw

            scored_components[
                component
            ] = raw.map(
                lambda value: (
                    _raw_to_scored(
                        component,
                        int(
                            value
                        ),
                    )
                    if value is not None
                    and not pd.isna(
                        value
                    )
                    else np.nan
                )
            )

        complete_components = raw_components.notna().all(
            axis=1
        )

        identity_mask = (
            casp.notna()
            & complete_components
        )

        if bool(
            identity_mask.any()
        ):

            recomputed = scored_components.loc[
                identity_mask,
                list(
                    COMPONENTS
                ),
            ].sum(
                axis=1
            ).astype(
                int
            )

            observed = casp.loc[
                identity_mask
            ].astype(
                int
            )

            if not observed.equals(
                recomputed
            ):
                raise RuntimeError(
                    "Observed CASP-19 scoring identity failed in training data."
                )

        # --------------------------------------------------------------------------
        # Total-score donor pools
        # --------------------------------------------------------------------------

        observed_casp = casp.dropna().astype(
            int
        )

        if observed_casp.empty:
            raise ValueError(
                f"No observed CASP-19 values found in W{wave} training data."
            )

        self.casp_marginal_ = ScalarDonorPool(
            values=observed_casp.to_numpy(
                dtype=int
            )
        )

        donor_frame = pd.DataFrame(
            {
                "CASP19": casp,
                "LowerMobility": lower_mobility,
            },
            index=train_df.index,
        )

        donor_frame = donor_frame[
            donor_frame[
                "CASP19"
            ].notna()
            & donor_frame[
                "LowerMobility"
            ].notna()
        ].copy()

        self.casp_by_lower_mobility_ = {}

        for mobility_value, sub in donor_frame.groupby(
            "LowerMobility",
            observed=True,
            dropna=True,
        ):

            values = sub[
                "CASP19"
            ].astype(
                int
            ).to_numpy(
                dtype=int
            )

            self.casp_by_lower_mobility_[
                int(
                    mobility_value
                )
            ] = ScalarDonorPool(
                values=values
            )

        # --------------------------------------------------------------------------
        # Complete scored item donor rows
        # --------------------------------------------------------------------------

        complete_donors = scored_components[
            complete_components
            & casp.notna()
        ].copy()

        if complete_donors.empty:
            raise ValueError(
                f"No complete 19-item CASP donor rows found in W{wave} training data."
            )

        for component in COMPONENTS:
            complete_donors[
                component
            ] = complete_donors[
                component
            ].astype(
                int
            )

        complete_donors[
            "CASP19"
        ] = complete_donors[
            list(
                COMPONENTS
            )
        ].sum(
            axis=1
        ).astype(
            int
        )

        self.complete_scored_rows_ = complete_donors

        self.fitted_ = True

        return self

    # ----------------------------------------------------------------------------------
    # PARTIAL DONOR SELECTION
    # ----------------------------------------------------------------------------------

    def _partial_donor_pool(
        self,
        *,
        observed_components: tuple[str, ...],
        observed_subtotal: int,
    ) -> tuple[
        RowDonorPool,
        bool,
        str,
    ]:

        assert (
            self.complete_scored_rows_
            is not None
        )

        donors = self.complete_scored_rows_

        subtotal = donors[
            list(
                observed_components
            )
        ].sum(
            axis=1
        )

        matched = donors[
            subtotal.eq(
                observed_subtotal
            )
        ]

        if len(
            matched
        ) >= self.min_donors:

            return (
                RowDonorPool(
                    rows=matched
                ),
                False,
                "OBSERVED_SUBTOTAL",
            )

        return (
            RowDonorPool(
                rows=donors
            ),
            True,
            "MARGINAL_COMPLETE_CASP_DONORS",
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

        assert self.casp_col is not None
        assert self.lower_mobility_col is not None
        assert self.casp_marginal_ is not None

        required = [
            self.casp_col,
            self.lower_mobility_col,
            *self.component_cols.values(),
        ]

        for column in required:

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

        root_sequence = np.random.SeedSequence(
            int(
                seed
            )
        )

        exact_sequence, partial_sequence, total_sequence = root_sequence.spawn(
            3
        )

        # exact_sequence is intentionally unused; keeping independent streams
        # means later pathway counts do not alter partial/all-missing random draws.
        _ = exact_sequence

        rng_partial = np.random.default_rng(
            partial_sequence
        )

        rng_total = np.random.default_rng(
            total_sequence
        )

        casp_before = out[
            self.casp_col
        ].copy(
            deep=True
        )

        casp_valid_before = casp_before.map(
            lambda value: _valid_integer(
                value,
                allowed=VALID_CASP19,
            )
        )

        for column in [
            self.casp_col,
            *self.component_cols.values(),
        ]:

            if not pd.api.types.is_numeric_dtype(
                out[
                    column
                ]
            ):

                out[
                    column
                ] = out[
                    column
                ].astype(
                    object
                )

        audit_rows: list[
            dict[str, Any]
        ] = []

        for idx in out.index:

            original_casp = _valid_integer(
                out.loc[
                    idx,
                    self.casp_col,
                ],
                allowed=VALID_CASP19,
            )

            # Observed CASP-19 is sacred.
            if original_casp is not None:
                continue

            raw_values: dict[
                str,
                int | None,
            ] = {}

            scored_values: dict[
                str,
                int | None,
            ] = {}

            for component, column in self.component_cols.items():

                raw = _valid_integer(
                    out.loc[
                        idx,
                        column,
                    ],
                    allowed=VALID_RAW_ITEM,
                )

                raw_values[
                    component
                ] = raw

                scored_values[
                    component
                ] = (
                    _raw_to_scored(
                        component,
                        raw,
                    )
                    if raw is not None
                    else None
                )

            observed_components = tuple(
                component
                for component in COMPONENTS
                if scored_values[
                    component
                ]
                is not None
            )

            missing_components = tuple(
                component
                for component in COMPONENTS
                if scored_values[
                    component
                ]
                is None
            )

            # ----------------------------------------------------------------------
            # Case 1: all 19 components observed -> exact reconstruction
            # ----------------------------------------------------------------------

            if len(
                missing_components
            ) == 0:

                completed_casp = int(
                    sum(
                        int(
                            scored_values[
                                component
                            ]
                        )
                        for component in COMPONENTS
                    )
                )

                out.loc[
                    idx,
                    self.casp_col,
                ] = completed_casp

                audit_rows.append(
                    {
                        "row_index": idx,
                        "dataset_label": dataset_label,
                        "wave": self.wave,
                        "seed": int(seed),
                        "pathway": "exact_19_item_reconstruction",
                        "n_items_missing": 0,
                        "missing_components": "",
                        "observed_subtotal": completed_casp,
                        "conditioning_value": "",
                        "donor_pool_n": np.nan,
                        "used_fallback": False,
                        "fallback_level": "NONE",
                        "imputed_item_values": "",
                        "imputed_casp19": completed_casp,
                    }
                )

                continue

            # ----------------------------------------------------------------------
            # Case 2: partial item missingness
            # ----------------------------------------------------------------------

            if 0 < len(
                missing_components
            ) < 19:

                observed_subtotal = int(
                    sum(
                        int(
                            scored_values[
                                component
                            ]
                        )
                        for component in observed_components
                    )
                )

                pool, fallback, fallback_level = self._partial_donor_pool(
                    observed_components=observed_components,
                    observed_subtotal=observed_subtotal,
                )

                donor = pool.sample_row(
                    rng_partial
                )

                imputed_pairs = []

                for component in missing_components:

                    sampled_scored = int(
                        donor[
                            component
                        ]
                    )

                    sampled_raw = _scored_to_raw(
                        component,
                        sampled_scored,
                    )

                    column = self.component_cols[
                        component
                    ]

                    out.loc[
                        idx,
                        column,
                    ] = sampled_raw

                    raw_values[
                        component
                    ] = sampled_raw

                    scored_values[
                        component
                    ] = sampled_scored

                    imputed_pairs.append(
                        f"{component}={sampled_raw}"
                    )

                completed_casp = int(
                    sum(
                        int(
                            scored_values[
                                component
                            ]
                        )
                        for component in COMPONENTS
                    )
                )

                out.loc[
                    idx,
                    self.casp_col,
                ] = completed_casp

                audit_rows.append(
                    {
                        "row_index": idx,
                        "dataset_label": dataset_label,
                        "wave": self.wave,
                        "seed": int(seed),
                        "pathway": "P(missing_items|observed_CASP_subtotal)",
                        "n_items_missing": len(
                            missing_components
                        ),
                        "missing_components": "|".join(
                            missing_components
                        ),
                        "observed_subtotal": observed_subtotal,
                        "conditioning_value": f"subtotal={observed_subtotal}",
                        "donor_pool_n": pool.n,
                        "used_fallback": fallback,
                        "fallback_level": fallback_level,
                        "imputed_item_values": "|".join(
                            imputed_pairs
                        ),
                        "imputed_casp19": completed_casp,
                    }
                )

                continue

            # ----------------------------------------------------------------------
            # Case 3: all 19 items missing -> CASP19 | LowerMobility
            # ----------------------------------------------------------------------

            lower_mobility = _valid_integer(
                out.loc[
                    idx,
                    self.lower_mobility_col,
                ],
                allowed=VALID_LOWER_MOBILITY,
            )

            conditional_pool = (
                self.casp_by_lower_mobility_.get(
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
                self.casp_marginal_
                if fallback
                else conditional_pool
            )

            draw = int(
                pool.sample(
                    rng_total
                )
            )

            out.loc[
                idx,
                self.casp_col,
            ] = draw

            audit_rows.append(
                {
                    "row_index": idx,
                    "dataset_label": dataset_label,
                    "wave": self.wave,
                    "seed": int(seed),
                    "pathway": "P(CASP19|LowerMobility)",
                    "n_items_missing": 19,
                    "missing_components": "|".join(
                        COMPONENTS
                    ),
                    "observed_subtotal": np.nan,
                    "conditioning_value": (
                        f"LowerMobility={lower_mobility}"
                        if lower_mobility is not None
                        else "<MISSING_CONDITION>"
                    ),
                    "donor_pool_n": pool.n,
                    "used_fallback": fallback,
                    "fallback_level": (
                        "MARGINAL_CASP19"
                        if fallback
                        else "NONE"
                    ),
                    "imputed_item_values": "",
                    "imputed_casp19": draw,
                }
            )

        self._validate(
            out,
            casp_valid_before=casp_valid_before,
        )

        audit = pd.DataFrame(
            audit_rows,
            columns=[
                "row_index",
                "dataset_label",
                "wave",
                "seed",
                "pathway",
                "n_items_missing",
                "missing_components",
                "observed_subtotal",
                "conditioning_value",
                "donor_pool_n",
                "used_fallback",
                "fallback_level",
                "imputed_item_values",
                "imputed_casp19",
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

        assert self.casp_marginal_ is not None
        assert self.complete_scored_rows_ is not None

        rows = [
            {
                "distribution": "P(CASP19)",
                "condition": "<ALL>",
                "n_donors": self.casp_marginal_.n,
            },
            {
                "distribution": "Complete_19_Item_Donor_Rows",
                "condition": "<ALL>",
                "n_donors": len(
                    self.complete_scored_rows_
                ),
            },
        ]

        for mobility_value, pool in sorted(
            self.casp_by_lower_mobility_.items()
        ):

            rows.append(
                {
                    "distribution": "P(CASP19|LowerMobility)",
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
            or self.casp_col is None
            or self.lower_mobility_col is None
            or self.casp_marginal_ is None
            or self.complete_scored_rows_ is None
        ):

            raise RuntimeError(
                "Imputer has not been fitted. Call fit(train_df, wave) first."
            )

    def _validate(
        self,
        out: pd.DataFrame,
        *,
        casp_valid_before: pd.Series,
    ) -> None:

        assert self.casp_col is not None

        casp_after = out[
            self.casp_col
        ].map(
            lambda value: _valid_integer(
                value,
                allowed=VALID_CASP19,
            )
        )

        if bool(
            casp_after.isna().any()
        ):

            n_bad = int(
                casp_after.isna().sum()
            )

            raise RuntimeError(
                f"CASP-19 imputation validation failed: "
                f"{n_bad} missing/invalid CASP-19 values remain."
            )

        # Observed CASP-19 must never change.
        originally_observed = casp_valid_before.notna()

        before = casp_valid_before[
            originally_observed
        ].astype(int)

        after = casp_after[
            originally_observed
        ].astype(int)

        if not before.equals(
            after
        ):

            raise RuntimeError(
                "CASP-19 imputation validation failed: "
                "at least one originally observed CASP-19 value changed."
            )

        # For newly imputed cases where all 19 raw items are now present,
        # CASP-19 must equal the exact official item sum.
        scored_after = pd.DataFrame(
            index=out.index
        )

        for component, column in self.component_cols.items():

            raw = out[
                column
            ].map(
                lambda value: _valid_integer(
                    value,
                    allowed=VALID_RAW_ITEM,
                )
            )

            scored_after[
                component
            ] = raw.map(
                lambda value: (
                    _raw_to_scored(
                        component,
                        int(value),
                    )
                    if value is not None
                    and not pd.isna(value)
                    else np.nan
                )
            )

        all_items_present = scored_after.notna().all(
            axis=1
        )

        newly_imputed = casp_valid_before.isna()

        check = (
            all_items_present
            & newly_imputed
        )

        if bool(
            check.any()
        ):

            recomputed = scored_after.loc[
                check,
                list(COMPONENTS),
            ].sum(
                axis=1
            ).astype(int)

            completed = casp_after.loc[
                check
            ].astype(int)

            if not recomputed.equals(
                completed
            ):

                raise RuntimeError(
                    "CASP-19 imputation validation failed: "
                    "a component-derived CASP-19 value does not equal the exact "
                    "official 19-item score."
                )


# ======================================================================================
# REPRODUCIBLE SEED HELPER
# ======================================================================================

def make_casp19_seed(
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
        "train": 1601,
        "test": 1709,
        "full": 1801,
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
22_casp19_stochastic_imputer.py is a reusable module.

HARD RULE
---------
Dementia is never used by this imputer.

PIPELINE ORDER
--------------
Lower Mobility should already be completed.

K-FOLD
------

casp_imputer = CASP19StochasticImputer()

# FIT ON TRAIN ONLY
casp_imputer.fit(
    train_df,
    wave=2,
)

train_seed = make_casp19_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="train",
)

test_seed = make_casp19_seed(
    base_seed=2026,
    wave=2,
    fold=1,
    completion=1,
    split="test",
)

train_complete, train_audit = casp_imputer.transform(
    train_df,
    seed=train_seed,
    return_audit=True,
    dataset_label="W2_F1_train",
)

test_complete, test_audit = casp_imputer.transform(
    test_df,
    seed=test_seed,
    return_audit=True,
    dataset_label="W2_F1_test",
)


FINAL FULL-WAVE DISCOVERY
-------------------------

casp_imputer = CASP19StochasticImputer()

casp_imputer.fit(
    full_wave_df,
    wave=2,
)

for m in range(1, 21):

    seed = make_casp19_seed(
        base_seed=2026,
        wave=2,
        fold=0,
        completion=m,
        split="full",
    )

    completed = casp_imputer.transform(
        full_wave_df,
        seed=seed,
    )

    # Learn BN from completed data.
"""
    )

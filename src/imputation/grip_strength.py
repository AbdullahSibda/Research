from __future__ import annotations

"""
40_grip_strength_stochastic_imputer.py

Final reusable imputer for Harmonized ELSA dominant-hand Grip Strength.

LOCKED RULE
-----------
Target: r{wave}gripsum (continuous kg)

1) If missing Grip but documented dominant-hand maximum is observed:
       exact reconstruction from that hand maximum.

2) If dominant hand is unavailable but CURRENT opposite hand is observed:
       stochastic draw from the 100 nearest TRAINING opposite-hand donors.

   Script 39: ~65% CRPS improvement over marginal in every wave, 5/5 fold wins.

3) If both hand maxima are unavailable:
       stochastic draw from P(Grip | AgeBand, Sex)

   Script 39 CRPS improvement:
       W2 40.06%, W4 38.32%, W6 36.35%, W8 37.40%
       all 5/5 fold wins.

   Adding Lower Mobility gives little/no extra benefit, so it is excluded.

Sex (`ragender`) is an auxiliary imputation variable only; it is NOT a BN node.

NOT USED
--------
- Prior grip: excluded because it is time-varying, systematically declines, and
  could import longitudinal signal into a within-wave network.
- Future grip.
- Nurse / Proxy / SelfCompletion.
- Lower Mobility in the final Grip rule.
- Dementia, ever.

CV RULE
-------
Split original incomplete data first. Fit on TRAIN only. Apply TRAIN-derived
donor distributions to TRAIN and TEST.

Observed Grip values are never overwritten.
"""

from typing import Any, Iterable
import math
import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue


VALID_WAVES = (2, 4, 6, 8)
GRIP_SUM = lambda w: f"r{w}gripsum"
DOM_HAND = lambda w: f"r{w}domhand"
LEFT_MAX = lambda w: f"r{w}lgrip"
RIGHT_MAX = lambda w: f"r{w}rgrip"
AGE = lambda w: f"r{w}agey"

GENDER_COL = "ragender"

GRIP_MIN = 0.0
GRIP_MAX = 100.0
VALID_DOM_HAND = (1, 2)
VALID_GENDER = (1, 2)

MIN_DONORS = 30
OPPOSITE_HAND_NEIGHBORS = 100


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


def _valid_grip(value: Any) -> float | None:
    x = _to_number(value)
    if x is None or x < GRIP_MIN or x > GRIP_MAX:
        return None
    return float(x)


def _valid_category(value: Any, allowed: tuple[int, ...]) -> int | None:
    x = _to_number(value)
    if x is None or not float(x).is_integer():
        return None
    x = int(x)
    return x if x in allowed else None


def _valid_age(value: Any) -> float | None:
    x = _to_number(value)
    if x is None or x < 50 or x > 120:
        return None
    return float(x)


def _age_band(age: float | None) -> str | None:
    if age is None:
        return None
    if age < 60:
        return "50-59"
    if age < 70:
        return "60-69"
    if age < 80:
        return "70-79"
    return "80+"


def _nearest_indices(values: np.ndarray, query: float, k: int) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.array([], dtype=int)
    k_eff = min(int(k), n)
    distances = np.abs(values - float(query))
    if k_eff == n:
        return np.arange(n, dtype=int)
    return np.argpartition(distances, k_eff - 1)[:k_eff]


class GripStrengthStochasticImputer:
    """Final hierarchical Grip Strength imputer."""

    def __init__(
        self,
        *,
        min_donors: int = MIN_DONORS,
        opposite_hand_neighbors: int = OPPOSITE_HAND_NEIGHBORS,
    ) -> None:
        if min_donors < 1:
            raise ValueError("min_donors must be >= 1")
        if opposite_hand_neighbors < 1:
            raise ValueError("opposite_hand_neighbors must be >= 1")

        self.min_donors = int(min_donors)
        self.opposite_hand_neighbors = int(opposite_hand_neighbors)

        self.wave: int | None = None
        self.target_col: str | None = None
        self.dom_col: str | None = None
        self.left_col: str | None = None
        self.right_col: str | None = None
        self.age_col: str | None = None

        self.marginal_grip_: np.ndarray | None = None
        self.by_age_sex_: dict[tuple[str, int], np.ndarray] = {}
        self.opp_values_: np.ndarray | None = None
        self.opp_grip_: np.ndarray | None = None
        self.fitted_ = False

    def fit(
        self,
        train_df: pd.DataFrame,
        wave: int,
    ) -> "GripStrengthStochasticImputer":

        if wave not in VALID_WAVES:
            raise ValueError(f"wave must be one of {VALID_WAVES}")

        self.wave = int(wave)
        self.target_col = GRIP_SUM(wave)
        self.dom_col = DOM_HAND(wave)
        self.left_col = LEFT_MAX(wave)
        self.right_col = RIGHT_MAX(wave)
        self.age_col = AGE(wave)

        required = (
            self.target_col,
            self.dom_col,
            self.left_col,
            self.right_col,
            self.age_col,
            GENDER_COL,
        )
        for col in required:
            if col not in train_df.columns:
                raise KeyError(f"Missing required training column: {col}")

        grip = train_df[self.target_col].map(_valid_grip)
        dom = train_df[self.dom_col].map(
            lambda v: _valid_category(v, VALID_DOM_HAND)
        )
        left = train_df[self.left_col].map(_valid_grip)
        right = train_df[self.right_col].map(_valid_grip)
        age = train_df[self.age_col].map(_valid_age)
        age_band = age.map(_age_band)
        sex = train_df[GENDER_COL].map(
            lambda v: _valid_category(v, VALID_GENDER)
        )

        observed = grip.dropna().to_numpy(dtype=float)
        if len(observed) < self.min_donors:
            raise ValueError(
                f"W{wave}: only {len(observed)} observed Grip donors."
            )
        self.marginal_grip_ = observed.copy()

        # P(Grip | AgeBand, Sex)
        cond = pd.DataFrame(
            {"Grip": grip, "AgeBand": age_band, "Sex": sex},
            index=train_df.index,
        ).dropna()

        self.by_age_sex_ = {}
        for (ab, sx), sub in cond.groupby(
            ["AgeBand", "Sex"],
            observed=True,
        ):
            self.by_age_sex_[(str(ab), int(sx))] = (
                sub["Grip"].to_numpy(dtype=float)
            )

        # Opposite-hand donor database.
        opposite = pd.Series(np.nan, index=train_df.index, dtype=float)
        opposite.loc[dom.eq(1)] = left.loc[dom.eq(1)]
        opposite.loc[dom.eq(2)] = right.loc[dom.eq(2)]

        opp = pd.DataFrame(
            {"Opposite": opposite, "Grip": grip},
            index=train_df.index,
        ).dropna()

        self.opp_values_ = opp["Opposite"].to_numpy(dtype=float)
        self.opp_grip_ = opp["Grip"].to_numpy(dtype=float)

        self.fitted_ = True
        return self

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

        assert self.wave is not None
        assert self.target_col is not None
        assert self.dom_col is not None
        assert self.left_col is not None
        assert self.right_col is not None
        assert self.age_col is not None
        assert self.marginal_grip_ is not None
        assert self.opp_values_ is not None
        assert self.opp_grip_ is not None

        required = (
            self.target_col,
            self.dom_col,
            self.left_col,
            self.right_col,
            self.age_col,
            GENDER_COL,
        )
        for col in required:
            if col not in df.columns:
                raise KeyError(f"Missing transform column: {col}")

        out = df.copy(deep=True) if copy else df
        rng = np.random.default_rng(int(seed))

        before = out[self.target_col].copy(deep=True)
        valid_before = before.map(_valid_grip)

        if not pd.api.types.is_numeric_dtype(out[self.target_col]):
            out[self.target_col] = out[self.target_col].astype(object)

        audit_rows: list[dict[str, Any]] = []

        for idx in out.index:
            if _valid_grip(out.loc[idx, self.target_col]) is not None:
                continue

            dom = _valid_category(
                out.loc[idx, self.dom_col],
                VALID_DOM_HAND,
            )
            left = _valid_grip(out.loc[idx, self.left_col])
            right = _valid_grip(out.loc[idx, self.right_col])
            age = _valid_age(out.loc[idx, self.age_col])
            ab = _age_band(age)
            sex = _valid_category(
                out.loc[idx, GENDER_COL],
                VALID_GENDER,
            )

            dominant = None
            opposite = None
            if dom == 1:
                dominant, opposite = right, left
            elif dom == 2:
                dominant, opposite = left, right

            if dominant is not None:
                imputed = float(dominant)
                pathway = "EXACT_DOMINANT_HAND_MAX_RECONSTRUCTION"
                stochastic = False
                donor_n = 0
                fallback = False
                fallback_level = "NONE"
                k_used = None

            elif opposite is not None:
                (
                    imputed,
                    donor_n,
                    fallback,
                    fallback_level,
                    k_used,
                ) = self._draw_opposite(opposite, rng)

                pathway = "NEAREST_OPPOSITE_HAND_DONORS"
                stochastic = True

            else:
                candidate = (
                    self.by_age_sex_.get((ab, sex))
                    if ab is not None and sex is not None
                    else None
                )

                if (
                    candidate is not None
                    and len(candidate) >= self.min_donors
                ):
                    pool = candidate
                    fallback = False
                    fallback_level = "NONE"
                else:
                    pool = self.marginal_grip_
                    fallback = True
                    fallback_level = "MARGINAL_GRIP"

                imputed = float(rng.choice(pool))
                donor_n = len(pool)
                pathway = "GRIP_GIVEN_AGEBAND_SEX"
                stochastic = True
                k_used = None

            out.loc[idx, self.target_col] = imputed

            audit_rows.append(
                {
                    "row_index": idx,
                    "dataset_label": dataset_label,
                    "wave": self.wave,
                    "seed": int(seed),
                    "pathway": pathway,
                    "stochastic": stochastic,
                    "dominant_hand": dom,
                    "left_max": left,
                    "right_max": right,
                    "opposite_hand_value": opposite,
                    "age": age,
                    "age_band": ab,
                    "sex": sex,
                    "selected_neighbor_k": k_used,
                    "donor_pool_n": int(donor_n),
                    "used_fallback": bool(fallback),
                    "fallback_level": fallback_level,
                    "imputed_grip": float(imputed),
                }
            )

        self._validate(out, valid_before)

        audit = pd.DataFrame(audit_rows)

        if return_audit:
            return out, audit
        return out

    def _draw_opposite(
        self,
        opposite: float,
        rng: np.random.Generator,
    ) -> tuple[float, int, bool, str, int | None]:

        assert self.marginal_grip_ is not None
        assert self.opp_values_ is not None
        assert self.opp_grip_ is not None

        if len(self.opp_grip_) < self.min_donors:
            pool = self.marginal_grip_
            return (
                float(rng.choice(pool)),
                len(pool),
                True,
                "MARGINAL_GRIP",
                None,
            )

        idx = _nearest_indices(
            self.opp_values_,
            opposite,
            self.opposite_hand_neighbors,
        )
        pool = self.opp_grip_[idx]

        if len(pool) < self.min_donors:
            pool = self.marginal_grip_
            return (
                float(rng.choice(pool)),
                len(pool),
                True,
                "MARGINAL_GRIP",
                None,
            )

        return (
            float(rng.choice(pool)),
            len(pool),
            False,
            "NONE",
            len(pool),
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
                df,
                seed=int(seed),
                copy=copy,
                return_audit=True,
                dataset_label=dataset_label,
            )
            yield int(seed), completed, audit

    def donor_summary(self) -> pd.DataFrame:
        self._check_fitted()

        assert self.marginal_grip_ is not None
        assert self.opp_grip_ is not None

        rows = [
            {
                "distribution": "P(Grip)",
                "condition": "<ALL>",
                "n_donors": len(self.marginal_grip_),
            },
            {
                "distribution": "Opposite-hand donor database",
                "condition": f"k={self.opposite_hand_neighbors}",
                "n_donors": len(self.opp_grip_),
            },
        ]

        for key, donors in sorted(self.by_age_sex_.items()):
            rows.append(
                {
                    "distribution": "P(Grip|AgeBand,Sex)",
                    "condition": f"{key[0]}|sex={key[1]}",
                    "n_donors": len(donors),
                }
            )

        return pd.DataFrame(rows)

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(
                "Call fit(train_df, wave) before transform()."
            )

    def _validate(
        self,
        out: pd.DataFrame,
        valid_before: pd.Series,
    ) -> None:
        assert self.target_col is not None

        after = out[self.target_col].map(_valid_grip)

        if after.isna().any():
            raise RuntimeError(
                f"{int(after.isna().sum())} missing/invalid Grip values remain."
            )

        observed = valid_before.notna()

        if not np.allclose(
            valid_before.loc[observed].to_numpy(dtype=float),
            after.loc[observed].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                "At least one originally observed Grip value changed."
            )


def make_grip_seed(
    *,
    base_seed: int,
    wave: int,
    fold: int,
    completion: int,
    split: str,
) -> int:

    split = split.lower().strip()

    if split not in {"train", "test", "full"}:
        raise ValueError(
            "split must be 'train', 'test', or 'full'."
        )

    split_code = {
        "train": 4001,
        "test": 4013,
        "full": 4021,
    }[split]

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


if __name__ == "__main__":
    print(
        """
40_grip_strength_stochastic_imputer.py

Final rule:
  1) exact dominant-hand max if available;
  2) otherwise 100-nearest current opposite-hand TRAIN donors;
  3) otherwise P(Grip | AgeBand, Sex), marginal fallback.

Prior grip is never used.
Dementia is never used.
ragender is auxiliary only and is not a BN node.
"""
    )

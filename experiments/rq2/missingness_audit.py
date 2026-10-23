from __future__ import annotations

from pathlib import Path
import argparse
import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue


# ======================================================================================
# CONFIG
# ======================================================================================

WAVES = (2, 4, 6, 8)
AGE_MIN = 50

# The script lives at <repo>/experiments/rq2/missingness_audit.py.
# Resolve data and outputs relative to the repository, not the shell's cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DTA_CANDIDATES = (
    PROJECT_ROOT / "Datasets" / "gh_elsa_h.dta",
    PROJECT_ROOT / "gh_elsa_h.dta",
)

# Override locations from the CLI when working with restricted ELSA data elsewhere.
DTA_OVERRIDE: Path | None = None
NURSE_DIR_OVERRIDE: Path | None = None

# Optional raw combined W8/W9 nurse file. The script searches automatically.
# This is only needed to distinguish whether harmonized r8gripsum was actually
# measured during the Wave 8 nurse fieldwork or Wave 9 nurse fieldwork.
NURSE_W8W9_PATTERNS = (
    "*elsa_nurse_w8w9*.dta",
    "*nurse*w8w9*.dta",
    "*nurse*w8*w9*.dta",
    "*w8w9*nurse*.dta",
)

OUTPUT_DIR = PROJECT_ROOT / "DataProcessing" / "missingness_final_audit"


# ======================================================================================
# FROZEN 17-PREDICTOR PANEL
# ======================================================================================

PREDICTORS = {
    "Age": ("r", "agey"),
    "Education_Level": ("all", "raeduc_e"),
    "Total_Family_Wealth": ("h", "atotb"),
    "Total_Word_Recall": ("r", "tr20"),
    "Orientation": ("r", "orient"),
    "CASP19_Quality_of_Life": ("r", "casp19"),
    "Friend_Support_Lack": ("r", "fsupport6"),
    "Grip_Strength": ("r", "gripsum"),
    "Lower_Mobility": ("r", "lowermob"),
    "BMI": ("r", "mbmi"),
    "Hearing": ("r", "hearing"),
    "Eyesight": ("r", "sight"),
    "Depression_CESD": ("r", "cesd"),
    "Loneliness": ("r", "lnlys3"),
    "Social_Participation": ("r", "socyr"),
    "Managing_Meds": ("r", "medsa"),
    "Vigorous_Activity": ("r", "vgactx_e"),
}

CASP19_COMPONENTS = (
    "ageprv", "outcon", "freepln", "flftout", "cando", "famprv", "plsslf",
    "hlthstp", "mshtge", "lookfwd", "lfemean", "enjdo", "enjcmpny", "snshappy",
    "fenerg", "chsnth", "lftrnout", "lifeopp", "futregd",
)


# ======================================================================================
# GENERIC HELPERS
# ======================================================================================

def find_source_dta() -> Path:
    if DTA_OVERRIDE is not None:
        if not DTA_OVERRIDE.is_file():
            raise FileNotFoundError(f"ELSA source DTA not found: {DTA_OVERRIDE}")
        return DTA_OVERRIDE.resolve()
    for path in DTA_CANDIDATES:
        if path.exists():
            return path.resolve()

    tried = "\n  ".join(str(p) for p in DTA_CANDIDATES)
    raise FileNotFoundError(
        "Could not find gh_elsa_h.dta. Tried:\n  "
        + tried
        + "\nPass --dta PATH if your source file is elsewhere."
    )


def is_stata_missing(v: Any) -> bool:
    return isinstance(v, StataMissingValue)


def is_plain_nan(v: Any) -> bool:
    if is_stata_missing(v):
        return False
    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def numeric_value(v: Any) -> float | None:
    if is_stata_missing(v) or is_plain_nan(v):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def raw_display(v: Any) -> str:
    if is_stata_missing(v):
        return str(v)
    if is_plain_nan(v):
        return "<NaN>"
    x = numeric_value(v)
    if x is not None and float(x).is_integer():
        return str(int(x))
    return str(v)


def exact_value_label(
    column: str,
    value: Any,
    value_labels: dict[str, dict[float, str]],
) -> str:
    if is_stata_missing(value):
        return str(value)

    num = numeric_value(value)
    if num is None:
        return ""

    mapping = value_labels.get(column, {})

    if num in mapping:
        return str(mapping[num])

    if float(num).is_integer() and int(num) in mapping:
        return str(mapping[int(num)])

    return ""


def special_tag_from_label(label: str) -> str | None:
    m = re.match(
        r"^(\.[A-Za-z])(?:\s*[:=-]|\b)",
        str(label).strip(),
    )
    return m.group(1).lower() if m else None


def special_tag(
    column: str,
    value: Any,
    value_labels: dict[str, dict[float, str]],
) -> str | None:
    if is_stata_missing(value):
        txt = str(value).strip().lower()
        return txt if txt.startswith(".") else None

    return special_tag_from_label(
        exact_value_label(column, value, value_labels)
    )


def substantive_numeric(
    column: str,
    value: Any,
    value_labels: dict[str, dict[float, str]],
    *,
    allow_negative: bool = False,
) -> bool:
    """
    Whether a cell is an observed substantive numeric value.

    Negative wealth is valid.
    Special labelled missing states are not observed.
    """
    if is_stata_missing(value) or is_plain_nan(value):
        return False

    if special_tag(column, value, value_labels) is not None:
        return False

    x = numeric_value(value)
    if x is None:
        return False

    if allow_negative:
        return True

    return x >= 0


def education_observed(
    value: Any,
    value_labels: dict[str, dict[float, str]],
) -> bool:
    """
    raeduc_e code -15 ('Other') is retained as a genuine category.
    DK/refusal/other missing states are not observed.
    """
    col = "raeduc_e"

    if is_stata_missing(value) or is_plain_nan(value):
        return False

    x = numeric_value(value)
    label = exact_value_label(col, value, value_labels).strip().lower()

    if x == -15 or label == "other" or label.endswith(":other"):
        return True

    if special_tag(col, value, value_labels) is not None:
        return False

    return x is not None and x >= 0


def predictor_column(name: str, wave: int) -> str:
    prefix, stem = PREDICTORS[name]

    if prefix == "all":
        return stem

    return f"{prefix}{wave}{stem}"


def mask_observed_series(
    name: str,
    wave: int,
    df: pd.DataFrame,
    value_labels: dict[str, dict[float, str]],
) -> pd.Series:
    col = predictor_column(name, wave)

    if name == "Education_Level":
        return df[col].map(
            lambda v: education_observed(v, value_labels)
        )

    if name == "Total_Family_Wealth":
        return df[col].map(
            lambda v: substantive_numeric(
                col,
                v,
                value_labels,
                allow_negative=True,
            )
        )

    return df[col].map(
        lambda v: substantive_numeric(
            col,
            v,
            value_labels,
        )
    )


def target_observed_series(
    col: str,
    df: pd.DataFrame,
    value_labels: dict[str, dict[float, str]],
) -> pd.Series:
    return (
        df[col].map(
            lambda v: substantive_numeric(
                col,
                v,
                value_labels,
            )
        )
        &
        df[col].map(
            lambda v: numeric_value(v) in {0, 1}
        )
    )


def num_series(
    df: pd.DataFrame,
    col: str,
) -> pd.Series:
    if col not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return df[col].map(numeric_value)


def pct(n: int, d: int) -> float:
    return float(100.0 * n / d) if d else float("nan")


def dementia_stats(
    df: pd.DataFrame,
    mask: pd.Series,
    col: str,
    value_labels: dict[str, dict[float, str]],
) -> tuple[int, int, float]:
    ok = target_observed_series(
        col,
        df,
        value_labels,
    )

    m = mask & ok
    n = int(m.sum())

    if n == 0:
        return 0, 0, float("nan")

    vals = num_series(
        df.loc[m],
        col,
    )

    cases = int(vals.eq(1).sum())

    return (
        n,
        cases,
        100.0 * cases / n,
    )


# ======================================================================================
# REQUIRED COLUMNS
# ======================================================================================

def requested_columns() -> set[str]:
    cols = {
        "idauniq",
        "raeduc_e",
        "r10alzdeme",
        "inw10",
    }

    for w in WAVES:

        cols.update(
            {
                f"inw{w}",
                f"inw{w}sc",
                f"inw{w}n",
                f"r{w}proxy",
                f"r{w}iwstat",
                f"r{w}alzdeme",
                f"r{w}imrc",
                f"r{w}dlrc",
            }
        )

        for name in PREDICTORS:
            cols.add(
                predictor_column(name, w)
            )

        cols.update(
            f"r{w}{stem}"
            for stem in CASP19_COMPONENTS
        )

    return cols


# ======================================================================================
# 1. CASP19: EXACT NUMBER OF MISSING COMPONENTS
# ======================================================================================

def casp_component_audit(
    df: pd.DataFrame,
    value_labels: dict[str, dict[float, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    detail_rows: list[dict[str, Any]] = []

    for w in WAVES:

        wave = num_series(
            df,
            f"inw{w}",
        ).eq(1)

        score_col = f"r{w}casp19"

        component_cols = [
            f"r{w}{stem}"
            for stem in CASP19_COMPONENTS
        ]

        for idx in df.index[wave]:

            observed_flags = [
                substantive_numeric(
                    c,
                    df.at[idx, c],
                    value_labels,
                )
                for c in component_cols
            ]

            n_observed = int(
                sum(observed_flags)
            )

            n_missing = (
                len(component_cols)
                - n_observed
            )

            score_observed = substantive_numeric(
                score_col,
                df.at[idx, score_col],
                value_labels,
            )

            detail_rows.append(
                {
                    "idauniq":
                        raw_display(
                            df.at[idx, "idauniq"]
                        ),

                    "Wave":
                        w,

                    "CASP19_Score_Observed":
                        score_observed,

                    "CASP19_Raw":
                        raw_display(
                            df.at[idx, score_col]
                        ),

                    "N_Components_Observed":
                        n_observed,

                    "N_Components_Missing":
                        n_missing,

                    "SelfCompletion":
                        numeric_value(
                            df.at[
                                idx,
                                f"inw{w}sc",
                            ]
                        ),

                    "Proxy":
                        numeric_value(
                            df.at[
                                idx,
                                f"r{w}proxy",
                            ]
                        ),
                }
            )

    detail = pd.DataFrame(
        detail_rows
    )

    summary_rows = []

    for w in WAVES:

        d = detail[
            detail["Wave"] == w
        ]

        for n_missing in range(
            0,
            20,
        ):

            x = d[
                d["N_Components_Missing"]
                == n_missing
            ]

            score_missing = x[
                ~x["CASP19_Score_Observed"]
            ]

            summary_rows.append(
                {
                    "Wave":
                        w,

                    "N_CASP_Components_Missing":
                        n_missing,

                    "N_All_Wave_Respondents":
                        len(x),

                    "N_CASP_Score_Missing":
                        len(score_missing),

                    "N_Score_Missing_SC_Yes":
                        int(
                            (
                                score_missing[
                                    "SelfCompletion"
                                ]
                                == 1
                            ).sum()
                        ),

                    "N_Score_Missing_SC_No":
                        int(
                            (
                                score_missing[
                                    "SelfCompletion"
                                ]
                                == 0
                            ).sum()
                        ),

                    "N_Score_Missing_Proxy":
                        int(
                            (
                                score_missing[
                                    "Proxy"
                                ]
                                == 1
                            ).sum()
                        ),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    return detail, summary


# ======================================================================================
# 2. CORRECTED TOTAL-WORD-RECALL CLASSIFICATION
# ======================================================================================

def recall_reason(
    w: int,
    idx: Any,
    df: pd.DataFrame,
    value_labels: dict[str, dict[float, str]],
) -> str:

    score_col = f"r{w}tr20"
    im_col = f"r{w}imrc"
    dl_col = f"r{w}dlrc"

    score_v = df.at[idx, score_col]
    im_v = df.at[idx, im_col]
    dl_v = df.at[idx, dl_col]

    score_ok = substantive_numeric(
        score_col,
        score_v,
        value_labels,
    )

    im_ok = substantive_numeric(
        im_col,
        im_v,
        value_labels,
    )

    dl_ok = substantive_numeric(
        dl_col,
        dl_v,
        value_labels,
    )

    proxy = (
        numeric_value(
            df.at[idx, f"r{w}proxy"]
        )
        == 1
    )

    if score_ok:
        if im_ok and dl_ok:
            return "OBSERVED_COMPLETE"
        return "OBSERVED_SCORE_COMPONENT_INCONSISTENCY_REVIEW"

    score_tag = special_tag(
        score_col,
        score_v,
        value_labels,
    )

    if proxy or score_tag == ".p":
        return "PROXY_STRUCTURAL"

    if im_ok and not dl_ok:
        return "DELAYED_RECALL_MISSING_ONLY"

    if dl_ok and not im_ok:
        return "IMMEDIATE_RECALL_MISSING_ONLY"

    if im_ok and dl_ok:
        return "SCORE_MISSING_BUT_BOTH_COMPONENTS_OBSERVED_REVIEW"

    im_tag = special_tag(
        im_col,
        im_v,
        value_labels,
    )

    dl_tag = special_tag(
        dl_col,
        dl_v,
        value_labels,
    )

    tags = sorted(
        set(
            t
            for t in (
                score_tag,
                im_tag,
                dl_tag,
            )
            if t is not None
        )
    )

    if tags:
        return (
            "BOTH_COMPONENTS_MISSING_SPECIAL_"
            + "_".join(
                t.replace(".", "")
                for t in tags
            )
        )

    return "BOTH_COMPONENTS_MISSING_OTHER"


def recall_corrected_audit(
    df: pd.DataFrame,
    value_labels: dict[str, dict[float, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rows = []

    for w in WAVES:

        wave = num_series(
            df,
            f"inw{w}",
        ).eq(1)

        for idx in df.index[wave]:

            rows.append(
                {
                    "idauniq":
                        raw_display(
                            df.at[idx, "idauniq"]
                        ),

                    "Wave":
                        w,

                    "Recall_Raw":
                        raw_display(
                            df.at[
                                idx,
                                f"r{w}tr20",
                            ]
                        ),

                    "Immediate_Raw":
                        raw_display(
                            df.at[
                                idx,
                                f"r{w}imrc",
                            ]
                        ),

                    "Delayed_Raw":
                        raw_display(
                            df.at[
                                idx,
                                f"r{w}dlrc",
                            ]
                        ),

                    "Proxy":
                        numeric_value(
                            df.at[
                                idx,
                                f"r{w}proxy",
                            ]
                        ),

                    "Classification":
                        recall_reason(
                            w,
                            idx,
                            df,
                            value_labels,
                        ),
                }
            )

    detail = pd.DataFrame(
        rows
    )

    summary = (
        detail.groupby(
            [
                "Wave",
                "Classification",
            ],
            dropna=False,
        )
        .size()
        .rename("Count")
        .reset_index()
    )

    return detail, summary


# ======================================================================================
# 3. W8 GRIP: ACTUAL W8 VS W9 NURSE MEASUREMENT TIMING
# ======================================================================================

def find_w8w9_nurse_dta() -> Path | None:
    roots = (
        (NURSE_DIR_OVERRIDE,) if NURSE_DIR_OVERRIDE is not None
        else (PROJECT_ROOT / "Datasets", PROJECT_ROOT)
    )

    candidates: list[Path] = []

    for root in roots:

        if not root.exists():
            continue

        for pattern in NURSE_W8W9_PATTERNS:

            candidates.extend(
                root.rglob(pattern)
            )

    unique = []
    seen = set()

    for p in candidates:

        try:
            rp = p.resolve()
        except Exception:
            rp = p

        if rp in seen:
            continue

        seen.add(rp)
        unique.append(rp)

    return unique[0] if unique else None


def resolve_case_insensitive(
    columns: Iterable[str],
    desired: str,
) -> str | None:

    lookup = {
        str(c).lower(): str(c)
        for c in columns
    }

    return lookup.get(
        desired.lower()
    )


def w8_grip_timing_audit(
    df: pd.DataFrame,
    value_labels: dict[str, dict[float, str]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:

    source = find_w8w9_nurse_dta()

    info = {
        "Raw_Nurse_File_Found":
            bool(source),

        "Raw_Nurse_File":
            str(source)
            if source
            else "",

        "Timing_Status":
            "",

        "Duplicate_IDs_In_Raw_Nurse_File":
            np.nan,
    }

    base_mask = (
        num_series(
            df,
            "inw8",
        ).eq(1)
        &
        num_series(
            df,
            "r8agey",
        ).ge(AGE_MIN)
    )

    grip_ok = df[
        "r8gripsum"
    ].map(
        lambda v: substantive_numeric(
            "r8gripsum",
            v,
            value_labels,
        )
    )

    base = df.loc[
        base_mask,
        [
            "idauniq",
            "r8gripsum",
            "inw8n",
        ],
    ].copy()

    base["Grip_Observed"] = (
        grip_ok.loc[
            base.index
        ].values
    )

    base["Nurse_Visit_Year"] = np.nan
    base["Measurement_Source"] = (
        "UNRESOLVED_NO_RAW_W8W9_NURSE_FILE"
    )

    if source is None:

        info["Timing_Status"] = (
            "Could not find the raw combined W8/W9 nurse DTA. "
            "The harmonized file alone cannot distinguish W8 from W9 nurse timing."
        )

        summary = (
            base.groupby(
                [
                    "Grip_Observed",
                    "Measurement_Source",
                ],
                dropna=False,
            )
            .size()
            .rename("Count")
            .reset_index()
        )

        return base, summary, info

    try:

        with pd.io.stata.StataReader(
            source,
            convert_categoricals=False,
            convert_missing=True,
            preserve_dtypes=True,
        ) as reader:

            columns = list(
                reader.variable_labels().keys()
            )

            id_col = resolve_case_insensitive(
                columns,
                "idauniq",
            )

            year_col = resolve_case_insensitive(
                columns,
                "visyear",
            )

            if id_col is None or year_col is None:

                info["Timing_Status"] = (
                    "Raw combined nurse file was found, "
                    "but idauniq and/or visyear could not be located."
                )

                summary = (
                    base.groupby(
                        [
                            "Grip_Observed",
                            "Measurement_Source",
                        ],
                        dropna=False,
                    )
                    .size()
                    .rename("Count")
                    .reset_index()
                )

                return base, summary, info

            nurse = reader.read(
                columns=[
                    id_col,
                    year_col,
                ],
                convert_categoricals=False,
                convert_missing=True,
                preserve_dtypes=True,
            )

    except Exception as exc:

        info["Timing_Status"] = (
            "Raw nurse file was found but could not be read: "
            + repr(exc)
        )

        summary = (
            base.groupby(
                [
                    "Grip_Observed",
                    "Measurement_Source",
                ],
                dropna=False,
            )
            .size()
            .rename("Count")
            .reset_index()
        )

        return base, summary, info

    nurse = nurse.rename(
        columns={
            id_col: "idauniq_raw_nurse",
            year_col: "visyear_raw_nurse",
        }
    )

    nurse["idauniq_num"] = nurse[
        "idauniq_raw_nurse"
    ].map(numeric_value)

    nurse["visyear_num"] = nurse[
        "visyear_raw_nurse"
    ].map(numeric_value)

    duplicate_ids = int(
        nurse[
            "idauniq_num"
        ]
        .dropna()
        .duplicated()
        .sum()
    )

    info[
        "Duplicate_IDs_In_Raw_Nurse_File"
    ] = duplicate_ids

    nurse = (
        nurse.sort_values(
            by="visyear_num",
            na_position="last",
        )
        .drop_duplicates(
            subset="idauniq_num",
            keep="first",
        )
    )

    base["idauniq_num"] = base[
        "idauniq"
    ].map(numeric_value)

    merged = base.merge(
        nurse[
            [
                "idauniq_num",
                "visyear_num",
            ]
        ],
        on="idauniq_num",
        how="left",
        validate="many_to_one",
    )

    merged["Nurse_Visit_Year"] = merged[
        "visyear_num"
    ]

    def classify_year(
        year: Any,
    ) -> str:

        y = (
            float(year)
            if pd.notna(year)
            else np.nan
        )

        if not np.isfinite(y):
            return "UNRESOLVED_NO_RAW_NURSE_MATCH_OR_YEAR"

        if y in {2016, 2017}:
            return "ACTUAL_W8_NURSE_MEASUREMENT"

        if y in {2018, 2019}:
            return "ACTUAL_W9_NURSE_MEASUREMENT"

        return (
            "OTHER_NURSE_YEAR_"
            + str(int(y))
        )

    merged[
        "Measurement_Source"
    ] = merged[
        "Nurse_Visit_Year"
    ].map(classify_year)

    info["Timing_Status"] = (
        "Resolved from raw combined W8/W9 nurse VISYEAR."
    )

    summary = (
        merged.groupby(
            [
                "Grip_Observed",
                "Measurement_Source",
                "Nurse_Visit_Year",
            ],
            dropna=False,
        )
        .size()
        .rename("Count")
        .reset_index()
    )

    return merged, summary, info


# ======================================================================================
# 4. DEMENTIA PREVALENCE BY MODULE / ROUTING STATUS
# ======================================================================================

def dementia_by_module_status(
    df: pd.DataFrame,
    value_labels: dict[str, dict[float, str]],
) -> pd.DataFrame:

    rows = []

    for w in WAVES:

        wave = num_series(
            df,
            f"inw{w}",
        ).eq(1)

        age50 = (
            wave
            &
            num_series(
                df,
                f"r{w}agey",
            ).ge(AGE_MIN)
        )

        sc = num_series(
            df,
            f"inw{w}sc",
        )

        nurse = num_series(
            df,
            f"inw{w}n",
        )

        proxy = num_series(
            df,
            f"r{w}proxy",
        )

        groups = {
            "SELF_COMPLETION=1":
                age50 & sc.eq(1),

            "SELF_COMPLETION=0":
                age50 & sc.eq(0),

            "NURSE=1":
                age50 & nurse.eq(1),

            "NURSE=0":
                age50 & nurse.eq(0),

            "PROXY=0":
                age50 & ~proxy.eq(1),

            "PROXY=1":
                age50 & proxy.eq(1),

            "SC1_NURSE1":
                age50
                & sc.eq(1)
                & nurse.eq(1),

            "SC1_NURSE0":
                age50
                & sc.eq(1)
                & nurse.eq(0),

            "SC0_NURSE1":
                age50
                & sc.eq(0)
                & nurse.eq(1),

            "SC0_NURSE0":
                age50
                & sc.eq(0)
                & nurse.eq(0),
        }

        for group, mask in groups.items():

            same_n, same_cases, same_prev = (
                dementia_stats(
                    df,
                    mask,
                    f"r{w}alzdeme",
                    value_labels,
                )
            )

            w10_n, w10_cases, w10_prev = (
                dementia_stats(
                    df,
                    mask,
                    "r10alzdeme",
                    value_labels,
                )
            )

            rows.append(
                {
                    "Wave":
                        w,

                    "Group":
                        group,

                    "Group_N":
                        int(mask.sum()),

                    "SameWave_Target_Observed_N":
                        same_n,

                    "SameWave_Dementia_Cases":
                        same_cases,

                    "SameWave_Dementia_Prevalence_Pct":
                        same_prev,

                    "W10_Target_Observed_N":
                        w10_n,

                    "W10_Dementia_Cases":
                        w10_cases,

                    "W10_Dementia_Prevalence_Pct":
                        w10_prev,
                }
            )

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# 5. 18-NODE ROW COMPLETENESS
# ======================================================================================

def node_observation_matrix(
    wave: int,
    df: pd.DataFrame,
    value_labels: dict[str, dict[float, str]],
    *,
    target_mode: str,
) -> tuple[pd.DataFrame, list[str]]:

    obs = pd.DataFrame(
        index=df.index
    )

    node_names = []

    for name in PREDICTORS:

        obs[name] = mask_observed_series(
            name,
            wave,
            df,
            value_labels,
        )

        node_names.append(name)

    if target_mode == "SAME_WAVE":

        target_name = (
            f"Dementia_W{wave}"
        )

        target_col = (
            f"r{wave}alzdeme"
        )

    elif target_mode == "W10":

        target_name = (
            "Dementia_W10"
        )

        target_col = (
            "r10alzdeme"
        )

    else:
        raise ValueError(
            target_mode
        )

    obs[target_name] = target_observed_series(
        target_col,
        df,
        value_labels,
    )

    node_names.append(
        target_name
    )

    assert len(node_names) == 18

    return obs, node_names


def node_count_distribution(
    df: pd.DataFrame,
    value_labels: dict[str, dict[float, str]],
    *,
    target_mode: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    distribution_rows = []
    detail_rows = []

    for w in WAVES:

        obs, node_names = (
            node_observation_matrix(
                w,
                df,
                value_labels,
                target_mode=target_mode,
            )
        )

        wave = num_series(
            df,
            f"inw{w}",
        ).eq(1)

        age50 = (
            wave
            &
            num_series(
                df,
                f"r{w}agey",
            ).ge(AGE_MIN)
        )

        cohorts = {
            "WAVE_RESPONDENTS":
                wave,

            "AGE50_WAVE_RESPONDENTS":
                age50,
        }

        observed_count = obs[
            node_names
        ].sum(axis=1).astype(int)

        missing_count = (
            18
            - observed_count
        )

        for cohort_name, mask in cohorts.items():

            d = int(mask.sum())

            for n_obs in range(
                18,
                -1,
                -1,
            ):

                n = int(
                    (
                        mask
                        &
                        observed_count.eq(
                            n_obs
                        )
                    ).sum()
                )

                distribution_rows.append(
                    {
                        "Wave":
                            w,

                        "Target_Mode":
                            target_mode,

                        "Cohort":
                            cohort_name,

                        "Observed_Nodes":
                            n_obs,

                        "Missing_Nodes":
                            18 - n_obs,

                        "N_Participants":
                            n,

                        "Percent":
                            pct(n, d),
                    }
                )

        for idx in df.index[age50]:

            missing_nodes = [
                node
                for node in node_names
                if not bool(
                    obs.at[idx, node]
                )
            ]

            detail_rows.append(
                {
                    "idauniq":
                        raw_display(
                            df.at[
                                idx,
                                "idauniq",
                            ]
                        ),

                    "Wave":
                        w,

                    "Target_Mode":
                        target_mode,

                    "Observed_Nodes":
                        int(
                            observed_count.at[
                                idx
                            ]
                        ),

                    "Missing_Nodes":
                        int(
                            missing_count.at[
                                idx
                            ]
                        ),

                    "Missing_Node_Names":
                        "; ".join(
                            missing_nodes
                        ),
                }
            )

    return (
        pd.DataFrame(
            distribution_rows
        ),
        pd.DataFrame(
            detail_rows
        ),
    )


# ======================================================================================
# 6. "HOW MANY HAVE A LOT OF MISSING VALUES?"
# ======================================================================================

def heavy_missingness_summary(
    detail: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    thresholds = (
        1,
        2,
        3,
        5,
        8,
        10,
        12,
        15,
    )

    for (
        target_mode,
        wave,
    ), g in detail.groupby(
        [
            "Target_Mode",
            "Wave",
        ]
    ):

        total = len(g)

        for threshold in thresholds:

            n = int(
                (
                    g["Missing_Nodes"]
                    >= threshold
                ).sum()
            )

            rows.append(
                {
                    "Target_Mode":
                        target_mode,

                    "Wave":
                        wave,

                    "Threshold":
                        f">={threshold} missing of 18",

                    "N_Participants":
                        n,

                    "Percent":
                        pct(
                            n,
                            total,
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def longitudinal_participant_missingness(
    same_wave_detail: pd.DataFrame,
) -> pd.DataFrame:
    """
    Participant-level missingness across W2/W4/W6/W8.

    Only age-50+ waves in which the person actually responded are present in
    same_wave_detail, so nonparticipation/attrition is NOT counted as a missing cell.
    """

    if same_wave_detail.empty:
        return pd.DataFrame()

    rows = []

    for idauniq, g in same_wave_detail.groupby(
        "idauniq"
    ):

        g = g.sort_values(
            "Wave"
        )

        n_waves = len(g)

        expected = (
            18
            * n_waves
        )

        observed = int(
            g["Observed_Nodes"].sum()
        )

        missing = int(
            g["Missing_Nodes"].sum()
        )

        rows.append(
            {
                "idauniq":
                    idauniq,

                "N_Age50_Responded_Waves":
                    n_waves,

                "Waves":
                    ",".join(
                        str(int(w))
                        for w in g["Wave"]
                    ),

                "Expected_Node_Cells":
                    expected,

                "Observed_Node_Cells":
                    observed,

                "Missing_Node_Cells":
                    missing,

                "Missing_Pct_Of_Expected":
                    pct(
                        missing,
                        expected,
                    ),

                "Max_Missing_In_One_Wave":
                    int(
                        g[
                            "Missing_Nodes"
                        ].max()
                    ),

                "N_Waves_With_3Plus_Missing":
                    int(
                        (
                            g[
                                "Missing_Nodes"
                            ]
                            >= 3
                        ).sum()
                    ),

                "N_Waves_With_5Plus_Missing":
                    int(
                        (
                            g[
                                "Missing_Nodes"
                            ]
                            >= 5
                        ).sum()
                    ),

                "N_Waves_With_8Plus_Missing":
                    int(
                        (
                            g[
                                "Missing_Nodes"
                            ]
                            >= 8
                        ).sum()
                    ),
            }
        )

    out = pd.DataFrame(
        rows
    )

    return out.sort_values(
        [
            "Missing_Pct_Of_Expected",
            "Missing_Node_Cells",
        ],
        ascending=[
            False,
            False,
        ],
    )


def longitudinal_missingness_bands(
    longitudinal: pd.DataFrame,
) -> pd.DataFrame:

    if longitudinal.empty:
        return pd.DataFrame()

    bins = [
        -0.001,
        0,
        5,
        10,
        20,
        30,
        40,
        50,
        60,
        75,
        100.0001,
    ]

    labels = [
        "0%",
        ">0-5%",
        ">5-10%",
        ">10-20%",
        ">20-30%",
        ">30-40%",
        ">40-50%",
        ">50-60%",
        ">60-75%",
        ">75%",
    ]

    x = longitudinal.copy()

    x["Missingness_Band"] = pd.cut(
        x[
            "Missing_Pct_Of_Expected"
        ],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    out = (
        x.groupby(
            "Missingness_Band",
            observed=False,
        )
        .size()
        .rename(
            "N_Participants"
        )
        .reset_index()
    )

    total = len(x)

    out["Percent"] = (
        100.0
        * out[
            "N_Participants"
        ]
        / total
    )

    return out


# ======================================================================================
# 7. REPORT
# ======================================================================================

def make_report(
    source: Path,
    casp_summary: pd.DataFrame,
    recall_summary: pd.DataFrame,
    grip_summary: pd.DataFrame,
    grip_info: dict[str, Any],
    module_dementia: pd.DataFrame,
    same_dist: pd.DataFrame,
    future_dist: pd.DataFrame,
    heavy: pd.DataFrame,
    long_bands: pd.DataFrame,
) -> str:

    lines = []

    sep = "=" * 120
    sub = "-" * 120

    lines.append(sep)
    lines.append(
        "ELSA FINAL MISSINGNESS AUDIT — SCRIPT 12"
    )
    lines.append(sep)

    lines.append(
        f"Source DTA: {source}"
    )

    lines.append(
        f"Age threshold: {AGE_MIN}+"
    )

    lines.append(
        "NO CLEANING OR IMPUTATION IS PERFORMED."
    )

    lines.append("")

    lines.append(
        "PURPOSE"
    )
    lines.append(sub)

    lines.extend(
        [
            "1. Count CASP19 missing components exactly (0 through 19).",
            "2. Correct the earlier Total Word Recall classification: recall is NOT a self-completion measure.",
            "3. Resolve whether harmonized W8 grip values were actually measured in W8 or W9, if the raw combined nurse DTA is present.",
            "4. Compare dementia prevalence by self-completion, nurse, and proxy status.",
            "5. Count how many participants have 18, 17, 16 ... 1 (and 0) observed nodes.",
            "6. Quantify heavy missingness within waves and longitudinally without treating wave nonresponse as item missingness.",
        ]
    )

    lines.append("")
    lines.append(
        "FULL 18-NODE ROW COUNTS — AGE 50+"
    )
    lines.append(sub)

    for label, table in (
        (
            "CONTEMPORANEOUS: 17 predictors + same-wave dementia",
            same_dist,
        ),
        (
            "FUTURE TARGET: 17 predictors + W10 dementia",
            future_dist,
        ),
    ):

        lines.append(label)

        x = table[
            (
                table["Cohort"]
                == "AGE50_WAVE_RESPONDENTS"
            )
            &
            (
                table["Observed_Nodes"]
                == 18
            )
        ]

        lines.append(
            x[
                [
                    "Wave",
                    "N_Participants",
                    "Percent",
                ]
            ]
            .to_string(
                index=False,
            )
        )

        lines.append("")

    lines.append(
        "EXACT CONTEMPORANEOUS NODE-COUNT DISTRIBUTION — AGE 50+"
    )
    lines.append(sub)

    lines.append(
        same_dist[
            same_dist["Cohort"]
            == "AGE50_WAVE_RESPONDENTS"
        ][
            [
                "Wave",
                "Observed_Nodes",
                "Missing_Nodes",
                "N_Participants",
                "Percent",
            ]
        ].to_string(
            index=False,
        )
    )

    lines.append("")
    lines.append(
        "EXACT FUTURE-TARGET NODE-COUNT DISTRIBUTION — AGE 50+"
    )
    lines.append(sub)

    lines.append(
        future_dist[
            future_dist["Cohort"]
            == "AGE50_WAVE_RESPONDENTS"
        ][
            [
                "Wave",
                "Observed_Nodes",
                "Missing_Nodes",
                "N_Participants",
                "Percent",
            ]
        ].to_string(
            index=False,
        )
    )

    lines.append("")
    lines.append(
        "HEAVY MISSINGNESS THRESHOLDS — AGE 50+"
    )
    lines.append(sub)

    lines.append(
        heavy.to_string(
            index=False,
        )
    )

    lines.append("")
    lines.append(
        "LONGITUDINAL PARTICIPANT MISSINGNESS BANDS"
    )
    lines.append(sub)

    lines.append(
        "Only age-50+ waves actually responded to are counted; attrition/nonparticipation is not treated as a missing node."
    )

    lines.append(
        long_bands.to_string(
            index=False,
        )
        if not long_bands.empty
        else "<none>"
    )

    lines.append("")
    lines.append(
        "CASP19 — EXACT MISSING COMPONENT COUNTS"
    )
    lines.append(sub)

    lines.append(
        casp_summary.to_string(
            index=False,
        )
    )

    lines.append("")
    lines.append(
        "TOTAL WORD RECALL — CORRECTED CLASSIFICATION"
    )
    lines.append(sub)

    lines.append(
        recall_summary.to_string(
            index=False,
        )
    )

    lines.append("")
    lines.append(
        "W8 GRIP ACTUAL NURSE-TIMING AUDIT"
    )
    lines.append(sub)

    for k, v in grip_info.items():
        lines.append(
            f"{k}: {v}"
        )

    lines.append("")

    lines.append(
        grip_summary.to_string(
            index=False,
        )
        if not grip_summary.empty
        else "<none>"
    )

    lines.append("")
    lines.append(
        "DEMENTIA PREVALENCE BY MODULE / ROUTING STATUS"
    )
    lines.append(sub)

    lines.append(
        module_dementia.to_string(
            index=False,
        )
    )

    lines.append("")
    lines.append(
        "INTERPRETATION GUARDRAILS"
    )
    lines.append(sub)

    lines.extend(
        [
            "1. A low observed-node count does not imply that every absent value should be imputed.",
            "2. Wave nonresponse is not counted as item-level missingness in participant-level longitudinal summaries.",
            "3. Same-wave and W10-target completeness are reported separately because they answer different study questions.",
            "4. Proxy, self-completion nonresponse, nurse nonresponse, and physical inability are potentially informative mechanisms.",
            "5. The W8/W9 grip source cannot be inferred from harmonized r8gripsum alone; raw nurse VISYEAR is used only if the combined nurse DTA is found.",
            "6. Dementia remains an outcome and should not be imputed in the final analysis pipeline.",
        ]
    )

    lines.append("")
    lines.append(
        "NEXT"
    )
    lines.append(sub)

    lines.append(
        "Use Script 12 results together with Scripts 10-11 to freeze the variable-by-variable cleaning/imputation policy before creating analysis-ready datasets."
    )

    return "\n".join(
        lines
    )


# ======================================================================================
# MAIN
# ======================================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the ELSA missingness audit and Phase-1 cohort file."
    )
    parser.add_argument(
        "--dta", type=Path, default=None,
        help="Optional explicit path to licensed gh_elsa_h.dta."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <repo>/DataProcessing/missingness_final_audit)."
    )
    parser.add_argument(
        "--nurse-dir", type=Path, default=None,
        help="Optional directory to search for the combined W8/W9 nurse DTA."
    )
    return parser.parse_args()


def main() -> None:
    global DTA_OVERRIDE, NURSE_DIR_OVERRIDE, OUTPUT_DIR
    args = parse_args()
    DTA_OVERRIDE = args.dta.expanduser().resolve() if args.dta else None
    NURSE_DIR_OVERRIDE = args.nurse_dir.expanduser().resolve() if args.nurse_dir else None
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir.expanduser().resolve()

    source = find_source_dta()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 120)
    print(
        "ELSA FINAL MISSINGNESS AUDIT — SCRIPT 12"
    )
    print("=" * 120)

    print(
        f"Source: {source}"
    )

    print(
        "Reading harmonized metadata and selected variables while preserving Stata missing tags..."
    )

    req = requested_columns()

    with pd.io.stata.StataReader(
        source,
        convert_categoricals=False,
        convert_missing=True,
        preserve_dtypes=True,
    ) as reader:

        variable_labels = (
            reader.variable_labels()
        )

        value_labels = (
            reader.value_labels()
        )

        available = set(
            variable_labels
        )

        missing_required = sorted(
            req - available
        )

        if missing_required:
            raise KeyError(
                "Required columns missing from source DTA:\n  "
                + "\n  ".join(
                    missing_required
                )
            )

        df = reader.read(
            columns=sorted(req),
            convert_categoricals=False,
            convert_missing=True,
            preserve_dtypes=True,
        )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        "Auditing exact CASP19 component missingness..."
    )

    casp_detail, casp_summary = (
        casp_component_audit(
            df,
            value_labels,
        )
    )

    casp_detail.to_csv(
        OUTPUT_DIR
        / "12a_casp19_case_detail.csv",
        index=False,
    )

    casp_summary.to_csv(
        OUTPUT_DIR
        / "12b_casp19_missing_component_counts.csv",
        index=False,
    )

    print(
        "Correcting Total Word Recall missingness classification..."
    )

    recall_detail, recall_summary = (
        recall_corrected_audit(
            df,
            value_labels,
        )
    )

    recall_detail.to_csv(
        OUTPUT_DIR
        / "12c_recall_missingness_corrected_detail.csv",
        index=False,
    )

    recall_summary.to_csv(
        OUTPUT_DIR
        / "12d_recall_missingness_corrected_summary.csv",
        index=False,
    )

    print(
        "Checking W8/W9 grip measurement timing..."
    )

    grip_detail, grip_summary, grip_info = (
        w8_grip_timing_audit(
            df,
            value_labels,
        )
    )

    grip_detail.to_csv(
        OUTPUT_DIR
        / "12e_w8_grip_measurement_timing_detail.csv",
        index=False,
    )

    grip_summary.to_csv(
        OUTPUT_DIR
        / "12f_w8_grip_measurement_timing_summary.csv",
        index=False,
    )

    print(
        "Calculating dementia prevalence by module/routing status..."
    )

    module_dementia = (
        dementia_by_module_status(
            df,
            value_labels,
        )
    )

    module_dementia.to_csv(
        OUTPUT_DIR
        / "12g_dementia_by_module_status.csv",
        index=False,
    )

    print(
        "Counting contemporaneous 18-node row completeness..."
    )

    same_dist, same_detail = (
        node_count_distribution(
            df,
            value_labels,
            target_mode="SAME_WAVE",
        )
    )

    same_dist.to_csv(
        OUTPUT_DIR
        / "12h_node_count_distribution_contemporaneous.csv",
        index=False,
    )

    same_detail.to_csv(
        OUTPUT_DIR
        / "12i_participant_wave_completeness_contemporaneous.csv",
        index=False,
    )

    print(
        "Counting W10-target 18-node row completeness..."
    )

    future_dist, future_detail = (
        node_count_distribution(
            df,
            value_labels,
            target_mode="W10",
        )
    )

    future_dist.to_csv(
        OUTPUT_DIR
        / "12j_node_count_distribution_w10_target.csv",
        index=False,
    )

    future_detail.to_csv(
        OUTPUT_DIR
        / "12k_participant_wave_completeness_w10_target.csv",
        index=False,
    )

    print(
        "Summarising participants with substantial within-wave missingness..."
    )

    combined_detail = pd.concat(
        [
            same_detail,
            future_detail,
        ],
        ignore_index=True,
    )

    heavy = heavy_missingness_summary(
        combined_detail
    )

    heavy.to_csv(
        OUTPUT_DIR
        / "12l_heavy_missingness_thresholds.csv",
        index=False,
    )

    longitudinal = (
        longitudinal_participant_missingness(
            same_detail
        )
    )

    longitudinal.to_csv(
        OUTPUT_DIR
        / "12m_longitudinal_participant_missingness.csv",
        index=False,
    )

    long_bands = (
        longitudinal_missingness_bands(
            longitudinal
        )
    )

    long_bands.to_csv(
        OUTPUT_DIR
        / "12n_longitudinal_missingness_bands.csv",
        index=False,
    )

    report = make_report(
        source,
        casp_summary,
        recall_summary,
        grip_summary,
        grip_info,
        module_dementia,
        same_dist,
        future_dist,
        heavy,
        long_bands,
    )

    report_path = (
        OUTPUT_DIR
        / "12_missingness_final_audit_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print("")
    print("=" * 120)
    print(
        "DONE — NO CLEANING OR IMPUTATION PERFORMED"
    )
    print("=" * 120)

    print(
        f"Report: {report_path}"
    )

    print("")
    print(
        "Key outputs:"
    )

    for name in (
        "12b_casp19_missing_component_counts.csv",
        "12d_recall_missingness_corrected_summary.csv",
        "12f_w8_grip_measurement_timing_summary.csv",
        "12g_dementia_by_module_status.csv",
        "12h_node_count_distribution_contemporaneous.csv",
        "12j_node_count_distribution_w10_target.csv",
        "12l_heavy_missingness_thresholds.csv",
        "12m_longitudinal_participant_missingness.csv",
        "12n_longitudinal_missingness_bands.csv",
    ):
        print(
            f"  {OUTPUT_DIR / name}"
        )


if __name__ == "__main__":
    main()

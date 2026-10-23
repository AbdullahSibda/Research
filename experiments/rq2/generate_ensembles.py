from __future__ import annotations

"""
generate_ensembles.py

RQ2 MASTER PIPELINE — PHASE 1
==============================
Generate the leakage-safe cross-validation ensembles used by the later
M-convergence and custom holdout-inference experiments.

LOCKED DESIGN
-------------
* Waves: 2, 4, 6, 8.
* Frozen cohort: Age >= 50, Dementia observed, <= 10/17 substantive predictors
  missing (the canonical completeness file from the missingness phase is used).
* K = 5 folds by default, stratified on observed Dementia and fixed once per wave.
* The ORIGINAL INCOMPLETE cohort is split before any imputation.
* For each fold/completion, the imputer chain is fitted sequentially on TRAIN
  only.  The fitted stage is then applied separately to TRAIN and TEST.
* Dementia is never imputed and is never a conditioning variable in the frozen
  imputer modules.  Dementia + process nodes are checked for exact preservation.
* M_initial_max = 25 by default.  Re-running with --m-max 30, 35, ... extends
  the same deterministic ensemble; existing completed work is skipped.
* One graph is learned per completed TRAIN dataset per requested model.
* The completed canonical TRAIN/TEST node matrices are saved once and reused by
  later scripts.  Model-specific transformations are TRAIN-fitted only.

MODEL REPRESENTATIONS
---------------------
Native discrete state representation (used by Discrete and Hybrid, and forced
numerically in the Continuous baseline):
    Education          : -15 -> 0 (Other), 1..6 -> 1..6        [7 states]
    Orientation        : 0-2 / 3 / 4                           [3 states]
    CES-D              : 0-1 / 2-3 / 4-8                       [3 states]
    Lower Mobility     : 0 / 1 / 2-4                           [3 states]
    Hearing            : 1-2 / 3 / 4-5                         [3 states]
    Eyesight           : 1-2 / 3 / 4-6                         [3 states]
    Vigorous Activity  : 5 / 4 / 3 / 2 -> 0 / 1 / 2 / 3       [4 states]
    Social Participation, Managing Meds, Dementia,
    Proxy, SelfCompletion, Nurse                               [binary]

Continuous-space nodes:
    Age, Wealth, Recall, CASP19, FriendSupport, Grip, BMI, Loneliness

Wealth is transformed for models that treat it continuously:
    Wealth* = asinh(Wealth / c)
where c is fitted from the COMPLETED TRAIN split only using the frozen robust
IQR/MAD/SD scale rule.

Discrete baseline:
    native discrete states above + three TRAIN-fitted ordinal quantile bins for
    each continuous-space node.  The discretizer is tie-aware: identical values
    are never split merely to force exact thirds.

Continuous baseline:
    all 21 nodes are supplied numerically to the frozen Gaussian scorer.  Native
    discrete nodes use the locked state codes above; continuous-space nodes keep
    their values, with Wealth transformed as above.  This is the intentional
    Gaussian-forcing baseline.

Hybrid V3:
    locked native discrete states + continuous-space values, with Wealth
    transformed as above.  Hybrid V3 performs its own train-fitted internal
    standardisation of continuous nodes.

PROCESS NODES
-------------
Proxy, SelfCompletion and Nurse are carried unchanged into every model as
ordinary BN nodes.  There are no forced-root or direction restrictions.

OUTPUT ROOT
-----------
Results/MasterPipeline/CV_Ensembles/

Important outputs:
    run_config.json
    fold_assignments/W2.csv ... W8.csv
    fold_summary.csv
    ensemble_index.csv
    completion_index.csv
    errors.csv

Per wave/fold/completion:
    W2/Fold01/M001/
        completed_train.csv.gz
        completed_test.csv.gz
        completion_summary.json
        representation_metadata.json
        DiscreteBaseline/
            edges.csv
            search_summary.json
            score_history.csv
        ContinuousBaseline/
            ...
        HybridV3/
            ...

The completed CSVs contain ONLY idauniq + the 21 final BN nodes, not the wider
auxiliary imputation columns.

SMOKE TEST EXAMPLES
-------------------
Preflight only:
    python generate_ensembles.py --preflight-only

Tiny end-to-end discrete smoke test:
    python generate_ensembles.py --waves 2 --m-max 1 --models discrete

Primary parallel run (recommended on the 4-core / 8-thread laptop):
    python generate_ensembles.py --workers 2

Three-worker throughput test:
    python generate_ensembles.py --workers 3

Extend the same ensemble later if M=25 has not converged:
    python generate_ensembles.py --m-max 30 --workers 2
"""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback

import networkx as nx
import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue
from sklearn.model_selection import StratifiedKFold


# ======================================================================================
# FROZEN DESIGN CONSTANTS
# ======================================================================================

DEFAULT_WAVES = (2, 4, 6, 8)
DEFAULT_FOLDS = 5
DEFAULT_M_MAX = 25
DEFAULT_BASE_SEED = 2026
MAX_MISSING_PREDICTORS = 10

FROZEN_RETAINED_COUNTS = {
    2: 9162,
    4: 10738,
    6: 10365,
    8: 8352,
}

EXPECTED_IMPUTER_FILES = {
    "small_categorical": "small_categorical.py",
    "education": "education.py",
    "wealth": "wealth.py",
    "recall": "recall.py",
    "orientation": "orientation.py",
    "cesd": "cesd.py",
    "casp19": "casp19.py",
    "loneliness": "loneliness.py",
    "friend_support": "friend_support.py",
    "social_participation": "social_participation.py",
    "bmi": "bmi.py",
    "grip_strength": "grip_strength.py",
    "vigorous_activity": "vigorous_activity.py",
}

EXPECTED_MODEL_FILES = {
    "discrete": "SLdiscrete.py",
    "continuous": "SLcontinuous.py",
    "hybrid": "SLhybrid.py",
}

SUBSTANTIVE_NODES = [
    "Age",
    "Education",
    "Wealth",
    "Recall",
    "Orientation",
    "CASP19",
    "FriendSupport",
    "Grip",
    "LowerMobility",
    "BMI",
    "Hearing",
    "Eyesight",
    "CESD",
    "Loneliness",
    "SocialParticipation",
    "ManagingMeds",
    "VigorousActivity",
]

PROCESS_NODES = ["Proxy", "SelfCompletion", "Nurse"]
OUTCOME_NODE = "Dementia"

CONTINUOUS_NODES = [
    "Age",
    "Wealth",
    "Recall",
    "CASP19",
    "FriendSupport",
    "Grip",
    "BMI",
    "Loneliness",
]

NATIVE_DISCRETE_NODES = [
    "Education",
    "Orientation",
    "LowerMobility",
    "Hearing",
    "Eyesight",
    "CESD",
    "SocialParticipation",
    "ManagingMeds",
    "VigorousActivity",
    OUTCOME_NODE,
    *PROCESS_NODES,
]

NODE_ORDER = [*SUBSTANTIVE_NODES, OUTCOME_NODE, *PROCESS_NODES]

HYBRID_CARDINALITIES = {
    "Education": 7,
    "Orientation": 3,
    "LowerMobility": 3,
    "Hearing": 3,
    "Eyesight": 3,
    "CESD": 3,
    "SocialParticipation": 2,
    "ManagingMeds": 2,
    "VigorousActivity": 4,
    "Dementia": 2,
    "Proxy": 2,
    "SelfCompletion": 2,
    "Nurse": 2,
}

DISCRETE_BASELINE_CARDINALITIES = {
    **HYBRID_CARDINALITIES,
    **{node: 3 for node in CONTINUOUS_NODES},
}

MODEL_DIR_NAMES = {
    "discrete": "DiscreteBaseline",
    "continuous": "ContinuousBaseline",
    "hybrid": "HybridV3",
}

# Search settings preserve the frozen search implementation belonging to each model.
DEFAULT_BASELINE_MAX_ITER = 300
DEFAULT_HYBRID_MAX_ITER = 150
DEFAULT_TABU_LENGTH = 10
DEFAULT_MAX_PARENTS = 3
DEFAULT_IMPROVEMENT_TOLERANCE = 1e-10


# ======================================================================================
# GENERIC HELPERS
# ======================================================================================

def jsonable(value: Any) -> Any:
    """Convert common numpy/path/dataclass values to JSON-safe Python objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, sort_keys=True)
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def to_number(value: Any) -> float | None:
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


def valid_continuous(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> bool:
    x = to_number(value)
    if x is None:
        return False
    if minimum is not None and x < minimum:
        return False
    if maximum is not None and x > maximum:
        return False
    return True


def valid_integer(
    value: Any,
    *,
    allowed: tuple[int, ...] | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> bool:
    x = to_number(value)
    if x is None or not float(x).is_integer():
        return False
    xi = int(x)
    if allowed is not None and xi not in allowed:
        return False
    if minimum is not None and xi < minimum:
        return False
    if maximum is not None and xi > maximum:
        return False
    return True


def normalized_scalar(value: Any) -> tuple[str, Any]:
    if isinstance(value, StataMissingValue):
        return ("STATA_MISSING", str(value))
    try:
        if pd.isna(value):
            return ("NAN", None)
    except Exception:
        pass
    x = to_number(value)
    if x is not None:
        return ("NUM", x)
    return ("OTHER", str(value))


def count_series_changes(before: pd.Series, after: pd.Series) -> int:
    if len(before) != len(after):
        raise RuntimeError("Series length changed during imputation.")
    b = before.map(normalized_scalar).tolist()
    a = after.map(normalized_scalar).tolist()
    return int(sum(x != y for x, y in zip(b, a)))


def derive_seed(base_seed: int, *parts: int) -> int:
    seq = np.random.SeedSequence([int(base_seed), *map(int, parts), 99173])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def robust_scale_iqr(values: Sequence[float]) -> float:
    """Frozen robust scale rule from 05_continuous_representation_audit.py."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("Cannot fit Wealth asinh scale from an empty array.")
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr > 0:
        return iqr / 1.349
    mad = float(np.median(np.abs(x - np.median(x))))
    if mad > 0:
        return mad / 0.6744897501960817
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return sd if sd > 0 else 1.0


def elapsed_string(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


# ======================================================================================
# PROJECT / MODULE LOADING
# ======================================================================================

def find_project_root(explicit: str | None = None) -> Path:
    """Find repository root without requiring access to restricted ELSA data."""
    starts = [Path(explicit).expanduser().resolve()] if explicit else [
        Path(__file__).resolve().parent, Path.cwd().resolve()
    ]
    for start in starts:
        for root in (start, *start.parents):
            if (root / "src" / "imputation").is_dir() and (root / "src" / "hybrid_bn").is_dir():
                return root
    raise FileNotFoundError(
        "Could not find the GitHub repository containing src/imputation and "
        "src/hybrid_bn. Supply --project-root."
    )


def load_module_from_file(runtime_name: str, path: Path):
    """Load a frozen model source without rewriting its implementation."""
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(runtime_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[runtime_name] = module
    spec.loader.exec_module(module)
    return module


def load_all_imputers(root: Path) -> dict[str, Any]:
    """Import the 13 unmodified variable-specific imputation modules."""
    repo = str(root)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import importlib
    return {
        key: importlib.import_module(f"src.imputation.{Path(filename).stem}")
        for key, filename in EXPECTED_IMPUTER_FILES.items()
    }


# Use the consolidated RQ1 model entry points. Hybrid V3 includes its corrected
# CC scorer inside the HybridV3_CDv3Candidate package; no external scorer files.
def resolve_model_sources(root: Path) -> dict[str, Path]:
    folder = root / "src" / "hybrid_bn"
    names = {
        "discrete": ("SLdiscrete.py", "discrete.py"),
        "continuous": ("SLcontinuous.py", "continuous.py"),
        "hybrid": ("SLhybrid.py",),
    }
    found = {}
    for key, candidates in names.items():
        hits = [folder / name for name in candidates if (folder / name).is_file()]
        if len(hits) > 1:
            # Prefer historical filenames when both exist; do not guess across variants.
            hits = [hits[0]]
        if hits:
            found[key] = hits[0]
    return found


def load_models(root: Path, requested: set[str]) -> dict[str, Any]:
    resolved = resolve_model_sources(root)
    needed = set(requested)
    missing = sorted(needed - resolved.keys())
    if missing:
        raise FileNotFoundError(
            "RQ2 uses frozen model implementations not found in src/hybrid_bn: "
            f"{missing}. The GitHub model API/version must be verified before "
            "replacing these with differently named implementations."
        )
    if "hybrid" in requested:
        package_dir = root / "src" / "hybrid_bn" / "HybridV3_CDv3Candidate"
        for required in (
            "__init__.py", "cc.py", "cd.py", "common.py", "dc.py", "dd.py",
            "engine.py", "preprocessing.py", "search.py",
        ):
            if not (package_dir / required).is_file():
                raise FileNotFoundError(
                    f"RQ2's frozen Hybrid scorer requires {package_dir / required}"
                )
    return {
        key: load_module_from_file(f"rq2_pipeline_model_{key}", resolved[key])
        for key in sorted(requested)
    }


def find_completeness_file(root: Path) -> Path:
    """Find locally generated participant-level cohort eligibility data."""
    filename = "12i_participant_wave_completeness_contemporaneous.csv"
    preferred = [
        root / "DataProcessing" / "missingness_final_audit" / filename,
        root / "DataProcessing" / "Outputs" / "12_missingness_final_audit" / filename,
        root / "DataMissingness" / "Outputs" / filename,
        root / "Datasets" / filename,
    ]
    for path in preferred:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Missing local cohort file {filename}. Generate it with the RQ2 "
        "missingness audit. Never commit this participant-level file."
    )


# ======================================================================================
# SOURCE COLUMN DEFINITIONS / FROZEN COHORT
# ======================================================================================

def target_column_map(wave: int) -> dict[str, str]:
    return {
        "Age": f"r{wave}agey",
        "Education": "raeduc_e",
        "Wealth": f"h{wave}atotb",
        "Recall": f"r{wave}tr20",
        "Orientation": f"r{wave}orient",
        "CASP19": f"r{wave}casp19",
        "FriendSupport": f"r{wave}fsupport6",
        "Grip": f"r{wave}gripsum",
        "LowerMobility": f"r{wave}lowermob",
        "BMI": f"r{wave}mbmi",
        "Hearing": f"r{wave}hearing",
        "Eyesight": f"r{wave}sight",
        "CESD": f"r{wave}cesd",
        "Loneliness": f"r{wave}lnlys3",
        "SocialParticipation": f"r{wave}socyr",
        "ManagingMeds": f"r{wave}medsa",
        "VigorousActivity": f"r{wave}vgactx_e",
    }


def protected_column_map(wave: int) -> dict[str, str]:
    return {
        "Dementia": f"r{wave}alzdeme",
        "Proxy": f"r{wave}proxy",
        "SelfCompletion": f"inw{wave}sc",
        "Nurse": f"inw{wave}n",
    }


def target_validator(variable: str) -> Callable[[Any], bool]:
    validators: dict[str, Callable[[Any], bool]] = {
        "Age": lambda v: valid_continuous(v, minimum=50.0, maximum=120.0),
        "Education": lambda v: valid_integer(v, allowed=(-15, 1, 2, 3, 4, 5, 6)),
        "Wealth": lambda v: valid_continuous(v),
        "Recall": lambda v: valid_integer(v, minimum=0, maximum=20),
        "Orientation": lambda v: valid_integer(v, minimum=0, maximum=4),
        "CASP19": lambda v: valid_integer(v, minimum=0, maximum=57),
        "FriendSupport": lambda v: valid_continuous(v, minimum=1.0, maximum=4.0),
        "Grip": lambda v: valid_continuous(v, minimum=0.0, maximum=100.0),
        "LowerMobility": lambda v: valid_integer(v, allowed=(0, 1, 2, 3, 4)),
        "BMI": lambda v: valid_continuous(v, minimum=5.0, maximum=100.0),
        "Hearing": lambda v: valid_integer(v, allowed=(1, 2, 3, 4, 5)),
        "Eyesight": lambda v: valid_integer(v, allowed=(1, 2, 3, 4, 5, 6)),
        "CESD": lambda v: valid_integer(v, minimum=0, maximum=8),
        "Loneliness": lambda v: valid_continuous(v, minimum=1.0, maximum=3.0),
        "SocialParticipation": lambda v: valid_integer(v, allowed=(0, 1)),
        "ManagingMeds": lambda v: valid_integer(v, allowed=(0, 1)),
        "VigorousActivity": lambda v: valid_integer(v, allowed=(2, 3, 4, 5)),
    }
    return validators[variable]


def valid_mask_for_target(series: pd.Series, variable: str) -> pd.Series:
    return series.map(target_validator(variable)).astype(bool)


def gather_required_columns(modules: dict[str, Any], waves: tuple[int, ...]) -> set[str]:
    """Mirror the validated integrity-audit source-column set."""
    m = modules
    required: set[str] = {"idauniq", "raeduc_e", "raedyrs_e", "ragender"}

    for wave in waves:
        required.update(target_column_map(wave).values())
        required.update(protected_column_map(wave).values())

        required.update({
            m["recall"].TOTAL(wave),
            m["recall"].IMMEDIATE(wave),
            m["recall"].DELAYED(wave),
            m["recall"].AGE(wave),
        })

        required.add(m["orientation"].ORIENT(wave))
        for builder in m["orientation"].COMPONENT_BUILDERS.values():
            required.add(builder(wave))

        required.add(m["casp19"].CASP19(wave))
        for builder in m["casp19"].COMPONENT_BUILDERS.values():
            required.add(builder(wave))

        required.add(m["friend_support"].FSUPPORT6(wave))
        for builder in m["friend_support"].COMPONENT_BUILDERS.values():
            required.add(builder(wave))

        required.update({
            m["bmi"].BMI(wave),
            m["bmi"].HEIGHT(wave),
            m["bmi"].WEIGHT(wave),
            m["bmi"].LOWER_MOBILITY(wave),
        })
        for prior_wave in (2, 4, 6):
            if prior_wave < wave:
                required.add(m["bmi"].HEIGHT(prior_wave))

        required.update({
            m["grip_strength"].GRIP_SUM(wave),
            m["grip_strength"].DOM_HAND(wave),
            m["grip_strength"].LEFT_MAX(wave),
            m["grip_strength"].RIGHT_MAX(wave),
            m["grip_strength"].AGE(wave),
            m["grip_strength"].GENDER_COL,
        })

        required.update({
            m["cesd"].CESD(wave),
            m["cesd"].LOWER_MOBILITY(wave),
            m["loneliness"].LONELINESS(wave),
            m["loneliness"].LOWER_MOBILITY(wave),
            m["social_participation"].SOCIAL_PARTICIPATION(wave),
            m["vigorous_activity"].VIG(wave),
            m["vigorous_activity"].LOWER_MOBILITY(wave),
            m["wealth"].WEALTH_COL(wave),
        })

    return required


def canonical_retained_ids(
    completeness: pd.DataFrame,
    raw_df: pd.DataFrame,
    wave: int,
) -> set[Any]:
    wc = completeness[completeness["Wave"].eq(wave)].copy()
    base = wc[wc["Missing_Nodes"].le(MAX_MISSING_PREDICTORS)][["idauniq"]].copy()
    dementia_col = f"r{wave}alzdeme"
    dementia = raw_df[["idauniq", dementia_col]].copy()
    dementia["DementiaValid"] = dementia[dementia_col].map(
        lambda value: valid_integer(value, allowed=(0, 1))
    )
    merged = base.merge(
        dementia[["idauniq", "DementiaValid"]],
        on="idauniq",
        how="left",
        validate="one_to_one",
    )
    retained = merged[merged["DementiaValid"].fillna(False)]
    return set(retained["idauniq"].tolist())


# ======================================================================================
# FINAL IMPUTER CHAIN
# ======================================================================================

def instantiate_chain(modules: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "key": "small_categorical",
            "targets": ["LowerMobility", "ManagingMeds", "Hearing", "Eyesight"],
            "factory": lambda: modules["small_categorical"].SmallCategoricalStochasticImputer(),
            "fit_uses_wave": True,
        },
        {
            "key": "education",
            "targets": ["Education"],
            "factory": lambda: modules["education"].EducationConditionalStochasticImputer(),
            "fit_uses_wave": False,
        },
        {
            "key": "wealth",
            "targets": ["Wealth"],
            "factory": lambda: modules["wealth"].WealthEducationConditionalImputer(),
            "fit_uses_wave": True,
        },
        {
            "key": "recall",
            "targets": ["Recall"],
            "factory": lambda: modules["recall"].RecallStochasticImputer(),
            "fit_uses_wave": True,
        },
        {
            "key": "orientation",
            "targets": ["Orientation"],
            "factory": lambda: modules["orientation"].OrientationStochasticImputer(),
            "fit_uses_wave": True,
        },
        {
            "key": "cesd",
            "targets": ["CESD"],
            "factory": lambda: modules["cesd"].CESDStochasticImputer(),
            "fit_uses_wave": True,
        },
        {
            "key": "casp19",
            "targets": ["CASP19"],
            "factory": lambda: modules["casp19"].CASP19StochasticImputer(),
            "fit_uses_wave": True,
        },
        {
            "key": "loneliness",
            "targets": ["Loneliness"],
            "factory": lambda: modules["loneliness"].LonelinessStochasticImputer(),
            "fit_uses_wave": True,
        },
        {
            "key": "friend_support",
            "targets": ["FriendSupport"],
            "factory": lambda: modules["friend_support"].FriendSupportStochasticImputer(),
            "fit_uses_wave": True,
        },
        {
            "key": "social_participation",
            "targets": ["SocialParticipation"],
            "factory": lambda: modules["social_participation"].SocialParticipationStochasticImputer(),
            "fit_uses_wave": True,
        },
        {
            "key": "bmi",
            "targets": ["BMI"],
            "factory": lambda: modules["bmi"].BMIStochasticImputer(),
            "fit_uses_wave": True,
        },
        {
            "key": "grip_strength",
            "targets": ["Grip"],
            "factory": lambda: modules["grip_strength"].GripStrengthStochasticImputer(),
            "fit_uses_wave": True,
        },
        {
            "key": "vigorous_activity",
            "targets": ["VigorousActivity"],
            "factory": lambda: modules["vigorous_activity"].VigorousActivityStochasticImputer(),
            "fit_uses_wave": True,
        },
    ]


def assert_protected_unchanged(
    before: pd.DataFrame,
    after: pd.DataFrame,
    wave: int,
    label: str,
) -> None:
    for col in protected_column_map(wave).values():
        changes = count_series_changes(
            before[col].reset_index(drop=True),
            after[col].reset_index(drop=True),
        )
        if changes:
            raise RuntimeError(
                f"{label}: protected column {col!r} changed in {changes} rows."
            )


def assert_observed_targets_preserved(
    original: pd.DataFrame,
    completed: pd.DataFrame,
    wave: int,
    label: str,
) -> None:
    cols = target_column_map(wave)
    for variable, col in cols.items():
        mask = valid_mask_for_target(original[col], variable).to_numpy(bool)
        if not np.any(mask):
            continue
        b = pd.to_numeric(original.loc[mask, col], errors="coerce").to_numpy(float)
        a = pd.to_numeric(completed.loc[mask, col], errors="coerce").to_numpy(float)
        if not np.allclose(b, a, rtol=0.0, atol=1e-12, equal_nan=False):
            changed = int(np.sum(~np.isclose(b, a, rtol=0.0, atol=1e-12)))
            raise RuntimeError(
                f"{label}: originally observed {variable} changed in {changed} rows."
            )


def assert_final_targets_complete(df: pd.DataFrame, wave: int, label: str) -> None:
    cols = target_column_map(wave)
    failures = []
    for variable, col in cols.items():
        valid = valid_mask_for_target(df[col], variable)
        bad = int((~valid).sum())
        if bad:
            failures.append(f"{variable}={bad}")
    if failures:
        raise RuntimeError(
            f"{label}: final imputation left invalid/missing substantive nodes: "
            + ", ".join(failures)
        )


def complete_train_test(
    *,
    train_original: pd.DataFrame,
    test_original: pd.DataFrame,
    wave: int,
    fold: int,
    completion: int,
    base_seed: int,
    chain: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Sequentially fit every imputer on current TRAIN only, then fill TRAIN/TEST."""
    train = train_original.copy(deep=True)
    test = test_original.copy(deep=True)
    stage_rows: list[dict[str, Any]] = []

    protected_train = train_original[list(protected_column_map(wave).values())].copy()
    protected_test = test_original[list(protected_column_map(wave).values())].copy()

    for stage_number, stage in enumerate(chain, start=1):
        started = time.perf_counter()
        imputer = stage["factory"]()

        # Count only the target(s) owned by this stage before filling.
        before_train_missing = 0
        before_test_missing = 0
        for variable in stage["targets"]:
            col = target_column_map(wave)[variable]
            before_train_missing += int((~valid_mask_for_target(train[col], variable)).sum())
            before_test_missing += int((~valid_mask_for_target(test[col], variable)).sum())

        if stage["fit_uses_wave"]:
            imputer.fit(train, wave=wave)
        else:
            imputer.fit(train)

        train_seed = derive_seed(
            base_seed, wave, fold, completion, stage_number, 1
        )
        test_seed = derive_seed(
            base_seed, wave, fold, completion, stage_number, 2
        )

        train = imputer.transform(
            train,
            seed=train_seed,
            copy=True,
            return_audit=False,
            dataset_label=f"W{wave}_F{fold}_M{completion}_train_{stage['key']}",
        )
        test = imputer.transform(
            test,
            seed=test_seed,
            copy=True,
            return_audit=False,
            dataset_label=f"W{wave}_F{fold}_M{completion}_test_{stage['key']}",
        )

        # No stage is permitted to alter Dementia/process columns.
        assert_protected_unchanged(
            protected_train.reset_index(drop=True),
            train[list(protected_column_map(wave).values())].reset_index(drop=True),
            wave,
            f"W{wave} F{fold} M{completion} train after {stage['key']}",
        )
        assert_protected_unchanged(
            protected_test.reset_index(drop=True),
            test[list(protected_column_map(wave).values())].reset_index(drop=True),
            wave,
            f"W{wave} F{fold} M{completion} test after {stage['key']}",
        )

        after_train_missing = 0
        after_test_missing = 0
        for variable in stage["targets"]:
            col = target_column_map(wave)[variable]
            after_train_missing += int((~valid_mask_for_target(train[col], variable)).sum())
            after_test_missing += int((~valid_mask_for_target(test[col], variable)).sum())

        stage_rows.append({
            "stage": stage_number,
            "imputer": stage["key"],
            "targets": list(stage["targets"]),
            "train_seed": train_seed,
            "test_seed": test_seed,
            "train_missing_before": before_train_missing,
            "train_missing_after": after_train_missing,
            "test_missing_before": before_test_missing,
            "test_missing_after": after_test_missing,
            "runtime_seconds": float(time.perf_counter() - started),
        })

    assert_final_targets_complete(train, wave, f"W{wave} F{fold} M{completion} TRAIN")
    assert_final_targets_complete(test, wave, f"W{wave} F{fold} M{completion} TEST")
    assert_observed_targets_preserved(
        train_original.reset_index(drop=True),
        train.reset_index(drop=True),
        wave,
        f"W{wave} F{fold} M{completion} TRAIN",
    )
    assert_observed_targets_preserved(
        test_original.reset_index(drop=True),
        test.reset_index(drop=True),
        wave,
        f"W{wave} F{fold} M{completion} TEST",
    )

    return train, test, stage_rows


# ======================================================================================
# CANONICAL 21-NODE DATASET + LOCKED REPRESENTATIONS
# ======================================================================================

def _numeric_required(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        bad = int(np.sum(~np.isfinite(values)))
        raise ValueError(f"{name} contains {bad} non-finite values.")
    return values


def build_canonical_node_frame(completed_raw: pd.DataFrame, wave: int) -> pd.DataFrame:
    out = pd.DataFrame(index=completed_raw.index)
    out["idauniq"] = completed_raw["idauniq"].to_numpy()

    for node, source in target_column_map(wave).items():
        values = _numeric_required(completed_raw[source], source)
        out[node] = values

    for node, source in protected_column_map(wave).items():
        values = _numeric_required(completed_raw[source], source)
        if not np.all(values == np.floor(values)) or values.min() < 0 or values.max() > 1:
            unique = sorted(set(values.tolist()))
            raise ValueError(
                f"Process/outcome column {source!r} must be binary 0/1 in the frozen cohort; "
                f"observed values={unique[:20]}"
            )
        out[node] = values.astype(np.int64)

    out = out[["idauniq", *NODE_ORDER]].reset_index(drop=True)
    if out["idauniq"].duplicated().any():
        raise RuntimeError("Canonical node frame contains duplicate idauniq values.")
    return out


def locked_native_discrete_frame(canonical: pd.DataFrame) -> pd.DataFrame:
    """Return all 21 nodes, applying only the frozen native-discrete recodes."""
    frame = canonical[NODE_ORDER].copy()

    # Education: preserve Other as its own nominal state; numeric labels are internal only.
    educ_map = {-15: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    education_raw = _numeric_required(frame["Education"], "Education")
    if not np.all(education_raw == np.floor(education_raw)):
        raise ValueError("Education contains non-integer values.")
    mapped = pd.Series(education_raw.astype(int)).map(educ_map)
    if mapped.isna().any():
        raise ValueError("Education contains an unexpected raw state.")
    frame["Education"] = mapped.to_numpy(np.int64)

    orientation = _numeric_required(frame["Orientation"], "Orientation").astype(int)
    frame["Orientation"] = np.select(
        [orientation <= 2, orientation == 3, orientation == 4],
        [0, 1, 2],
        default=-1,
    ).astype(np.int64)

    cesd = _numeric_required(frame["CESD"], "CESD").astype(int)
    frame["CESD"] = np.select(
        [cesd <= 1, (cesd >= 2) & (cesd <= 3), cesd >= 4],
        [0, 1, 2],
        default=-1,
    ).astype(np.int64)

    mobility = _numeric_required(frame["LowerMobility"], "LowerMobility").astype(int)
    frame["LowerMobility"] = np.select(
        [mobility == 0, mobility == 1, mobility >= 2],
        [0, 1, 2],
        default=-1,
    ).astype(np.int64)

    hearing = _numeric_required(frame["Hearing"], "Hearing").astype(int)
    frame["Hearing"] = np.select(
        [hearing <= 2, hearing == 3, hearing >= 4],
        [0, 1, 2],
        default=-1,
    ).astype(np.int64)

    eyesight = _numeric_required(frame["Eyesight"], "Eyesight").astype(int)
    frame["Eyesight"] = np.select(
        [eyesight <= 2, eyesight == 3, eyesight >= 4],
        [0, 1, 2],
        default=-1,
    ).astype(np.int64)

    vigorous = _numeric_required(frame["VigorousActivity"], "VigorousActivity").astype(int)
    vigorous_map = {5: 0, 4: 1, 3: 2, 2: 3}
    mapped_vig = pd.Series(vigorous).map(vigorous_map)
    if mapped_vig.isna().any():
        raise ValueError("VigorousActivity contains an unexpected raw state.")
    frame["VigorousActivity"] = mapped_vig.to_numpy(np.int64)

    for node in ("SocialParticipation", "ManagingMeds", "Dementia", *PROCESS_NODES):
        values = _numeric_required(frame[node], node)
        if not np.all(values == np.floor(values)) or values.min() < 0 or values.max() > 1:
            raise ValueError(f"{node} must be binary 0/1.")
        frame[node] = values.astype(np.int64)

    # Validate frozen cardinalities exactly.
    for node, cardinality in HYBRID_CARDINALITIES.items():
        values = frame[node].to_numpy(np.int64)
        if values.min() < 0 or values.max() >= cardinality:
            raise ValueError(
                f"{node} recoding produced an out-of-range state for cardinality={cardinality}."
            )

    # Continuous nodes are numeric/finitely valued but otherwise unchanged here.
    for node in CONTINUOUS_NODES:
        frame[node] = _numeric_required(frame[node], node)

    return frame[NODE_ORDER]


def fit_tie_aware_three_bins(values: Sequence[float]) -> dict[str, Any]:
    """Fit two cut points near 1/3 and 2/3 without ever splitting tied values.

    Candidate cuts lie between adjacent unique training values.  An O(U) dynamic
    scan selects the distinct pair minimising absolute deviation of cumulative
    training mass from 1/3 and 2/3.
    """
    x = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Quantile discretizer received non-finite training values.")

    unique, counts = np.unique(x, return_counts=True)
    if len(unique) < 3:
        raise ValueError(
            "A three-state baseline requires at least three distinct training values."
        )

    cumulative = np.cumsum(counts)[:-1] / float(len(x))  # cuts after unique[0..-2]
    first_cost = np.abs(cumulative - 1.0 / 3.0)
    second_cost = np.abs(cumulative - 2.0 / 3.0)

    # Need i < j. Track best first-cut candidate available before each j.
    best_i = 0
    best_i_cost = float(first_cost[0])
    best_pair: tuple[int, int] | None = None
    best_total = float("inf")

    for j in range(1, len(cumulative)):
        total = best_i_cost + float(second_cost[j])
        if total < best_total - 1e-15:
            best_total = total
            best_pair = (best_i, j)
        # Candidate j may serve as the first cut only for a later second cut.
        if float(first_cost[j]) < best_i_cost - 1e-15:
            best_i = j
            best_i_cost = float(first_cost[j])

    if best_pair is None:
        raise RuntimeError("Could not choose two distinct tie-aware quantile cuts.")

    i, j = best_pair

    def midpoint(a: float, b: float) -> float:
        # Stable midpoint that avoids needless overflow from (a+b)/2.
        return float(a + (b - a) / 2.0)

    cut1 = midpoint(float(unique[i]), float(unique[i + 1]))
    cut2 = midpoint(float(unique[j]), float(unique[j + 1]))
    if cut1 >= cut2:
        raise RuntimeError("Tie-aware discretizer produced non-increasing cut points.")

    states = np.searchsorted(np.asarray([cut1, cut2]), x, side="right")
    counts3 = np.bincount(states.astype(np.int64), minlength=3)
    if np.any(counts3 == 0):
        raise RuntimeError("Tie-aware discretizer produced an empty training state.")

    return {
        "cutpoints": [cut1, cut2],
        "train_counts": counts3.astype(int).tolist(),
        "train_proportions": (counts3 / counts3.sum()).astype(float).tolist(),
        "distinct_values": int(len(unique)),
        "objective_absolute_deviation": float(best_total),
    }


def apply_three_bins(values: Sequence[float], metadata: Mapping[str, Any]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Quantile discretizer received non-finite values at transform time.")
    cuts = np.asarray(metadata["cutpoints"], dtype=float)
    if cuts.shape != (2,) or cuts[0] >= cuts[1]:
        raise ValueError("Invalid saved three-bin cutpoints.")
    return np.searchsorted(cuts, x, side="right").astype(np.int64)


def build_model_representations(
    train_canonical: pd.DataFrame,
    test_canonical: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    train_locked = locked_native_discrete_frame(train_canonical)
    test_locked = locked_native_discrete_frame(test_canonical)

    # Wealth transform is fitted from completed TRAIN only.
    wealth_scale = robust_scale_iqr(train_locked["Wealth"].to_numpy(float))
    if not math.isfinite(wealth_scale) or wealth_scale <= 0:
        raise ValueError(f"Invalid Wealth asinh scale: {wealth_scale}")

    continuous_train = train_locked.copy()
    continuous_test = test_locked.copy()
    continuous_train["Wealth"] = np.arcsinh(
        continuous_train["Wealth"].to_numpy(float) / wealth_scale
    )
    continuous_test["Wealth"] = np.arcsinh(
        continuous_test["Wealth"].to_numpy(float) / wealth_scale
    )

    # Same numeric representation feeds both the Gaussian-forcing baseline and Hybrid.
    hybrid_train = continuous_train.copy()
    hybrid_test = continuous_test.copy()

    # Discrete baseline: only the continuous-space nodes are train-fitted into 3 bins.
    discrete_train = train_locked.copy()
    discrete_test = test_locked.copy()
    quantile_meta: dict[str, Any] = {}
    for node in CONTINUOUS_NODES:
        meta = fit_tie_aware_three_bins(train_locked[node].to_numpy(float))
        quantile_meta[node] = meta
        discrete_train[node] = apply_three_bins(train_locked[node].to_numpy(float), meta)
        discrete_test[node] = apply_three_bins(test_locked[node].to_numpy(float), meta)

    for node, cardinality in DISCRETE_BASELINE_CARDINALITIES.items():
        values = discrete_train[node].to_numpy(np.int64)
        if values.min() < 0 or values.max() >= cardinality:
            raise ValueError(f"Discrete baseline TRAIN has invalid state for {node}.")
        test_values = discrete_test[node].to_numpy(np.int64)
        if test_values.min() < 0 or test_values.max() >= cardinality:
            raise ValueError(f"Discrete baseline TEST has invalid state for {node}.")
        discrete_train[node] = values
        discrete_test[node] = test_values

    metadata = {
        "wealth_asinh": {
            "formula": "asinh(Wealth / c)",
            "scale_rule": "IQR/1.349; fallback MAD/0.6744897501960817; fallback sample SD; fallback 1",
            "c_train_only": float(wealth_scale),
        },
        "discrete_baseline_quantile_bins": quantile_meta,
        "native_discrete_cardinalities": dict(HYBRID_CARDINALITIES),
        "discrete_baseline_cardinalities": dict(DISCRETE_BASELINE_CARDINALITIES),
        "continuous_nodes": list(CONTINUOUS_NODES),
        "native_discrete_nodes": list(NATIVE_DISCRETE_NODES),
        "node_order": list(NODE_ORDER),
    }

    train = {
        "discrete": discrete_train[NODE_ORDER],
        "continuous": continuous_train[NODE_ORDER],
        "hybrid": hybrid_train[NODE_ORDER],
    }
    test = {
        "discrete": discrete_test[NODE_ORDER],
        "continuous": continuous_test[NODE_ORDER],
        "hybrid": hybrid_test[NODE_ORDER],
    }
    return train, test, metadata


# ======================================================================================
# GRAPH LEARNING
# ======================================================================================

def save_edges(edges: Sequence[tuple[str, str]], path: Path) -> None:
    frame = pd.DataFrame(list(edges), columns=["Parent", "Child"])
    if not frame.empty:
        frame = frame.sort_values(["Parent", "Child"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def learn_one_graph(
    *,
    model_key: str,
    train_data: pd.DataFrame,
    modules: dict[str, Any],
    root: Path,
    model_out: Path,
    baseline_max_iter: int,
    hybrid_max_iter: int,
    tabu_length: int,
    max_parents: int,
    improvement_tolerance: float,
    save_diagnostics: bool,
    label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    model_out.mkdir(parents=True, exist_ok=True)

    if model_key == "discrete":
        module = modules["discrete"]
        scorer = module.DiscreteScoringEngine(
            train_data,
            cardinalities=DISCRETE_BASELINE_CARDINALITIES,
        )
        search = module.TabuGreedyHillClimbing(
            NODE_ORDER,
            scorer,
            tabu_length=tabu_length,
            max_iter=baseline_max_iter,
            max_parents=max_parents,
            improvement_tolerance=improvement_tolerance,
        )
        search_name = "SLdiscrete.TabuGreedyHillClimbing"

    elif model_key == "continuous":
        module = modules["continuous"]
        scorer = module.ContinuousScoringEngine(train_data)
        search = module.TabuGreedyHillClimbing(
            NODE_ORDER,
            scorer,
            tabu_length=tabu_length,
            max_iter=baseline_max_iter,
            max_parents=max_parents,
            improvement_tolerance=improvement_tolerance,
        )
        search_name = "SLcontinuous.TabuGreedyHillClimbing"

    elif model_key == "hybrid":
        module = modules["hybrid"]
        scorer = module.HybridV3ScoringEngine(
            train_data,
            discrete_nodes=NATIVE_DISCRETE_NODES,
            continuous_nodes=CONTINUOUS_NODES,
            cardinalities=HYBRID_CARDINALITIES,
        )
        search = module.AtomicAdd2TabuGreedyHillClimbing(
            NODE_ORDER,
            scorer,
            tabu_length=tabu_length,
            max_iter=hybrid_max_iter,
            max_parents=max_parents,
            improvement_tolerance=improvement_tolerance,
            enable_add2=False,
            progress_label=f"Hybrid V3 {label}",
        )
        search_name = "HybridV3.TabuGreedyHillClimbing_no_add2"

    else:
        raise KeyError(model_key)

    edges = search.search()
    graph = nx.DiGraph()
    graph.add_nodes_from(NODE_ORDER)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError(f"{label} {model_key}: learned graph is not a DAG.")
    if any(graph.in_degree(node) > max_parents for node in NODE_ORDER):
        raise RuntimeError(f"{label} {model_key}: learned graph exceeded max_parents.")

    save_edges(edges, model_out / "edges.csv")

    score_history = getattr(search, "score_history_", [])
    pd.DataFrame(
        {
            "Step": np.arange(len(score_history), dtype=int),
            "Score": np.asarray(score_history, dtype=float),
        }
    ).to_csv(model_out / "score_history.csv", index=False)

    summary: dict[str, Any] = {
        "model": model_key,
        "search_class": search_name,
        "best_score": float(getattr(search, "best_score_", np.nan)),
        "edge_count": int(len(edges)),
        "iterations_recorded": int(max(len(score_history) - 1, 0)),
        "max_parents": int(max_parents),
        "tabu_length": int(tabu_length),
        "improvement_tolerance": float(improvement_tolerance),
        "runtime_seconds": float(time.perf_counter() - started),
    }

    if model_key == "hybrid":
        try:
            summary["hybrid_preprocessing"] = scorer.preprocessing_metadata()
        except Exception as exc:
            summary["hybrid_preprocessing_error"] = f"{type(exc).__name__}: {exc}"

        if save_diagnostics:
            # These are diagnostic outputs only; structure learning is already complete.
            for filename, method_name in (
                ("score_info.csv", "score_info_dataframe"),
                ("cd_laplace_diagnostics.csv", "laplace_diagnostics_dataframe"),
                ("cd_model_diagnostics.csv", "cd_model_diagnostics_dataframe"),
                ("dc_mixture_diagnostics.csv", "mixture_diagnostics_dataframe"),
            ):
                try:
                    frame = getattr(scorer, method_name)()
                    frame.to_csv(model_out / filename, index=False)
                except Exception as exc:
                    summary.setdefault("diagnostic_export_errors", []).append(
                        f"{method_name}: {type(exc).__name__}: {exc}"
                    )

            # CC diagnostics are available only if at least one CC family was requested.
            try:
                cc_engine = getattr(getattr(scorer, "_cc", None), "engine", None)
                if cc_engine is not None:
                    if hasattr(cc_engine, "contextual_refinement_dataframe"):
                        cc_engine.contextual_refinement_dataframe().to_csv(
                            model_out / "cc_contextual_refinement.csv", index=False
                        )
                    diagnostics = getattr(cc_engine, "diagnostics", None)
                    if isinstance(diagnostics, dict):
                        rows = []
                        for (child, parents), value in diagnostics.items():
                            row = {
                                "child": child,
                                "parents": "|".join(parents),
                            }
                            if is_dataclass(value):
                                row.update(asdict(value))
                            elif isinstance(value, Mapping):
                                row.update(value)
                            else:
                                row["diagnostic"] = repr(value)
                            rows.append(row)
                        pd.DataFrame(rows).to_csv(
                            model_out / "cc_diagnostics.csv", index=False
                        )
            except Exception as exc:
                summary.setdefault("diagnostic_export_errors", []).append(
                    f"CC diagnostics: {type(exc).__name__}: {exc}"
                )

    write_json(model_out / "search_summary.json", summary)
    return summary


# ======================================================================================
# FOLD ASSIGNMENTS / RUN CONFIG
# ======================================================================================

def make_or_load_fold_assignments(
    *,
    wave_original: pd.DataFrame,
    wave: int,
    folds: int,
    base_seed: int,
    path: Path,
) -> pd.DataFrame:
    dementia_col = f"r{wave}alzdeme"
    base = wave_original[["idauniq", dementia_col]].copy()
    base["Dementia"] = pd.to_numeric(base[dementia_col], errors="raise").astype(int)
    base = base[["idauniq", "Dementia"]].sort_values("idauniq").reset_index(drop=True)

    if path.exists():
        existing = pd.read_csv(path)
        required = {"idauniq", "Dementia", "Fold"}
        if set(existing.columns) != required:
            raise RuntimeError(f"Existing fold assignment has unexpected columns: {path}")
        existing = existing.sort_values("idauniq").reset_index(drop=True)
        if not np.array_equal(existing["idauniq"].to_numpy(), base["idauniq"].to_numpy()):
            raise RuntimeError(f"Existing fold assignment IDs do not match frozen W{wave} cohort.")
        if not np.array_equal(existing["Dementia"].to_numpy(int), base["Dementia"].to_numpy(int)):
            raise RuntimeError(f"Existing fold assignment Dementia values changed for W{wave}.")
        observed_folds = sorted(existing["Fold"].unique().tolist())
        if observed_folds != list(range(1, folds + 1)):
            raise RuntimeError(
                f"Existing W{wave} fold assignment was not built with K={folds}: {observed_folds}"
            )
        return existing

    y = base["Dementia"].to_numpy(int)
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=derive_seed(base_seed, wave, 5001),
    )
    fold_values = np.empty(len(base), dtype=np.int64)
    dummy = np.zeros((len(base), 1), dtype=float)
    for fold_index, (_, test_idx) in enumerate(splitter.split(dummy, y), start=1):
        fold_values[test_idx] = fold_index
    base["Fold"] = fold_values

    path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(path, index=False)
    return base


def build_code_hashes(root: Path) -> dict[str, str]:
    """Hash the exact imputer/model sources for resume provenance checks."""
    hashes: dict[str, str] = {}
    for filename in EXPECTED_IMPUTER_FILES.values():
        path = root / "src" / "imputation" / filename
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    resolved = resolve_model_sources(root)
    for key in EXPECTED_MODEL_FILES:
        if key in resolved:
            path = resolved[key]
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    package = root / "src" / "hybrid_bn" / "HybridV3_CDv3Candidate"
    for filename in (
        "__init__.py", "cc.py", "cd.py", "common.py", "dc.py", "dd.py",
        "engine.py", "preprocessing.py", "search.py",
    ):
        path = package / filename
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def locked_design_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "folds": int(args.folds),
        "fold_method": "StratifiedKFold on observed Dementia; shuffle=True; fixed per wave",
        "base_seed": int(args.base_seed),
        "frozen_cohort_max_missing_predictors": int(MAX_MISSING_PREDICTORS),
        "frozen_retained_counts": dict(FROZEN_RETAINED_COUNTS),
        "node_order": list(NODE_ORDER),
        "substantive_nodes": list(SUBSTANTIVE_NODES),
        "process_nodes": list(PROCESS_NODES),
        "outcome_node": OUTCOME_NODE,
        "continuous_nodes": list(CONTINUOUS_NODES),
        "native_discrete_nodes": list(NATIVE_DISCRETE_NODES),
        "hybrid_cardinalities": dict(HYBRID_CARDINALITIES),
        "discrete_baseline_cardinalities": dict(DISCRETE_BASELINE_CARDINALITIES),
        "wealth_transform": "asinh(Wealth / c), c fitted from completed TRAIN only by robust IQR/MAD/SD rule",
        "discrete_baseline_bins": "3-state tie-aware TRAIN-fitted quantile approximation for continuous-space nodes",
        "baseline_search": {
            "class": "frozen baseline TabuGreedyHillClimbing",
            "max_iter": int(args.baseline_max_iter),
            "tabu_length": int(args.tabu_length),
            "max_parents": int(args.max_parents),
            "improvement_tolerance": float(args.improvement_tolerance),
        },
        "hybrid_search": {
            "class": "AtomicAdd2TabuGreedyHillClimbing",
            "max_iter": int(args.hybrid_max_iter),
            "tabu_length": int(args.tabu_length),
            "max_parents": int(args.max_parents),
            "improvement_tolerance": float(args.improvement_tolerance),
            "enable_add2": False,
        },
        "imputation": {
            "split_before_fill": True,
            "fit_on_train_only": True,
            "train_test_draws_use_distinct_deterministic_seeds": True,
            "dementia_imputed": False,
            "dementia_used_as_imputation_predictor": False,
            "process_nodes_imputed": False,
        },
    }


def establish_run_config(
    *,
    output_root: Path,
    root: Path,
    args: argparse.Namespace,
    read_only: bool = False,
) -> None:
    path = output_root / "run_config.json"
    locked = locked_design_payload(args)
    hashes = build_code_hashes(root)

    if path.exists():
        existing = read_json(path)
        if existing.get("locked_design") != jsonable(locked):
            raise RuntimeError(
                "Existing Results/MasterPipeline/CV_Ensembles/run_config.json has a different "
                "locked design. Use a fresh output directory or restore the original settings."
            )
        if existing.get("code_sha256") != hashes:
            raise RuntimeError(
                "Model/imputer source hashes differ from the existing run_config.json. "
                "Do not mix results produced by different code versions in one ensemble folder."
            )
        if read_only:
            return
        history = list(existing.get("m_max_requested_history", []))
        if int(args.m_max) not in history:
            history.append(int(args.m_max))
        existing["m_max_requested_history"] = sorted(set(history))
        wave_history = [tuple(map(int, w)) for w in existing.get("waves_requested_history", [])]
        requested_waves = tuple(map(int, args.waves))
        if requested_waves not in wave_history:
            wave_history.append(requested_waves)
        existing["waves_requested_history"] = [list(w) for w in wave_history]
        existing["last_requested_models"] = sorted(args.models)
        write_json(path, existing)
        return

    if read_only:
        raise RuntimeError(
            "Parallel worker could not find run_config.json. The coordinator preflight must run first."
        )

    payload = {
        "script": Path(__file__).name,
        "project_root": str(root),
        "output_root": str(output_root),
        "locked_design": locked,
        "m_initial_max": DEFAULT_M_MAX,
        "m_convergence_checkpoints_planned": [10, 15, 20, 25],
        "m_extension_rule": "If not converged at 25, rerun this script with --m-max 30, 35, ...; hard cap 50 is evaluated later by convergence script.",
        "m_max_requested_history": [int(args.m_max)],
        "waves_requested_history": [list(map(int, args.waves))],
        "last_requested_models": sorted(args.models),
        "code_sha256": hashes,
    }
    write_json(path, payload)


# ======================================================================================
# OUTPUT / RESUME HELPERS
# ======================================================================================

def completed_data_paths(completion_dir: Path) -> tuple[Path, Path]:
    return (
        completion_dir / "completed_train.csv.gz",
        completion_dir / "completed_test.csv.gz",
    )


def save_completed_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    completion_dir: Path,
) -> None:
    train_path, test_path = completed_data_paths(completion_dir)
    completion_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False, compression="gzip", float_format="%.17g")
    test.to_csv(test_path, index=False, compression="gzip", float_format="%.17g")


def load_completed_data(completion_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path, test_path = completed_data_paths(completion_dir)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    expected = ["idauniq", *NODE_ORDER]
    if list(train.columns) != expected or list(test.columns) != expected:
        raise RuntimeError(f"Saved completed data has unexpected columns in {completion_dir}")
    return train, test


def model_is_complete(completion_dir: Path, model_key: str) -> bool:
    folder = completion_dir / MODEL_DIR_NAMES[model_key]
    return (folder / "edges.csv").exists() and (folder / "search_summary.json").exists()


def rebuild_indexes(output_root: Path) -> None:
    graph_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for summary_path in sorted(output_root.glob("W*/Fold*/M*/completion_summary.json")):
        try:
            payload = read_json(summary_path)
            completion_rows.append(payload)
        except Exception:
            continue

    for summary_path in sorted(output_root.glob("W*/Fold*/M*/*/search_summary.json")):
        try:
            payload = read_json(summary_path)
            parts = summary_path.parts
            # Find W#, Fold##, M### robustly from path components.
            wave_part = next(p for p in parts if p.startswith("W") and p[1:].isdigit())
            fold_part = next(p for p in parts if p.startswith("Fold"))
            m_part = next(p for p in parts if p.startswith("M") and p[1:].isdigit())
            payload.update({
                "Wave": int(wave_part[1:]),
                "Fold": int(fold_part.replace("Fold", "")),
                "Completion": int(m_part[1:]),
                "ModelDir": summary_path.parent.name,
                "SummaryPath": str(summary_path),
            })
            graph_rows.append(payload)
        except Exception:
            continue

    for error_path in sorted(output_root.glob("W*/Fold*/M*/FAILED_*.json")):
        try:
            error_rows.append(read_json(error_path))
        except Exception:
            continue

    pd.DataFrame(graph_rows).to_csv(output_root / "ensemble_index.csv", index=False)
    pd.DataFrame(completion_rows).to_csv(output_root / "completion_index.csv", index=False)
    pd.DataFrame(error_rows).to_csv(output_root / "errors.csv", index=False)



# ======================================================================================
# OUTER PROCESS PARALLELISM
# ======================================================================================

def _forward_common_cli_args(args: argparse.Namespace) -> list[str]:
    """Reconstruct the scientifically relevant CLI for a child process."""
    out: list[str] = []
    if args.project_root is not None:
        out += ["--project-root", str(args.project_root)]
    out += ["--folds", str(int(args.folds))]
    out += ["--m-max", str(int(args.m_max))]
    out += ["--base-seed", str(int(args.base_seed))]
    out += ["--models", *list(args.models)]
    out += ["--baseline-max-iter", str(int(args.baseline_max_iter))]
    out += ["--hybrid-max-iter", str(int(args.hybrid_max_iter))]
    out += ["--tabu-length", str(int(args.tabu_length))]
    out += ["--max-parents", str(int(args.max_parents))]
    out += ["--improvement-tolerance", repr(float(args.improvement_tolerance))]
    if args.save_diagnostics:
        out.append("--save-diagnostics")
    if args.continue_on_error:
        out.append("--continue-on-error")
    return out


def _parallel_child_environment() -> dict[str, str]:
    """One numerical thread per Hybrid process; outer jobs provide parallelism."""
    env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _tail_text(path: Path, lines: int = 30) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-int(lines):])
    except Exception:
        return ""


def run_parallel_coordinator(args: argparse.Namespace) -> None:
    """Run independent wave/fold jobs in separate Python processes.

    Each subprocess owns one (wave, fold) and processes M=1..m_max serially.
    This leaves the scientific scorer/search unchanged while allowing safe
    coarse-grained parallelism. Global indexes/config are written only by the
    coordinator, never concurrently by workers.
    """
    root = find_project_root(args.project_root)
    output_root = root / "Results" / "MasterPipeline" / "CV_Ensembles"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "parallel_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve()
    python_exe = Path(sys.executable).resolve()
    env = _parallel_child_environment()
    common = _forward_common_cli_args(args)

    print("=" * 110)
    print("RQ2 PHASE 1 — PARALLEL COORDINATOR")
    print("=" * 110)
    print(f"Project root : {root}")
    print(f"Output root  : {output_root}")
    print(f"Workers      : {args.workers}")
    print(f"Waves        : {tuple(args.waves)}")
    print(f"Folds        : {args.folds}")
    print(f"M requested  : 1..{args.m_max}")
    print(f"Models       : {', '.join(args.models)}")
    print("Worker BLAS  : 1 numerical thread per process")
    print("Parallel unit: one complete (wave, fold) job; completions remain serial within a fold")

    # One ordinary preflight process owns global run_config/fold_summary writes.
    preflight_cmd = [
        str(python_exe), str(script),
        "--waves", *[str(int(w)) for w in args.waves],
        *common,
        "--workers", "1",
        "--preflight-only",
    ]
    preflight_log = logs_root / "coordinator_preflight.log"
    print("\n[COORDINATOR] Running one serial preflight before workers start...")
    with preflight_log.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            preflight_cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(script.parent),
            check=False,
        )
    if result.returncode != 0:
        tail = _tail_text(preflight_log)
        raise RuntimeError(
            f"Parallel preflight failed (exit {result.returncode}). Log: {preflight_log}\n{tail}"
        )
    print(f"  [OK] preflight | log={preflight_log}")

    folds = [int(args.fold_only)] if args.fold_only is not None else list(range(1, int(args.folds) + 1))
    jobs = [(int(w), int(f)) for w in args.waves for f in folds]
    if not jobs:
        print("No wave/fold jobs requested.")
        return

    print(f"\n[COORDINATOR] Dispatching {len(jobs)} wave/fold jobs across {args.workers} workers...")
    started = time.perf_counter()

    failures: list[dict[str, Any]] = []
    pending = list(jobs)
    active: dict[int, dict[str, Any]] = {}
    creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    last_status = time.perf_counter()

    def launch_one(job: tuple[int, int]) -> None:
        wave, fold = job
        log_path = logs_root / f"W{wave}_F{fold:02d}.log"
        cmd = [
            str(python_exe), str(script),
            "--waves", str(wave),
            *common,
            "--workers", "1",
            "--fold-only", str(fold),
            "--internal-worker",
        ]
        log_handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(script.parent),
            creationflags=creationflags,
        )
        active[int(proc.pid)] = {
            "proc": proc,
            "wave": wave,
            "fold": fold,
            "log_path": log_path,
            "log_handle": log_handle,
            "started": time.perf_counter(),
        }
        print(f"  [START] W{wave} F{fold:02d} | pid={proc.pid} | log={log_path}")

    try:
        while pending or active:
            while pending and len(active) < int(args.workers):
                launch_one(pending.pop(0))

            time.sleep(2.0)
            finished_pids: list[int] = []
            for pid, info in active.items():
                proc = info["proc"]
                returncode = proc.poll()
                if returncode is None:
                    continue
                info["log_handle"].close()
                runtime = float(time.perf_counter() - info["started"])
                if int(returncode) == 0:
                    print(
                        f"  [DONE] W{info['wave']} F{info['fold']:02d} | "
                        f"runtime={elapsed_string(runtime)} | log={info['log_path']}"
                    )
                else:
                    failure = {
                        "wave": int(info["wave"]),
                        "fold": int(info["fold"]),
                        "returncode": int(returncode),
                        "runtime_seconds": runtime,
                        "log_path": str(info["log_path"]),
                    }
                    failures.append(failure)
                    print(
                        f"  [FAIL] W{info['wave']} F{info['fold']:02d} | "
                        f"exit={returncode} | log={info['log_path']}"
                    )
                    print(_tail_text(info["log_path"], 20))
                finished_pids.append(pid)

            for pid in finished_pids:
                active.pop(pid, None)

            now = time.perf_counter()
            if active and now - last_status >= 600.0:
                running = ", ".join(
                    f"W{info['wave']}F{info['fold']:02d} "
                    f"({elapsed_string(now - info['started'])})"
                    for info in active.values()
                )
                print(f"  [STATUS] active={len(active)} | pending={len(pending)} | {running}")
                last_status = now

    except KeyboardInterrupt:
        print("\n[COORDINATOR] Ctrl+C received. Terminating active workers; completed graphs remain reusable...")
        for info in active.values():
            proc = info["proc"]
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        deadline = time.time() + 10.0
        for info in active.values():
            proc = info["proc"]
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except Exception:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            try:
                info["log_handle"].close()
            except Exception:
                pass
        rebuild_indexes(output_root)
        print("[COORDINATOR] Workers stopped. Re-run with --workers 2 or --workers 3 to resume.")
        raise

    # No workers are alive now, so global indexes can be rebuilt without races.
    rebuild_indexes(output_root)
    runtime = float(time.perf_counter() - started)

    print("\n" + "=" * 110)
    print("PARALLEL PHASE 1 COORDINATOR COMPLETE")
    print("=" * 110)
    print(f"Runtime       : {elapsed_string(runtime)}")
    print(f"Fold jobs     : {len(jobs)}")
    print(f"Failures      : {len(failures)}")
    print(f"Output root   : {output_root}")
    print(f"Worker logs   : {logs_root}")
    print("=" * 110)

    if failures and not args.continue_on_error:
        raise RuntimeError(
            f"{len(failures)} parallel wave/fold job(s) failed. See logs under {logs_root}."
        )


# ======================================================================================
# MAIN
# ======================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ2 Phase 1: generate leakage-safe CV imputation/graph ensembles."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional path to the Research-GitHub repository root.",
    )
    parser.add_argument(
        "--waves",
        type=int,
        nargs="+",
        default=list(DEFAULT_WAVES),
        help="Subset of waves to run (default: 2 4 6 8).",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--m-max",
        type=int,
        default=DEFAULT_M_MAX,
        help="Generate completions 1..M. Re-run with a larger value to extend the same ensemble.",
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of independent wave/fold worker processes. Use 2 as the safe default "
            "on the 4-core i5-11300H; use 3 for a throughput test. workers=1 preserves "
            "the original serial execution."
        ),
    )
    parser.add_argument(
        "--fold-only",
        type=int,
        default=None,
        help="Run only one fold number (mainly for targeted reruns/debugging).",
    )
    parser.add_argument(
        "--internal-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("discrete", "continuous", "hybrid"),
        default=["discrete", "continuous", "hybrid"],
        help="Model(s) to learn for each completed training dataset.",
    )
    parser.add_argument("--baseline-max-iter", type=int, default=DEFAULT_BASELINE_MAX_ITER)
    parser.add_argument("--hybrid-max-iter", type=int, default=DEFAULT_HYBRID_MAX_ITER)
    parser.add_argument("--tabu-length", type=int, default=DEFAULT_TABU_LENGTH)
    parser.add_argument("--max-parents", type=int, default=DEFAULT_MAX_PARENTS)
    parser.add_argument(
        "--improvement-tolerance",
        type=float,
        default=DEFAULT_IMPROVEMENT_TOLERANCE,
    )
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Save detailed Hybrid scorer diagnostics for every graph (can be large).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed completion/model and continue instead of stopping immediately.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate files/cohorts/folds/config, then stop before imputation/structure learning.",
    )
    args = parser.parse_args()

    waves = tuple(int(w) for w in args.waves)
    invalid_waves = sorted(set(waves) - set(DEFAULT_WAVES))
    if invalid_waves:
        raise ValueError(f"Only waves {DEFAULT_WAVES} are supported; got {invalid_waves}")
    if args.folds < 2:
        raise ValueError("--folds must be >= 2")
    if args.m_max < 1:
        raise ValueError("--m-max must be >= 1")
    if args.m_max > 50:
        raise ValueError(
            "This locked design uses 50 as the hard safety cap. Do not request M > 50."
        )
    if args.max_parents != 3:
        raise ValueError(
            "The frozen RQ2 model family supports max_parents=3. Do not change --max-parents."
        )
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.fold_only is not None and not (1 <= int(args.fold_only) <= int(args.folds)):
        raise ValueError(f"--fold-only must be between 1 and {args.folds}")

    args.waves = list(waves)
    args.models = sorted(set(args.models))
    requested_models = set(args.models)

    # In parallel mode this process is only a lightweight coordinator. It launches
    # one ordinary preflight child, then independent per-fold worker subprocesses.
    if args.workers > 1 and not args.preflight_only and not args.internal_worker:
        run_parallel_coordinator(args)
        return

    root = find_project_root(args.project_root)
    output_root = root / "Results" / "MasterPipeline" / "CV_Ensembles"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("RQ2 PHASE 1 — CV ENSEMBLE GENERATION")
    print("=" * 110)
    print(f"Project root : {root}")
    print(f"Output root  : {output_root}")
    print(f"Waves        : {waves}")
    print(f"Folds        : {args.folds} (stratified on Dementia)")
    print(f"M requested  : 1..{args.m_max}")
    print(f"Models       : {', '.join(args.models)}")
    print(f"Workers      : {args.workers}" + (" (internal fold worker)" if args.internal_worker else ""))
    if args.fold_only is not None:
        print(f"Fold scope   : Fold {int(args.fold_only):02d} only")
    print("Split rule   : ORIGINAL incomplete frozen cohort split before any imputation")
    print("Imputation   : sequential TRAIN-only fit; independent deterministic TRAIN/TEST draws")
    print("Process nodes: Proxy + SelfCompletion + Nurse, unchanged ordinary BN nodes")

    # Load all imputers because source-column discovery depends on their frozen constants.
    print("\n[1/5] Loading frozen imputer modules...")
    imputers = load_all_imputers(root)
    chain = instantiate_chain(imputers)
    for key, filename in EXPECTED_IMPUTER_FILES.items():
        print(f"  [OK] {key:<22} src/imputation/{filename}")

    print("\n[2/5] Loading requested model modules...")
    models = load_models(root, requested_models)
    for key in args.models:
        print(f"  [OK] {key}")

    # Establish config BEFORE expensive work, including source hashes.
    print("\n[3/5] Locking/checking run configuration and source hashes...")
    establish_run_config(
        output_root=output_root, root=root, args=args, read_only=bool(args.internal_worker)
    )
    print("  [OK] run_config.json")

    dta = root / "Datasets" / "gh_elsa_h.dta"
    if not dta.exists():
        raise FileNotFoundError(f"Dataset not found: {dta}")

    completeness_path = find_completeness_file(root)
    completeness = pd.read_csv(completeness_path)
    required = gather_required_columns(imputers, waves)

    print("\n[4/5] Loading selected ELSA source columns...")
    print(f"  DTA          : {dta}")
    print(f"  Cohort file  : {completeness_path}")
    print(f"  Columns      : {len(required):,}")

    with pd.io.stata.StataReader(
        dta,
        convert_categoricals=False,
        convert_missing=True,
        preserve_dtypes=True,
    ) as reader:
        labels = reader.variable_labels()
        missing_columns = sorted(required - set(labels))
        if missing_columns:
            raise KeyError(
                "Required DTA columns are missing:\n  " + "\n  ".join(missing_columns)
            )
        raw_df = reader.read(
            columns=sorted(required),
            convert_categoricals=False,
            convert_missing=True,
            preserve_dtypes=True,
        )
    print(f"  Loaded rows  : {len(raw_df):,}")

    print("\n[5/5] Freezing cohorts and fold assignments...")
    wave_frames: dict[int, pd.DataFrame] = {}
    fold_assignments: dict[int, pd.DataFrame] = {}
    fold_summary_rows: list[dict[str, Any]] = []

    for wave in waves:
        cohort_ids = canonical_retained_ids(completeness, raw_df, wave)
        actual_n = len(cohort_ids)
        expected_n = FROZEN_RETAINED_COUNTS[wave]
        if actual_n != expected_n:
            raise RuntimeError(
                f"W{wave}: frozen cohort mismatch. Expected {expected_n:,}, got {actual_n:,}."
            )

        wave_original = raw_df[raw_df["idauniq"].isin(cohort_ids)].copy()
        wave_original = wave_original.sort_values("idauniq").reset_index(drop=True)
        if len(wave_original) != actual_n:
            raise RuntimeError(f"W{wave}: cohort ID count and selected row count differ.")

        assignment_path = output_root / "fold_assignments" / f"W{wave}.csv"
        assignment = make_or_load_fold_assignments(
            wave_original=wave_original,
            wave=wave,
            folds=args.folds,
            base_seed=args.base_seed,
            path=assignment_path,
        )

        wave_frames[wave] = wave_original
        fold_assignments[wave] = assignment

        for fold in range(1, args.folds + 1):
            sub = assignment[assignment["Fold"].eq(fold)]
            train = assignment[~assignment["Fold"].eq(fold)]
            fold_summary_rows.append({
                "Wave": wave,
                "Fold": fold,
                "Test_N": int(len(sub)),
                "Test_Dementia_1": int(sub["Dementia"].sum()),
                "Test_Dementia_Prevalence": float(sub["Dementia"].mean()),
                "Train_N": int(len(train)),
                "Train_Dementia_1": int(train["Dementia"].sum()),
                "Train_Dementia_Prevalence": float(train["Dementia"].mean()),
            })

        print(
            f"  W{wave}: N={actual_n:,}, Dementia=1 "
            f"{int(assignment['Dementia'].sum()):,}, folds saved"
        )

    if not args.internal_worker:
        pd.DataFrame(fold_summary_rows).to_csv(output_root / "fold_summary.csv", index=False)

    if args.preflight_only:
        if not args.internal_worker:
            rebuild_indexes(output_root)
        print("\nPREFLIGHT PASSED. No imputation or graph learning was run.")
        return

    folds_to_run = (int(args.fold_only),) if args.fold_only is not None else tuple(range(1, args.folds + 1))
    total_planned = len(waves) * len(folds_to_run) * args.m_max
    print("\n" + "=" * 110)
    print(f"STARTING COMPLETION ENSEMBLE | wave-fold-completions requested: {total_planned:,}")
    print("Existing completed datasets/graphs are reused; safe to stop and resume.")
    print("=" * 110)

    overall_started = time.perf_counter()
    completed_counter = 0
    error_count = 0

    for wave in waves:
        wave_original = wave_frames[wave]
        assignment = fold_assignments[wave]
        assignment_map = dict(zip(assignment["idauniq"], assignment["Fold"]))
        fold_vector = wave_original["idauniq"].map(assignment_map).to_numpy(int)

        for fold in folds_to_run:
            test_mask = fold_vector == fold
            train_original = wave_original.loc[~test_mask].reset_index(drop=True)
            test_original = wave_original.loc[test_mask].reset_index(drop=True)

            if len(train_original) + len(test_original) != len(wave_original):
                raise RuntimeError(f"W{wave} F{fold}: train/test partition mismatch.")

            for completion in range(1, args.m_max + 1):
                completed_counter += 1
                label = f"W{wave} F{fold} M{completion:03d}"
                completion_dir = (
                    output_root / f"W{wave}" / f"Fold{fold:02d}" / f"M{completion:03d}"
                )
                completion_dir.mkdir(parents=True, exist_ok=True)
                train_path, test_path = completed_data_paths(completion_dir)

                print("\n" + "-" * 110)
                print(
                    f"{label} | completion {completed_counter}/{total_planned} | "
                    f"elapsed={elapsed_string(time.perf_counter() - overall_started)}"
                )
                print("-" * 110)

                try:
                    if train_path.exists() and test_path.exists():
                        train_canonical, test_canonical = load_completed_data(completion_dir)
                        stage_rows: list[dict[str, Any]] = []
                        imputation_runtime = 0.0
                        print("  [REUSE] completed TRAIN/TEST node matrices")
                    else:
                        impute_started = time.perf_counter()
                        train_completed_raw, test_completed_raw, stage_rows = complete_train_test(
                            train_original=train_original,
                            test_original=test_original,
                            wave=wave,
                            fold=fold,
                            completion=completion,
                            base_seed=args.base_seed,
                            chain=chain,
                        )
                        train_canonical = build_canonical_node_frame(train_completed_raw, wave)
                        test_canonical = build_canonical_node_frame(test_completed_raw, wave)
                        save_completed_data(train_canonical, test_canonical, completion_dir)
                        imputation_runtime = float(time.perf_counter() - impute_started)
                        print(
                            f"  [OK] imputation + canonical save | "
                            f"{elapsed_string(imputation_runtime)}"
                        )

                    # Fit all model-specific preprocessing from completed TRAIN only.
                    rep_train, _, rep_meta = build_model_representations(
                        train_canonical, test_canonical
                    )
                    write_json(completion_dir / "representation_metadata.json", rep_meta)

                    completion_summary = {
                        "Wave": wave,
                        "Fold": fold,
                        "Completion": completion,
                        "Train_N": int(len(train_canonical)),
                        "Test_N": int(len(test_canonical)),
                        "Train_Dementia_1": int(train_canonical["Dementia"].sum()),
                        "Test_Dementia_1": int(test_canonical["Dementia"].sum()),
                        "imputation_runtime_seconds": float(imputation_runtime),
                        "imputation_stage_summary": stage_rows,
                        "completed_train_path": str(train_path),
                        "completed_test_path": str(test_path),
                        "representation_metadata_path": str(
                            completion_dir / "representation_metadata.json"
                        ),
                    }
                    write_json(completion_dir / "completion_summary.json", completion_summary)

                    for model_key in args.models:
                        model_dir = completion_dir / MODEL_DIR_NAMES[model_key]
                        if model_is_complete(completion_dir, model_key):
                            print(f"  [REUSE] {MODEL_DIR_NAMES[model_key]} graph")
                            continue

                        print(f"  [RUN] {MODEL_DIR_NAMES[model_key]} structure learning...")
                        model_started = time.perf_counter()
                        try:
                            summary = learn_one_graph(
                                model_key=model_key,
                                train_data=rep_train[model_key],
                                modules=models,
                                root=root,
                                model_out=model_dir,
                                baseline_max_iter=args.baseline_max_iter,
                                hybrid_max_iter=args.hybrid_max_iter,
                                tabu_length=args.tabu_length,
                                max_parents=args.max_parents,
                                improvement_tolerance=args.improvement_tolerance,
                                save_diagnostics=args.save_diagnostics,
                                label=label,
                            )
                            print(
                                f"  [OK] {MODEL_DIR_NAMES[model_key]} | "
                                f"edges={summary['edge_count']} | "
                                f"score={summary['best_score']:.3f} | "
                                f"runtime={elapsed_string(time.perf_counter() - model_started)}"
                            )
                        except Exception as exc:
                            error_count += 1
                            failure = {
                                "Wave": wave,
                                "Fold": fold,
                                "Completion": completion,
                                "Model": model_key,
                                "Stage": "structure_learning",
                                "ErrorType": type(exc).__name__,
                                "Error": str(exc),
                                "Traceback": traceback.format_exc(),
                            }
                            write_json(
                                completion_dir / f"FAILED_{MODEL_DIR_NAMES[model_key]}.json",
                                failure,
                            )
                            print(
                                f"  [FAIL] {MODEL_DIR_NAMES[model_key]}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            if not args.continue_on_error:
                                if not args.internal_worker:
                                    rebuild_indexes(output_root)
                                raise

                except Exception as exc:
                    error_count += 1
                    failure = {
                        "Wave": wave,
                        "Fold": fold,
                        "Completion": completion,
                        "Model": None,
                        "Stage": "imputation_or_representation",
                        "ErrorType": type(exc).__name__,
                        "Error": str(exc),
                        "Traceback": traceback.format_exc(),
                    }
                    write_json(completion_dir / "FAILED_completion.json", failure)
                    print(f"  [FAIL] completion: {type(exc).__name__}: {exc}")
                    if not args.continue_on_error:
                        if not args.internal_worker:
                            rebuild_indexes(output_root)
                        raise

                # Keep aggregate indexes useful in serial mode. Parallel workers
                # never touch global indexes concurrently; the coordinator rebuilds once.
                if not args.internal_worker:
                    rebuild_indexes(output_root)

    if not args.internal_worker:
        rebuild_indexes(output_root)
    runtime = time.perf_counter() - overall_started

    print("\n" + "=" * 110)
    print("PHASE 1 COMPLETE")
    print("=" * 110)
    print(f"Runtime      : {elapsed_string(runtime)}")
    print(f"Errors logged : {error_count}")
    print(f"Output root  : {output_root}")
    print("Next script  : 02_m_convergence.py (reads these graph ensembles; no new graph learning required)")
    print("=" * 110)


if __name__ == "__main__":
    main()

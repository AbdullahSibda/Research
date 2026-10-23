"""
heldout_inference.py
==========================

RQ2 Phase 2: leakage-safse held-out probabilistic inference from the frozen
Phase-1 graph/completion ensembles.

This runner DOES NOT relearn any graph.  For every Wave x Fold x Completion it
loads the saved completed TRAIN matrix and the saved DAG, refits only the local
conditional families required by a pseudo-target's Markov blanket, and predicts
originally-observed held-out pseudo-target values.

Locked scientific design implemented here
-----------------------------------------
* Waves: 2, 4, 6, 8.
* Outer CV: the exact Phase-1 5-fold assignments are reused.
* Ensemble: M=25 stochastic completion/graph members per Wave x Fold.
* Pseudo-targets:
    Continuous: Recall, Grip, CASP19, Loneliness
    Discrete  : Dementia, ManagingMeds, LowerMobility, CESD
* Only originally-observed target values are scored.
* Exactly one pseudo-target is hidden; all other completed nodes are evidence.
* Inference uses the exact BN full conditional / Markov blanket:

      p(T | X_-T) propto p(T | Pa(T)) * prod_{C in Ch(T)} p(C | Pa(C)).

* Discrete targets are enumerated and exactly normalised over legal states.
* Continuous Hybrid targets are normalised by vectorised one-dimensional
  Gauss-Legendre quadrature.  The Continuous baseline has an analytically
  Gaussian full conditional and is normalised exactly.
* M member predictive distributions are averaged BEFORE scoring.  Log densities
  are combined with log-sum-exp; logs are never averaged.
* Five held-out folds are concatenated into one complete OOF prediction set
  BEFORE F1/log-loss/NLPD/RMSE are calculated.
* Targets are NEVER averaged together.  Results remain Wave x Target x Model.
* Categorical metrics:
    - binary target: positive-class F1 + log loss
    - multiclass target: macro-F1 + log loss
    - continuous targets for three-way comparison: common-bin macro-F1 + log loss
* Dementia adds ROC-AUC, PR-AUC, F1@0.5, and threshold-vs-F1 diagnostics.
* Native continuous comparison (Continuous vs Hybrid): NLPD + RMSE.
* CIs/model contrasts use paired held-out-participant bootstrap within each
  Wave x Target.  (No cross-wave target pooling is performed.)

Leakage rule
------------
Seven targets can safely use the already-saved Phase-1 completed TEST evidence
because their observed values were not used to construct other imputed BN
variables.  LowerMobility is the exception: CESD, CASP19, Loneliness, BMI and
VigorousActivity can be imputed conditional on it.  For LowerMobility only,
this runner reconstructs the ORIGINAL incomplete fold, hides r{wave}lowermob
before completion, then replays the exact frozen TRAIN-only imputer chain with
Phase-1 seeds.  The generated LowerMobility value may drive later imputers, but
the true held-out value never does.  The generated target itself is not used as
BN evidence.

Common-bin harmonisation
------------------------
The Discrete baseline's continuous-node cutpoints vary slightly across some of
the 25 imputations.  Within each Wave x Fold x continuous target, this runner
uses the component-wise median of the 25 TRAIN-fitted cutpoint pairs as ONE
common evaluation partition.  For each Discrete member the DAG remains fixed,
but the target and all child local factors are refit using that common target
state representation.  Continuous/Hybrid predictive densities are integrated
over the same common intervals.

Output layout
-------------
Results/MasterPipeline/CV_HeldoutInference/
    run_config.json
    common_cutpoints.json
    diagnostics/
    target_safe_test/
    members/W2/Fold01/M001/HybridV3/Recall.csv.gz
    ensemble/W2/Fold01/HybridV3/Recall.csv.gz
    oof/W2/HybridV3/Recall.csv.gz
    metrics/metrics_long.csv
    metrics/model_comparisons.csv
    metrics/dementia_threshold_curves.csv
    plots/*.png

The runner is resumable: an existing member prediction file is reused unless
--force is supplied.

Typical use
-----------
From the Research Question 2 project root or Experiments folder:

    python Experiments/heldout_inference.py --preflight-only
    python Experiments/heldout_inference.py

The script expects generate_ensembles.py beside it in Experiments/ and
reuses its frozen preprocessing/imputation helper functions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


# ======================================================================================
# LOCKED PHASE-2 DESIGN
# ======================================================================================

CONTINUOUS_TARGETS: tuple[str, ...] = ("Recall", "Grip", "CASP19", "Loneliness")
DISCRETE_TARGETS: tuple[str, ...] = ("Dementia", "ManagingMeds", "LowerMobility", "CESD")
PRIMARY_TARGETS: tuple[str, ...] = (*CONTINUOUS_TARGETS, *DISCRETE_TARGETS)

# All except LowerMobility can use the already-saved completed TEST evidence.
SAVED_TEST_SAFE_TARGETS = frozenset(
    {"Recall", "Grip", "CASP19", "Loneliness", "Dementia", "ManagingMeds", "CESD"}
)
TARGET_AWARE_REIMPUTE = frozenset({"LowerMobility"})

MODEL_DIR_NAMES = {
    "discrete": "DiscreteBaseline",
    "continuous": "ContinuousBaseline",
    "hybrid": "HybridV3",
}
MODEL_DISPLAY = {
    "discrete": "Discrete",
    "continuous": "Continuous",
    "hybrid": "Hybrid",
}
MODEL_ORDER = ("discrete", "continuous", "hybrid")

# Fixed native state meanings of the four chosen discrete pseudo-targets.
DISCRETE_TARGET_CARDINALITY = {
    "Dementia": 2,
    "ManagingMeds": 2,
    "LowerMobility": 3,
    "CESD": 3,
}
BINARY_TARGETS = frozenset({"Dementia", "ManagingMeds"})

PROB_FLOOR = 1e-300
METRIC_PROB_FLOOR = 1e-15
DEFAULT_BOOTSTRAP_REPS = 1000
DEFAULT_INTEGRATION_NODES_PER_BIN = 32
DEFAULT_INTEGRATION_TAIL_SD = 8.0
DEFAULT_INTEGRATION_CHUNK = 128
LONELINESS_INTEGRATION_NODES_PER_BIN = 256
LONELINESS_INTEGRATION_CHUNK = 32
TAIL_LOG_GAP_WARNING = 18.0


# ======================================================================================
# GENERIC UTILITIES
# ======================================================================================

def elapsed_string(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module_from_file(runtime_name: str, path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(runtime_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[runtime_name] = module
    spec.loader.exec_module(module)
    return module


def locate_phase1_script() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "generate_ensembles.py",
        Path.cwd().resolve() / "generate_ensembles.py",
        Path.cwd().resolve() / "Experiments" / "generate_ensembles.py",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "Could not locate generate_ensembles.py. Put this Phase-2 runner "
        "beside it in Experiments/ or run from the project/Experiments directory."
    )


def load_phase1_module():
    path = locate_phase1_script()
    return load_module_from_file("rq2_phase1_helpers", path), path


def parse_int_list(values: Sequence[int] | None, default: Sequence[int]) -> tuple[int, ...]:
    if values is None:
        return tuple(map(int, default))
    return tuple(sorted(set(map(int, values))))


def ensure_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def clip_and_normalize_probabilities(probs: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    p = np.maximum(p, float(floor))
    sums = p.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(sums)) or np.any(sums <= 0):
        raise FloatingPointError("Invalid probability row sum.")
    return p / sums


def load_graph(path: Path, nodes: Sequence[str]) -> nx.DiGraph:
    edges = pd.read_csv(path)
    if not {"Parent", "Child"}.issubset(edges.columns):
        if edges.shape[1] < 2:
            raise ValueError(f"Invalid edge file: {path}")
        edges = edges.rename(columns={edges.columns[0]: "Parent", edges.columns[1]: "Child"})
    g = nx.DiGraph()
    g.add_nodes_from(map(str, nodes))
    g.add_edges_from((str(a), str(b)) for a, b in edges[["Parent", "Child"]].itertuples(index=False, name=None))
    if not nx.is_directed_acyclic_graph(g):
        raise RuntimeError(f"Saved graph is not a DAG: {path}")
    unknown = (set(g.nodes) - set(nodes)) | (set(nodes) - set(g.nodes))
    if unknown:
        raise RuntimeError(f"Graph node mismatch at {path}: {sorted(unknown)}")
    return g


def target_source_column(phase1, wave: int, target: str) -> str:
    if target == "Dementia":
        return phase1.protected_column_map(wave)["Dementia"]
    return phase1.target_column_map(wave)[target]


def target_observed_mask(phase1, raw_test: pd.DataFrame, wave: int, target: str) -> pd.Series:
    col = target_source_column(phase1, wave, target)
    if target == "Dementia":
        return raw_test[col].map(lambda v: phase1.valid_integer(v, allowed=(0, 1))).astype(bool)
    return phase1.valid_mask_for_target(raw_test[col], target).astype(bool)


def raw_target_values(raw_test: pd.DataFrame, phase1, wave: int, target: str, mask: pd.Series) -> np.ndarray:
    col = target_source_column(phase1, wave, target)
    return pd.to_numeric(raw_test.loc[mask, col], errors="coerce").to_numpy(dtype=float)


def recode_discrete_truth(target: str, raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=float)
    if target in {"Dementia", "ManagingMeds"}:
        out = x.astype(np.int64)
    elif target == "LowerMobility":
        out = np.select([x == 0, x == 1, x >= 2], [0, 1, 2], default=-1).astype(np.int64)
    elif target == "CESD":
        out = np.select([x <= 1, (x >= 2) & (x <= 3), x >= 4], [0, 1, 2], default=-1).astype(np.int64)
    else:
        raise KeyError(target)
    k = DISCRETE_TARGET_CARDINALITY[target]
    if np.any((out < 0) | (out >= k)):
        raise ValueError(f"Invalid recoded truth for {target}.")
    return out


def apply_common_bins(values: Sequence[float], cutpoints: Sequence[float]) -> np.ndarray:
    cuts = np.asarray(cutpoints, dtype=float)
    if cuts.shape != (2,) or cuts[0] >= cuts[1]:
        raise ValueError(f"Invalid common cutpoints: {cuts}")
    x = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Cannot bin non-finite target values.")
    return np.searchsorted(cuts, x, side="right").astype(np.int64)


# ======================================================================================
# PHASE-1 RECONSTRUCTION / INPUT VALIDATION
# ======================================================================================

def resolve_paths(args, phase1) -> tuple[Path, Path, Path]:
    root = phase1.find_project_root(args.project_root)
    p1_root = (
        Path(args.phase1_root).expanduser().resolve()
        if args.phase1_root
        else (root / "Results" / "MasterPipeline" / "CV_Ensembles").resolve()
    )
    out_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (root / "Results" / "MasterPipeline" / "CV_HeldoutInference").resolve()
    )
    if not p1_root.exists():
        raise FileNotFoundError(f"Phase-1 output root not found: {p1_root}")
    return root, p1_root, out_root


def verify_phase1_hashes(root: Path, phase1_root: Path, allow_mismatch: bool) -> list[dict[str, Any]]:
    cfg_path = phase1_root / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = read_json(cfg_path)
    rows = []
    mismatches = []
    for rel, expected in cfg.get("code_sha256", {}).items():
        path = root / rel
        actual = sha256_file(path) if path.exists() else None
        ok = actual == expected
        row = {"path": rel, "expected": expected, "actual": actual, "match": ok}
        rows.append(row)
        if not ok:
            mismatches.append(row)
    if mismatches and not allow_mismatch:
        text = "\n".join(
            f"  {r['path']}: expected={r['expected']} actual={r['actual']}" for r in mismatches
        )
        raise RuntimeError(
            "Frozen Phase-1 model/imputer source hashes do not match run_config.json:\n"
            + text
            + "\nUse --allow-source-hash-mismatch only if you deliberately accept this provenance break."
        )
    return rows


def load_raw_inputs(phase1, root: Path, waves: tuple[int, ...]):
    imputers = phase1.load_all_imputers(root)
    chain = phase1.instantiate_chain(imputers)
    completeness_path = phase1.find_completeness_file(root)
    completeness = pd.read_csv(completeness_path)
    dta = root / "Datasets" / "gh_elsa_h.dta"
    if not dta.exists():
        raise FileNotFoundError(dta)
    required = phase1.gather_required_columns(imputers, waves)
    print(f"Loading ELSA source columns ({len(required):,}) from {dta} ...")
    with pd.io.stata.StataReader(
        dta,
        convert_categoricals=False,
        convert_missing=True,
        preserve_dtypes=True,
    ) as reader:
        labels = reader.variable_labels()
        missing = sorted(required - set(labels))
        if missing:
            raise KeyError("Required DTA columns are missing:\n  " + "\n  ".join(missing))
        raw = reader.read(
            columns=sorted(required),
            convert_categoricals=False,
            convert_missing=True,
            preserve_dtypes=True,
        )
    return imputers, chain, completeness, raw, completeness_path, dta


def reconstruct_wave_frames(
    phase1,
    phase1_root: Path,
    completeness: pd.DataFrame,
    raw_df: pd.DataFrame,
    waves: tuple[int, ...],
    run_config: Mapping[str, Any],
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    wave_frames: dict[int, pd.DataFrame] = {}
    assignments: dict[int, pd.DataFrame] = {}
    expected_counts = {
        int(k): int(v)
        for k, v in run_config["locked_design"]["frozen_retained_counts"].items()
    }
    for wave in waves:
        ids = phase1.canonical_retained_ids(completeness, raw_df, wave)
        if wave in expected_counts and len(ids) != expected_counts[wave]:
            raise RuntimeError(
                f"W{wave}: frozen cohort mismatch; expected {expected_counts[wave]:,}, got {len(ids):,}."
            )
        frame = raw_df[raw_df["idauniq"].isin(ids)].copy()
        frame = frame.sort_values("idauniq").reset_index(drop=True)
        assignment_path = phase1_root / "fold_assignments" / f"W{wave}.csv"
        assignment = pd.read_csv(assignment_path)
        if set(assignment["idauniq"]) != set(frame["idauniq"]):
            raise RuntimeError(f"W{wave}: fold assignment IDs do not match the frozen cohort.")
        if assignment["idauniq"].duplicated().any():
            raise RuntimeError(f"W{wave}: duplicate idauniq in fold assignment.")
        wave_frames[wave] = frame
        assignments[wave] = assignment
    return wave_frames, assignments


def split_wave_fold(wave_frame: pd.DataFrame, assignment: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    amap = dict(zip(assignment["idauniq"], assignment["Fold"]))
    fold_vector = wave_frame["idauniq"].map(amap).to_numpy(dtype=int)
    test_mask = fold_vector == int(fold)
    train = wave_frame.loc[~test_mask].reset_index(drop=True)
    test = wave_frame.loc[test_mask].reset_index(drop=True)
    if len(train) + len(test) != len(wave_frame):
        raise RuntimeError("Train/test split mismatch.")
    return train, test


def phase1_completion_dir(phase1_root: Path, wave: int, fold: int, completion: int) -> Path:
    return phase1_root / f"W{wave}" / f"Fold{fold:02d}" / f"M{completion:03d}"


def load_completed_canonical(completion_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(completion_dir / "completed_train.csv.gz")
    test = pd.read_csv(completion_dir / "completed_test.csv.gz")
    return train, test


def compute_common_cutpoints(
    phase1_root: Path,
    waves: tuple[int, ...],
    folds: int,
    m_max: int,
    targets: Sequence[str],
    *,
    selected_folds: Sequence[int] | None = None,
    selected_completions: Sequence[int] | None = None,
) -> tuple[dict[tuple[int, int, str], tuple[float, float]], list[dict[str, Any]]]:
    out: dict[tuple[int, int, str], tuple[float, float]] = {}
    diagnostics: list[dict[str, Any]] = []
    continuous_targets = [t for t in targets if t in CONTINUOUS_TARGETS]
    for wave in waves:
        for fold in (selected_folds if selected_folds is not None else range(1, folds + 1)):
            for target in continuous_targets:
                all_cuts: list[tuple[float, float]] = []
                for m in (selected_completions if selected_completions is not None else range(1, m_max + 1)):
                    meta_path = phase1_completion_dir(phase1_root, wave, fold, m) / "representation_metadata.json"
                    if not meta_path.exists():
                        raise FileNotFoundError(meta_path)
                    meta = read_json(meta_path)
                    cuts = tuple(map(float, meta["discrete_baseline_quantile_bins"][target]["cutpoints"]))
                    if len(cuts) != 2 or cuts[0] >= cuts[1]:
                        raise ValueError(f"Invalid saved cutpoints for W{wave} F{fold} M{m} {target}: {cuts}")
                    all_cuts.append((cuts[0], cuts[1]))
                matrix = np.asarray(all_cuts, dtype=float)
                med = tuple(np.median(matrix, axis=0).astype(float).tolist())
                if med[0] >= med[1]:
                    raise RuntimeError(f"Median cutpoints are invalid for W{wave} F{fold} {target}: {med}")
                out[(wave, fold, target)] = (float(med[0]), float(med[1]))
                counts = Counter(all_cuts)
                diagnostics.append({
                    "wave": wave,
                    "fold": fold,
                    "target": target,
                    "common_cut1": med[0],
                    "common_cut2": med[1],
                    "n_unique_member_signatures": len(counts),
                    "member_signature_counts": {
                        f"[{a:.15g},{b:.15g}]": int(n) for (a, b), n in sorted(counts.items())
                    },
                })
    return out, diagnostics


def save_common_cutpoints(out_root: Path, common: Mapping[tuple[int, int, str], tuple[float, float]], diagnostics):
    payload: dict[str, Any] = {}
    for (wave, fold, target), cuts in sorted(common.items()):
        payload.setdefault(f"W{wave}", {}).setdefault(f"Fold{fold:02d}", {})[target] = list(cuts)
    write_json(out_root / "common_cutpoints.json", payload)
    pd.DataFrame(diagnostics).to_csv(out_root / "diagnostics" / "common_cutpoint_audit.csv", index=False)


def metadata_matches_saved(rebuilt: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    c1 = float(rebuilt["wealth_asinh"]["c_train_only"])
    c2 = float(saved["wealth_asinh"]["c_train_only"])
    if not math.isclose(c1, c2, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Wealth transform mismatch: rebuilt={c1}, saved={c2}")
    for node, meta in rebuilt["discrete_baseline_quantile_bins"].items():
        a = np.asarray(meta["cutpoints"], dtype=float)
        b = np.asarray(saved["discrete_baseline_quantile_bins"][node]["cutpoints"], dtype=float)
        if not np.allclose(a, b, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Discrete representation cutpoint mismatch for {node}: {a} vs {b}")


def dataframe_numeric_equal(a: pd.DataFrame, b: pd.DataFrame, label: str, atol: float = 1e-12) -> None:
    if list(a.columns) != list(b.columns):
        raise RuntimeError(f"{label}: column mismatch")
    if len(a) != len(b):
        raise RuntimeError(f"{label}: row-count mismatch {len(a)} != {len(b)}")
    if "idauniq" in a.columns:
        if not np.array_equal(a["idauniq"].to_numpy(), b["idauniq"].to_numpy()):
            raise RuntimeError(f"{label}: idauniq order mismatch")
        cols = [c for c in a.columns if c != "idauniq"]
    else:
        cols = list(a.columns)
    aa = a[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    bb = b[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.allclose(aa, bb, rtol=0.0, atol=atol, equal_nan=True):
        diff = np.abs(aa - bb)
        finite = np.isfinite(diff)
        maxdiff = float(np.max(diff[finite])) if finite.any() else float("nan")
        raise RuntimeError(f"{label}: numeric mismatch; max abs difference={maxdiff}")


def lower_mobility_safe_test_cache_path(out_root: Path, wave: int, fold: int, completion: int) -> Path:
    return (
        out_root / "target_safe_test" / f"W{wave}" / f"Fold{fold:02d}" /
        f"M{completion:03d}_LowerMobility_hidden.csv.gz"
    )


def reconstruct_lower_mobility_safe_test(
    *,
    phase1,
    chain,
    train_original: pd.DataFrame,
    test_original: pd.DataFrame,
    saved_train_canonical: pd.DataFrame,
    wave: int,
    fold: int,
    completion: int,
    base_seed: int,
    cache_path: Path,
    force: bool,
) -> pd.DataFrame:
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    hidden_test = test_original.copy(deep=True)
    target_col = phase1.target_column_map(wave)["LowerMobility"]
    hidden_test[target_col] = np.nan

    train_completed_raw, test_completed_raw, _ = phase1.complete_train_test(
        train_original=train_original,
        test_original=hidden_test,
        wave=wave,
        fold=fold,
        completion=completion,
        base_seed=base_seed,
        chain=chain,
    )
    replay_train = phase1.build_canonical_node_frame(train_completed_raw, wave)
    safe_test = phase1.build_canonical_node_frame(test_completed_raw, wave)

    # This is a high-value reproducibility assertion: TRAIN is unchanged by target hiding,
    # so the exact Phase-1 completion must be reproduced from the frozen seeds/code.
    dataframe_numeric_equal(
        replay_train.reset_index(drop=True),
        saved_train_canonical.reset_index(drop=True),
        f"W{wave} F{fold} M{completion} replayed TRAIN vs saved TRAIN",
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    safe_test.to_csv(cache_path, index=False, compression="gzip")
    return safe_test


# ======================================================================================
# DISCRETE BASELINE: FIXED-DAG MLE CPT PREDICTIVE FACTORS
# ======================================================================================

@dataclass
class DiscreteCPTFactor:
    child: str
    parents: tuple[str, ...]
    child_cardinality: int
    parent_cardinalities: tuple[int, ...]
    table: np.ndarray  # shape (q, r)

    @staticmethod
    def _codes(frame: pd.DataFrame, parents: tuple[str, ...], cards: tuple[int, ...]) -> np.ndarray:
        if not parents:
            return np.zeros(len(frame), dtype=np.int64)
        matrix = frame[list(parents)].to_numpy(dtype=np.int64)
        code = np.zeros(len(frame), dtype=np.int64)
        mult = 1
        for j in range(len(parents) - 1, -1, -1):
            vals = matrix[:, j]
            card = cards[j]
            if np.any((vals < 0) | (vals >= card)):
                raise ValueError(f"Out-of-range parent state for {parents[j]}")
            code += vals * mult
            mult *= card
        return code

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        child: str,
        parents: Sequence[str],
        cardinalities: Mapping[str, int],
    ) -> "DiscreteCPTFactor":
        parents = tuple(sorted(map(str, parents)))
        r = int(cardinalities[child])
        pcards = tuple(int(cardinalities[p]) for p in parents)
        q = int(np.prod(pcards)) if pcards else 1
        parent_codes = cls._codes(train, parents, pcards)
        y = train[child].to_numpy(dtype=np.int64)
        if np.any((y < 0) | (y >= r)):
            raise ValueError(f"Out-of-range child state for {child}")
        counts = np.bincount(parent_codes * r + y, minlength=q * r).reshape(q, r).astype(float)
        totals = counts.sum(axis=1, keepdims=True)
        table = np.empty_like(counts)
        seen = totals[:, 0] > 0
        table[seen] = counts[seen] / totals[seen]
        table[~seen] = 1.0 / r  # MLE undefined for unseen parent config; neutral backoff.
        table = np.maximum(table, PROB_FLOOR)
        table /= table.sum(axis=1, keepdims=True)
        return cls(child, parents, r, pcards, table)

    def log_prob(self, frame: pd.DataFrame) -> np.ndarray:
        code = self._codes(frame, self.parents, self.parent_cardinalities)
        y = frame[self.child].to_numpy(dtype=np.int64)
        p = self.table[code, y]
        return np.log(np.maximum(p, PROB_FLOOR))


class DiscreteDAGModel:
    def __init__(self, train: pd.DataFrame, graph: nx.DiGraph, cardinalities: Mapping[str, int]):
        self.train = train.reset_index(drop=True)
        self.graph = graph
        self.cardinalities = {str(k): int(v) for k, v in cardinalities.items()}
        self._factors: dict[str, DiscreteCPTFactor] = {}

    def factor(self, child: str) -> DiscreteCPTFactor:
        if child not in self._factors:
            parents = tuple(self.graph.predecessors(child))
            self._factors[child] = DiscreteCPTFactor.fit(
                self.train, child, parents, self.cardinalities
            )
        return self._factors[child]

    def log_unnormalized(self, target: str, frames: pd.DataFrame) -> np.ndarray:
        values = self.factor(target).log_prob(frames)
        for child in self.graph.successors(target):
            values = values + self.factor(str(child)).log_prob(frames)
        return values

    def predict_states(self, target: str, evidence: pd.DataFrame, cardinality: int) -> np.ndarray:
        n = len(evidence)
        k = int(cardinality)
        expanded = pd.DataFrame(
            np.repeat(evidence.to_numpy(), k, axis=0),
            columns=evidence.columns,
        )
        expanded[target] = np.tile(np.arange(k, dtype=np.int64), n)
        logq = self.log_unnormalized(target, expanded).reshape(n, k)
        bad = ~np.isfinite(logsumexp(logq, axis=1))
        if np.any(bad):
            raise FloatingPointError(f"Discrete full conditional has no finite state for {int(bad.sum())} rows.")
        logp = logq - logsumexp(logq, axis=1, keepdims=True)
        return np.exp(logp)


# ======================================================================================
# CONTINUOUS BASELINE: FIXED-DAG LINEAR-GAUSSIAN FULL CONDITIONAL
# ======================================================================================

@dataclass
class LinearGaussianFactor:
    child: str
    parents: tuple[str, ...]
    intercept: float
    coefficients: dict[str, float]
    variance: float

    @classmethod
    def fit(cls, train: pd.DataFrame, child: str, parents: Sequence[str], minimum_variance: float = 1e-12):
        parents = tuple(sorted(map(str, parents)))
        y = train[child].to_numpy(dtype=float)
        n = len(train)
        if not parents:
            intercept = float(y.mean())
            coefs: dict[str, float] = {}
            residual = y - intercept
        else:
            X = np.column_stack([np.ones(n), train[list(parents)].to_numpy(dtype=float)])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            intercept = float(beta[0])
            coefs = {p: float(b) for p, b in zip(parents, beta[1:])}
            residual = y - X @ beta
        var = max(float(residual @ residual / n), float(minimum_variance))
        return cls(str(child), parents, intercept, coefs, var)

    def conditional_mean(self, frame: pd.DataFrame, exclude_parent: str | None = None) -> np.ndarray:
        mu = np.full(len(frame), self.intercept, dtype=float)
        for p, b in self.coefficients.items():
            if p == exclude_parent:
                continue
            mu += b * frame[p].to_numpy(dtype=float)
        return mu


class ContinuousDAGModel:
    def __init__(self, train: pd.DataFrame, graph: nx.DiGraph):
        self.train = train.reset_index(drop=True)
        self.graph = graph
        self._factors: dict[str, LinearGaussianFactor] = {}

    def factor(self, child: str) -> LinearGaussianFactor:
        if child not in self._factors:
            self._factors[child] = LinearGaussianFactor.fit(
                self.train, child, tuple(self.graph.predecessors(child))
            )
        return self._factors[child]

    def full_conditional_gaussian(self, target: str, evidence: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        target_factor = self.factor(target)
        mu0 = target_factor.conditional_mean(evidence)
        precision = np.full(len(evidence), 1.0 / target_factor.variance, dtype=float)
        numerator = mu0 / target_factor.variance

        for child in self.graph.successors(target):
            f = self.factor(str(child))
            beta_t = float(f.coefficients.get(target, 0.0))
            if np.isclose(beta_t, 0.0, rtol=1e-09, atol=1e-09):
                # This should not happen if target is a graph parent, but keep a strict guard.
                raise RuntimeError(f"Target {target} is a parent of {child} but fitted coefficient is missing.")
            a = f.conditional_mean(evidence, exclude_parent=target)
            y_obs = evidence[str(child)].to_numpy(dtype=float)
            precision += (beta_t * beta_t) / f.variance
            numerator += beta_t * (y_obs - a) / f.variance

        var = 1.0 / precision
        mean = numerator * var
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(var)) or np.any(var <= 0):
            raise FloatingPointError("Invalid continuous-baseline full conditional.")
        return mean, var


# ======================================================================================
# HYBRID V3: SCORE-ALIGNED LOCAL PREDICTIVE FACTORS
# ======================================================================================

@dataclass
class HybridDDPredictive:
    child: str
    parents: tuple[str, ...]
    child_cardinality: int
    parent_cardinalities: tuple[int, ...]
    table: np.ndarray

    @classmethod
    def fit(cls, data: pd.DataFrame, child: str, parents: Sequence[str], cards: Mapping[str, int], ess: float):
        parents = tuple(sorted(map(str, parents)))
        r = int(cards[child])
        pcards = tuple(int(cards[p]) for p in parents)
        q = int(np.prod(pcards)) if pcards else 1
        parent_codes = DiscreteCPTFactor._codes(data, parents, pcards)
        y = data[child].to_numpy(dtype=np.int64)
        counts = np.bincount(parent_codes * r + y, minlength=q * r).reshape(q, r).astype(float)
        alpha_j = float(ess) / q
        alpha_jk = float(ess) / (q * r)
        totals = counts.sum(axis=1, keepdims=True)
        table = (counts + alpha_jk) / (totals + alpha_j)
        table = clip_and_normalize_probabilities(table)
        return cls(child, parents, r, pcards, table)

    def log_prob(self, frame: pd.DataFrame) -> np.ndarray:
        code = DiscreteCPTFactor._codes(frame, self.parents, self.parent_cardinalities)
        y = frame[self.child].to_numpy(dtype=np.int64)
        return np.log(np.maximum(self.table[code, y], PROB_FLOOR))


@dataclass
class DCGroupPredictive:
    k: int
    beta: np.ndarray  # k x p (or 1 x p)
    variances: np.ndarray  # k
    gamma: np.ndarray | None  # (k-1) x p for k>1

    def logpdf(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.k == 1:
            mu = X @ self.beta[0]
            v = max(float(self.variances[0]), 1e-15)
            return -0.5 * (np.log(2.0 * np.pi * v) + (y - mu) ** 2 / v)
        H = helmert(self.k, full=False).T
        gate_coef = H @ np.asarray(self.gamma, dtype=float)
        logits = X @ gate_coef.T
        log_gate = logits - logsumexp(logits, axis=1, keepdims=True)
        means = X @ self.beta.T
        log_normal = -0.5 * (
            np.log(2.0 * np.pi)
            + np.log(self.variances)[None, :]
            + (y[:, None] - means) ** 2 / self.variances[None, :]
        )
        return logsumexp(log_gate + log_normal, axis=1)


@dataclass
class DCPredictiveFamily:
    child: str
    c_parents: tuple[str, ...]
    d_parents: tuple[str, ...]
    groups: dict[tuple[int, ...], DCGroupPredictive]
    fallback: DCGroupPredictive

    @staticmethod
    def _key(row: Sequence[int]) -> tuple[int, ...]:
        return tuple(int(v) for v in row)

    def log_prob(self, frame: pd.DataFrame) -> np.ndarray:
        X = np.ones((len(frame), 1), dtype=float)
        if self.c_parents:
            X = np.column_stack([np.ones(len(frame)), frame[list(self.c_parents)].to_numpy(dtype=float)])
        y = frame[self.child].to_numpy(dtype=float)
        if not self.d_parents:
            return self.fallback.logpdf(X, y)
        matrix = frame[list(self.d_parents)].to_numpy(dtype=np.int64)
        result = np.empty(len(frame), dtype=float)
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i, row in enumerate(matrix):
            groups[self._key(row)].append(i)
        for key, idxs in groups.items():
            idx = np.asarray(idxs, dtype=int)
            model = self.groups.get(key, self.fallback)
            result[idx] = model.logpdf(X[idx], y[idx])
        return result


def nig_posterior_mean_predictive(
    X: np.ndarray,
    y: np.ndarray,
    *,
    coefficient_prior_variance: float,
    intercept_prior_variance: float,
    inverse_gamma_shape: float,
    inverse_gamma_scale: float,
) -> DCGroupPredictive:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    variances0 = np.full(p, float(coefficient_prior_variance), dtype=float)
    variances0[0] = float(intercept_prior_variance)
    precision = X.T @ X + np.diag(1.0 / variances0)
    rhs = X.T @ y
    beta = np.linalg.solve(precision, rhs)
    a0 = float(inverse_gamma_shape)
    b0 = float(inverse_gamma_scale)
    an = a0 + 0.5 * n
    quad = float(y @ y - beta @ rhs)
    bn = b0 + 0.5 * max(quad, 0.0)
    sigma2_mean = float(bn / (an - 1.0))
    return DCGroupPredictive(1, beta[None, :], np.asarray([max(sigma2_mean, 1e-15)]), None)


def fit_dc_mixture_map(dc, X: np.ndarray, y: np.ndarray, k: int, *, coefficient_prior_variance: float,
                       intercept_prior_variance: float, inverse_gamma_shape: float, inverse_gamma_scale: float) -> DCGroupPredictive:
    objective, gradient, _, responsibilities = dc._mixture_problem(
        X,
        y,
        k,
        coefficient_prior_variance=coefficient_prior_variance,
        intercept_prior_variance=intercept_prior_variance,
        inverse_gamma_shape=inverse_gamma_shape,
        inverse_gamma_scale=inverse_gamma_scale,
    )
    starts = dc._mixture_initializations(X, y, k)
    p = X.shape[1]
    beta_size = k * p
    variance_size = k
    bounds = (
        [(None, None)] * beta_size
        + [(-8.0, 4.0)] * variance_size
        + [(None, None)] * ((k - 1) * p)
    )
    best = None
    best_fun = np.inf
    for initial in starts:
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds,
            options={"maxiter": dc.mixture_max_iter, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50},
        )
        if np.isfinite(result.fun) and float(result.fun) < best_fun:
            best_fun = float(result.fun)
            best = result
    if best is None:
        raise FloatingPointError("DC selected mixture refit failed at every deterministic start.")
    theta = np.asarray(best.x, dtype=float)
    grad_norm = float(np.linalg.norm(gradient(theta), ord=np.inf))
    if not best.success and grad_norm > 1e-3:
        raise FloatingPointError(f"DC selected mixture refit did not converge; grad={grad_norm:.3e}")
    eff = responsibilities(theta).sum(axis=0)
    if float(np.min(eff)) < dc.mixture_min_effective_count:
        raise FloatingPointError("DC selected mixture refit produced a near-empty component.")
    betas = theta[:beta_size].reshape(k, p)
    start = beta_size
    logv = theta[start:start + variance_size]
    gamma = theta[start + variance_size:].reshape(k - 1, p)
    return DCGroupPredictive(k, betas, np.exp(logv), gamma)


def fit_dc_group_with_hard_k(
    dc,
    X: np.ndarray,
    y: np.ndarray,
    *,
    child: str,
    all_parents: tuple[str, ...],
    configuration: Any,
    coefficient_prior_variance: float,
    intercept_prior_variance: float,
    inverse_gamma_shape: float,
    inverse_gamma_scale: float,
) -> tuple[DCGroupPredictive, int]:
    before = len(dc.mixture_diagnostics)
    _, _, selected_k = dc._score_continuous_group(
        child=child,
        all_parents=all_parents,
        configuration=configuration,
        X=X,
        y=y,
        coefficient_prior_variance=coefficient_prior_variance,
        intercept_prior_variance=intercept_prior_variance,
        inverse_gamma_shape=inverse_gamma_shape,
        inverse_gamma_scale=inverse_gamma_scale,
    )
    # Explicitly preserve the frozen Phase-1 hard-K policy.
    module = sys.modules.get(dc.__class__.__module__)
    if module is not None and bool(getattr(module, "DC_BMA_ENABLED", False)):
        raise RuntimeError("DC_BMA_ENABLED must be False for Phase-2 score-aligned refitting.")
    if selected_k == 1:
        model = nig_posterior_mean_predictive(
            X,
            y,
            coefficient_prior_variance=coefficient_prior_variance,
            intercept_prior_variance=intercept_prior_variance,
            inverse_gamma_shape=inverse_gamma_shape,
            inverse_gamma_scale=inverse_gamma_scale,
        )
    else:
        model = fit_dc_mixture_map(
            dc,
            X,
            y,
            selected_k,
            coefficient_prior_variance=coefficient_prior_variance,
            intercept_prior_variance=intercept_prior_variance,
            inverse_gamma_shape=inverse_gamma_shape,
            inverse_gamma_scale=inverse_gamma_scale,
        )
    # Sanity: exactly one candidate must be marked selected in the just-added diagnostics.
    added = dc.mixture_diagnostics[before:]
    marked = [d for d in added if bool(d.selected)]
    if len(marked) != 1 or int(marked[0].n_components) != int(selected_k):
        raise RuntimeError("DC hard-K diagnostic/refit selection mismatch.")
    return model, int(selected_k)


def normalize_config_key(configuration: Any) -> tuple[int, ...]:
    if isinstance(configuration, tuple):
        return tuple(int(x) for x in configuration)
    if isinstance(configuration, (list, np.ndarray)):
        return tuple(int(x) for x in configuration)
    return (int(configuration),)


def fit_dc_predictive_family(engine, child: str, parents: Sequence[str]) -> DCPredictiveFamily:
    dc = engine.dc
    if dc is None:
        raise RuntimeError("Hybrid DC scorer unavailable.")
    parents = tuple(sorted(map(str, parents)))
    c_parents = tuple(p for p in parents if p in engine._continuous_set)
    d_parents = tuple(p for p in parents if p in engine._discrete_set)
    if not d_parents:
        raise ValueError("DC predictive fit requires at least one discrete parent.")

    groups: dict[tuple[int, ...], DCGroupPredictive] = {}
    for configuration, group in dc._iter_groups(d_parents):
        X = dc._design_matrix(group, c_parents)
        y = group[child].to_numpy(dtype=float)
        model, _ = fit_dc_group_with_hard_k(
            dc,
            X,
            y,
            child=child,
            all_parents=parents,
            configuration=configuration,
            coefficient_prior_variance=dc.clg_coefficient_prior_variance,
            intercept_prior_variance=dc.clg_intercept_prior_variance,
            inverse_gamma_shape=dc.clg_inverse_gamma_shape,
            inverse_gamma_scale=dc.clg_inverse_gamma_scale,
        )
        groups[normalize_config_key(configuration)] = model

    # Score-consistent families are undefined for a discrete-parent configuration that
    # never occurred in TRAIN.  In the common case where every legal configuration was
    # observed, no backoff can ever be used, so do NOT pay for another K=1..3 Laplace
    # selection/refit.  Only fit a deterministic global C-parent backoff when the TRAIN
    # support is genuinely incomplete.
    possible = int(dc._configuration_count(d_parents))
    if len(groups) >= possible:
        fallback = next(iter(groups.values()))  # unreachable for legal test configurations
    else:
        Xall = dc._design_matrix(dc.data, c_parents)
        yall = dc.data[child].to_numpy(dtype=float)
        fallback, _ = fit_dc_group_with_hard_k(
            dc,
            Xall,
            yall,
            child=child,
            all_parents=parents,
            configuration="__fallback__",
            coefficient_prior_variance=dc.clg_coefficient_prior_variance,
            intercept_prior_variance=dc.clg_intercept_prior_variance,
            inverse_gamma_shape=dc.clg_inverse_gamma_shape,
            inverse_gamma_scale=dc.clg_inverse_gamma_scale,
        )
    return DCPredictiveFamily(child, c_parents, d_parents, groups, fallback)


class HybridDAGModel:
    def __init__(
        self,
        *,
        train_external: pd.DataFrame,
        graph: nx.DiGraph,
        hybrid_module,
        discrete_nodes: Sequence[str],
        continuous_nodes: Sequence[str],
        cardinalities: Mapping[str, int],
    ) -> None:
        self.graph = graph
        self.discrete_nodes = tuple(map(str, discrete_nodes))
        self.continuous_nodes = tuple(map(str, continuous_nodes))
        self.cardinalities = {str(k): int(v) for k, v in cardinalities.items()}
        self.engine = hybrid_module.HybridV3ScoringEngine(
            train_external.copy(),
            discrete_nodes=self.discrete_nodes,
            continuous_nodes=self.continuous_nodes,
            cardinalities=self.cardinalities,
        )
        # Phase-1 used the hard-K DC score; do not permit an accidental source change.
        dc_module = sys.modules.get(self.engine.dc.__class__.__module__) if self.engine.dc is not None else None
        if dc_module is not None and bool(getattr(dc_module, "DC_BMA_ENABLED", False)):
            raise RuntimeError("Frozen Hybrid Phase-1 semantics require DC_BMA_ENABLED=False.")
        self._factors: dict[str, tuple[str, Any, tuple[str, ...]]] = {}

    def factor(self, child: str):
        child = str(child)
        if child in self._factors:
            return self._factors[child]
        parents = tuple(sorted(map(str, self.graph.predecessors(child))))
        quadrant = self.engine.quadrant(child, parents)
        c_parents = tuple(p for p in parents if p in set(self.continuous_nodes))
        d_parents = tuple(p for p in parents if p in set(self.discrete_nodes))

        if quadrant == "DD":
            factor = HybridDDPredictive.fit(
                self.engine.data,
                child,
                d_parents,
                self.cardinalities,
                float(self.engine.dd.equivalent_sample_size),
            )
        elif quadrant == "CD":
            factor = self.engine.cd.fit_predictive_family(
                child,
                c_parents,
                d_parents,
                all_parents=parents,
            )
        elif quadrant == "DC":
            factor = fit_dc_predictive_family(self.engine, child, parents)
        elif quadrant == "CC":
            # The corrected Final-V3 scorer owns both refit and predictive density.
            factor = None
            self.engine.cc.engine.score_local(child, c_parents)
        else:
            raise KeyError(quadrant)
        self._factors[child] = (quadrant, factor, parents)
        return self._factors[child]

    def prepare_target(self, target: str) -> None:
        self.factor(target)
        for child in self.graph.successors(target):
            self.factor(str(child))

    def _factor_log_prob(
        self,
        child: str,
        external_frame: pd.DataFrame,
        transformed_frame: pd.DataFrame,
        continuous_external: pd.DataFrame,
    ) -> np.ndarray:
        quadrant, factor, parents = self.factor(child)
        if quadrant == "DD":
            return factor.log_prob(transformed_frame)
        if quadrant == "CD":
            return factor.log_prob(transformed_frame)
        if quadrant == "DC":
            return factor.log_prob(transformed_frame)
        if quadrant == "CC":
            c_parents = tuple(p for p in parents if p in set(self.continuous_nodes))
            return np.asarray(
                self.engine.cc.engine.local_predictive_logpdf(child, c_parents, continuous_external),
                dtype=float,
            )
        raise KeyError(quadrant)

    def log_unnormalized(self, target: str, external_frame: pd.DataFrame) -> np.ndarray:
        self.prepare_target(target)
        transformed = self.engine.transform_new_data(external_frame)
        continuous_external = external_frame.loc[:, list(self.continuous_nodes)].copy()
        total = self._factor_log_prob(target, external_frame, transformed, continuous_external)
        for child in self.graph.successors(target):
            total = total + self._factor_log_prob(str(child), external_frame, transformed, continuous_external)
        if total.shape != (len(external_frame),):
            raise RuntimeError("Hybrid local-factor evaluation returned wrong shape.")
        return total

    def predict_discrete(self, target: str, evidence_external: pd.DataFrame, cardinality: int) -> np.ndarray:
        n = len(evidence_external)
        k = int(cardinality)
        expanded = pd.DataFrame(
            np.repeat(evidence_external.to_numpy(), k, axis=0),
            columns=evidence_external.columns,
        )
        expanded[target] = np.tile(np.arange(k, dtype=np.int64), n)
        logq = self.log_unnormalized(target, expanded).reshape(n, k)
        logz = logsumexp(logq, axis=1)
        if np.any(~np.isfinite(logz)):
            raise FloatingPointError("Hybrid discrete full conditional failed to normalize.")
        return np.exp(logq - logz[:, None])


# ======================================================================================
# CONTINUOUS-TARGET INTEGRATION FOR HYBRID
# ======================================================================================

@dataclass
class QuadratureGrid:
    nodes: np.ndarray
    weights: np.ndarray
    bin_index: np.ndarray
    lo: float
    hi: float
    cutpoints: tuple[float, float]


def build_quadrature_grid(
    train_target: Sequence[float],
    cutpoints: Sequence[float],
    *,
    nodes_per_bin: int,
    tail_sd: float,
) -> QuadratureGrid:
    x = np.asarray(train_target, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Hybrid integration training target contains non-finite values.")
    if len(x) < 2:
        raise ValueError("Need at least two training values for integration support.")
    c1, c2 = map(float, cutpoints)
    if c1 >= c2:
        raise ValueError("Common cutpoints must be increasing.")
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = max(float(np.ptp(x)), 1.0)
    lo = min(float(np.min(x)) - float(tail_sd) * sd, c1 - 2.0 * sd)
    hi = max(float(np.max(x)) + float(tail_sd) * sd, c2 + 2.0 * sd)
    if not lo < c1 < c2 < hi:
        raise RuntimeError(f"Invalid integration support {lo}, {c1}, {c2}, {hi}")

    z, w = leggauss(int(nodes_per_bin))
    nodes_list = []
    weights_list = []
    bins = []
    for j, (a, b) in enumerate(((lo, c1), (c1, c2), (c2, hi))):
        mapped = 0.5 * (b - a) * z + 0.5 * (a + b)
        mapped_w = 0.5 * (b - a) * w
        nodes_list.append(mapped)
        weights_list.append(mapped_w)
        bins.append(np.full(len(mapped), j, dtype=np.int64))
    return QuadratureGrid(
        nodes=np.concatenate(nodes_list),
        weights=np.concatenate(weights_list),
        bin_index=np.concatenate(bins),
        lo=float(lo),
        hi=float(hi),
        cutpoints=(c1, c2),
    )


def hybrid_continuous_predictions(
    *,
    model: HybridDAGModel,
    target: str,
    evidence_external: pd.DataFrame,
    true_values: np.ndarray,
    train_target_values: np.ndarray,
    cutpoints: tuple[float, float],
    nodes_per_bin: int,
    tail_sd: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    grid = build_quadrature_grid(
        train_target_values,
        cutpoints,
        nodes_per_bin=nodes_per_bin,
        tail_sd=tail_sd,
    )
    n = len(evidence_external)
    probs = np.empty((n, 3), dtype=float)
    log_density_true = np.empty(n, dtype=float)
    pred_mean = np.empty(n, dtype=float)
    tail_warn = np.zeros(n, dtype=bool)
    log_quad_w = np.log(grid.weights)

    target_col_idx = list(evidence_external.columns).index(target)
    qn = len(grid.nodes)

    for start in range(0, n, int(chunk_size)):
        stop = min(n, start + int(chunk_size))
        base = evidence_external.iloc[start:stop].copy().reset_index(drop=True)
        b = len(base)
        expanded_np = np.repeat(base.to_numpy(), qn, axis=0)
        expanded_np[:, target_col_idx] = np.tile(grid.nodes, b)
        expanded = pd.DataFrame(expanded_np, columns=base.columns)
        logq = model.log_unnormalized(target, expanded).reshape(b, qn)
        if np.any(np.isnan(logq)):
            raise FloatingPointError("NaN in Hybrid continuous-target quadrature log q.")

        bin_logz = np.empty((b, 3), dtype=float)
        for j in range(3):
            mask = grid.bin_index == j
            bin_logz[:, j] = logsumexp(logq[:, mask] + log_quad_w[mask][None, :], axis=1)
        total_logz = logsumexp(bin_logz, axis=1)
        if np.any(~np.isfinite(total_logz)):
            raise FloatingPointError("Hybrid continuous-target quadrature normalization failed.")
        probs[start:stop] = np.exp(bin_logz - total_logz[:, None])

        normalized_w = np.exp(logq + log_quad_w[None, :] - total_logz[:, None])
        pred_mean[start:stop] = normalized_w @ grid.nodes

        truth_frame = base.copy()
        truth_frame[target] = true_values[start:stop]
        logq_true = model.log_unnormalized(target, truth_frame)
        log_density_true[start:stop] = logq_true - total_logz

        # Tail diagnostic only; support is TRAIN-defined and never depends on held-out truth.
        endpoints_np = np.repeat(base.to_numpy(), 2, axis=0)
        endpoints_np[:, target_col_idx] = np.tile(np.asarray([grid.lo, grid.hi]), b)
        endpoints = pd.DataFrame(endpoints_np, columns=base.columns)
        end_logq = model.log_unnormalized(target, endpoints).reshape(b, 2)
        max_grid = np.max(logq, axis=1)
        tail_warn[start:stop] = np.max(end_logq, axis=1) > (max_grid - TAIL_LOG_GAP_WARNING)

    probs = clip_and_normalize_probabilities(probs, floor=PROB_FLOOR)
    diagnostics = {
        "integration_nodes_total": int(len(grid.nodes)),
        "nodes_per_common_bin": int(nodes_per_bin),
        "support_lo": float(grid.lo),
        "support_hi": float(grid.hi),
        "tail_sd": float(tail_sd),
        "tail_warning_count": int(tail_warn.sum()),
        "tail_warning_fraction": float(tail_warn.mean()) if len(tail_warn) else 0.0,
    }
    return probs, log_density_true, pred_mean, diagnostics


# ======================================================================================
# REPRESENTATIONS AND PER-COMPLETION PREDICTION
# ======================================================================================

def harmonized_discrete_train(
    phase1,
    train_canonical: pd.DataFrame,
    base_discrete_train: pd.DataFrame,
    target: str,
    common_cuts: tuple[float, float] | None,
) -> pd.DataFrame:
    train = base_discrete_train.copy()
    if target in CONTINUOUS_TARGETS:
        if common_cuts is None:
            raise ValueError(f"Common cutpoints required for {target}")
        locked = phase1.locked_native_discrete_frame(train_canonical)
        train[target] = apply_common_bins(locked[target].to_numpy(dtype=float), common_cuts)
    return train


def placeholder_hide_target(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    out = frame.copy()
    # All model representations are numeric; zero is a finite placeholder and is
    # overwritten before any target-dependent factor is evaluated.
    out[target] = 0
    return out


def eligible_evidence_and_truth(
    *,
    phase1,
    raw_test: pd.DataFrame,
    canonical_test: pd.DataFrame,
    representation_test: pd.DataFrame,
    wave: int,
    target: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    mask = target_observed_mask(phase1, raw_test, wave, target)
    ids = raw_test.loc[mask, "idauniq"].to_numpy()
    true_raw = raw_target_values(raw_test, phase1, wave, target, mask)

    rep = representation_test.copy()
    rep.insert(0, "idauniq", canonical_test["idauniq"].to_numpy())
    indexed = rep.set_index("idauniq", drop=True)
    try:
        evidence = indexed.loc[ids].reset_index(drop=True)
    except KeyError as exc:
        raise RuntimeError(f"Completed TEST representation missing eligible IDs for {target}.") from exc

    if target in CONTINUOUS_TARGETS:
        true_state = np.full(len(ids), -1, dtype=np.int64)  # filled by common bins later
    else:
        true_state = recode_discrete_truth(target, true_raw)
    return evidence, ids, true_raw, true_state


def integration_settings(model_key: str, target: str, args) -> tuple[int, int]:
    """Use converged quadrature only for Hybrid Loneliness; preserve other defaults."""
    if model_key == "hybrid" and target == "Loneliness":
        return LONELINESS_INTEGRATION_NODES_PER_BIN, LONELINESS_INTEGRATION_CHUNK
    return int(args.integration_nodes_per_bin), int(args.integration_chunk_size)


def integration_marker_path(member_path: Path) -> Path:
    """Sidecar prevents silent reuse of pre-correction Hybrid Loneliness members."""
    return member_path.with_name(member_path.name + ".quadrature.json")


def loneliness_member_valid(member_path: Path, args) -> bool:
    if not member_path.exists():
        return False
    marker = integration_marker_path(member_path)
    if not marker.exists():
        return False
    try:
        payload = read_json(marker)
        return (payload.get("model") == "hybrid"
                and payload.get("target") == "Loneliness"
                and int(payload.get("nodes_per_bin", -1)) == LONELINESS_INTEGRATION_NODES_PER_BIN
                and int(payload.get("total_nodes", -1)) == 3 * LONELINESS_INTEGRATION_NODES_PER_BIN
                and float(payload.get("tail_sd", -1)) == float(args.integration_tail_sd)
                and bool(payload.get("complete", False)))
    except (ValueError, TypeError, OSError, KeyError):
        return False


def member_output_path(out_root: Path, wave: int, fold: int, completion: int, model_key: str, target: str) -> Path:
    return (
        out_root / "members" / f"W{wave}" / f"Fold{fold:02d}" / f"M{completion:03d}" /
        MODEL_DIR_NAMES[model_key] / f"{target}.csv.gz"
    )


def save_member_prediction(
    path: Path,
    *,
    ids: np.ndarray,
    wave: int,
    fold: int,
    completion: int,
    target: str,
    model_key: str,
    true_raw: np.ndarray,
    true_state: np.ndarray,
    probs: np.ndarray,
    log_density_true: np.ndarray | None = None,
    predictive_mean: np.ndarray | None = None,
    common_cuts: tuple[float, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "idauniq": ids,
        "wave": np.full(len(ids), int(wave)),
        "fold": np.full(len(ids), int(fold)),
        "completion": np.full(len(ids), int(completion)),
        "target": np.full(len(ids), target),
        "model": np.full(len(ids), MODEL_DISPLAY[model_key]),
        "true_raw": true_raw,
        "true_state": true_state,
    }
    for j in range(probs.shape[1]):
        payload[f"p{j}"] = probs[:, j]
    if log_density_true is not None:
        payload["log_density_true"] = log_density_true
    if predictive_mean is not None:
        payload["predictive_mean"] = predictive_mean
    if common_cuts is not None:
        payload["common_cut1"] = np.full(len(ids), common_cuts[0])
        payload["common_cut2"] = np.full(len(ids), common_cuts[1])
    pd.DataFrame(payload).to_csv(path, index=False, compression="gzip")


def predict_one_model_target(
    *,
    phase1,
    model_key: str,
    model_module,
    root: Path,
    train_canonical: pd.DataFrame,
    test_canonical: pd.DataFrame,
    raw_test: pd.DataFrame,
    graph: nx.DiGraph,
    rep_train: Mapping[str, pd.DataFrame],
    rep_test: Mapping[str, pd.DataFrame],
    run_config: Mapping[str, Any],
    target: str,
    common_cuts: tuple[float, float] | None,
    integration_nodes_per_bin: int,
    integration_tail_sd: float,
    integration_chunk: int,
    model_cache: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    node_order = list(run_config["locked_design"]["node_order"])
    cardinalities_discrete = {
        str(k): int(v) for k, v in run_config["locked_design"]["discrete_baseline_cardinalities"].items()
    }
    hybrid_cards = {
        str(k): int(v) for k, v in run_config["locked_design"]["hybrid_cardinalities"].items()
    }
    continuous_nodes = tuple(run_config["locked_design"]["continuous_nodes"])
    native_discrete_nodes = tuple(run_config["locked_design"]["native_discrete_nodes"])

    evidence, ids, true_raw, true_state = eligible_evidence_and_truth(
        phase1=phase1,
        raw_test=raw_test,
        canonical_test=test_canonical,
        representation_test=rep_test[model_key],
        wave=int(raw_test.attrs["wave"]),
        target=target,
    )

    if target in CONTINUOUS_TARGETS:
        if common_cuts is None:
            raise ValueError("Continuous target requires common cutpoints.")
        true_state = apply_common_bins(true_raw, common_cuts)
        categorical_k = 3
    else:
        categorical_k = DISCRETE_TARGET_CARDINALITY[target]

    evidence = placeholder_hide_target(evidence[node_order], target)
    diagnostics: dict[str, Any] = {}
    if model_cache is None:
        model_cache = {}

    if model_key == "discrete":
        # A common-bin continuous target changes the TRAIN representation of that
        # target and therefore any local child factor containing it as a parent.
        # Native discrete targets all share the original Phase-1 representation.
        cache_key = f"discrete:{target}" if target in CONTINUOUS_TARGETS else "discrete:native"
        model = model_cache.get(cache_key)
        if model is None:
            train_d = harmonized_discrete_train(
                phase1,
                train_canonical,
                rep_train["discrete"],
                target,
                common_cuts,
            )
            model = DiscreteDAGModel(train_d[node_order], graph, cardinalities_discrete)
            model_cache[cache_key] = model
        probs = model.predict_states(target, evidence, categorical_k)
        return ids, true_raw, true_state, probs, None, None, diagnostics

    if model_key == "continuous":
        model = model_cache.get("continuous")
        if model is None:
            model = ContinuousDAGModel(rep_train["continuous"][node_order], graph)
            model_cache["continuous"] = model
        mean, var = model.full_conditional_gaussian(target, evidence)
        sd = np.sqrt(var)
        if target in CONTINUOUS_TARGETS:
            c1, c2 = common_cuts
            z1 = (c1 - mean) / sd
            z2 = (c2 - mean) / sd
            p0 = norm.cdf(z1)
            p1 = norm.cdf(z2) - norm.cdf(z1)
            p2 = 1.0 - norm.cdf(z2)
            probs = clip_and_normalize_probabilities(np.column_stack([p0, p1, p2]))
            log_density = norm.logpdf(true_raw, loc=mean, scale=sd)
            return ids, true_raw, true_state, probs, log_density, mean, diagnostics
        # Gaussian-forced legal-state probabilities use midpoint/Voronoi intervals.
        k = categorical_k
        boundaries = np.arange(k - 1, dtype=float) + 0.5
        cdfs = [norm.cdf((b - mean) / sd) for b in boundaries]
        cols = []
        if k == 2:
            cols = [cdfs[0], 1.0 - cdfs[0]]
        else:
            cols = [cdfs[0]]
            for j in range(1, k - 1):
                cols.append(cdfs[j] - cdfs[j - 1])
            cols.append(1.0 - cdfs[-1])
        probs = clip_and_normalize_probabilities(np.column_stack(cols))
        return ids, true_raw, true_state, probs, None, None, diagnostics

    if model_key == "hybrid":
        model = model_cache.get("hybrid")
        if model is None:
            model = HybridDAGModel(
                train_external=rep_train["hybrid"][node_order],
                graph=graph,
                hybrid_module=model_module,
                discrete_nodes=native_discrete_nodes,
                continuous_nodes=continuous_nodes,
                cardinalities=hybrid_cards,
            )
            model_cache["hybrid"] = model
        model.prepare_target(target)
        if target in CONTINUOUS_TARGETS:
            probs, log_density, pred_mean, qdiag = hybrid_continuous_predictions(
                model=model,
                target=target,
                evidence_external=evidence,
                true_values=true_raw,
                train_target_values=rep_train["hybrid"][target].to_numpy(dtype=float),
                cutpoints=common_cuts,
                nodes_per_bin=integration_nodes_per_bin,
                tail_sd=integration_tail_sd,
                chunk_size=integration_chunk,
            )
            diagnostics.update(qdiag)
            return ids, true_raw, true_state, probs, log_density, pred_mean, diagnostics
        probs = model.predict_discrete(target, evidence, categorical_k)
        return ids, true_raw, true_state, probs, None, None, diagnostics

    raise KeyError(model_key)


# ======================================================================================
# MEMBER -> FOLD ENSEMBLE -> OOF AGGREGATION
# ======================================================================================

def ensemble_output_path(out_root: Path, wave: int, fold: int, model_key: str, target: str) -> Path:
    return out_root / "ensemble" / f"W{wave}" / f"Fold{fold:02d}" / MODEL_DIR_NAMES[model_key] / f"{target}.csv.gz"


def oof_output_path(out_root: Path, wave: int, model_key: str, target: str) -> Path:
    return out_root / "oof" / f"W{wave}" / MODEL_DIR_NAMES[model_key] / f"{target}.csv.gz"


def probability_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c.startswith("p") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def aggregate_fold_members(
    *,
    out_root: Path,
    wave: int,
    fold: int,
    target: str,
    model_key: str,
    completions: Sequence[int],
    require_all: bool = True,
) -> Path | None:
    paths = [member_output_path(out_root, wave, fold, m, model_key, target) for m in completions]
    missing = [p for p in paths if not p.exists()]
    if missing:
        if require_all:
            raise FileNotFoundError(f"Missing {len(missing)} member predictions for W{wave} F{fold} {model_key} {target}")
        return None
    frames = [pd.read_csv(p).sort_values("idauniq").reset_index(drop=True) for p in paths]
    ref = frames[0]
    pcols = probability_columns(ref)
    for i, f in enumerate(frames[1:], start=2):
        if not np.array_equal(f["idauniq"].to_numpy(), ref["idauniq"].to_numpy()):
            raise RuntimeError(f"Member ID mismatch while aggregating W{wave} F{fold} {target} {model_key}, member {i}")
        if not np.array_equal(f["true_state"].to_numpy(), ref["true_state"].to_numpy()):
            raise RuntimeError("Member true-state mismatch.")
        if not np.allclose(f["true_raw"], ref["true_raw"], rtol=0.0, atol=1e-12):
            raise RuntimeError("Member true-raw mismatch.")
        if probability_columns(f) != pcols:
            raise RuntimeError("Member probability-column mismatch.")

    out = ref[["idauniq", "wave", "fold", "target", "model", "true_raw", "true_state"]].copy()
    parr = np.stack([f[pcols].to_numpy(dtype=float) for f in frames], axis=0)
    mean_p = parr.mean(axis=0)
    mean_p = clip_and_normalize_probabilities(mean_p)
    for j, col in enumerate(pcols):
        out[col] = mean_p[:, j]
    out["pred_state"] = np.argmax(mean_p, axis=1).astype(np.int64)
    out["ensemble_members"] = len(frames)

    if all("log_density_true" in f.columns for f in frames):
        logs = np.stack([f["log_density_true"].to_numpy(dtype=float) for f in frames], axis=0)
        out["ensemble_log_density_true"] = logsumexp(logs, axis=0) - math.log(len(frames))
    if all("predictive_mean" in f.columns for f in frames):
        means = np.stack([f["predictive_mean"].to_numpy(dtype=float) for f in frames], axis=0)
        out["ensemble_predictive_mean"] = means.mean(axis=0)
    if "common_cut1" in ref.columns:
        out["common_cut1"] = ref["common_cut1"]
        out["common_cut2"] = ref["common_cut2"]

    path = ensemble_output_path(out_root, wave, fold, model_key, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, compression="gzip")
    return path


def build_oof(
    *,
    out_root: Path,
    wave: int,
    target: str,
    model_key: str,
    folds: Sequence[int],
    require_all: bool = True,
) -> Path | None:
    paths = [ensemble_output_path(out_root, wave, f, model_key, target) for f in folds]
    missing = [p for p in paths if not p.exists()]
    if missing:
        if require_all:
            raise FileNotFoundError(f"Missing fold ensembles for W{wave} {target} {model_key}: {missing}")
        return None
    frames = [pd.read_csv(p) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    if out["idauniq"].duplicated().any():
        raise RuntimeError(f"OOF duplicate participant within W{wave} {target} {model_key}")
    out = out.sort_values("idauniq").reset_index(drop=True)
    path = oof_output_path(out_root, wave, model_key, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, compression="gzip")
    return path


# ======================================================================================
# METRICS + PAIRED BOOTSTRAP
# ======================================================================================

def categorical_metrics(frame: pd.DataFrame, target: str) -> dict[str, float]:
    pcols = probability_columns(frame)
    probs = frame[pcols].to_numpy(dtype=float)
    y = frame["true_state"].to_numpy(dtype=np.int64)
    pred = np.argmax(probs, axis=1).astype(np.int64)
    k = len(pcols)
    if target in BINARY_TARGETS:
        # Natural 0.5 operating point for binary targets.
        pred_bin = (probs[:, 1] >= 0.5).astype(np.int64)
        f1 = float(f1_score(y, pred_bin, pos_label=1, average="binary", zero_division=0))
    else:
        f1 = float(
            f1_score(y, pred, labels=list(range(k)), average="macro", zero_division=0)
        )
    ptrue = probs[np.arange(len(y)), y]
    logloss = float(-np.mean(np.log(np.clip(ptrue, METRIC_PROB_FLOOR, 1.0))))
    out = {"F1": f1, "LogLoss": logloss}
    if target == "Dementia":
        p1 = probs[:, 1]
        if len(np.unique(y)) == 2:
            out["ROC_AUC"] = float(roc_auc_score(y, p1))
            out["PR_AUC"] = float(average_precision_score(y, p1))
        else:
            out["ROC_AUC"] = float("nan")
            out["PR_AUC"] = float("nan")
        out["F1_at_0.5"] = f1
        precision, recall, thresholds = precision_recall_curve(y, p1)
        f = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        if len(thresholds):
            idx = int(np.nanargmax(f[:-1]))
            out["Oracle_Max_F1"] = float(f[idx])
            out["Oracle_Threshold"] = float(thresholds[idx])
        else:
            out["Oracle_Max_F1"] = float(f[0]) if len(f) else float("nan")
            out["Oracle_Threshold"] = float("nan")
    return out


def native_continuous_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if "ensemble_log_density_true" not in frame.columns or "ensemble_predictive_mean" not in frame.columns:
        raise ValueError("Native continuous columns missing.")
    y = frame["true_raw"].to_numpy(dtype=float)
    logp = frame["ensemble_log_density_true"].to_numpy(dtype=float)
    mean = frame["ensemble_predictive_mean"].to_numpy(dtype=float)
    return {
        "NLPD": float(-np.mean(logp)),
        "RMSE": float(np.sqrt(np.mean((mean - y) ** 2))),
    }


def align_model_frames(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    common = None
    for f in frames.values():
        ids = set(f["idauniq"].tolist())
        common = ids if common is None else common & ids
    if common is None:
        return {}
    common_sorted = sorted(common)
    aligned = {}
    for model, f in frames.items():
        x = f.set_index("idauniq").loc[common_sorted].reset_index()
        aligned[model] = x
    ref = next(iter(aligned.values()))
    for model, x in aligned.items():
        if not np.array_equal(x["true_state"].to_numpy(), ref["true_state"].to_numpy()):
            raise RuntimeError(f"Aligned truth mismatch for {model}")
        if not np.allclose(x["true_raw"], ref["true_raw"], atol=1e-12, rtol=0.0):
            raise RuntimeError(f"Aligned raw truth mismatch for {model}")
    return aligned


def percentile_ci(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def bootstrap_wave_target(
    *,
    wave: int,
    target: str,
    categorical_frames: Mapping[str, pd.DataFrame],
    native_frames: Mapping[str, pd.DataFrame],
    reps: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    # ---------------- Categorical/all-three view ----------------
    aligned = align_model_frames(categorical_frames)
    if aligned:
        n = len(next(iter(aligned.values())))
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, wave, stable_int(target), 101]))
        bootstrap_values: dict[str, dict[str, list[float]]] = {
            m: defaultdict(list) for m in aligned
        }
        point_values: dict[str, dict[str, float]] = {
            m: categorical_metrics(f, target) for m, f in aligned.items()
        }
        for _ in range(int(reps)):
            idx = rng.integers(0, n, size=n)
            for m, f in aligned.items():
                vals = categorical_metrics(f.iloc[idx].reset_index(drop=True), target)
                for metric, value in vals.items():
                    if metric == "Oracle_Threshold":
                        continue
                    bootstrap_values[m][metric].append(value)
        for m, vals in point_values.items():
            for metric, value in vals.items():
                if metric == "Oracle_Threshold":
                    # Threshold itself is exploratory; no CI needed.
                    lo = hi = float("nan")
                else:
                    lo, hi = percentile_ci(bootstrap_values[m].get(metric, []))
                metric_rows.append({
                    "wave": wave,
                    "target": target,
                    "view": "categorical_common_support",
                    "model": MODEL_DISPLAY[m],
                    "metric": metric,
                    "value": value,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                    "bootstrap_reps": reps,
                })

        # Primary RQ2 contrasts: Hybrid against each baseline.
        if "hybrid" in aligned:
            for baseline in ("discrete", "continuous"):
                if baseline not in aligned:
                    continue
                common_metrics = set(point_values["hybrid"]) & set(point_values[baseline])
                for metric in sorted(common_metrics):
                    if metric in {"Oracle_Threshold"}:
                        continue
                    higher_better = metric in {"F1", "ROC_AUC", "PR_AUC", "F1_at_0.5", "Oracle_Max_F1"}
                    observed = (
                        point_values["hybrid"][metric] - point_values[baseline][metric]
                        if higher_better
                        else point_values[baseline][metric] - point_values["hybrid"][metric]
                    )
                    hb = np.asarray(bootstrap_values["hybrid"].get(metric, []), dtype=float)
                    bb = np.asarray(bootstrap_values[baseline].get(metric, []), dtype=float)
                    if higher_better:
                        diff = hb - bb
                    else:
                        diff = bb - hb
                    lo, hi = percentile_ci(diff)
                    comparison_rows.append({
                        "wave": wave,
                        "target": target,
                        "view": "categorical_common_support",
                        "metric": metric,
                        "contrast": f"Hybrid - {MODEL_DISPLAY[baseline]}",
                        "advantage_definition": "positive = Hybrid better",
                        "advantage": observed,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": n,
                        "bootstrap_reps": reps,
                    })

    # ---------------- Native continuous view ----------------
    if target in CONTINUOUS_TARGETS:
        native_aligned = align_model_frames(native_frames)
        if native_aligned:
            n = len(next(iter(native_aligned.values())))
            rng = np.random.default_rng(np.random.SeedSequence([base_seed, wave, stable_int(target), 202]))
            point = {m: native_continuous_metrics(f) for m, f in native_aligned.items()}
            boot: dict[str, dict[str, list[float]]] = {m: defaultdict(list) for m in native_aligned}
            for _ in range(int(reps)):
                idx = rng.integers(0, n, size=n)
                for m, f in native_aligned.items():
                    vals = native_continuous_metrics(f.iloc[idx].reset_index(drop=True))
                    for metric, value in vals.items():
                        boot[m][metric].append(value)
            for m, vals in point.items():
                for metric, value in vals.items():
                    lo, hi = percentile_ci(boot[m][metric])
                    metric_rows.append({
                        "wave": wave,
                        "target": target,
                        "view": "native_continuous",
                        "model": MODEL_DISPLAY[m],
                        "metric": metric,
                        "value": value,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": n,
                        "bootstrap_reps": reps,
                    })
            if "hybrid" in native_aligned and "continuous" in native_aligned:
                for metric in ("NLPD", "RMSE"):
                    observed = point["continuous"][metric] - point["hybrid"][metric]
                    diff = np.asarray(boot["continuous"][metric]) - np.asarray(boot["hybrid"][metric])
                    lo, hi = percentile_ci(diff)
                    comparison_rows.append({
                        "wave": wave,
                        "target": target,
                        "view": "native_continuous",
                        "metric": metric,
                        "contrast": "Hybrid - Continuous",
                        "advantage_definition": "positive = Hybrid better (lower loss/error)",
                        "advantage": observed,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": n,
                        "bootstrap_reps": reps,
                    })
    return metric_rows, comparison_rows


def build_dementia_threshold_curves(out_root: Path, waves: Sequence[int], models: Sequence[str]) -> pd.DataFrame:
    rows = []
    thresholds = np.linspace(0.0, 1.0, 201)
    for wave in waves:
        for model_key in models:
            path = oof_output_path(out_root, wave, model_key, "Dementia")
            if not path.exists():
                continue
            f = pd.read_csv(path)
            y = f["true_state"].to_numpy(dtype=np.int64)
            p1 = f["p1"].to_numpy(dtype=float)
            for t in thresholds:
                pred = (p1 >= t).astype(np.int64)
                score = float(f1_score(y, pred, pos_label=1, average="binary", zero_division=0))
                rows.append({
                    "wave": wave,
                    "model": MODEL_DISPLAY[model_key],
                    "threshold": float(t),
                    "f1": score,
                })
    return pd.DataFrame(rows)


# ======================================================================================
# PLOTS
# ======================================================================================

def generate_plots(out_root: Path, metrics: pd.DataFrame, threshold_curves: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for target in PRIMARY_TARGETS:
        # Common-support categorical metrics.
        for metric in ("F1", "LogLoss"):
            sub = metrics[
                (metrics["target"] == target)
                & (metrics["view"] == "categorical_common_support")
                & (metrics["metric"] == metric)
            ].copy()
            if sub.empty:
                continue
            plt.figure(figsize=(8, 5))
            for model in ["Discrete", "Continuous", "Hybrid"]:
                x = sub[sub["model"] == model].sort_values("wave")
                if x.empty:
                    continue
                waves = x["wave"].to_numpy(dtype=int)
                vals = x["value"].to_numpy(dtype=float)
                lo = x["ci_low"].to_numpy(dtype=float)
                hi = x["ci_high"].to_numpy(dtype=float)
                if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                    yerr = np.vstack([vals - lo, hi - vals])
                    plt.errorbar(waves, vals, yerr=yerr, marker="o", capsize=3, label=model)
                else:
                    plt.plot(waves, vals, marker="o", label=model)
            plt.xticks([2, 4, 6, 8])
            plt.xlabel("ELSA Wave")
            plt.ylabel(metric)
            plt.title(f"{target}: {metric} on held-out OOF predictions")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"{target}_categorical_{metric}.png", dpi=180)
            plt.close()

        if target in CONTINUOUS_TARGETS:
            for metric in ("NLPD", "RMSE"):
                sub = metrics[
                    (metrics["target"] == target)
                    & (metrics["view"] == "native_continuous")
                    & (metrics["metric"] == metric)
                ].copy()
                if sub.empty:
                    continue
                plt.figure(figsize=(8, 5))
                for model in ["Continuous", "Hybrid"]:
                    x = sub[sub["model"] == model].sort_values("wave")
                    if x.empty:
                        continue
                    waves = x["wave"].to_numpy(dtype=int)
                    vals = x["value"].to_numpy(dtype=float)
                    lo = x["ci_low"].to_numpy(dtype=float)
                    hi = x["ci_high"].to_numpy(dtype=float)
                    if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                        yerr = np.vstack([vals - lo, hi - vals])
                        plt.errorbar(waves, vals, yerr=yerr, marker="o", capsize=3, label=model)
                    else:
                        plt.plot(waves, vals, marker="o", label=model)
                plt.xticks([2, 4, 6, 8])
                plt.xlabel("ELSA Wave")
                plt.ylabel(metric)
                plt.title(f"{target}: native continuous {metric}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_dir / f"{target}_native_{metric}.png", dpi=180)
                plt.close()

    if not threshold_curves.empty:
        for wave in sorted(threshold_curves["wave"].unique()):
            sub = threshold_curves[threshold_curves["wave"] == wave]
            plt.figure(figsize=(8, 5))
            for model in ["Discrete", "Continuous", "Hybrid"]:
                x = sub[sub["model"] == model]
                if x.empty:
                    continue
                plt.plot(x["threshold"], x["f1"], label=model)
            plt.xlabel("Dementia probability threshold")
            plt.ylabel("Positive-class F1")
            plt.title(f"Dementia threshold vs F1 — Wave {wave}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"Dementia_W{wave}_threshold_vs_F1.png", dpi=180)
            plt.close()


# ======================================================================================
# PREFLIGHT
# ======================================================================================

def preflight(
    *,
    args,
    phase1,
    phase1_path: Path,
    root: Path,
    phase1_root: Path,
    out_root: Path,
    run_config: Mapping[str, Any],
    waves: tuple[int, ...],
    targets: tuple[str, ...],
    models: tuple[str, ...],
    chain,
    wave_frames: Mapping[int, pd.DataFrame],
    assignments: Mapping[int, pd.DataFrame],
    common_cutpoints,
    model_modules,
) -> None:
    print("\n" + "=" * 100)
    print("RQ2 PHASE-2 PREFLIGHT")
    print("=" * 100)
    print(f"Project root : {root}")
    print(f"Phase-1 root : {phase1_root}")
    print(f"Phase-2 root : {out_root}")
    print(f"Phase-1 code : {phase1_path}")
    print(f"Waves        : {waves}")
    print(f"Targets      : {targets}")
    print(f"Models       : {models}")
    print(f"M            : {args.m_max}")

    folds = (int(args.fold_only),) if args.fold_only is not None else tuple(
        range(1, int(run_config["locked_design"]["folds"]) + 1)
    )
    completions = (int(args.completion_only),) if args.completion_only is not None else tuple(
        range(1, args.m_max + 1)
    )
    print(f"Selected folds : {folds}; completions: {completions}")
    missing = []
    for wave in waves:
        for fold in folds:
            for m in completions:
                cdir = phase1_completion_dir(phase1_root, wave, fold, m)
                required = [
                    cdir / "completed_train.csv.gz",
                    cdir / "completed_test.csv.gz",
                    cdir / "representation_metadata.json",
                ]
                for model_key in models:
                    required.append(cdir / MODEL_DIR_NAMES[model_key] / "edges.csv")
                missing.extend(str(p) for p in required if not p.exists())
    if missing:
        print(f"[FAIL] Missing {len(missing)} required Phase-1 files. First 20:")
        for p in missing[:20]:
            print("  ", p)
        raise FileNotFoundError("Phase-1 output tree is incomplete.")
    print("[OK] Required completion/train/test/graph files exist.")

    for wave in waves:
        print(f"[OK] W{wave}: frozen N={len(wave_frames[wave]):,}; folds={sorted(assignments[wave]['Fold'].unique())}")

    if any(t in CONTINUOUS_TARGETS for t in targets):
        n_common = sum(1 for k in common_cutpoints if k[0] in waves and k[2] in targets)
        print(f"[OK] Common train-only evaluation cutpoints constructed for {n_common} Wave x Fold x Target groups.")

    # Representative fixed-DAG reconstruction parity for the two simple baselines.
    wave = waves[0]
    fold = folds[0]
    m = completions[0]
    cdir = phase1_completion_dir(phase1_root, wave, fold, m)
    train_can, test_can = load_completed_canonical(cdir)
    rep_train, _, rebuilt_meta = phase1.build_model_representations(train_can, test_can)
    metadata_matches_saved(rebuilt_meta, read_json(cdir / "representation_metadata.json"))
    print("[OK] Representative Phase-1 representation rebuilt exactly from saved completed TRAIN.")

    nodes = run_config["locked_design"]["node_order"]
    if "discrete" in models:
        g = load_graph(cdir / "DiscreteBaseline" / "edges.csv", nodes)
        scorer = model_modules["discrete"].DiscreteScoringEngine(
            rep_train["discrete"],
            cardinalities=run_config["locked_design"]["discrete_baseline_cardinalities"],
        )
        score = float(sum(scorer.score_local(n, list(g.predecessors(n))) for n in nodes))
        saved = float(read_json(cdir / "DiscreteBaseline" / "search_summary.json")["best_score"])
        if not math.isclose(score, saved, rel_tol=0.0, abs_tol=1e-7):
            raise RuntimeError(f"Discrete representative fixed-DAG score parity failed: {score} vs {saved}")
        print(f"[OK] Discrete fixed-DAG refit score parity: delta={score - saved:.3e}")

    if "continuous" in models:
        g = load_graph(cdir / "ContinuousBaseline" / "edges.csv", nodes)
        scorer = model_modules["continuous"].ContinuousScoringEngine(rep_train["continuous"])
        score = float(sum(scorer.score_local(n, list(g.predecessors(n))) for n in nodes))
        saved = float(read_json(cdir / "ContinuousBaseline" / "search_summary.json")["best_score"])
        if not math.isclose(score, saved, rel_tol=0.0, abs_tol=1e-7):
            raise RuntimeError(f"Continuous representative fixed-DAG score parity failed: {score} vs {saved}")
        print(f"[OK] Continuous fixed-DAG refit score parity: delta={score - saved:.3e}")

    if "hybrid" in models:
        dc_mod_name = model_modules["hybrid"].HybridV3ScoringEngine.__module__
        # Public class is imported from package; inspect instantiated engine instead.
        h = HybridDAGModel(
            train_external=rep_train["hybrid"],
            graph=load_graph(cdir / "HybridV3" / "edges.csv", nodes),
            hybrid_module=model_modules["hybrid"],
            discrete_nodes=run_config["locked_design"]["native_discrete_nodes"],
            continuous_nodes=run_config["locked_design"]["continuous_nodes"],
            cardinalities=run_config["locked_design"]["hybrid_cardinalities"],
        )
        # Exercise one representative target family without running search.
        test_target = targets[0]
        h.prepare_target(test_target)
        print(f"[OK] Hybrid fixed-DAG local-family refit succeeds for representative target {test_target}.")

    print("\nPREFLIGHT PASSED.")
    print("No graph structure was relearned.")


# ======================================================================================
# MAIN PIPELINE
# ======================================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RQ2 Phase 2: fixed-DAG, leakage-safe held-out probabilistic inference."
    )
    p.add_argument("--project-root", default=None)
    p.add_argument("--phase1-root", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--waves", nargs="+", type=int, default=[2, 4, 6, 8])
    p.add_argument("--fold-only", type=int, default=None)
    p.add_argument("--completion-only", type=int, default=None)
    p.add_argument("--m-max", type=int, default=25)
    p.add_argument(
        "--models", nargs="+", choices=list(MODEL_ORDER), default=list(MODEL_ORDER)
    )
    p.add_argument(
        "--targets", nargs="+", choices=list(PRIMARY_TARGETS), default=list(PRIMARY_TARGETS)
    )
    p.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    p.add_argument("--bootstrap-seed", type=int, default=20261006)
    p.add_argument("--integration-nodes-per-bin", type=int, default=DEFAULT_INTEGRATION_NODES_PER_BIN)
    p.add_argument("--integration-tail-sd", type=float, default=DEFAULT_INTEGRATION_TAIL_SD)
    p.add_argument("--integration-chunk-size", type=int, default=DEFAULT_INTEGRATION_CHUNK)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--skip-bootstrap", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--force", action="store_true", help="Recompute existing member predictions and target-safe test caches.")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--allow-source-hash-mismatch", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.m_max <= 0:
        raise ValueError("--m-max must be positive.")
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive.")
    if args.integration_nodes_per_bin < 8:
        raise ValueError("--integration-nodes-per-bin should be >= 8.")
    if args.integration_tail_sd <= 0:
        raise ValueError("--integration-tail-sd must be positive.")
    if args.integration_chunk_size <= 0:
        raise ValueError("--integration-chunk-size must be positive.")

    phase1, phase1_path = load_phase1_module()
    root, phase1_root, out_root = resolve_paths(args, phase1)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "diagnostics").mkdir(parents=True, exist_ok=True)
    (out_root / "metrics").mkdir(parents=True, exist_ok=True)

    run_config_path = phase1_root / "run_config.json"
    run_config = read_json(run_config_path)
    locked = run_config["locked_design"]
    folds_n = int(locked["folds"])
    base_seed = int(locked["base_seed"])

    waves = parse_int_list(args.waves, (2, 4, 6, 8))
    invalid_waves = [w for w in waves if w not in (2, 4, 6, 8)]
    if invalid_waves:
        raise ValueError(f"Unsupported waves: {invalid_waves}")
    models = tuple(m for m in MODEL_ORDER if m in set(args.models))
    targets = tuple(t for t in PRIMARY_TARGETS if t in set(args.targets))
    if not models or not targets:
        raise ValueError("At least one model and one target are required.")
    folds_to_run = (int(args.fold_only),) if args.fold_only is not None else tuple(range(1, folds_n + 1))
    if any(f < 1 or f > folds_n for f in folds_to_run):
        raise ValueError("--fold-only out of range.")
    completions_to_run = (
        (int(args.completion_only),)
        if args.completion_only is not None
        else tuple(range(1, args.m_max + 1))
    )
    if any(m < 1 or m > args.m_max for m in completions_to_run):
        raise ValueError("--completion-only out of range.")

    print("=" * 110)
    print("RQ2 PHASE 2 — HELD-OUT PROBABILISTIC INFERENCE")
    print("=" * 110)
    print(f"Project root : {root}")
    print(f"Phase-1 root : {phase1_root}")
    print(f"Phase-2 root : {out_root}")
    print(f"Waves        : {waves}")
    print(f"Folds        : {folds_to_run}")
    print(f"M max        : {args.m_max}")
    print(f"Models       : {models}")
    print(f"Targets      : {targets}")
    print("Graphs       : FIXED Phase-1 DAGs; NO structure relearning")
    print("Fold metrics : stitch OOF folds first, then score")
    print("Target agg.  : NONE (targets remain separate)")
    print("Hybrid Loneliness quadrature: 256 nodes/bin (768 total); chunk 32")

    hash_rows = verify_phase1_hashes(root, phase1_root, args.allow_source_hash_mismatch)
    pd.DataFrame(hash_rows).to_csv(out_root / "diagnostics" / "source_hash_check.csv", index=False)
    mismatches = sum(not bool(r["match"]) for r in hash_rows)
    print(f"Source hashes: {len(hash_rows) - mismatches}/{len(hash_rows)} match Phase-1 run_config")

    # Exact frozen model sources.
    model_modules = phase1.load_models(root, set(models))

    # Raw data are needed to identify originally-observed targets, and to replay
    # target-hidden LowerMobility evidence.
    _, chain, completeness, raw_df, _, _ = load_raw_inputs(phase1, root, waves)
    wave_frames, assignments = reconstruct_wave_frames(
        phase1, phase1_root, completeness, raw_df, waves, run_config
    )

    common_cutpoints, cut_diag = compute_common_cutpoints(
        phase1_root, waves, folds_n, args.m_max, targets,
        selected_folds=folds_to_run, selected_completions=completions_to_run,
    )
    save_common_cutpoints(out_root, common_cutpoints, cut_diag)

    phase2_config = {
        "script": Path(__file__).name,
        "phase1_script": str(phase1_path),
        "project_root": str(root),
        "phase1_root": str(phase1_root),
        "output_root": str(out_root),
        "phase1_run_config_sha256": sha256_file(run_config_path),
        "waves": list(waves),
        "folds": folds_n,
        "m_max": int(args.m_max),
        "models": list(models),
        "targets": list(targets),
        "continuous_targets": list(CONTINUOUS_TARGETS),
        "discrete_targets": list(DISCRETE_TARGETS),
        "target_aware_reimputation": list(TARGET_AWARE_REIMPUTE),
        "saved_test_safe_targets": sorted(SAVED_TEST_SAFE_TARGETS),
        "common_cutpoint_rule": "component-wise median of M TRAIN-fitted Discrete-baseline cutpoint pairs within Wave x Fold x target",
        "inference": "single-hidden-node Markov-blanket/full-conditional",
        "ensemble_rule": "average member predictive distributions before scoring; log densities via logsumexp",
        "fold_rule": "concatenate 5 OOF fold predictions before metrics",
        "target_aggregation": "none",
        "categorical_metrics": "binary positive-class F1 or multiclass macro-F1 + log loss",
        "native_continuous_metrics": "NLPD + RMSE",
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.bootstrap_seed),
        "hybrid_continuous_integration": {
            "method": "vectorized Gauss-Legendre 1D quadrature split at common cutpoints",
            "nodes_per_bin": int(args.integration_nodes_per_bin),
            "tail_support": "TRAIN min/max expanded by tail_sd * TRAIN population SD",
            "tail_sd": float(args.integration_tail_sd),
            "chunk_size": int(args.integration_chunk_size),
        },
        "target_specific_inference_overrides": {
            "Loneliness": {
                "model": "Hybrid",
                "nodes_per_common_bin": LONELINESS_INTEGRATION_NODES_PER_BIN,
                "integration_nodes_total": 3 * LONELINESS_INTEGRATION_NODES_PER_BIN,
                "chunk_size": LONELINESS_INTEGRATION_CHUNK,
                "tail_sd": float(args.integration_tail_sd),
                "provenance": "integrated high-resolution quadrature correction",
            }
        },
        "source_hash_mismatches": int(mismatches),
    }
    write_json(out_root / "run_config.json", phase2_config)

    preflight(
        args=args,
        phase1=phase1,
        phase1_path=phase1_path,
        root=root,
        phase1_root=phase1_root,
        out_root=out_root,
        run_config=run_config,
        waves=waves,
        targets=targets,
        models=models,
        chain=chain,
        wave_frames=wave_frames,
        assignments=assignments,
        common_cutpoints=common_cutpoints,
        model_modules=model_modules,
    )
    if args.preflight_only:
        return

    diagnostics_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    if not args.aggregate_only:
        overall_started = time.perf_counter()
        planned = len(waves) * len(folds_to_run) * len(completions_to_run)
        cell_counter = 0

        for wave in waves:
            wave_frame = wave_frames[wave]
            assignment = assignments[wave]
            for fold in folds_to_run:
                train_original, test_original = split_wave_fold(wave_frame, assignment, fold)
                # Store wave on frame without changing its columns; helper reads attrs only.
                test_original.attrs["wave"] = int(wave)

                for completion in completions_to_run:
                    cell_counter += 1
                    label = f"W{wave} F{fold:02d} M{completion:03d}"
                    print("\n" + "-" * 110)
                    print(
                        f"{label} | completion cell {cell_counter}/{planned} | "
                        f"elapsed={elapsed_string(time.perf_counter() - overall_started)}"
                    )
                    print("-" * 110)
                    cdir = phase1_completion_dir(phase1_root, wave, fold, completion)
                    try:
                        train_can, saved_test_can = load_completed_canonical(cdir)
                        rep_train, rep_test_saved, rebuilt_meta = phase1.build_model_representations(
                            train_can, saved_test_can
                        )
                        metadata_matches_saved(rebuilt_meta, read_json(cdir / "representation_metadata.json"))

                        # LowerMobility gets a separate target-hidden evidence completion.
                        safe_lower_test = None
                        rep_test_lower = None
                        if "LowerMobility" in targets:
                            cache = lower_mobility_safe_test_cache_path(out_root, wave, fold, completion)
                            safe_lower_test = reconstruct_lower_mobility_safe_test(
                                phase1=phase1,
                                chain=chain,
                                train_original=train_original,
                                test_original=test_original,
                                saved_train_canonical=train_can,
                                wave=wave,
                                fold=fold,
                                completion=completion,
                                base_seed=base_seed,
                                cache_path=cache,
                                force=args.force,
                            )
                            _, rep_test_lower, lower_meta = phase1.build_model_representations(
                                train_can, safe_lower_test
                            )
                            metadata_matches_saved(lower_meta, read_json(cdir / "representation_metadata.json"))

                        for model_key in models:
                            graph = load_graph(
                                cdir / MODEL_DIR_NAMES[model_key] / "edges.csv",
                                locked["node_order"],
                            )
                            # Reuse fixed-DAG parameter fits across pseudo-targets within
                            # this completion/model.  This is especially important for
                            # Hybrid V3, where the same local family can appear in several
                            # Markov blankets and can be expensive to refit.
                            prediction_model_cache: dict[str, Any] = {}
                            for target in targets:
                                out_path = member_output_path(
                                    out_root, wave, fold, completion, model_key, target
                                )
                                if out_path.exists() and not args.force:
                                    valid_reuse = (loneliness_member_valid(out_path, args)
                                                   if (model_key == "hybrid" and target == "Loneliness")
                                                   else True)
                                    if valid_reuse:
                                        print(f"  [REUSE] {MODEL_DIR_NAMES[model_key]:<18} {target}")
                                        continue
                                    print("  [RECOMPUTE] Hybrid Loneliness: missing/stale high-resolution provenance")
                                started = time.perf_counter()
                                test_can = safe_lower_test if target == "LowerMobility" else saved_test_can
                                rep_test = rep_test_lower if target == "LowerMobility" else rep_test_saved
                                if test_can is None or rep_test is None:
                                    raise RuntimeError("Target-safe test representation is unavailable.")
                                test_original.attrs["wave"] = int(wave)
                                cuts = common_cutpoints.get((wave, fold, target))
                                integration_nodes, integration_chunk = integration_settings(model_key, target, args)
                                (
                                    ids,
                                    true_raw,
                                    true_state,
                                    probs,
                                    log_density,
                                    pred_mean,
                                    diag,
                                ) = predict_one_model_target(
                                    phase1=phase1,
                                    model_key=model_key,
                                    model_module=model_modules[model_key],
                                    root=root,
                                    train_canonical=train_can,
                                    test_canonical=test_can,
                                    raw_test=test_original,
                                    graph=graph,
                                    rep_train=rep_train,
                                    rep_test=rep_test,
                                    run_config=run_config,
                                    target=target,
                                    common_cuts=cuts,
                                    integration_nodes_per_bin=integration_nodes,
                                    integration_tail_sd=args.integration_tail_sd,
                                    integration_chunk=integration_chunk,
                                    model_cache=prediction_model_cache,
                                )
                                save_member_prediction(
                                    out_path,
                                    ids=ids,
                                    wave=wave,
                                    fold=fold,
                                    completion=completion,
                                    target=target,
                                    model_key=model_key,
                                    true_raw=true_raw,
                                    true_state=true_state,
                                    probs=probs,
                                    log_density_true=log_density,
                                    predictive_mean=pred_mean,
                                    common_cuts=cuts,
                                )
                                if model_key == "hybrid" and target == "Loneliness":
                                    # Marker is written ONLY after the member file is successfully saved.
                                    if int(diag.get("nodes_per_common_bin", -1)) != integration_nodes:
                                        raise RuntimeError(f"Hybrid Loneliness quadrature mismatch: {diag}")
                                    if int(diag.get("integration_nodes_total", -1)) != 3 * integration_nodes:
                                        raise RuntimeError(f"Hybrid Loneliness quadrature total mismatch: {diag}")
                                    write_json(integration_marker_path(out_path), {
                                        "complete": True, "model": "hybrid", "target": "Loneliness",
                                        "nodes_per_bin": integration_nodes,
                                        "total_nodes": 3 * integration_nodes,
                                        "chunk_size": integration_chunk,
                                        "tail_sd": float(args.integration_tail_sd),
                                    })
                                runtime = float(time.perf_counter() - started)
                                diagnostics_rows.append({
                                    "wave": wave,
                                    "fold": fold,
                                    "completion": completion,
                                    "model": MODEL_DISPLAY[model_key],
                                    "target": target,
                                    "n_eligible": len(ids),
                                    "runtime_seconds": runtime,
                                    **diag,
                                })
                                print(
                                    f"  [OK] {MODEL_DIR_NAMES[model_key]:<18} {target:<15} "
                                    f"N={len(ids):,} runtime={elapsed_string(runtime)}"
                                )
                    except Exception as exc:
                        row = {
                            "wave": wave,
                            "fold": fold,
                            "completion": completion,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                        error_rows.append(row)
                        pd.DataFrame(error_rows).to_csv(out_root / "diagnostics" / "errors.csv", index=False)
                        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
                        if not args.continue_on_error:
                            raise

        if diagnostics_rows:
            diag_path = out_root / "diagnostics" / "member_runtime_diagnostics.csv"
            existing = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
            combined = pd.concat([existing, pd.DataFrame(diagnostics_rows)], ignore_index=True)
            combined = combined.drop_duplicates(
                ["wave", "fold", "completion", "model", "target"], keep="last"
            )
            combined.to_csv(diag_path, index=False)
        if error_rows:
            pd.DataFrame(error_rows).to_csv(out_root / "diagnostics" / "errors.csv", index=False)

    # ----------------------------------------------------------------------------------
    # Aggregate M member distributions -> fold ensemble -> complete OOF predictions.
    # Only perform strict final aggregation when the full M/fold scope is available.
    # ----------------------------------------------------------------------------------
    full_completion_scope = args.completion_only is None
    full_fold_scope = args.fold_only is None
    require_all = full_completion_scope and full_fold_scope
    completions_all = tuple(range(1, args.m_max + 1))
    folds_all = tuple(range(1, folds_n + 1))

    if "hybrid" in models and "Loneliness" in targets:
        stale = []
        for wave in waves:
            for fold in folds_all:
                for completion in completions_all:
                    member = member_output_path(out_root, wave, fold, completion, "hybrid", "Loneliness")
                    if member.exists() and not loneliness_member_valid(member, args):
                        stale.append(str(member))
        if stale:
            raise RuntimeError(
                f"Found {len(stale)} stale Hybrid Loneliness members without valid "
                f"high-resolution quadrature provenance. Re-run prediction without "
                f"--aggregate-only. First example: {stale[0]}"
            )

    print("\n" + "=" * 110)
    print("AGGREGATING MEMBER PREDICTIVE DISTRIBUTIONS")
    print("=" * 110)
    for wave in waves:
        for fold in folds_all:
            for model_key in models:
                for target in targets:
                    path = aggregate_fold_members(
                        out_root=out_root,
                        wave=wave,
                        fold=fold,
                        target=target,
                        model_key=model_key,
                        completions=completions_all,
                        require_all=require_all,
                    )
                    if path is not None:
                        print(f"[ENSEMBLE] W{wave} F{fold:02d} {MODEL_DIR_NAMES[model_key]} {target}")
        for model_key in models:
            for target in targets:
                path = build_oof(
                    out_root=out_root,
                    wave=wave,
                    target=target,
                    model_key=model_key,
                    folds=folds_all,
                    require_all=require_all,
                )
                if path is not None:
                    print(f"[OOF]      W{wave} {MODEL_DIR_NAMES[model_key]} {target}")

    if not require_all:
        print(
            "\nPartial fold/completion scope requested. Member/fold outputs were saved, but final metrics "
            "are only rebuilt when all requested M members and all 5 folds are present."
        )
        # It may still be possible if previous runs already completed everything; test below.

    # ----------------------------------------------------------------------------------
    # Metrics + paired bootstrap on complete OOF sets.
    # ----------------------------------------------------------------------------------
    all_metric_rows: list[dict[str, Any]] = []
    all_comparison_rows: list[dict[str, Any]] = []
    can_build_metrics = True
    for wave in waves:
        for target in targets:
            for model_key in models:
                if not oof_output_path(out_root, wave, model_key, target).exists():
                    can_build_metrics = False
    if not can_build_metrics:
        print("\nOOF sets are incomplete; skipping final metrics/bootstrap/plots for now.")
        return

    print("\n" + "=" * 110)
    print("CALCULATING OOF METRICS")
    print("=" * 110)
    for wave in waves:
        for target in targets:
            cat_frames = {
                m: pd.read_csv(oof_output_path(out_root, wave, m, target)) for m in models
            }
            native_frames = {
                m: f for m, f in cat_frames.items()
                if target in CONTINUOUS_TARGETS
                and m in {"continuous", "hybrid"}
                and "ensemble_log_density_true" in f.columns
            }
            reps = 1 if args.skip_bootstrap else int(args.bootstrap_reps)
            metric_rows, comparison_rows = bootstrap_wave_target(
                wave=wave,
                target=target,
                categorical_frames=cat_frames,
                native_frames=native_frames,
                reps=reps,
                base_seed=args.bootstrap_seed,
            )
            if args.skip_bootstrap:
                for r in metric_rows:
                    r["ci_low"] = float("nan")
                    r["ci_high"] = float("nan")
                    r["bootstrap_reps"] = 0
                for r in comparison_rows:
                    r["ci_low"] = float("nan")
                    r["ci_high"] = float("nan")
                    r["bootstrap_reps"] = 0
            all_metric_rows.extend(metric_rows)
            all_comparison_rows.extend(comparison_rows)
            print(f"[METRICS] W{wave} {target}")

    metrics = pd.DataFrame(all_metric_rows)
    comparisons = pd.DataFrame(all_comparison_rows)
    metrics.to_csv(out_root / "metrics" / "metrics_long.csv", index=False)
    comparisons.to_csv(out_root / "metrics" / "model_comparisons.csv", index=False)

    threshold_curves = pd.DataFrame()
    if "Dementia" in targets:
        threshold_curves = build_dementia_threshold_curves(out_root, waves, models)
        threshold_curves.to_csv(out_root / "metrics" / "dementia_threshold_curves.csv", index=False)

    if not args.skip_plots:
        generate_plots(out_root, metrics, threshold_curves)
        print(f"Plots saved to: {out_root / 'plots'}")

    print("\n" + "=" * 110)
    print("PHASE-2 INFERENCE COMPLETE")
    print("=" * 110)
    print(f"Metrics     : {out_root / 'metrics' / 'metrics_long.csv'}")
    print(f"Comparisons : {out_root / 'metrics' / 'model_comparisons.csv'}")
    print("Targets remain separate; no cross-target macro score was calculated.")


if __name__ == "__main__":
    main()

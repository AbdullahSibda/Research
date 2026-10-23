"""
RQ2 Phase 2 — M-convergence and completed-wave structural snapshot.

Reads graph outputs created by 01_cv_generate_ensembles.py. It performs NO
imputation and NO structure learning.

Primary purpose:
  * quantify how adjacency probabilities change as M grows (default 10,15,20,25)
  * keep direction stability separate from adjacency stability
  * summarize graph density and cross-completion similarity
  * surface Dementia-neighbourhood stability for inspection

Important design choice:
M convergence is assessed WITHIN each CV fold and then summarized across folds.
The 5 folds are not pooled as if they were 125 exchangeable imputations.

No automatic convergence threshold is imposed here. This script reports the
quantities needed to lock a threshold/decision transparently.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

PROCESS_NODES = ["Proxy", "SelfCompletion", "Nurse"]
OUTCOME_NODE = "Dementia"
SUBSTANTIVE_NODES = [
    "Age", "Education", "Wealth", "Recall", "Orientation", "CASP19",
    "FriendSupport", "Grip", "LowerMobility", "BMI", "Hearing", "Eyesight",
    "CESD", "Loneliness", "SocialParticipation", "ManagingMeds",
    "VigorousActivity", OUTCOME_NODE,
]
NODE_ORDER = [
    "Age", "Education", "Wealth", "Recall", "Orientation", "CASP19",
    "FriendSupport", "Grip", "LowerMobility", "BMI", "Hearing", "Eyesight",
    "CESD", "Loneliness", "SocialParticipation", "ManagingMeds",
    "VigorousActivity", OUTCOME_NODE, *PROCESS_NODES,
]

MODEL_DIRS = {
    "discrete": "DiscreteBaseline",
    "continuous": "ContinuousBaseline",
    "hybrid": "HybridV3",
}
MODEL_LABELS = {
    "discrete": "Discrete baseline",
    "continuous": "Continuous baseline",
    "hybrid": "Hybrid V3",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ2 M-convergence analysis")
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--waves", nargs="+", type=int, default=[2, 4, 6, 8])
    p.add_argument("--models", nargs="+", choices=list(MODEL_DIRS), default=list(MODEL_DIRS))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--checkpoints", nargs="+", type=int, default=[10, 15, 20, 25])
    p.add_argument("--allow-incomplete", action="store_true",
                   help="Analyze available prefixes even if the largest checkpoint is not complete.")
    p.add_argument("--top", type=int, default=15,
                   help="Number of top edges shown in the terminal report.")
    return p.parse_args()


def find_project_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    here = Path(__file__).resolve().parent
    # GitHub layout: experiments/rq2/m_convergence.py, three levels below repo root.
    candidates = [here.parents[2], here.parent, here, Path.cwd().resolve(), Path.cwd().resolve().parent]
    for root in candidates:
        if (root / "Results" / "MasterPipeline" / "CV_Ensembles").exists():
            return root
    raise FileNotFoundError("Could not locate project root containing Results/MasterPipeline/CV_Ensembles")


def edge_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def read_edges(path: Path) -> Set[Tuple[str, str]]:
    df = pd.read_csv(path)
    if list(df.columns) != ["Parent", "Child"]:
        raise ValueError(f"Unexpected edge file columns in {path}: {list(df.columns)}")
    edges = {(str(r.Parent), str(r.Child)) for r in df.itertuples(index=False)}
    if any(a == b for a, b in edges):
        raise ValueError(f"Self-edge found in {path}")
    return edges


def all_pairs(nodes: Sequence[str]) -> List[Tuple[str, str]]:
    return [tuple(sorted(x)) for x in itertools.combinations(nodes, 2)]  # type: ignore[list-item]


ALL_ADJ_PAIRS = all_pairs(NODE_ORDER)
SUBSTANTIVE_ADJ_PAIRS = all_pairs(SUBSTANTIVE_NODES)
ALL_DIRECTED = [(a, b) for a in NODE_ORDER for b in NODE_ORDER if a != b]


def graph_path(cv_root: Path, wave: int, fold: int, m: int, model: str) -> Path:
    return cv_root / f"W{wave}" / f"Fold{fold:02d}" / f"M{m:03d}" / MODEL_DIRS[model] / "edges.csv"


def available_prefix(cv_root: Path, wave: int, fold: int, model: str, max_probe: int = 200) -> int:
    n = 0
    for m in range(1, max_probe + 1):
        if graph_path(cv_root, wave, fold, m, model).exists():
            n = m
        else:
            break
    return n


def load_graphs(cv_root: Path, wave: int, fold: int, model: str, m: int) -> List[Set[Tuple[str, str]]]:
    out = []
    for i in range(1, m + 1):
        path = graph_path(cv_root, wave, fold, i, model)
        if not path.exists():
            raise FileNotFoundError(path)
        out.append(read_edges(path))
    return out


def adjacency_probabilities(graphs: Sequence[Set[Tuple[str, str]]], pairs=ALL_ADJ_PAIRS) -> Dict[Tuple[str, str], float]:
    m = len(graphs)
    counts = {p: 0 for p in pairs}
    for edges in graphs:
        adj = {edge_key(a, b) for a, b in edges}
        for p in adj:
            if p in counts:
                counts[p] += 1
    return {p: c / m for p, c in counts.items()}


def directed_probabilities(graphs: Sequence[Set[Tuple[str, str]]]) -> Dict[Tuple[str, str], float]:
    m = len(graphs)
    counts = {p: 0 for p in ALL_DIRECTED}
    for edges in graphs:
        for p in edges:
            if p in counts:
                counts[p] += 1
    return {p: c / m for p, c in counts.items()}


def jaccard(a: Set, b: Set) -> float:
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def pairwise_jaccards(graphs: Sequence[Set[Tuple[str, str]]], directed: bool) -> List[float]:
    reps = []
    for e in graphs:
        reps.append(set(e) if directed else {edge_key(a, b) for a, b in e})
    return [jaccard(reps[i], reps[j]) for i in range(len(reps)) for j in range(i + 1, len(reps))]


def finite_mean(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def finite_sd(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0 if len(arr) == 1 else float("nan")


def fmt(x: float, d: int = 3) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.{d}f}"


def analyze_wave(
    *, cv_root: Path, out_root: Path, wave: int, models: Sequence[str], folds: int,
    checkpoints: Sequence[int], allow_incomplete: bool, top_n: int,
) -> str:
    checkpoints = sorted(set(map(int, checkpoints)))
    max_cp = max(checkpoints)
    wave_out = out_root / f"W{wave}"
    wave_out.mkdir(parents=True, exist_ok=True)

    # Cache graphs by model/fold up to available max needed.
    cache: Dict[Tuple[str, int], List[Set[Tuple[str, str]]]] = {}
    availability_rows = []
    for model in models:
        for fold in range(1, folds + 1):
            avail = available_prefix(cv_root, wave, fold, model)
            availability_rows.append({"Wave": wave, "Model": model, "Fold": fold, "AvailablePrefix": avail})
            needed = min(avail, max_cp) if allow_incomplete else max_cp
            if not allow_incomplete and avail < max_cp:
                raise RuntimeError(
                    f"W{wave} {model} Fold{fold:02d} has only M={avail}; need M={max_cp}. "
                    "Use --allow-incomplete for a provisional snapshot."
                )
            if needed > 0:
                cache[(model, fold)] = load_graphs(cv_root, wave, fold, model, needed)
    pd.DataFrame(availability_rows).to_csv(wave_out / "availability.csv", index=False)

    graph_summary_rows = []
    stability_rows = []
    directed_rows = []
    fold_conv_rows = []
    conv_rows = []
    dementia_rows = []
    process_rows = []
    similarity_rows = []

    # Graph-level summaries and probabilities.
    for model in models:
        model_fold_probs: Dict[int, Dict[int, Dict[Tuple[str, str], float]]] = {}
        model_fold_dir: Dict[int, Dict[int, Dict[Tuple[str, str], float]]] = {}

        for fold in range(1, folds + 1):
            graphs_all = cache.get((model, fold), [])
            if not graphs_all:
                continue
            usable_cps = [cp for cp in checkpoints if cp <= len(graphs_all)]
            if not usable_cps:
                continue
            model_fold_probs[fold] = {}
            model_fold_dir[fold] = {}

            final_graphs = graphs_all[:max(usable_cps)]
            edge_counts = [len(g) for g in final_graphs]
            sub_counts = [sum(a in SUBSTANTIVE_NODES and b in SUBSTANTIVE_NODES for a, b in g) for g in final_graphs]
            adj_j = pairwise_jaccards(final_graphs, directed=False)
            dir_j = pairwise_jaccards(final_graphs, directed=True)
            graph_summary_rows.append({
                "Wave": wave, "Model": model, "Fold": fold, "M": len(final_graphs),
                "EdgeCountMean": finite_mean(edge_counts), "EdgeCountSD": finite_sd(edge_counts),
                "EdgeCountMin": min(edge_counts), "EdgeCountMax": max(edge_counts),
                "SubstantiveEdgeCountMean": finite_mean(sub_counts), "SubstantiveEdgeCountSD": finite_sd(sub_counts),
                "AdjacencyJaccardMean": finite_mean(adj_j), "AdjacencyJaccardSD": finite_sd(adj_j),
                "DirectedJaccardMean": finite_mean(dir_j), "DirectedJaccardSD": finite_sd(dir_j),
            })

            for cp in usable_cps:
                ap = adjacency_probabilities(graphs_all[:cp])
                dp = directed_probabilities(graphs_all[:cp])
                model_fold_probs[fold][cp] = ap
                model_fold_dir[fold][cp] = dp
                for (a, b), p in ap.items():
                    stability_rows.append({"Wave": wave, "Model": model, "Fold": fold, "M": cp,
                                           "NodeA": a, "NodeB": b, "AdjacencyProbability": p,
                                           "SubstantivePair": a in SUBSTANTIVE_NODES and b in SUBSTANTIVE_NODES})
                for (a, b), p in dp.items():
                    if p > 0:
                        directed_rows.append({"Wave": wave, "Model": model, "Fold": fold, "M": cp,
                                              "Parent": a, "Child": b, "DirectedProbability": p})

            # Within-fold checkpoint changes.
            for prev, curr in zip(usable_cps[:-1], usable_cps[1:]):
                for scope, pairs in (("all_21_nodes", ALL_ADJ_PAIRS), ("substantive_18_nodes", SUBSTANTIVE_ADJ_PAIRS)):
                    diffs = np.asarray([abs(model_fold_probs[fold][curr][p] - model_fold_probs[fold][prev][p]) for p in pairs])
                    fold_conv_rows.append({
                        "Wave": wave, "Model": model, "Fold": fold, "Scope": scope,
                        "M_prev": prev, "M_curr": curr,
                        "MeanAbsDelta": float(np.mean(diffs)),
                        "MedianAbsDelta": float(np.median(diffs)),
                        "P95AbsDelta": float(np.quantile(diffs, 0.95)),
                        "MaxAbsDelta": float(np.max(diffs)),
                        "CountDeltaGT_0_10": int(np.sum(diffs > 0.10 + 1e-12)),
                        "CountDeltaGT_0_20": int(np.sum(diffs > 0.20 + 1e-12)),
                        "PairCount": int(len(pairs)),
                    })

        # Cross-fold mean stability at each checkpoint.
        for cp in checkpoints:
            valid_folds = [f for f in model_fold_probs if cp in model_fold_probs[f]]
            if not valid_folds:
                continue
            for p in ALL_ADJ_PAIRS:
                vals = [model_fold_probs[f][cp][p] for f in valid_folds]
                a, b = p
                row = {
                    "Wave": wave, "Model": model, "M": cp, "NodeA": a, "NodeB": b,
                    "MeanAdjacencyProbability": float(np.mean(vals)),
                    "SDFoldAdjacencyProbability": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "MinFoldAdjacencyProbability": float(np.min(vals)),
                    "MaxFoldAdjacencyProbability": float(np.max(vals)),
                    "Folds": len(vals),
                    "SubstantivePair": a in SUBSTANTIVE_NODES and b in SUBSTANTIVE_NODES,
                }
                # Add direction probabilities in both orientations.
                uv = [model_fold_dir[f][cp][(a, b)] for f in valid_folds]
                vu = [model_fold_dir[f][cp][(b, a)] for f in valid_folds]
                row["MeanDirection_A_to_B"] = float(np.mean(uv))
                row["MeanDirection_B_to_A"] = float(np.mean(vu))
                denom = row["MeanAdjacencyProbability"]
                row["DirectionalConsistencyGivenAdjacency"] = (
                    float(max(row["MeanDirection_A_to_B"], row["MeanDirection_B_to_A"]) / denom)
                    if denom > 0 else float("nan")
                )
                # Reuse stability output as cross-fold summary by Fold=0.
                stability_rows.append({
                    "Wave": wave, "Model": model, "Fold": 0, "M": cp,
                    "NodeA": a, "NodeB": b, "AdjacencyProbability": row["MeanAdjacencyProbability"],
                    "SubstantivePair": row["SubstantivePair"],
                })
                if cp == max_cp or (allow_incomplete and cp == max([c for c in checkpoints if c <= min(len(cache.get((model, f), [])) for f in valid_folds)])):
                    if OUTCOME_NODE in p:
                        other = b if a == OUTCOME_NODE else a
                        dementia_rows.append({**row, "OtherNode": other})
                    if a in PROCESS_NODES or b in PROCESS_NODES:
                        process_rows.append(row)

    # Aggregate fold convergence metrics.
    fconv = pd.DataFrame(fold_conv_rows)
    if not fconv.empty:
        for keys, grp in fconv.groupby(["Wave", "Model", "Scope", "M_prev", "M_curr"], sort=True):
            wave_k, model, scope, prev, curr = keys
            conv_rows.append({
                "Wave": wave_k, "Model": model, "Scope": scope, "M_prev": prev, "M_curr": curr,
                "MeanAbsDelta_AcrossFolds": float(grp["MeanAbsDelta"].mean()),
                "P95AbsDelta_AcrossFoldPairs": float(np.quantile(
                    np.concatenate([
                        np.asarray([
                            abs(
                                adjacency_probabilities(cache[(model, int(r.Fold))][:int(curr)])[p] -
                                adjacency_probabilities(cache[(model, int(r.Fold))][:int(prev)])[p]
                            ) for p in (ALL_ADJ_PAIRS if scope == "all_21_nodes" else SUBSTANTIVE_ADJ_PAIRS)
                        ]) for r in grp.itertuples(index=False)
                    ]), 0.95
                )),
                "MaxAbsDelta_AcrossFolds": float(grp["MaxAbsDelta"].max()),
                "MeanCountDeltaGT_0_10": float(grp["CountDeltaGT_0_10"].mean()),
                "MaxCountDeltaGT_0_10": int(grp["CountDeltaGT_0_10"].max()),
                "MeanCountDeltaGT_0_20": float(grp["CountDeltaGT_0_20"].mean()),
                "MaxCountDeltaGT_0_20": int(grp["CountDeltaGT_0_20"].max()),
                "Folds": int(len(grp)),
            })

    # Cross-model similarity of final mean adjacency vectors.
    stable_df = pd.DataFrame(stability_rows)
    if not stable_df.empty:
        final_cp = max_cp
        cross = stable_df[(stable_df["Fold"] == 0) & (stable_df["M"] == final_cp)]
        for m1, m2 in itertools.combinations(models, 2):
            a = cross[cross.Model == m1].set_index(["NodeA", "NodeB"])["AdjacencyProbability"]
            b = cross[cross.Model == m2].set_index(["NodeA", "NodeB"])["AdjacencyProbability"]
            joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            if len(joined):
                similarity_rows.append({
                    "Wave": wave, "M": final_cp, "ModelA": m1, "ModelB": m2,
                    "PearsonAdjacencyStability": float(joined.a.corr(joined.b, method="pearson")),
                    "SpearmanAdjacencyStability": float(joined.a.corr(joined.b, method="spearman")),
                    "MeanAbsDifference": float(np.mean(np.abs(joined.a - joined.b))),
                })

    # Save outputs.
    pd.DataFrame(graph_summary_rows).to_csv(wave_out / "graph_summary.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(wave_out / "adjacency_stability.csv", index=False)
    pd.DataFrame(directed_rows).to_csv(wave_out / "directed_stability_nonzero.csv", index=False)
    pd.DataFrame(fold_conv_rows).to_csv(wave_out / "fold_convergence.csv", index=False)
    pd.DataFrame(conv_rows).to_csv(wave_out / "convergence_summary.csv", index=False)
    pd.DataFrame(dementia_rows).to_csv(wave_out / "dementia_neighborhood.csv", index=False)
    pd.DataFrame(process_rows).to_csv(wave_out / "process_neighborhood.csv", index=False)
    pd.DataFrame(similarity_rows).to_csv(wave_out / "model_similarity.csv", index=False)

    # Human-readable terminal/report summary.
    lines: List[str] = []
    lines.append("=" * 104)
    lines.append(f"WAVE {wave} — CV GRAPH ENSEMBLE SNAPSHOT / M-CONVERGENCE")
    lines.append("=" * 104)
    lines.append("M-convergence is assessed within fold; fold summaries are then averaged. No pass/fail threshold is applied.")
    lines.append("")

    avail_df = pd.DataFrame(availability_rows)
    lines.append("AVAILABLE PREFIXES")
    for model in models:
        vals = avail_df[avail_df.Model == model].sort_values("Fold")
        text = ", ".join(f"F{int(r.Fold):02d}=M{int(r.AvailablePrefix)}" for r in vals.itertuples(index=False))
        lines.append(f"  {MODEL_LABELS[model]:20s}: {text}")
    lines.append("")

    gs = pd.DataFrame(graph_summary_rows)
    lines.append(f"GRAPH STRUCTURE AT M={max_cp}")
    for model in models:
        g = gs[gs.Model == model]
        if g.empty:
            continue
        lines.append(
            f"  {MODEL_LABELS[model]:20s} edges={g.EdgeCountMean.mean():.1f} mean/graph "
            f"(fold means {g.EdgeCountMean.min():.1f}–{g.EdgeCountMean.max():.1f}); "
            f"adj-Jaccard={g.AdjacencyJaccardMean.mean():.3f}; directed-Jaccard={g.DirectedJaccardMean.mean():.3f}"
        )
    lines.append("")

    conv = pd.DataFrame(conv_rows)
    lines.append("ADJACENCY M-CONVERGENCE — substantive 18-node graph")
    if conv.empty:
        lines.append("  No complete checkpoint transitions available.")
    else:
        c = conv[conv.Scope == "substantive_18_nodes"]
        for model in models:
            lines.append(f"  {MODEL_LABELS[model]}:")
            cm = c[c.Model == model].sort_values(["M_prev", "M_curr"])
            for r in cm.itertuples(index=False):
                lines.append(
                    f"    M{int(r.M_prev):02d}->M{int(r.M_curr):02d}: "
                    f"mean|Δp|={r.MeanAbsDelta_AcrossFolds:.3f}, "
                    f"p95={r.P95AbsDelta_AcrossFoldPairs:.3f}, max={r.MaxAbsDelta_AcrossFolds:.3f}, "
                    f"pairs >.10/fold={r.MeanCountDeltaGT_0_10:.1f}"
                )
    lines.append("")

    # Top substantive adjacencies at final checkpoint.
    if not stable_df.empty:
        cross = stable_df[(stable_df.Fold == 0) & (stable_df.M == max_cp) & (stable_df.SubstantivePair == True)].copy()
        lines.append(f"TOP SUBSTANTIVE ADJACENCIES AT M={max_cp} — mean stability across 5 folds")
        for model in models:
            lines.append(f"  {MODEL_LABELS[model]}:")
            cm = cross[cross.Model == model].sort_values("AdjacencyProbability", ascending=False).head(top_n)
            for r in cm.itertuples(index=False):
                lines.append(f"    {r.NodeA} -- {r.NodeB}: {r.AdjacencyProbability:.3f}")
        lines.append("")

    dem = pd.DataFrame(dementia_rows)
    lines.append(f"DEMENTIA NEIGHBOURHOOD AT M={max_cp} — mean adjacency stability across folds")
    if dem.empty:
        lines.append("  No final-checkpoint Dementia summary available.")
    else:
        for model in models:
            lines.append(f"  {MODEL_LABELS[model]}:")
            dm = dem[dem.Model == model].sort_values("MeanAdjacencyProbability", ascending=False)
            dm = dm[dm.MeanAdjacencyProbability > 0]
            for r in dm.itertuples(index=False):
                other = r.OtherNode
                # Orient direction values relative to Dementia.
                if r.NodeA == OUTCOME_NODE:
                    d_to_o = r.MeanDirection_A_to_B
                    o_to_d = r.MeanDirection_B_to_A
                else:
                    d_to_o = r.MeanDirection_B_to_A
                    o_to_d = r.MeanDirection_A_to_B
                lines.append(
                    f"    {other:20s} adj={r.MeanAdjacencyProbability:.3f} "
                    f"[fold {r.MinFoldAdjacencyProbability:.2f}–{r.MaxFoldAdjacencyProbability:.2f}] "
                    f"| {other}->D={o_to_d:.3f}, D->{other}={d_to_o:.3f}"
                )
    lines.append("")

    sim = pd.DataFrame(similarity_rows)
    if not sim.empty:
        lines.append(f"MODEL-TO-MODEL ADJACENCY-STABILITY SIMILARITY AT M={max_cp}")
        for r in sim.itertuples(index=False):
            lines.append(
                f"  {MODEL_LABELS[r.ModelA]} vs {MODEL_LABELS[r.ModelB]}: "
                f"Spearman={r.SpearmanAdjacencyStability:.3f}, mean|Δp|={r.MeanAbsDifference:.3f}"
            )
        lines.append("")

    lines.append(f"Saved detailed CSVs to: {wave_out}")
    report = "\n".join(lines)
    (wave_out / "snapshot_report.txt").write_text(report + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    root = find_project_root(args.project_root)
    cv_root = root / "Results" / "MasterPipeline" / "CV_Ensembles"
    out_root = root / "Results" / "MasterPipeline" / "M_Convergence"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("RQ2 PHASE 2 — M-CONVERGENCE / STRUCTURAL SNAPSHOT")
    print("=" * 104)
    print(f"Project root : {root}")
    print(f"Input root   : {cv_root}")
    print(f"Output root  : {out_root}")
    print(f"Waves        : {tuple(args.waves)}")
    print(f"Models       : {', '.join(args.models)}")
    print(f"Checkpoints  : {tuple(sorted(set(args.checkpoints)))}")
    print()

    reports = []
    for wave in args.waves:
        report = analyze_wave(
            cv_root=cv_root, out_root=out_root, wave=wave, models=args.models,
            folds=args.folds, checkpoints=args.checkpoints,
            allow_incomplete=args.allow_incomplete, top_n=args.top,
        )
        print(report)
        print()
        reports.append(report)

    (out_root / "latest_terminal_report.txt").write_text("\n\n".join(reports) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

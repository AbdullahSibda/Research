from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # experiments/rq2/ -> repository root

ENSEMBLE_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "MasterPipeline"
    / "CV_Ensembles"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "ScientificResults"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


MODEL_FOLDERS = {
    "DiscreteBaseline": "Discrete",
    "ContinuousBaseline": "Continuous",
    "HybridV3": "Hybrid",
}

EXPECTED_WAVES = [2, 4, 6, 8]
EXPECTED_FOLDS = 5
EXPECTED_COMPLETIONS = 25

# These are descriptive thresholds only.
# Full frequencies are always retained in the CSV outputs.
MODERATE_STABILITY = 0.50
HIGH_STABILITY = 0.80


# =============================================================================
# Helpers
# =============================================================================

def normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def infer_edge_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Robustly identify the source/target columns in edges.csv.

    Supports names such as:
        source,target
        parent,child
        from,to
        u,v
    """

    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    candidate_pairs = [
        ("source", "target"),
        ("parent", "child"),
        ("from", "to"),
        ("u", "v"),
        ("start", "end"),
    ]

    for left, right in candidate_pairs:
        if left in normalized and right in normalized:
            return normalized[left], normalized[right]

    # If the file contains exactly two columns, use them.
    if len(df.columns) == 2:
        return str(df.columns[0]), str(df.columns[1])

    raise ValueError(
        "Could not identify edge columns. "
        f"Found columns: {list(df.columns)}"
    )


def extract_number(text: str, prefix: str) -> int:
    match = re.search(
        rf"{re.escape(prefix)}0*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            f"Could not extract number from {text!r} "
            f"using prefix {prefix!r}."
        )
    return int(match.group(1))


def canonical_pair(
    a: str,
    b: str,
) -> tuple[str, str]:
    """
    Represent an undirected adjacency in a stable alphabetical order.
    """
    return tuple(sorted((str(a), str(b))))


def find_dementia_node(nodes: set[str]) -> str | None:
    """
    Automatically find the dementia variable without hard-coding its
    exact final name.
    """
    matches = [
        node
        for node in nodes
        if "dement" in normalize_column_name(node)
        or "alzdeme" in normalize_column_name(node)
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print(
            "WARNING: Multiple dementia-like nodes found:",
            matches,
        )

    return matches[0] if matches else None


# =============================================================================
# 1. Load all learned graphs
# =============================================================================

def load_all_edges() -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_rows: list[dict] = []
    graph_rows: list[dict] = []

    pattern = "W*/Fold*/M*/*/edges.csv"
    files = sorted(ENSEMBLE_ROOT.glob(pattern))

    print(f"Scanning: {ENSEMBLE_ROOT}")
    print(f"Candidate edges.csv files found: {len(files)}")

    for edges_path in files:
        model_folder = edges_path.parent.name

        if model_folder not in MODEL_FOLDERS:
            continue

        completion_folder = edges_path.parent.parent.name
        fold_folder = edges_path.parent.parent.parent.name
        wave_folder = edges_path.parent.parent.parent.parent.name

        wave = extract_number(wave_folder, "W")
        fold = extract_number(fold_folder, "Fold")
        completion = extract_number(completion_folder, "M")

        model = MODEL_FOLDERS[model_folder]

        df = pd.read_csv(edges_path)

        source_column, target_column = infer_edge_columns(df)

        edges = (
            df[[source_column, target_column]]
            .rename(
                columns={
                    source_column: "Source",
                    target_column: "Target",
                }
            )
            .dropna()
            .astype(str)
            .drop_duplicates()
        )

        edges = edges.loc[
            edges["Source"] != edges["Target"]
        ].copy()

        graph_id = (
            f"W{wave}_F{fold:02d}_M{completion:03d}_{model}"
        )

        graph_rows.append(
            {
                "GraphID": graph_id,
                "Wave": wave,
                "Fold": fold,
                "Completion": completion,
                "Model": model,
                "EdgeCount": int(len(edges)),
                "EdgesFile": str(
                    edges_path.relative_to(PROJECT_ROOT)
                ),
            }
        )

        for row in edges.itertuples(index=False):
            source = str(row.Source)
            target = str(row.Target)

            a, b = canonical_pair(source, target)

            edge_rows.append(
                {
                    "GraphID": graph_id,
                    "Wave": wave,
                    "Fold": fold,
                    "Completion": completion,
                    "Model": model,
                    "Source": source,
                    "Target": target,
                    "NodeA": a,
                    "NodeB": b,
                    "DirectedEdge": f"{source} -> {target}",
                    "Adjacency": f"{a} -- {b}",
                }
            )

    edges = pd.DataFrame(edge_rows)
    graphs = pd.DataFrame(graph_rows)

    if graphs.empty:
        raise RuntimeError(
            "No graph files were loaded. Check ENSEMBLE_ROOT."
        )

    return edges, graphs


# =============================================================================
# 2. Inventory / integrity checks
# =============================================================================

def graph_inventory(
    graphs: pd.DataFrame,
) -> pd.DataFrame:
    inventory = (
        graphs.groupby(
            ["Wave", "Model"],
            as_index=False,
        )
        .agg(
            GraphCount=("GraphID", "nunique"),
            MeanEdgeCount=("EdgeCount", "mean"),
            SDEdgeCount=("EdgeCount", "std"),
            MinEdgeCount=("EdgeCount", "min"),
            MaxEdgeCount=("EdgeCount", "max"),
        )
    )

    inventory["ExpectedGraphs"] = (
        EXPECTED_FOLDS * EXPECTED_COMPLETIONS
    )

    inventory["Complete"] = (
        inventory["GraphCount"]
        == inventory["ExpectedGraphs"]
    )

    return inventory


# =============================================================================
# 3. Directed edge stability
# =============================================================================

def directed_edge_stability(
    edges: pd.DataFrame,
    graphs: pd.DataFrame,
) -> pd.DataFrame:

    # Frequency within each fold across the 25 completed datasets.
    fold_counts = (
        edges.groupby(
            [
                "Wave",
                "Model",
                "Fold",
                "Source",
                "Target",
            ]
        )
        .size()
        .rename("Appearances")
        .reset_index()
    )

    graph_denominator = (
        graphs.groupby(
            ["Wave", "Model", "Fold"]
        )["GraphID"]
        .nunique()
        .rename("GraphsInFold")
        .reset_index()
    )

    fold_counts = fold_counts.merge(
        graph_denominator,
        on=["Wave", "Model", "Fold"],
        how="left",
    )

    fold_counts["FoldFrequency"] = (
        fold_counts["Appearances"]
        / fold_counts["GraphsInFold"]
    )

    # Need explicit zeroes for folds where a candidate edge never appears.
    all_edges = (
        edges[
            ["Wave", "Model", "Source", "Target"]
        ]
        .drop_duplicates()
    )

    all_folds = (
        graphs[
            ["Wave", "Model", "Fold"]
        ]
        .drop_duplicates()
    )

    grid = all_edges.merge(
        all_folds,
        on=["Wave", "Model"],
        how="left",
    )

    fold_grid = grid.merge(
        fold_counts[
            [
                "Wave",
                "Model",
                "Fold",
                "Source",
                "Target",
                "Appearances",
                "GraphsInFold",
                "FoldFrequency",
            ]
        ],
        on=[
            "Wave",
            "Model",
            "Fold",
            "Source",
            "Target",
        ],
        how="left",
    )

    fold_grid["Appearances"] = (
        fold_grid["Appearances"].fillna(0)
    )

    fold_grid["GraphsInFold"] = (
        fold_grid["GraphsInFold"].fillna(
            EXPECTED_COMPLETIONS
        )
    )

    fold_grid["FoldFrequency"] = (
        fold_grid["FoldFrequency"].fillna(0.0)
    )

    summary = (
        fold_grid.groupby(
            ["Wave", "Model", "Source", "Target"],
            as_index=False,
        )
        .agg(
            TotalAppearances=("Appearances", "sum"),
            MeanFoldFrequency=("FoldFrequency", "mean"),
            SDFoldFrequency=("FoldFrequency", "std"),
            MinFoldFrequency=("FoldFrequency", "min"),
            MaxFoldFrequency=("FoldFrequency", "max"),
            FoldsPresent=(
                "FoldFrequency",
                lambda x: int(np.sum(np.asarray(x) > 0)),
            ),
            FoldsMajority=(
                "FoldFrequency",
                lambda x: int(
                    np.sum(
                        np.asarray(x)
                        >= MODERATE_STABILITY
                    )
                ),
            ),
            FoldsHigh=(
                "FoldFrequency",
                lambda x: int(
                    np.sum(
                        np.asarray(x)
                        >= HIGH_STABILITY
                    )
                ),
            ),
        )
    )

    denominators = (
        graphs.groupby(
            ["Wave", "Model"]
        )["GraphID"]
        .nunique()
        .rename("Graphs")
        .reset_index()
    )

    summary = summary.merge(
        denominators,
        on=["Wave", "Model"],
        how="left",
    )

    summary["RawFrequency"] = (
        summary["TotalAppearances"]
        / summary["Graphs"]
    )

    # Equal number of completions per fold means this should equal
    # MeanFoldFrequency. Keep both as an integrity check.
    summary["FrequencyDifferenceCheck"] = (
        summary["RawFrequency"]
        - summary["MeanFoldFrequency"]
    ).abs()

    summary["DirectedEdge"] = (
        summary["Source"]
        + " -> "
        + summary["Target"]
    )

    return summary.sort_values(
        ["Wave", "Model", "RawFrequency"],
        ascending=[True, True, False],
    )


# =============================================================================
# 4. Undirected adjacency stability + direction consistency
# =============================================================================

def adjacency_stability(
    edges: pd.DataFrame,
    graphs: pd.DataFrame,
) -> pd.DataFrame:

    # One adjacency can occur only once per DAG, but enforce this anyway.
    adjacency_edges = (
        edges[
            [
                "GraphID",
                "Wave",
                "Model",
                "Fold",
                "NodeA",
                "NodeB",
                "Source",
                "Target",
            ]
        ]
        .drop_duplicates(
            [
                "GraphID",
                "NodeA",
                "NodeB",
            ]
        )
    )

    fold_counts = (
        adjacency_edges.groupby(
            [
                "Wave",
                "Model",
                "Fold",
                "NodeA",
                "NodeB",
            ]
        )
        .size()
        .rename("Appearances")
        .reset_index()
    )

    denominators = (
        graphs.groupby(
            ["Wave", "Model", "Fold"]
        )["GraphID"]
        .nunique()
        .rename("GraphsInFold")
        .reset_index()
    )

    pair_universe = (
        adjacency_edges[
            ["Wave", "Model", "NodeA", "NodeB"]
        ]
        .drop_duplicates()
    )

    fold_universe = (
        graphs[
            ["Wave", "Model", "Fold"]
        ]
        .drop_duplicates()
    )

    grid = pair_universe.merge(
        fold_universe,
        on=["Wave", "Model"],
        how="left",
    )

    fold_grid = grid.merge(
        fold_counts,
        on=[
            "Wave",
            "Model",
            "Fold",
            "NodeA",
            "NodeB",
        ],
        how="left",
    ).merge(
        denominators,
        on=["Wave", "Model", "Fold"],
        how="left",
    )

    fold_grid["Appearances"] = (
        fold_grid["Appearances"].fillna(0)
    )

    fold_grid["AdjacencyFrequency"] = (
        fold_grid["Appearances"]
        / fold_grid["GraphsInFold"]
    )

    adjacency_summary = (
        fold_grid.groupby(
            ["Wave", "Model", "NodeA", "NodeB"],
            as_index=False,
        )
        .agg(
            TotalAppearances=("Appearances", "sum"),
            MeanFoldFrequency=(
                "AdjacencyFrequency",
                "mean",
            ),
            SDFoldFrequency=(
                "AdjacencyFrequency",
                "std",
            ),
            MinFoldFrequency=(
                "AdjacencyFrequency",
                "min",
            ),
            MaxFoldFrequency=(
                "AdjacencyFrequency",
                "max",
            ),
            FoldsPresent=(
                "AdjacencyFrequency",
                lambda x: int(np.sum(np.asarray(x) > 0)),
            ),
            FoldsMajority=(
                "AdjacencyFrequency",
                lambda x: int(
                    np.sum(
                        np.asarray(x)
                        >= MODERATE_STABILITY
                    )
                ),
            ),
            FoldsHigh=(
                "AdjacencyFrequency",
                lambda x: int(
                    np.sum(
                        np.asarray(x)
                        >= HIGH_STABILITY
                    )
                ),
            ),
        )
    )

    graph_counts = (
        graphs.groupby(
            ["Wave", "Model"]
        )["GraphID"]
        .nunique()
        .rename("Graphs")
        .reset_index()
    )

    adjacency_summary = adjacency_summary.merge(
        graph_counts,
        on=["Wave", "Model"],
        how="left",
    )

    adjacency_summary["AdjacencyFrequency"] = (
        adjacency_summary["TotalAppearances"]
        / adjacency_summary["Graphs"]
    )

    # Direction counts
    direction_counts = (
        adjacency_edges.groupby(
            [
                "Wave",
                "Model",
                "NodeA",
                "NodeB",
                "Source",
                "Target",
            ]
        )
        .size()
        .rename("DirectionCount")
        .reset_index()
    )

    direction_lookup = defaultdict(dict)

    for row in direction_counts.itertuples(
        index=False
    ):
        key = (
            row.Wave,
            row.Model,
            row.NodeA,
            row.NodeB,
        )

        direction_lookup[key][
            (row.Source, row.Target)
        ] = int(row.DirectionCount)

    rows = []

    for row in adjacency_summary.itertuples(
        index=False
    ):
        key = (
            row.Wave,
            row.Model,
            row.NodeA,
            row.NodeB,
        )

        directions = direction_lookup[key]

        count_ab = directions.get(
            (row.NodeA, row.NodeB),
            0,
        )
        count_ba = directions.get(
            (row.NodeB, row.NodeA),
            0,
        )

        total = count_ab + count_ba

        if total > 0:
            prop_ab = count_ab / total
            prop_ba = count_ba / total
        else:
            prop_ab = np.nan
            prop_ba = np.nan

        if count_ab > count_ba:
            dominant_source = row.NodeA
            dominant_target = row.NodeB
            dominant_count = count_ab
        elif count_ba > count_ab:
            dominant_source = row.NodeB
            dominant_target = row.NodeA
            dominant_count = count_ba
        else:
            dominant_source = ""
            dominant_target = ""
            dominant_count = count_ab

        direction_consistency = (
            dominant_count / total
            if total > 0
            else np.nan
        )

        values = row._asdict()

        values.update(
            {
                "NodeA_to_NodeB_Count": count_ab,
                "NodeB_to_NodeA_Count": count_ba,
                "NodeA_to_NodeB_ConditionalFrequency": prop_ab,
                "NodeB_to_NodeA_ConditionalFrequency": prop_ba,
                "DominantSource": dominant_source,
                "DominantTarget": dominant_target,
                "DirectionConsistency": direction_consistency,
                "DominantDirectedEdge": (
                    f"{dominant_source} -> "
                    f"{dominant_target}"
                    if dominant_source
                    else "Tie"
                ),
                "Adjacency": (
                    f"{row.NodeA} -- {row.NodeB}"
                ),
            }
        )

        rows.append(values)

    result = pd.DataFrame(rows)

    return result.sort_values(
        [
            "Wave",
            "Model",
            "AdjacencyFrequency",
            "DirectionConsistency",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    )


# =============================================================================
# 5. Cross-wave persistence
# =============================================================================

def cross_wave_persistence(
    adjacency: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        model,
        node_a,
        node_b,
    ), group in adjacency.groupby(
        ["Model", "NodeA", "NodeB"]
    ):

        by_wave = (
            group.set_index("Wave")[
                "AdjacencyFrequency"
            ].to_dict()
        )

        row = {
            "Model": model,
            "NodeA": node_a,
            "NodeB": node_b,
            "Adjacency": f"{node_a} -- {node_b}",
        }

        values = []

        for wave in EXPECTED_WAVES:
            value = float(
                by_wave.get(wave, 0.0)
            )
            row[f"Wave{wave}Frequency"] = value
            values.append(value)

        row["MeanAcrossWaves"] = float(
            np.mean(values)
        )
        row["MinAcrossWaves"] = float(
            np.min(values)
        )
        row["MaxAcrossWaves"] = float(
            np.max(values)
        )

        row["WavesAtLeast50"] = int(
            np.sum(
                np.asarray(values)
                >= MODERATE_STABILITY
            )
        )

        row["WavesAtLeast80"] = int(
            np.sum(
                np.asarray(values)
                >= HIGH_STABILITY
            )
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    return result.sort_values(
        [
            "Model",
            "WavesAtLeast80",
            "WavesAtLeast50",
            "MeanAcrossWaves",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    )


# =============================================================================
# 6. Dementia neighbourhood
# =============================================================================

def dementia_neighbourhood(
    adjacency: pd.DataFrame,
    all_nodes: set[str],
) -> pd.DataFrame:

    dementia_node = find_dementia_node(
        all_nodes
    )

    if dementia_node is None:
        print(
            "WARNING: Could not automatically identify "
            "the dementia node."
        )
        return pd.DataFrame()

    print(
        f"Dementia node identified as: "
        f"{dementia_node}"
    )

    dementia = adjacency.loc[
        (adjacency["NodeA"] == dementia_node)
        | (adjacency["NodeB"] == dementia_node)
    ].copy()

    dementia["Neighbour"] = np.where(
        dementia["NodeA"] == dementia_node,
        dementia["NodeB"],
        dementia["NodeA"],
    )

    dementia["DementiaDirection"] = np.where(
        dementia["DominantSource"] == dementia_node,
        "Dementia -> Neighbour",
        np.where(
            dementia["DominantTarget"]
            == dementia_node,
            "Neighbour -> Dementia",
            "Direction tie",
        ),
    )

    columns = [
        "Wave",
        "Model",
        "Neighbour",
        "AdjacencyFrequency",
        "MeanFoldFrequency",
        "SDFoldFrequency",
        "MinFoldFrequency",
        "MaxFoldFrequency",
        "FoldsPresent",
        "FoldsMajority",
        "FoldsHigh",
        "DominantDirectedEdge",
        "DirectionConsistency",
        "DementiaDirection",
    ]

    return dementia[
        columns
    ].sort_values(
        [
            "Model",
            "Wave",
            "AdjacencyFrequency",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )


# =============================================================================
# 7. Cross-model agreement
# =============================================================================

def cross_model_agreement(
    adjacency: pd.DataFrame,
) -> pd.DataFrame:

    pivot = adjacency.pivot_table(
        index=[
            "Wave",
            "NodeA",
            "NodeB",
        ],
        columns="Model",
        values="AdjacencyFrequency",
        fill_value=0.0,
    ).reset_index()

    for model in [
        "Discrete",
        "Continuous",
        "Hybrid",
    ]:
        if model not in pivot.columns:
            pivot[model] = 0.0

    pivot["Adjacency"] = (
        pivot["NodeA"]
        + " -- "
        + pivot["NodeB"]
    )

    pivot["MeanAcrossModels"] = pivot[
        ["Discrete", "Continuous", "Hybrid"]
    ].mean(axis=1)

    pivot["MinAcrossModels"] = pivot[
        ["Discrete", "Continuous", "Hybrid"]
    ].min(axis=1)

    pivot["MaxAcrossModels"] = pivot[
        ["Discrete", "Continuous", "Hybrid"]
    ].max(axis=1)

    pivot["RangeAcrossModels"] = (
        pivot["MaxAcrossModels"]
        - pivot["MinAcrossModels"]
    )

    pivot["ModelsAtLeast50"] = (
        pivot[
            [
                "Discrete",
                "Continuous",
                "Hybrid",
            ]
        ]
        >= MODERATE_STABILITY
    ).sum(axis=1)

    pivot["ModelsAtLeast80"] = (
        pivot[
            [
                "Discrete",
                "Continuous",
                "Hybrid",
            ]
        ]
        >= HIGH_STABILITY
    ).sum(axis=1)

    return pivot.sort_values(
        [
            "Wave",
            "ModelsAtLeast80",
            "ModelsAtLeast50",
            "MeanAcrossModels",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    )


# =============================================================================
# 8. Node connectivity
# =============================================================================

def node_connectivity(
    adjacency: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for row in adjacency.itertuples(
        index=False
    ):
        for node, neighbour in [
            (row.NodeA, row.NodeB),
            (row.NodeB, row.NodeA),
        ]:
            rows.append(
                {
                    "Wave": row.Wave,
                    "Model": row.Model,
                    "Node": node,
                    "Neighbour": neighbour,
                    "AdjacencyFrequency": (
                        row.AdjacencyFrequency
                    ),
                }
            )

    long = pd.DataFrame(rows)

    summary = (
        long.groupby(
            ["Wave", "Model", "Node"],
            as_index=False,
        )
        .agg(
            ExpectedDegree=(
                "AdjacencyFrequency",
                "sum",
            ),
            StableNeighbours50=(
                "AdjacencyFrequency",
                lambda x: int(
                    np.sum(
                        np.asarray(x)
                        >= MODERATE_STABILITY
                    )
                ),
            ),
            StableNeighbours80=(
                "AdjacencyFrequency",
                lambda x: int(
                    np.sum(
                        np.asarray(x)
                        >= HIGH_STABILITY
                    )
                ),
            ),
            MaximumAdjacencyFrequency=(
                "AdjacencyFrequency",
                "max",
            ),
        )
    )

    return summary.sort_values(
        [
            "Wave",
            "Model",
            "ExpectedDegree",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )


# =============================================================================
# 9. Text report
# =============================================================================

def build_report(
    inventory: pd.DataFrame,
    adjacency: pd.DataFrame,
    persistence: pd.DataFrame,
    dementia: pd.DataFrame,
    agreement: pd.DataFrame,
) -> str:

    lines: list[str] = []

    lines.append(
        "ELSA SCIENTIFIC STRUCTURE ANALYSIS"
    )
    lines.append("=" * 88)
    lines.append("")

    lines.append("GRAPH INVENTORY")
    lines.append("-" * 88)

    for row in inventory.itertuples(
        index=False
    ):
        lines.append(
            f"Wave {row.Wave} | {row.Model}: "
            f"{row.GraphCount} graphs "
            f"(expected {row.ExpectedGraphs}); "
            f"mean edges={row.MeanEdgeCount:.2f}; "
            f"range={row.MinEdgeCount}-"
            f"{row.MaxEdgeCount}"
        )

    lines.append("")
    lines.append(
        "IMPORTANT INTERPRETATION NOTE"
    )
    lines.append("-" * 88)
    lines.append(
        "The 25 completed datasets within each fold "
        "are not treated as 25 independent scientific "
        "replicates. Edge frequency is reported over "
        "the complete 125-graph wave ensemble, while "
        "fold-specific frequencies are retained to "
        "assess whether structural support generalises "
        "across held-out partitions."
    )

    lines.append("")
    lines.append(
        "TOP HYBRID ADJACENCIES BY WAVE"
    )
    lines.append("-" * 88)

    hybrid = adjacency.loc[
        adjacency["Model"] == "Hybrid"
    ]

    for wave in EXPECTED_WAVES:
        lines.append("")
        lines.append(f"Wave {wave}")

        top = (
            hybrid.loc[
                hybrid["Wave"] == wave
            ]
            .head(15)
        )

        for row in top.itertuples(
            index=False
        ):
            lines.append(
                f"  {row.Adjacency}: "
                f"freq={row.AdjacencyFrequency:.3f}; "
                f"fold range="
                f"{row.MinFoldFrequency:.2f}-"
                f"{row.MaxFoldFrequency:.2f}; "
                f"dominant={row.DominantDirectedEdge}; "
                f"direction consistency="
                f"{row.DirectionConsistency:.3f}"
            )

    lines.append("")
    lines.append(
        "MOST PERSISTENT HYBRID RELATIONSHIPS"
    )
    lines.append("-" * 88)

    persistent_hybrid = (
        persistence.loc[
            persistence["Model"] == "Hybrid"
        ]
        .head(25)
    )

    for row in persistent_hybrid.itertuples(
        index=False
    ):
        lines.append(
            f"{row.Adjacency}: "
            f"W2={row.Wave2Frequency:.2f}, "
            f"W4={row.Wave4Frequency:.2f}, "
            f"W6={row.Wave6Frequency:.2f}, "
            f"W8={row.Wave8Frequency:.2f}; "
            f"mean={row.MeanAcrossWaves:.3f}; "
            f"waves>=0.8={row.WavesAtLeast80}/4"
        )

    if not dementia.empty:
        lines.append("")
        lines.append(
            "HYBRID DEMENTIA NEIGHBOURHOOD"
        )
        lines.append("-" * 88)

        hd = dementia.loc[
            dementia["Model"] == "Hybrid"
        ]

        for wave in EXPECTED_WAVES:
            lines.append("")
            lines.append(f"Wave {wave}")

            wave_rows = hd.loc[
                hd["Wave"] == wave
            ].head(20)

            for row in wave_rows.itertuples(
                index=False
            ):
                lines.append(
                    f"  {row.Neighbour}: "
                    f"adjacency={row.AdjacencyFrequency:.3f}; "
                    f"fold range="
                    f"{row.MinFoldFrequency:.2f}-"
                    f"{row.MaxFoldFrequency:.2f}; "
                    f"{row.DominantDirectedEdge}; "
                    f"direction consistency="
                    f"{row.DirectionConsistency:.3f}"
                )

    lines.append("")
    lines.append(
        "STRONGEST THREE-MODEL AGREEMENT"
    )
    lines.append("-" * 88)

    strong_agreement = (
        agreement.loc[
            agreement["ModelsAtLeast50"] == 3
        ]
        .head(30)
    )

    for row in strong_agreement.itertuples(
        index=False
    ):
        lines.append(
            f"Wave {row.Wave} | {row.Adjacency}: "
            f"D={row.Discrete:.3f}, "
            f"C={row.Continuous:.3f}, "
            f"H={row.Hybrid:.3f}"
        )

    lines.append("")
    lines.append("=" * 88)
    lines.append("END OF REPORT")

    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    edges, graphs = load_all_edges()

    print()
    print("Loaded graph count:", len(graphs))
    print("Loaded edge rows:", len(edges))

    expected_total = (
        len(EXPECTED_WAVES)
        * EXPECTED_FOLDS
        * EXPECTED_COMPLETIONS
        * len(MODEL_FOLDERS)
    )

    print(
        "Expected total graphs:",
        expected_total,
    )

    if len(graphs) != expected_total:
        print(
            "WARNING: Graph count differs from expected."
        )

    all_nodes = set(edges["Source"]).union(
        set(edges["Target"])
    )

    print(
        "Unique nodes found:",
        len(all_nodes),
    )

    print(
        "Nodes:",
        sorted(all_nodes),
    )

    inventory = graph_inventory(graphs)

    directed = directed_edge_stability(
        edges,
        graphs,
    )

    adjacency = adjacency_stability(
        edges,
        graphs,
    )

    persistence = cross_wave_persistence(
        adjacency
    )

    dementia = dementia_neighbourhood(
        adjacency,
        all_nodes,
    )

    agreement = cross_model_agreement(
        adjacency
    )

    connectivity = node_connectivity(
        adjacency
    )

    edges.to_csv(
        OUTPUT_ROOT / "all_edges_long.csv",
        index=False,
    )

    graphs.to_csv(
        OUTPUT_ROOT / "graph_inventory_raw.csv",
        index=False,
    )

    inventory.to_csv(
        OUTPUT_ROOT / "graph_inventory.csv",
        index=False,
    )

    directed.to_csv(
        OUTPUT_ROOT
        / "directed_edge_stability.csv",
        index=False,
    )

    adjacency.to_csv(
        OUTPUT_ROOT
        / "adjacency_stability.csv",
        index=False,
    )

    persistence.to_csv(
        OUTPUT_ROOT
        / "cross_wave_persistence.csv",
        index=False,
    )

    dementia.to_csv(
        OUTPUT_ROOT
        / "dementia_neighbourhood.csv",
        index=False,
    )

    agreement.to_csv(
        OUTPUT_ROOT
        / "cross_model_agreement.csv",
        index=False,
    )

    connectivity.to_csv(
        OUTPUT_ROOT
        / "node_connectivity.csv",
        index=False,
    )

    report = build_report(
        inventory,
        adjacency,
        persistence,
        dementia,
        agreement,
    )

    report_path = (
        OUTPUT_ROOT
        / "scientific_analysis_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("Analysis complete.")
    print(f"Results written to: {OUTPUT_ROOT}")
    print()
    print(inventory.to_string(index=False))
    print()
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
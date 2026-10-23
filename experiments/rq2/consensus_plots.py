"""Report-only RQ2 figure generator (Abdullah Sibda research report)."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROCESS_NODES = {"Proxy", "Nurse", "SelfCompletion"}
WAVES = [2, 4, 6, 8]

def plot_dementia_heatmap(
    dementia: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Main scientific heatmap:
    process/missingness nodes are excluded so the figure focuses on substantive
    health dependencies. They remain in the network figures.
    """
    substantive = dementia.loc[
        ~dementia["Neighbour"].isin(PROCESS_NODES)
    ].copy()

    pivot = substantive.pivot_table(
        index="Neighbour",
        columns="Wave",
        values="AdjacencyFrequency",
        aggfunc="max",
        fill_value=0.0,
    )

    for wave in WAVES:
        if wave not in pivot.columns:
            pivot[wave] = 0.0

    pivot = pivot[WAVES]

    # Rank rows by mean Hybrid adjacency to Dementia across waves.
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False)
    pivot = pivot.drop(columns="_mean")

    values = pivot.to_numpy(dtype=float)

    # Report-ready typography, consistent with the RQ1.5 figures.
    # Extra width/row height accommodates 12 pt cell labels without crowding.
    fig_height = max(6.8, 0.42 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(5.2, fig_height))

    image = ax.imshow(
        values,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    # Subtle cell boundaries keep values visually separate.
    ax.set_xticks(np.arange(-0.5, len(WAVES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(np.arange(len(WAVES)))
    ax.set_xticklabels([f"Wave {wave}" for wave in WAVES], fontsize=11)

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index), fontsize=11)
    ax.tick_params(axis="both", which="both", length=0, pad=7)

    ax.set_title(
        "Dementia-neighbourhood stability",
        fontsize=14,
        fontweight="semibold",
        pad=15,
    )
    ax.set_xlabel("ELSA wave", fontsize=12, labelpad=10)
    ax.set_ylabel("Substantive neighbour", fontsize=12, labelpad=10)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="semibold",
                # Matplotlib's default viridis map is dark at the low end;
                # use white text there and dark text on the brighter cells.
                color="white" if value < 0.50 else "black",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label(
        "Adjacency frequency",
        fontsize=11,
        labelpad=8,
    )
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout(pad=0.7)

    pdf_path = output_dir / "hybrid_dementia_neighbourhood_heatmap.pdf"
    png_path = output_dir / "hybrid_dementia_neighbourhood_heatmap.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    pivot.to_csv(
        output_dir / "hybrid_dementia_neighbourhood_heatmap_values.csv"
    )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dementia", type=Path, help="Path to analyse_graph_ensembles.py output dementia_neighbourhood.csv")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    source = (args.dementia or root / "ScientificResults" / "dementia_neighbourhood.csv").expanduser().resolve()
    out = (args.output_dir or root / "ScientificPlots").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    dementia = pd.read_csv(source)
    dementia = dementia[dementia["Model"].astype(str) == "Hybrid"].copy()
    if dementia.empty:
        raise ValueError("No Hybrid Dementia-neighbourhood rows in supplied summary")
    plot_dementia_heatmap(dementia, out)
    print(f"Report Figure 11 saved to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

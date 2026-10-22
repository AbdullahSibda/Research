"""RQ1 continuous-ground-truth report figures.

Reproduce ONLY the two 4 x 1 final-paper learning curves: common-resolution
KL and structural Hamming distance (SHD). Means and +/- 1 SD are pooled
across seeds, using categorical sample-size positions.

Inputs: master_model_results.csv (complete, full benchmark results).
Outputs: vector PDF and 300-dpi PNG for each figure. No model fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# Figure configuration
# =============================================================================

GENERATOR_ORDER = ["unimodal", "skew", "multimodal", "skew_multi"]
GENERATOR_LABELS = {"unimodal": "Unimodal", "skew": "Skew", "multimodal": "Multimodal", "skew_multi": "Skew-Multimodal"}
MODEL_ORDER = ["Continuous", "Discrete", "Hybrid"]
MODEL_LABELS = {name: name for name in MODEL_ORDER}
MODEL_LINESTYLES = {"Continuous": "-", "Discrete": "--", "Hybrid": "-."}
MODEL_MARKERS = {"Continuous": "o", "Discrete": "s", "Hybrid": "^"}


# =============================================================================
# Repository paths and CLI
# =============================================================================

def discover_project_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents):
        if (candidate / "experiments" / "rq1").is_dir() or (candidate / "src" / "hybrid_bn").is_dir():
            return candidate
    raise FileNotFoundError("Could not detect repository root; provide --project-root.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory containing master_model_results.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination for the two final-report figures")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


# =============================================================================
# Data validation and seeded summaries
# =============================================================================

def load_results(path: Path) -> tuple[pd.DataFrame, list[int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}; pass --results-dir pointing to FULL original results.")
    master = pd.read_csv(path)
    required = {"Generator", "Seed", "Sample_Size", "Model", "Common_KL_K3", "SHD"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"master_model_results.csv missing columns: {missing}")
    master = master[master["Generator"].isin(GENERATOR_ORDER) & master["Model"].isin(MODEL_ORDER)].copy()
    master["Sample_Size"] = master["Sample_Size"].astype(int)
    sample_sizes = sorted(master["Sample_Size"].unique().tolist())
    if not sample_sizes:
        raise ValueError("No valid benchmark rows found")
    if master.groupby(["Generator", "Sample_Size", "Model"]).size().lt(10).any():
        print("[WARN] Some generator/sample-size/model groups have fewer than 10 seeds.")
    return master, sample_sizes



def metric_stats(master: pd.DataFrame, y_raw: str) -> pd.DataFrame:
    """Mean and SD across seeds for each generator x n x model cell."""
    return (
        master.groupby(["Generator", "Sample_Size", "Model"], as_index=False)
        .agg(
            Plot_Mean=(y_raw, "mean"),
            Plot_SD=(y_raw, "std"),
            Runs=("Seed", "size"),
        )
    )

def compact_sample_label(n: int) -> str:
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)

# =============================================================================
# Original publication-format 4 x 1 renderer (unchanged)
# =============================================================================

def single_column_plot(
    *,
    master: pd.DataFrame,
    y_raw: str,
    y_label: str,
    out_base: Path,
    sample_sizes: list[int],
    dpi: int,
) -> None:
    """
    Publication-oriented 4x1 version sized for one cas-dc column.

    The figure deliberately has no overall title: the LaTeX caption supplies
    that context and avoiding a title saves valuable vertical space.
    """
    stats = metric_stats(master, y_raw)

    # About one Elsevier column wide.  The height keeps all four panels legible
    # without consuming an entire page once the LaTeX caption is included.
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(5.2, 8.8),
        sharex=True,
    )

    x_positions = {n: i for i, n in enumerate(sample_sizes)}
    tick_positions = list(range(len(sample_sizes)))
    tick_labels = [compact_sample_label(n) for n in sample_sizes]

    legend_handles = []
    legend_labels = []

    for panel_index, (ax, generator) in enumerate(zip(axes, GENERATOR_ORDER)):
        g_stats = stats[stats["Generator"] == generator]

        for model in MODEL_ORDER:
            m_stats = (
                g_stats[g_stats["Model"] == model]
                .copy()
                .sort_values("Sample_Size")
            )
            if m_stats.empty:
                continue

            x = [x_positions[int(n)] for n in m_stats["Sample_Size"]]
            container = ax.errorbar(
                x,
                m_stats["Plot_Mean"],
                yerr=m_stats["Plot_SD"].fillna(0.0),
                linestyle=MODEL_LINESTYLES[model],
                linewidth=1.45,
                marker=MODEL_MARKERS[model],
                markersize=4.1,
                capsize=2.6,
                capthick=0.85,
                elinewidth=0.85,
                label=MODEL_LABELS[model],
            )

            if panel_index == 0:
                legend_handles.append(container.lines[0])
                legend_labels.append(MODEL_LABELS[model])

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            f"({panel_letter}) {GENERATOR_LABELS[generator]}",
            fontsize=15,
            pad=6,
            fontweight="semibold",
        )
        ax.set_xlim(-0.32, len(sample_sizes) - 0.68)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.55)
        ax.tick_params(axis="y", labelsize=12)
        ax.tick_params(axis="x", labelsize=12)

        # Keep the x labels only on the bottom panel to save space.
        if panel_index < len(GENERATOR_ORDER) - 1:
            ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels)
    axes[-1].set_xlabel("Sample size", fontsize=14, labelpad=7)

    # One shared y-axis label is much cleaner at single-column width.
    fig.supylabel(y_label, fontsize=14, x=0.012)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=12.5,
        handlelength=2.4,
        columnspacing=0.9,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.tight_layout(
        rect=(0.07, 0.03, 1.0, 0.955),
        h_pad=1.15,
    )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# Generate exactly the two report figures
# =============================================================================

def main() -> int:
    args = parse_args()
    root = discover_project_root(args.project_root)
    if args.results_dir is not None:
        results_dir = args.results_dir.expanduser().resolve()
    else:
        candidates = [root / "ContinuousResults" / "RQ1_Final_Benchmark", root / "Outputs" / "RQ1_Final_Benchmark"]
        results_dir = next((p for p in candidates if (p / "master_model_results.csv").is_file()), candidates[0])
    output_dir = (args.output_dir or (root / "results" / "rq1" / "figures" / "continuous")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master, sample_sizes = load_results(results_dir / "master_model_results.csv")

    for metric, label, filename in (
        ("Common_KL_K3", "Common-resolution KL", "rq1_continuous_common_kl_singlecol"),
        ("SHD", "SHD", "rq1_continuous_shd_singlecol"),
    ):
        single_column_plot(master=master, y_raw=metric, y_label=label,
                           out_base=output_dir / filename, sample_sizes=sample_sizes, dpi=args.dpi)
        print(f"Saved {filename}.pdf and {filename}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

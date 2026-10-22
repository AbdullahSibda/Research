"""RQ1.5 native-mixed final-report figures.

Produces exactly three figures (PNG and PDF): common-support K=3 KL
learning curves, SHD learning curves, and mature-sample directed-edge
quadrant F1. Reads full results only; does not retrain models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Figure configuration
# =============================================================================

GENERATOR_ORDER = ["unimodal", "skew", "multimodal", "skew_multi"]
GENERATOR_LABELS = {
    "unimodal": "Unimodal",
    "skew": "Skewed",
    "multimodal": "Multimodal",
    "skew_multi": "Skewed–multimodal",
}

MODEL_ORDER = ["Continuous", "Discrete", "Hybrid"]
MODEL_MARKERS = {
    "Continuous": "s",
    "Discrete": "o",
    "Hybrid": "D",
}
MODEL_LINESTYLES = {
    "Continuous": ":",
    "Discrete": "--",
    "Hybrid": "-",
}

SAMPLE_ORDER = [100, 500, 2000, 5000]

QUADRANT_ORDER = ["CC", "CD", "DC", "DD"]
QUADRANT_LABELS = {
    "CC": r"$C\rightarrow C$",
    "CD": r"$C\rightarrow D$",
    "DC": r"$D\rightarrow C$",
    "DD": r"$D\rightarrow D$",
}



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
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory containing full mixed_model_results.csv and mixed_quadrant_diagnostics.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quadrant-n", type=int, default=5000)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()

# =============================================================================
# Input validation and seeded summaries
# =============================================================================

def load_inputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = results_dir / "mixed_model_results.csv"
    quadrant_path = results_dir / "mixed_quadrant_diagnostics.csv"

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not quadrant_path.exists():
        raise FileNotFoundError(quadrant_path)

    model = pd.read_csv(model_path)
    quadrant = pd.read_csv(quadrant_path)

    required_model = {
        "Generator",
        "Seed",
        "Sample_Size",
        "Model",
        "Common_KL_K3",
        "SHD",
    }
    required_quadrant = {
        "Generator",
        "Seed",
        "Sample_Size",
        "Model",
        "Quadrant",
        "F1",
    }

    missing_model = sorted(required_model - set(model.columns))
    missing_quadrant = sorted(required_quadrant - set(quadrant.columns))

    if missing_model:
        raise ValueError(
            f"mixed_model_results.csv is missing columns: {missing_model}"
        )
    if missing_quadrant:
        raise ValueError(
            f"mixed_quadrant_diagnostics.csv is missing columns: {missing_quadrant}"
        )

    return model, quadrant

def learning_summary(
    model_results: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Mean ± 1 SD across the 10 seeded networks in each benchmark cell."""
    return (
        model_results.groupby(
            ["Generator", "Sample_Size", "Model"],
            as_index=False,
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "Mean",
                "std": "SD",
                "count": "Runs",
            }
        )
    )

def quadrant_summary(
    quadrant_results: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    """Pool the four generators at the selected sample size."""
    subset = quadrant_results.loc[
        quadrant_results["Sample_Size"] == sample_size
    ].copy()

    if subset.empty:
        available = sorted(
            quadrant_results["Sample_Size"].dropna().astype(int).unique()
        )
        raise ValueError(
            f"No quadrant rows found for n={sample_size}. "
            f"Available sample sizes: {available}"
        )

    return (
        subset.groupby(
            ["Quadrant", "Model"],
            as_index=False,
        )["F1"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "Mean_F1",
                "std": "SD_F1",
                "count": "Runs",
            }
        )
    )

# =============================================================================
# Original final-report renderers
# =============================================================================

def plot_learning_curves(
    summary: pd.DataFrame,
    *,
    metric_label: str,
    output_stem: Path,
    dpi: int = 300,
) -> None:
    """
    Four vertically stacked generator panels.

    Only one shared y-axis label is used for the whole figure.
    Error bars are ±1 SD across the 10 seeded networks.
    """
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(5.2, 8.8),
        sharex=True,
    )

    x = np.arange(len(SAMPLE_ORDER), dtype=float)

    for ax, generator in zip(axes, GENERATOR_ORDER):
        generator_data = summary.loc[
            summary["Generator"] == generator
        ]

        for model in MODEL_ORDER:
            rows = (
                generator_data.loc[
                    generator_data["Model"] == model
                ]
                .set_index("Sample_Size")
                .reindex(SAMPLE_ORDER)
            )

            if rows["Mean"].isna().any():
                missing_n = [
                    SAMPLE_ORDER[i]
                    for i, missing in enumerate(rows["Mean"].isna())
                    if missing
                ]
                raise RuntimeError(
                    f"Missing {metric_label} rows for "
                    f"{generator}/{model}: n={missing_n}"
                )

            ax.errorbar(
                x,
                rows["Mean"].to_numpy(dtype=float),
                yerr=rows["SD"].fillna(0.0).to_numpy(dtype=float),
                marker=MODEL_MARKERS[model],
                linestyle=MODEL_LINESTYLES[model],
                linewidth=2.0,
                markersize=6.5,
                capsize=3.2,
                label=model,
            )

        ax.set_title(
            GENERATOR_LABELS[generator],
            fontsize=15,
            pad=6,
            fontweight="semibold",
        )
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=12)

        if metric_label == "SHD":
            ax.set_ylim(bottom=0)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [f"{n:,}" for n in SAMPLE_ORDER],
        fontsize=12,
    )
    axes[-1].set_xlabel(
        "Sample size",
        fontsize=14,
        labelpad=7,
    )

    # One shared y-axis label for the complete figure.
    fig.supylabel(
        metric_label,
        x=0.012,
        fontsize=14,
    )

    # One shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=12.5,
        handlelength=2.4,
    )

    fig.tight_layout(
        rect=(0.07, 0.01, 1.0, 0.955),
        h_pad=1.15,
    )

    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

def plot_quadrant_bars(
    summary: pd.DataFrame,
    *,
    sample_size: int,
    output_stem: Path,
    dpi: int = 300,
) -> None:
    """
    Grouped bars of directed-edge F1 at the mature sample size.

    No error bars and no in-plot explanatory caption are used: the paper
    caption will carry that information.
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.8))

    x = np.arange(len(QUADRANT_ORDER), dtype=float)
    # Bars within each quadrant touch edge-to-edge.
    width = 0.24

    offsets = {
        "Continuous": -width,
        "Discrete": 0.0,
        "Hybrid": width,
    }

    hatches = {
        "Continuous": "..",
        "Discrete": "//",
        "Hybrid": "xx",
    }

    for model in MODEL_ORDER:
        rows = (
            summary.loc[
                summary["Model"] == model
            ]
            .set_index("Quadrant")
            .reindex(QUADRANT_ORDER)
        )

        if rows["Mean_F1"].isna().any():
            missing = [
                QUADRANT_ORDER[i]
                for i, is_missing in enumerate(rows["Mean_F1"].isna())
                if is_missing
            ]
            raise RuntimeError(
                f"Missing quadrant F1 rows for {model}: {missing}"
            )

        means = rows["Mean_F1"].to_numpy(dtype=float)

        bars = ax.bar(
            x + offsets[model],
            means,
            width=width,
            label=model,
            hatch=hatches[model],
        )



    ax.set_xticks(x)
    ax.set_xticklabels(
        [QUADRANT_LABELS[q] for q in QUADRANT_ORDER],
        fontsize=14,
    )
    ax.set_xlabel(
        "True parent → child variable type",
        fontsize=14,
        labelpad=8,
    )
    ax.set_ylabel(
        "Directed-edge F1",
        fontsize=14,
        labelpad=8,
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0.0, 1.06)
    ax.grid(axis="y", alpha=0.25)

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        fontsize=12.5,
        handlelength=1.8,
    )

    fig.tight_layout()

    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

# =============================================================================
# Generate exactly the three report figures
# =============================================================================

def main() -> int:
    args = parse_args()
    root = discover_project_root(args.project_root)
    if args.results_dir is not None:
        results_dir = args.results_dir.expanduser().resolve()
    else:
        candidates = [root / "MixedResults" / "RQ1_5_NativeMixed_LearningCurves", root / "results" / "rq1" / "mixed"]
        results_dir = next((p for p in candidates if (p / "mixed_model_results.csv").is_file()), candidates[0])
    output_dir = (args.output_dir or (root / "results" / "rq1" / "figures" / "mixed")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_results, quadrant_results = load_inputs(results_dir)
    counts = model_results.groupby(["Generator", "Sample_Size", "Model"]).size()
    if not (counts == 10).all():
        print("[WARN] Not all Generator x Sample_Size x Model cells contain 10 seeds.")

    plot_learning_curves(learning_summary(model_results, "Common_KL_K3"),
                         metric_label="Common-support KL divergence",
                         output_stem=output_dir / "rq1_mixed_kl_4x1", dpi=args.dpi)
    plot_learning_curves(learning_summary(model_results, "SHD"),
                         metric_label="SHD", output_stem=output_dir / "rq1_mixed_shd_4x1", dpi=args.dpi)
    plot_quadrant_bars(quadrant_summary(quadrant_results, args.quadrant_n),
                       sample_size=args.quadrant_n,
                       output_stem=output_dir / "rq1_mixed_quadrant_f1", dpi=args.dpi)
    for stem in ("rq1_mixed_kl_4x1", "rq1_mixed_shd_4x1", "rq1_mixed_quadrant_f1"):
        print(f"Saved {output_dir / stem}.pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Report-only RQ2 figure generator (Abdullah Sibda research report)."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Iterable

MODEL_ORDER = ["Discrete", "Continuous", "Hybrid"]
MODEL_LABELS = {
    "Discrete": "Discrete",
    "Continuous": "Continuous",
    "Hybrid": "Hybrid",
}
MODEL_LINESTYLES = {
    "Discrete": "--",
    "Continuous": "-",
    "Hybrid": "-.",
}
MODEL_MARKERS = {
    "Discrete": "s",
    "Continuous": "o",
    "Hybrid": "^",
}
WAVE_ORDER = [2, 4, 6, 8]





def norm_text(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())

def canonical_model(model: object) -> str:
    s = norm_text(model)
    if "hybrid" in s:
        return "Hybrid"
    if "continuous" in s:
        return "Continuous"
    if "discrete" in s:
        return "Discrete"
    return str(model)

def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(columns)
    exact = {c: c for c in cols}
    lower = {c.lower(): c for c in cols}
    normal = {norm_text(c): c for c in cols}
    for c in candidates:
        if c in exact:
            return exact[c]
        if c.lower() in lower:
            return lower[c.lower()]
        if norm_text(c) in normal:
            return normal[norm_text(c)]
    return None

def load_threshold_curves(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find threshold-curve file: {path}")

    df = pd.read_csv(path)
    wave_col = first_present(df.columns, ["wave"])
    model_col = first_present(df.columns, ["model"])
    threshold_col = first_present(df.columns, ["threshold", "cutoff", "probability_threshold"])
    f1_col = first_present(df.columns, ["f1", "f1_score", "macro_f1"])

    missing = [
        name for name, col in {
            "wave": wave_col,
            "model": model_col,
            "threshold": threshold_col,
            "f1": f1_col,
        }.items() if col is None
    ]
    if missing:
        raise ValueError(
            "dementia_threshold_curves.csv is missing required fields: " + ", ".join(missing)
        )

    out = pd.DataFrame({
        "wave": pd.to_numeric(df[wave_col], errors="coerce").astype("Int64"),
        "model": df[model_col].map(canonical_model),
        "threshold": pd.to_numeric(df[threshold_col], errors="coerce"),
        "F1": pd.to_numeric(df[f1_col], errors="coerce"),
    })
    out = out.dropna(subset=["wave", "model", "threshold", "F1"]).copy()
    out["wave"] = out["wave"].astype(int)
    out = out[out["model"].isin(MODEL_ORDER)].copy()
    out = out.sort_values(["wave", "model", "threshold"], kind="stable")
    return out

def save_figure(fig: plt.Figure, base_path: Path, dpi: int) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_f1_vs_threshold(curves: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    waves_present = [w for w in WAVE_ORDER if w in set(curves["wave"])]
    if not waves_present:
        raise ValueError("No Dementia threshold-curve waves found to plot.")

    fig, axes = plt.subplots(4, 1, figsize=(5.2, 11.2), sharex=True, sharey=True)
    axes = axes.ravel()

    legend_handles = []
    legend_labels = []

    for ax, wave in zip(axes, WAVE_ORDER):
        wdf = curves[curves["wave"] == wave].copy()
        if wdf.empty:
            ax.axis("off")
            continue

        peak_handles = []

        for model in MODEL_ORDER:
            mdf = wdf[wdf["model"] == model].copy().sort_values("threshold")
            if mdf.empty:
                continue

            line, = ax.plot(
                mdf["threshold"],
                mdf["F1"],
                linestyle=MODEL_LINESTYLES[model],
                linewidth=2.2,
                label=MODEL_LABELS[model],
            )

            # Mark the peak F1 on the curve.  The numerical peak values are
            # listed in a compact key inside the upper-right of each panel so
            # the curves themselves remain uncluttered.
            peak_idx = mdf["F1"].idxmax()
            peak = mdf.loc[peak_idx]
            peak_color = line.get_color()
            peak_marker = MODEL_MARKERS[model]

            ax.plot(
                [peak["threshold"]],
                [peak["F1"]],
                marker=peak_marker,
                markersize=10,
                linestyle="None",
                color=peak_color,
                markeredgecolor=peak_color,
                zorder=5,
            )

            peak_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=peak_marker,
                    linestyle="None",
                    markersize=9,
                    color=peak_color,
                    markeredgecolor=peak_color,
                    label=(
                        f"{MODEL_LABELS[model]}: "
                        f"t={peak['threshold']:.2f}, F1={peak['F1']:.2f}"
                    ),
                )
            )

            if wave == waves_present[0]:
                legend_handles.append(line)
                legend_labels.append(MODEL_LABELS[model])

        if peak_handles:
            ax.legend(
                handles=peak_handles,
                loc="upper right",
                fontsize=10,
                frameon=True,
                framealpha=0.90,
                borderpad=0.45,
                labelspacing=0.35,
                handletextpad=0.45,
                title="Peak values",
                title_fontsize=10,
            )

        ax.set_title(f"Wave {wave}", fontsize=14)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Decision threshold", fontsize=12)
        ax.set_ylabel("F1", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)

    # Hide unused axes if fewer than four waves somehow exist.
    for i in range(len(WAVE_ORDER), len(axes)):
        axes[i].axis("off")

    fig.suptitle("Dementia: F1 vs Threshold", fontsize=14, y=0.99)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_handles),
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.955),
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=1.5)

    out = out_dir / "dementia_f1_vs_threshold"
    save_figure(fig, out, dpi)
    return out.with_suffix(".pdf")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/MasterPipeline/CV_HeldoutInference"))
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_dir = (args.plot_dir or output_root / "plots" / "report").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = output_root / "metrics" / "dementia_threshold_curves.csv"
    plot_f1_vs_threshold(load_threshold_curves(threshold_path), output_dir, args.dpi)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

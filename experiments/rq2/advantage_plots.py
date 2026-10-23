"""Report-only RQ2 figure generator (Abdullah Sibda research report)."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

TARGET_ORDER = [
    "Dementia",
    "ManagingMeds",
    "LowerMobility",
    "CESD",
    "Recall",
    "Grip",
    "CASP19",
    "Loneliness",
]

TARGET_LABELS = {
    "Dementia": "Dementia",
    "ManagingMeds": "Managing Meds",
    "LowerMobility": "Lower Mobility",
    "CESD": "CESD",
    "Recall": "Recall",
    "Grip": "Grip",
    "CASP19": "CASP19",
    "Loneliness": "Loneliness",
}

WAVE_ORDER = [2, 4, 6, 8]

CONTRAST_ORDER = [
    "Hybrid - Continuous",
    "Hybrid - Discrete",
]

CONTRAST_LABELS = {
    "Hybrid - Continuous": "Hybrid vs Continuous",
    "Hybrid - Discrete": "Hybrid vs Discrete",
}

CONTRAST_MARKERS = {
    "Hybrid - Continuous": "o",
    "Hybrid - Discrete": "s",
}

# Small vertical offsets stop the two baseline contrasts from sitting directly
# on top of one another when both are available for a row.
CONTRAST_Y_OFFSETS = {
    "Hybrid - Continuous": +0.13,
    "Hybrid - Discrete": -0.13,
}

METRIC_LABELS = {
    "F1": "F1 advantage",
    "LogLoss": "Log-loss advantage",
    "NLPD": "NLPD advantage",
    "RMSE": "RMSE advantage",
}

METRIC_TITLES = {
    "F1": "Hybrid Paired Advantage — F1",
    "LogLoss": "Hybrid Paired Advantage — Log Loss",
    "NLPD": "Hybrid Paired Advantage — NLPD",
    "RMSE": "Hybrid Paired Advantage — RMSE",
}

# Dementia receives separate threshold/ranking diagnostics (fixed-threshold F1,
# oracle-max F1, ROC-AUC, PR-AUC, etc.), so its headline F1 contrast is omitted
# from the general F1 forest to keep that special-case interpretation separate.
PLOT_EXCLUDED_TARGETS_BY_METRIC = {
    "F1": {"Dementia"},
}





def ordered_targets(df: pd.DataFrame) -> list[str]:
    present = list(df["target"].dropna().astype(str).unique())
    ordered = [t for t in TARGET_ORDER if t in present]
    extras = sorted(t for t in present if t not in TARGET_ORDER)
    return ordered + extras

def build_rows(metric_df: pd.DataFrame) -> pd.DataFrame:
    """Build grouped target rows, with short wave-only tick labels."""
    targets = ordered_targets(metric_df)
    rows = []
    row_index = 0

    for target in targets:
        target_df = metric_df[metric_df["target"] == target]
        present_waves = [w for w in WAVE_ORDER if w in set(target_df["wave"].dropna().astype(int))]
        extras = sorted(
            set(target_df["wave"].dropna().astype(int)) - set(WAVE_ORDER)
        )
        for wave in present_waves + extras:
            rows.append(
                {
                    "target": target,
                    "wave": int(wave),
                    "row_index": row_index,
                    "row_label": f"W{int(wave)}",
                }
            )
            row_index += 1

    return pd.DataFrame(rows)

def symmetric_xlim(metric_df: pd.DataFrame) -> tuple[float, float]:
    vals = pd.concat(
        [
            metric_df["estimate_std"].abs(),
            metric_df["ci_low_std"].abs(),
            metric_df["ci_high_std"].abs(),
        ],
        ignore_index=True,
    ).dropna()

    if vals.empty:
        return (-1.0, 1.0)

    max_abs = float(vals.max())
    if max_abs <= 0:
        max_abs = 1.0

    # A little room for caps/points and for the arrow annotations at the top.
    lim = max_abs * 1.12
    return (-lim, lim)

def add_target_bands_and_separators(ax, rows: pd.DataFrame) -> None:
    # Very light alternating group shading improves scanning across 20-30 rows
    # without encoding any additional scientific information.
    targets = rows["target"].drop_duplicates().tolist()
    for i, target in enumerate(targets):
        grp = rows[rows["target"] == target]
        lo = float(grp["row_index"].min()) - 0.5
        hi = float(grp["row_index"].max()) + 0.5
        if i % 2 == 0:
            ax.axhspan(lo, hi, color="0.96", zorder=0)
        if i > 0:
            ax.axhline(lo, color="0.82", linewidth=0.8, zorder=0)

def wave_suffix(waves: tuple[int, ...]) -> str:
    return "" if waves == (2, 4, 6, 8) else "_" + "_".join(f"w{w}" for w in waves)

def save_figure(fig: plt.Figure, base: Path, formats: Iterable[str], dpi: int) -> None:
    for fmt in formats:
        fig.savefig(base.with_suffix("." + fmt), dpi=dpi if fmt == "png" else None,
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[saved] {base}")

def add_group_labels(ax, rows: pd.DataFrame) -> None:
    """Rotated target name once per grouped block; W2/W4/etc as y ticks."""
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for target in rows["target"].drop_duplicates():
        group = rows[rows["target"] == target]
        midpoint = 0.5 * (group["row_index"].min() + group["row_index"].max())
        ax.text(-0.245, midpoint, TARGET_LABELS.get(target, target),
                rotation=90, transform=trans, va="center", ha="center",
                fontsize=11, fontweight="bold", clip_on=False)

def plot_metric_forest(
    df: pd.DataFrame, metric: str, figure_dir: Path,
    formats: Iterable[str], dpi: int, waves: tuple[int, ...],
) -> Path | None:
    metric_df = df[(df["canonical_metric"] == metric) & (df["wave"].isin(waves))].copy()
    excluded = PLOT_EXCLUDED_TARGETS_BY_METRIC.get(metric, set())
    metric_df = metric_df[~metric_df["target"].isin(excluded)].copy()
    if metric_df.empty:
        print(f"[skip] no {metric} data for waves {waves}")
        return None
    rows = build_rows(metric_df)
    metric_df = metric_df.merge(rows, on=["target", "wave"], how="inner")
    n_rows = len(rows)
    # Larger lettering while preserving readable spacing for each row.
    fig_height = max(5.5, 2.35 + 0.40 * n_rows)
    fig, ax = plt.subplots(figsize=(5.35, fig_height))
    add_target_bands_and_separators(ax, rows)
    ax.axvline(0, color="black", lw=1.1, zorder=1)
    plotted = []
    for contrast in CONTRAST_ORDER:
        d = metric_df[metric_df["contrast"] == contrast].dropna(
            subset=["estimate_std", "ci_low_std", "ci_high_std"])
        if d.empty:
            continue
        plotted.append(contrast)
        y = d["row_index"].to_numpy(float) + CONTRAST_Y_OFFSETS[contrast]
        x = d["estimate_std"].to_numpy(float)
        xerr = np.vstack([x - d["ci_low_std"].to_numpy(float),
                          d["ci_high_std"].to_numpy(float) - x])
        ax.errorbar(x, y, xerr=xerr, fmt=CONTRAST_MARKERS[contrast],
                    markersize=5.3, capsize=3.1, capthick=1.45,
                    elinewidth=1.8, linewidth=0,
                    label=CONTRAST_LABELS[contrast], zorder=3)
    ax.set_xlim(*symmetric_xlim(metric_df))
    ax.set_ylim(-0.65, n_rows - 0.35)
    ax.invert_yaxis()
    ax.set_yticks(rows["row_index"].to_numpy(float))
    ax.set_yticklabels(rows["row_label"].tolist(), fontsize=11.5,
                       fontweight="semibold")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlabel(METRIC_LABELS[metric] + "  (positive = Hybrid better)",
                  fontsize=12.5, fontweight="bold", labelpad=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    add_group_labels(ax, rows)
    wave_title = ", ".join(map(str, waves))
    fig.suptitle(f"{METRIC_TITLES[metric]}  |  Waves {wave_title}",
                 fontsize=15.5, fontweight="bold", y=0.99)
    ax.text(0.01, 1.017, "← Baseline better", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10.5, fontweight="bold")
    ax.text(0.99, 1.017, "Hybrid better →", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10.5, fontweight="bold")
    if plotted:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.961),
                   ncol=len(plotted), frameon=False, fontsize=11,
                   columnspacing=1.3, handletextpad=0.4)
    # Explicit margins reserve space for rotated target headings.
    fig.subplots_adjust(left=0.295, right=0.97, bottom=0.075,
                        top=0.88 if n_rows >= 24 else 0.85)
    base = figure_dir / f"rq2_forest_{metric.lower()}{wave_suffix(waves)}"
    save_figure(fig, base, formats, dpi)
    return base

def summarise_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, contrast), g in df.groupby(
        ["canonical_metric", "contrast"], dropna=False
    ):
        excluded_targets = PLOT_EXCLUDED_TARGETS_BY_METRIC.get(str(metric), set())
        if excluded_targets:
            g = g[~g["target"].astype(str).isin(excluded_targets)].copy()
        if g.empty:
            continue
        sig_pos = int(((g["ci_low_std"] > 0) & (g["ci_high_std"] > 0)).sum())
        sig_neg = int(((g["ci_low_std"] < 0) & (g["ci_high_std"] < 0)).sum())
        nonsig = int(len(g) - sig_pos - sig_neg)
        point_pos = int((g["estimate_std"] > 0).sum())
        point_neg = int((g["estimate_std"] < 0).sum())
        point_zero = int(np.isclose(g["estimate_std"], 0.0).sum())
        rows.append(
            {
                "metric": metric,
                "contrast": contrast,
                "cells": len(g),
                "point_estimate_hybrid_better": point_pos,
                "point_estimate_baseline_better": point_neg,
                "point_estimate_ties": point_zero,
                "ci_significantly_hybrid_better": sig_pos,
                "ci_crosses_zero": nonsig,
                "ci_significantly_baseline_better": sig_neg,
            }
        )
    return pd.DataFrame(rows)

def load_report_comparisons(path: Path) -> pd.DataFrame:
    """Read paired participant-bootstrap contrasts straight from Phase 2.

    The sign is already standardised in the inference output: positive means
    Hybrid better. This function does not recompute bootstrap estimates.
    """
    raw = pd.read_csv(path)
    required = {"wave", "target", "view", "metric", "contrast", "advantage", "ci_low", "ci_high"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing comparison columns: {missing}")
    aliases = {
        "f1": "F1", "logloss": "LogLoss", "crossentropy": "LogLoss",
        "nlpd": "NLPD", "negativelogpredictivedensity": "NLPD",
        "rmse": "RMSE", "rootmeansquareerror": "RMSE",
    }
    df = raw.copy()
    df["canonical_metric"] = df["metric"].map(lambda x: aliases.get(re.sub(r"[^a-z0-9]", "", str(x).lower())))
    df = df[df["canonical_metric"].notna() & df["contrast"].isin(CONTRAST_ORDER)].copy()
    valid_view = (
        ((df["view"] == "categorical_common_support") & df["canonical_metric"].isin(["F1", "LogLoss"])) |
        ((df["view"] == "native_continuous") & df["canonical_metric"].isin(["NLPD", "RMSE"]))
    )
    df = df[valid_view].copy()
    if "advantage_definition" in df.columns:
        definitions = df["advantage_definition"].dropna().str.lower()
        if not definitions.str.startswith("positive = hybrid better").all():
            raise ValueError("Inconsistent paired advantage sign convention")
    for old, new in [("advantage", "estimate_std"), ("ci_low", "ci_low_std"), ("ci_high", "ci_high_std")]:
        df[new] = pd.to_numeric(df[old], errors="coerce")
    df["wave"] = pd.to_numeric(df["wave"], errors="raise").astype(int)
    keys = ["wave", "target", "view", "canonical_metric", "contrast"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate paired-advantage comparison cells")
    if (df["ci_low_std"] > df["ci_high_std"]).any():
        raise ValueError("Confidence interval lower bound exceeds upper bound")
    if df.empty:
        raise ValueError("No report-relevant comparisons found")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/MasterPipeline/CV_HeldoutInference"))
    parser.add_argument("--comparisons-csv", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=350)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    comparisons = (args.comparisons_csv or root / "metrics" / "model_comparisons.csv").expanduser().resolve()
    output = (args.figure_dir or root / "plots" / "report").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    df = load_report_comparisons(comparisons)
    for metric in ("LogLoss", "F1", "NLPD", "RMSE"):
        if plot_metric_forest(df, metric, output, ("pdf", "png"), args.dpi, (2, 4, 6, 8)) is None:
            raise RuntimeError(f"Missing report metric: {metric}")
    (output / "paired_advantage_figure_manifest.txt").write_text(
        "Figure 6: LogLoss\nFigure 7: F1 (Dementia excluded)\n"
        "Figure 8: NLPD\nFigure 9: RMSE\n"
        f"Source: {comparisons}\n", encoding="utf-8"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

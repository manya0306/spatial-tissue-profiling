"""Generate all quantitative plots and spatial prediction maps.

Reads only the saved outputs of ``run_experiments.py`` - it recomputes nothing
and fits nothing, so every figure shows the same out-of-fold numbers that are
in the metric tables.

Sample and gene choices for the spatial maps follow fixed rules stated in the
code and printed at runtime, so the examples shown are not selected for
appearance. A deliberately poor gene is always included.

Run from the project root, after run_experiments.py:

    python scripts\\make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR: Path = PROJECT_ROOT / "outputs" / "experiments"
FIGURE_DIR: Path = PROJECT_ROOT / "outputs" / "figures" / "experiments"

DPI = 200
N_MAP_GENES = 3          # best genes to map, by within-sample Pearson
N_SCATTER_GENES = 6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_experiment() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Load metrics, per-model predictions and ground truth."""
    required = [
        EXPERIMENT_DIR / "metrics_per_gene.csv",
        EXPERIMENT_DIR / "metrics_aggregate.csv",
        EXPERIMENT_DIR / "predictions" / "ground_truth.csv",
        EXPERIMENT_DIR / "run_config.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}\nRun: python scripts\\run_experiments.py"
            )

    per_gene = pd.read_csv(EXPERIMENT_DIR / "metrics_per_gene.csv")
    aggregate = pd.read_csv(EXPERIMENT_DIR / "metrics_aggregate.csv")
    truth = pd.read_csv(EXPERIMENT_DIR / "predictions" / "ground_truth.csv")

    with open(EXPERIMENT_DIR / "run_config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)

    predictions: Dict[str, pd.DataFrame] = {}
    for model in config["models"]:
        path = EXPERIMENT_DIR / "predictions" / f"{model}.csv"
        if path.exists():
            predictions[model] = pd.read_csv(path)

    return per_gene, aggregate, predictions, truth, config


def choose_best_model(aggregate: pd.DataFrame) -> str:
    """The model with the highest mean within-sample Pearson correlation.

    Chosen on the aggregate table, before looking at any individual gene.
    """
    ranked = aggregate.dropna(subset=["within_sample_pearson_mean"])
    if ranked.empty:
        return str(aggregate.iloc[0]["model"])
    return str(ranked.sort_values("within_sample_pearson_mean", ascending=False).iloc[0]["model"])


def condition_of(sample: str) -> str:
    """Map a sample identifier to its condition group."""
    if sample.endswith("_LS"):
        return "AD lesional"
    if sample.endswith("_NL"):
        return "AD non-lesional"
    return "Healthy"


def choose_map_samples(truth: pd.DataFrame) -> List[str]:
    """One sample per condition, each the median-sized section in its group.

    Picking by spot count is a rule fixed in advance and unrelated to how well
    any model performed, which is what stops this from being a display of
    favourable examples.
    """
    counts = truth.groupby("sample").size()
    chosen: List[str] = []

    for condition in ["AD lesional", "AD non-lesional", "Healthy"]:
        members = [s for s in counts.index if condition_of(s) == condition]
        if not members:
            continue
        ordered = sorted(members, key=lambda s: counts[s])
        chosen.append(ordered[len(ordered) // 2])

    return chosen


# ---------------------------------------------------------------------------
# Quantitative figures
# ---------------------------------------------------------------------------

def plot_model_comparison(aggregate: pd.DataFrame) -> Path:
    """Three-panel comparison of every model across the headline metrics."""
    frame = aggregate.copy().sort_values("MAE")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    panels = [
        ("MAE", "Mean absolute error (lower is better)", "#4C72B0"),
        ("R2_mean", "Mean R2 across genes (higher is better)", "#DD8452"),
        ("within_sample_pearson_mean", "Mean within-sample Pearson", "#55A868"),
    ]

    for axis, (column, title, colour) in zip(axes, panels):
        values = frame[column].to_numpy(dtype=float)
        positions = np.arange(len(frame))
        axis.barh(positions, values, color=colour)
        axis.set_yticks(positions)
        axis.set_yticklabels(frame["model"], fontsize=9)
        axis.set_title(title, fontsize=11)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.grid(axis="x", alpha=0.3)
        for position, value in zip(positions, values):
            if np.isfinite(value):
                axis.text(
                    value,
                    position,
                    f" {value:.3f}",
                    va="center",
                    fontsize=8,
                    ha="left" if value >= 0 else "right",
                )
        axis.invert_yaxis()

    figure.suptitle(
        "Model comparison - out-of-fold, sample-grouped cross-validation",
        fontsize=13,
    )
    figure.tight_layout()

    path = FIGURE_DIR / "model_comparison.png"
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_per_gene_metrics(per_gene: pd.DataFrame, model: str) -> List[Path]:
    """Per-gene MAE and R2 for every gene in the panel, none omitted."""
    frame = per_gene[per_gene["model"] == model].copy()
    paths: List[Path] = []

    for column, title, ascending in [
        ("MAE", "Per-gene MAE", True),
        ("R2", "Per-gene R2", False),
    ]:
        ordered = frame.sort_values(column, ascending=ascending)
        values = ordered[column].to_numpy(dtype=float)
        colours = ["#55A868" if v > 0 else "#C44E52" for v in values] if column == "R2" \
            else ["#4C72B0"] * len(values)

        figure, axis = plt.subplots(figsize=(14, 6))
        axis.bar(np.arange(len(ordered)), values, color=colours)
        axis.set_xticks(np.arange(len(ordered)))
        axis.set_xticklabels(ordered["gene"], rotation=90, fontsize=7)
        axis.set_ylabel(column)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.3)
        axis.set_title(
            f"{title} - {model} (all {len(ordered)} genes shown)", fontsize=12
        )
        figure.tight_layout()

        path = FIGURE_DIR / f"per_gene_{column.lower()}.png"
        figure.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    return paths


def plot_metric_heatmap(per_gene: pd.DataFrame) -> Path:
    """Gene x model heatmap of within-sample Pearson correlation."""
    pivot = per_gene.pivot_table(
        index="gene", columns="model", values="within_sample_pearson"
    )
    best_column = pivot.mean(axis=0).idxmax()
    pivot = pivot.sort_values(best_column, ascending=False)

    figure, axis = plt.subplots(figsize=(9, max(10, 0.22 * len(pivot))))
    limit = float(np.nanmax(np.abs(pivot.to_numpy()))) or 1.0

    image = axis.imshow(
        pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    axis.set_xticks(np.arange(pivot.shape[1]))
    axis.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    axis.set_yticks(np.arange(pivot.shape[0]))
    axis.set_yticklabels(pivot.index, fontsize=7)
    axis.set_title("Within-sample Pearson correlation by gene and model", fontsize=12)
    figure.colorbar(image, ax=axis, label="Pearson r", shrink=0.6)
    figure.tight_layout()

    path = FIGURE_DIR / "gene_model_heatmap.png"
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_scatter(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    per_gene: pd.DataFrame,
    model: str,
) -> Path:
    """Predicted versus measured expression for the strongest genes."""
    ranked = (
        per_gene[per_gene["model"] == model]
        .sort_values("within_sample_pearson", ascending=False)
        .head(N_SCATTER_GENES)
    )

    n = len(ranked)
    columns = 3
    rows = int(np.ceil(n / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, (_, row) in zip(axes, ranked.iterrows()):
        gene = row["gene"]
        measured = truth[gene].to_numpy(dtype=float)
        predicted = prediction[gene].to_numpy(dtype=float)
        valid = np.isfinite(measured) & np.isfinite(predicted)

        axis.scatter(measured[valid], predicted[valid], s=3, alpha=0.25, color="#4C72B0")
        low = float(min(measured[valid].min(), predicted[valid].min()))
        high = float(max(measured[valid].max(), predicted[valid].max()))
        axis.plot([low, high], [low, high], "--", color="grey", linewidth=1)

        axis.set_title(
            f"{gene}\nr={row['pearson']:.3f}  R2={row['R2']:.3f}", fontsize=10
        )
        axis.set_xlabel("Measured (log-normalised)", fontsize=8)
        axis.set_ylabel("Predicted", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.3)

    for axis in axes[n:]:
        axis.axis("off")

    figure.suptitle(f"Predicted vs measured - {model}, out-of-fold", fontsize=13)
    figure.tight_layout()

    path = FIGURE_DIR / "predicted_vs_measured.png"
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Spatial maps
# ---------------------------------------------------------------------------

def plot_spatial_map(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    sample: str,
    gene: str,
    model: str,
    metrics: Optional[pd.Series],
) -> Path:
    """Measured, predicted and absolute-error maps for one sample and gene.

    Measured and predicted share a single colour scale so the two panels are
    directly comparable; the error panel has its own scale.
    """
    mask = truth["sample"] == sample
    x = truth.loc[mask, "x_coordinate"].to_numpy(dtype=float)
    y = truth.loc[mask, "y_coordinate"].to_numpy(dtype=float)
    measured = truth.loc[mask, gene].to_numpy(dtype=float)
    predicted = prediction.loc[mask.to_numpy(), gene].to_numpy(dtype=float)
    error = np.abs(predicted - measured)

    shared_low = float(np.nanmin([measured.min(), predicted.min()]))
    shared_high = float(np.nanmax([measured.max(), predicted.max()]))

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    panels = [
        (measured, "Measured expression", "viridis", shared_low, shared_high),
        (predicted, "Predicted expression", "viridis", shared_low, shared_high),
        (error, "Absolute error", "magma", float(error.min()), float(error.max())),
    ]

    for axis, (values, title, cmap, low, high) in zip(axes, panels):
        scatter = axis.scatter(
            x, y, c=values, s=14, cmap=cmap, vmin=low, vmax=high, edgecolors="none"
        )
        axis.invert_yaxis()
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("X (pixels)", fontsize=8)
        axis.set_ylabel("Y (pixels)", fontsize=8)
        axis.tick_params(labelsize=7)
        figure.colorbar(scatter, ax=axis, shrink=0.75)

    subtitle = f"{sample} ({condition_of(sample)}) - {gene} - model: {model}"
    if metrics is not None:
        subtitle += (
            f"\nMAE={metrics['MAE']:.3f}  R2={metrics['R2']:.3f}  "
            f"within-sample r={metrics['within_sample_pearson']:.3f} "
            f"(metrics are across all samples, not this one alone)"
        )

    figure.suptitle(subtitle, fontsize=12)
    figure.tight_layout()

    path = FIGURE_DIR / "spatial_maps" / f"{sample}_{gene}_{model}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\nGENERATING FIGURES")
    print("=" * 78)

    try:
        per_gene, aggregate, predictions, truth, config = load_experiment()
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    best_model = choose_best_model(aggregate)
    print(f"Best model (mean within-sample Pearson): {best_model}")

    if best_model not in predictions:
        print(f"ERROR: no saved predictions for {best_model}")
        return 1

    best_predictions = predictions[best_model]
    written: List[Path] = []

    written.append(plot_model_comparison(aggregate))
    written.extend(plot_per_gene_metrics(per_gene, best_model))
    written.append(plot_metric_heatmap(per_gene))
    written.append(plot_scatter(truth, best_predictions, per_gene, best_model))

    # ------------------------------------------------------- spatial maps ---

    map_samples = choose_map_samples(truth)
    print(f"\nSpatial map samples (median section size per condition): {map_samples}")

    ranked = per_gene[per_gene["model"] == best_model].sort_values(
        "within_sample_pearson", ascending=False
    )
    best_genes = ranked.head(N_MAP_GENES)["gene"].tolist()
    worst_gene = ranked.tail(1)["gene"].iloc[0]

    map_genes = best_genes + [worst_gene]
    print(f"Genes mapped: {best_genes} (strongest) + {worst_gene} (weakest, shown for honesty)")

    metric_lookup = ranked.set_index("gene")

    for sample in map_samples:
        for gene in map_genes:
            path = plot_spatial_map(
                truth,
                best_predictions,
                sample,
                gene,
                best_model,
                metric_lookup.loc[gene] if gene in metric_lookup.index else None,
            )
            written.append(path)

    # ------------------------------------------------------------ summary ---

    manifest = {
        "best_model": best_model,
        "map_samples": map_samples,
        "map_sample_rule": "median spot count within each condition group",
        "map_genes_best": best_genes,
        "map_gene_worst": worst_gene,
        "map_gene_rule": "top-3 and bottom-1 by within-sample Pearson of the best model",
        "figures": [str(p.relative_to(PROJECT_ROOT)) for p in written],
    }
    with open(FIGURE_DIR / "figure_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nWrote {len(written)} figures to {FIGURE_DIR.relative_to(PROJECT_ROOT)}")
    for path in written:
        print(f"  {path.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

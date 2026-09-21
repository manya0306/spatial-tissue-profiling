
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs/AD_1_LS")

model_files = {
    "Image only": OUTPUT_DIR / "spatial_block_cv_results.csv",
    "Image + coordinates": OUTPUT_DIR / "spatial_features_cv_results.csv",
    "Image + neighbors": OUTPUT_DIR / "spatial_neighborhood_cv_results.csv",
}

all_results = []

for model_name, file_path in model_files.items():
    df = pd.read_csv(file_path)
    df["model"] = model_name
    all_results.append(df)

combined = pd.concat(all_results, ignore_index=True)

# Average metrics across spatial folds
summary = (
    combined
    .groupby(["model", "gene"])[["MAE", "R2"]]
    .mean()
    .reset_index()
)

# Overall average across all target genes
overall_summary = (
    summary
    .groupby("model")[["MAE", "R2"]]
    .mean()
    .reset_index()
)

# Compare against image-only baseline
baseline = (
    summary[summary["model"] == "Image only"]
    [["gene", "MAE", "R2"]]
    .rename(columns={
        "MAE": "baseline_MAE",
        "R2": "baseline_R2"
    })
)

comparison = summary.merge(baseline, on="gene")

comparison["MAE_change_vs_baseline"] = (
    comparison["MAE"] - comparison["baseline_MAE"]
)

comparison["R2_change_vs_baseline"] = (
    comparison["R2"] - comparison["baseline_R2"]
)

# Save results
summary.to_csv(
    OUTPUT_DIR / "model_comparison_by_gene.csv",
    index=False
)

overall_summary.to_csv(
    OUTPUT_DIR / "model_comparison_overall.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_DIR / "model_comparison_vs_baseline.csv",
    index=False
)

print("\nMODEL COMPARISON BY GENE")
print("=" * 70)
print(summary.to_string(index=False))

print("\nOVERALL MODEL COMPARISON")
print("=" * 70)
print(overall_summary.to_string(index=False))

# Plot mean R2
r2_pivot = summary.pivot(
    index="gene",
    columns="model",
    values="R2"
)

ax = r2_pivot.plot(
    kind="bar",
    figsize=(14, 6)
)

ax.axhline(0, linewidth=0.8)
ax.set_title("Mean Spatial Cross-Validation R²")
ax.set_xlabel("Gene")
ax.set_ylabel("Mean R²")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison_mean_r2.png",
    dpi=300
)

plt.close()

# Plot mean MAE
mae_pivot = summary.pivot(
    index="gene",
    columns="model",
    values="MAE"
)

ax = mae_pivot.plot(
    kind="bar",
    figsize=(14, 6)
)

ax.set_title("Mean Spatial Cross-Validation MAE")
ax.set_xlabel("Gene")
ax.set_ylabel("Mean MAE")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison_mean_mae.png",
    dpi=300
)

plt.close()

print("\nSaved comparison files and plots in the outputs folder.")

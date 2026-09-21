
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("outputs/AD_1_LS")

# Load model comparison results
summary_path = OUTPUT_DIR / "model_comparison_by_gene.csv"
summary = pd.read_csv(summary_path)

# Overall model metrics
overall = (
    summary
    .groupby("model")[["MAE", "R2"]]
    .mean()
    .reset_index()
)

# Identify image-only baseline
baseline = overall[
    overall["model"] == "Image only"
].iloc[0]

overall["MAE_reduction_vs_baseline"] = (
    (baseline["MAE"] - overall["MAE"])
    / baseline["MAE"] * 100
)

overall["R2_change_vs_baseline"] = (
    overall["R2"] - baseline["R2"]
)

# Round values for readability
overall_display = overall.copy()

for column in [
    "MAE",
    "R2",
    "MAE_reduction_vs_baseline",
    "R2_change_vs_baseline"
]:
    overall_display[column] = overall_display[column].round(4)

# Save polished summary
output_path = OUTPUT_DIR / "final_results_summary.csv"
overall_display.to_csv(output_path, index=False)

# Print results
print("\nFINAL RESULTS SUMMARY")
print("=" * 80)
print(overall_display.to_string(index=False))

print("\nINTERPRETATION")
print("=" * 80)

for _, row in overall_display.iterrows():
    print(
        f"{row['model']}: "
        f"MAE = {row['MAE']:.4f}, "
        f"R² = {row['R2']:.4f}, "
        f"MAE change = {row['MAE_reduction_vs_baseline']:.2f}%, "
        f"R² change = {row['R2_change_vs_baseline']:.4f}"
    )

print(f"\nSaved summary to: {output_path}")

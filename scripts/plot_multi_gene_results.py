
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs/AD_1_LS")

# Load results
results = pd.read_csv(
    OUTPUT_DIR / "multi_gene_holdout_results.csv"
)

print("\nLoaded results:")
print(results)

# -----------------------------
# Plot 1: MAE
# -----------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    results["gene"],
    results["MAE"]
)

plt.title("Spatial Holdout MAE Across Target Genes")
plt.xlabel("Gene")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=45)
plt.tight_layout()

mae_path = OUTPUT_DIR / "multi_gene_holdout_mae.png"

plt.savefig(
    mae_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# Plot 2: R²
# -----------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    results["gene"],
    results["R2"]
)

plt.axhline(
    y=0,
    linewidth=0.8
)

plt.title("Spatial Holdout R² Across Target Genes")
plt.xlabel("Gene")
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.tight_layout()

r2_path = OUTPUT_DIR / "multi_gene_holdout_r2.png"

plt.savefig(
    r2_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nSaved MAE plot: {mae_path}")
print(f"Saved R² plot: {r2_path}")

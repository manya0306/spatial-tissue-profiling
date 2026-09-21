from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATCH_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "AD_1_LS"
    / "patches"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "AD_1_LS"
OUTPUT_CSV = OUTPUT_DIR / "patch_quality.csv"
OUTPUT_PLOT = OUTPUT_DIR / "patch_quality_distribution.png"


results = []

for patch_path in PATCH_DIR.glob("*.png"):

    image = np.array(Image.open(patch_path).convert("RGB"))

    # Detect pixels that differ meaningfully from white background
    non_white = np.any(image < 235, axis=2)

    tissue_percentage = non_white.mean() * 100

    results.append({
        "barcode": patch_path.stem,
        "tissue_percentage": tissue_percentage,
        "patch_path": str(patch_path),
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "tissue_percentage",
    ascending=True,
)

results_df.to_csv(OUTPUT_CSV, index=False)


# Plot distribution
plt.figure(figsize=(10, 6))

plt.hist(
    results_df["tissue_percentage"],
    bins=20,
)

plt.xlabel("Non-white pixel percentage (%)")
plt.ylabel("Number of patches")
plt.title("AD_1_LS Patch Tissue Content Distribution")
plt.tight_layout()

plt.savefig(OUTPUT_PLOT, dpi=200)
plt.close()


print(f"Total patches analyzed: {len(results_df)}")
print(
    f"Minimum tissue percentage: "
    f"{results_df['tissue_percentage'].min():.2f}%"
)
print(
    f"Maximum tissue percentage: "
    f"{results_df['tissue_percentage'].max():.2f}%"
)
print(
    f"Mean tissue percentage: "
    f"{results_df['tissue_percentage'].mean():.2f}%"
)
print(f"Saved CSV: {OUTPUT_CSV}")
print(f"Saved plot: {OUTPUT_PLOT}")
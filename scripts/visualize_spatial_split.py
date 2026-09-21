
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "AD_1_LS"

coords = np.load(OUTPUT_DIR / "aligned_coords.npy")

x_coordinates = coords[:, 1]
threshold = np.percentile(x_coordinates, 80)

test_mask = x_coordinates >= threshold
train_mask = ~test_mask

plt.figure(figsize=(8, 6))

plt.scatter(
    coords[train_mask, 1],
    coords[train_mask, 0],
    label="Training spots",
    alpha=0.7,
)

plt.scatter(
    coords[test_mask, 1],
    coords[test_mask, 0],
    label="Held-out spots",
    alpha=0.9,
)

plt.axvline(
    threshold,
    linestyle="--",
    label="Holdout boundary",
)

plt.xlabel("Spatial X")
plt.ylabel("Spatial Y")
plt.title("Spatial Train/Test Split — AD_1_LS")
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()

output_path = OUTPUT_DIR / "spatial_train_test_split.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("Saved:", output_path)
print("Training spots:", train_mask.sum())
print("Testing spots:", test_mask.sum())

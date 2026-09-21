import tarfile
import io
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "GSM5907077_AD_1_LS.tar.gz"

SAMPLE_NAME = "AD_1_LS"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "AD_1_LS_spot_overlay.png"


# Read files from archive
with tarfile.open(DATA_PATH, "r:gz") as tar:
    image = Image.open(
        io.BytesIO(
            tar.extractfile(
                f"{SAMPLE_NAME}/spatial/tissue_hires_image.png"
            ).read()
        )
    ).convert("RGB")

    coordinates = pd.read_csv(
        tar.extractfile(
            f"{SAMPLE_NAME}/spatial/tissue_positions_list.csv"
        ),
        header=None,
    )

    scale_factors = json.loads(
        tar.extractfile(
            f"{SAMPLE_NAME}/spatial/scalefactors_json.json"
        ).read().decode()
    )


# Assign coordinate column names
coordinates.columns = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pixel_row",
    "pixel_col",
]

# Keep only spots located inside tissue
coordinates = coordinates[coordinates["in_tissue"] == 1].copy()

# Map full-resolution coordinates to hires image
scale = scale_factors["tissue_hires_scalef"]

coordinates["x"] = coordinates["pixel_col"] * scale
coordinates["y"] = coordinates["pixel_row"] * scale


# Plot overlay
plt.figure(figsize=(10, 10))

plt.imshow(image)
plt.scatter(
    coordinates["x"],
    coordinates["y"],
    s=5,
    alpha=0.6,
)

plt.axis("off")
plt.title(f"{SAMPLE_NAME}: Spatial Spot Overlay")

# Save result
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"Image size: {image.size}")
print(f"Tissue spots plotted: {len(coordinates)}")
print(f"Saved overlay to: {OUTPUT_PATH}")
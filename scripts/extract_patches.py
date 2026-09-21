import tarfile
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "GSM5907077_AD_1_LS.tar.gz"

SAMPLE_NAME = "AD_1_LS"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / SAMPLE_NAME / "patches"

PATCH_SIZE = 224


# Load files from archive
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


coordinates.columns = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pixel_row",
    "pixel_col",
]

coordinates = coordinates[coordinates["in_tissue"] == 1].copy()

scale = scale_factors["tissue_hires_scalef"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

half_size = PATCH_SIZE // 2
metadata = []

for _, spot in coordinates.iterrows():

    x = int(round(spot["pixel_col"] * scale))
    y = int(round(spot["pixel_row"] * scale))

    left = x - half_size
    upper = y - half_size
    right = x + half_size
    lower = y + half_size

    # White padding ensures every patch has identical dimensions
    patch = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE), "white")

    source_left = max(0, left)
    source_upper = max(0, upper)
    source_right = min(image.width, right)
    source_lower = min(image.height, lower)

    cropped = image.crop(
        (source_left, source_upper, source_right, source_lower)
    )

    paste_x = source_left - left
    paste_y = source_upper - upper

    patch.paste(cropped, (paste_x, paste_y))

    filename = f"{spot['barcode']}.png"
    patch.save(OUTPUT_DIR / filename)

    metadata.append({
        "barcode": spot["barcode"],
        "x": x,
        "y": y,
        "patch_path": str(OUTPUT_DIR / filename),
    })


metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(
    OUTPUT_DIR.parent / "patch_metadata.csv",
    index=False,
)

print(f"Image size: {image.size}")
print(f"Patches extracted: {len(metadata_df)}")
print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
print(f"Saved to: {OUTPUT_DIR}")
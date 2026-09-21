
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "normalized_immune_score"
    / "normalized_immune_scores.csv"
)

DATA_ROOT = PROJECT_ROOT / "data" / "extracted"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "image_features"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 96
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load pretrained ResNet18
# -----------------------------

print(f"Using device: {DEVICE}")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Remove the final classification layer.
model.fc = torch.nn.Identity()

model = model.to(DEVICE)
model.eval()

preprocess = weights.transforms()


# -----------------------------
# Load spot-level target data
# -----------------------------

scores = pd.read_csv(SCORE_FILE)

print(f"Total spots: {len(scores)}")
print(f"Total samples: {scores['sample'].nunique()}")


# -----------------------------
# Feature extraction function
# -----------------------------

def extract_sample_features(sample_df, sample_name):

    image_path = (
        DATA_ROOT
        / sample_name
        / "spatial"
        / "tissue_hires_image.png"
    )

    scale_path = (
        DATA_ROOT
        / sample_name
        / "spatial"
        / "scalefactors_json.json"
    )

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    if not scale_path.exists():
        raise FileNotFoundError(f"Missing scale file: {scale_path}")

    image = Image.open(image_path).convert("RGB")

    with open(scale_path, "r") as file:
        scale_factors = json.load(file)

    hires_scale = scale_factors["tissue_hires_scalef"]

    image_width, image_height = image.size

    print(f"\nProcessing: {sample_name}")
    print(f"Image size: {image.size}")
    print(f"Scale factor: {hires_scale}")
    print(f"Spots: {len(sample_df)}")

    patches = []
    valid_rows = []

    half_patch = PATCH_SIZE // 2

    for row_index, row in sample_df.iterrows():

        # Coordinates in the positions file are full-resolution pixels.
        x = int(round(row["x_coordinate"] * hires_scale))
        y = int(round(row["y_coordinate"] * hires_scale))

        left = x - half_patch
        upper = y - half_patch
        right = x + half_patch
        lower = y + half_patch

        # Skip spots whose crop is outside the image.
        if (
            left < 0
            or upper < 0
            or right > image_width
            or lower > image_height
        ):
            continue

        patch = image.crop((left, upper, right, lower))
        patch = preprocess(patch)

        patches.append(patch)
        valid_rows.append(row_index)

    if not patches:
        raise RuntimeError(f"No valid patches extracted for {sample_name}")

    patch_tensor = torch.stack(patches).to(DEVICE)

    features = []

    with torch.no_grad():

        for start in range(0, len(patch_tensor), BATCH_SIZE):

            batch = patch_tensor[start:start + BATCH_SIZE]

            batch_features = model(batch)

            features.append(batch_features.cpu().numpy())

    features = np.concatenate(features, axis=0)

    metadata = sample_df.loc[valid_rows].reset_index(drop=True)

    print(f"Valid patches: {len(metadata)}")
    print(f"Feature shape: {features.shape}")

    return metadata, features


# -----------------------------
# Process every sample
# -----------------------------

all_metadata = []
all_features = []

for sample_name in sorted(scores["sample"].unique()):

    sample_df = scores[
        scores["sample"] == sample_name
    ].reset_index(drop=True)

    metadata, features = extract_sample_features(
        sample_df,
        sample_name
    )

    all_metadata.append(metadata)
    all_features.append(features)


# -----------------------------
# Save combined dataset
# -----------------------------

metadata_df = pd.concat(
    all_metadata,
    ignore_index=True
)

feature_matrix = np.concatenate(
    all_features,
    axis=0
)

feature_path = OUTPUT_DIR / "resnet18_features.npy"
metadata_path = OUTPUT_DIR / "resnet18_metadata.csv"

np.save(feature_path, feature_matrix)
metadata_df.to_csv(metadata_path, index=False)

print("\nExtraction completed successfully.")
print(f"Final feature matrix: {feature_matrix.shape}")
print(f"Saved features: {feature_path}")
print(f"Saved metadata: {metadata_path}")

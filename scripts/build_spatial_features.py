
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "image_features"
    / "resnet18_features.npy"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "image_features"
    / "resnet18_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "spatial_features"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_NEIGHBORS = 6


# -----------------------------
# Load data
# -----------------------------

features = np.load(FEATURE_PATH)
metadata = pd.read_csv(METADATA_PATH)

assert len(features) == len(metadata)

print("Image features:", features.shape)
print("Metadata:", metadata.shape)


# -----------------------------
# Build neighborhood features
# -----------------------------

all_spatial_features = []
all_neighbor_scores = []
all_neighbor_distances = []

for sample_name in sorted(metadata["sample"].unique()):

    sample_indices = np.where(
        metadata["sample"].values == sample_name
    )[0]

    sample_metadata = metadata.iloc[sample_indices].reset_index(drop=True)
    sample_features = features[sample_indices]

    coordinates = sample_metadata[
        ["x_coordinate", "y_coordinate"]
    ].values

    # Use at most N_NEIGHBORS, accounting for small samples.
    k = min(N_NEIGHBORS + 1, len(coordinates))

    if k <= 1:
        print(f"Skipping {sample_name}: insufficient spots")
        continue

    nearest_neighbors = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean"
    )

    nearest_neighbors.fit(coordinates)

    distances, indices = nearest_neighbors.kneighbors(coordinates)

    # First neighbor is the spot itself.
    neighbor_indices = indices[:, 1:]
    neighbor_distances = distances[:, 1:]

    # Mean image feature vector from nearby spots.
    neighbor_features = sample_features[neighbor_indices].mean(axis=1)

    # Mean target score among neighbors.
    sample_scores = sample_metadata[
        "normalized_immune_score"
    ].values

    neighbor_scores = sample_scores[neighbor_indices].mean(axis=1)

    all_spatial_features.append(neighbor_features)
    all_neighbor_scores.append(neighbor_scores)
    all_neighbor_distances.append(
        neighbor_distances.mean(axis=1)
    )

    print(
        f"{sample_name}: "
        f"spots={len(sample_indices)}, "
        f"neighbors={k - 1}, "
        f"features={neighbor_features.shape}"
    )


# -----------------------------
# Combine and save
# -----------------------------

spatial_features = np.concatenate(
    all_spatial_features,
    axis=0
)

neighbor_scores = np.concatenate(
    all_neighbor_scores,
    axis=0
)

neighbor_distances = np.concatenate(
    all_neighbor_distances,
    axis=0
)

# Important:
# The processing order is sorted by sample, matching the loop above.
# Rebuild metadata in the same order to preserve row alignment.

ordered_metadata = pd.concat(
    [
        metadata[
            metadata["sample"] == sample_name
        ]
        for sample_name in sorted(metadata["sample"].unique())
    ],
    ignore_index=True
)

assert len(ordered_metadata) == len(spatial_features)

np.save(
    OUTPUT_DIR / "neighbor_image_features.npy",
    spatial_features
)

np.save(
    OUTPUT_DIR / "neighbor_mean_scores.npy",
    neighbor_scores
)

np.save(
    OUTPUT_DIR / "neighbor_mean_distances.npy",
    neighbor_distances
)

ordered_metadata.to_csv(
    OUTPUT_DIR / "spatial_metadata.csv",
    index=False
)

print("\nSpatial feature construction completed.")
print("Neighbor image features:", spatial_features.shape)
print("Neighbor scores:", neighbor_scores.shape)
print("Neighbor distances:", neighbor_distances.shape)

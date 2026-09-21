
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_FEATURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "image_features"
    / "resnet18_features.npy"
)

IMAGE_METADATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "image_features"
    / "resnet18_metadata.csv"
)

NEIGHBOR_FEATURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "spatial_features"
    / "neighbor_image_features.npy"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load data
# -----------------------------

image_features = np.load(IMAGE_FEATURE_PATH)
neighbor_features = np.load(NEIGHBOR_FEATURE_PATH)

metadata = pd.read_csv(IMAGE_METADATA_PATH)

assert len(image_features) == len(metadata)
assert len(neighbor_features) == len(metadata)

targets = metadata["normalized_immune_score"].to_numpy()
groups = metadata["sample"].to_numpy()

print("Image features:", image_features.shape)
print("Neighbor features:", neighbor_features.shape)
print("Targets:", targets.shape)
print("Samples:", len(np.unique(groups)))


# -----------------------------
# Define models
# -----------------------------

models = {
    "Image only": image_features,
    "Image + spatial context": np.concatenate(
        [image_features, neighbor_features],
        axis=1
    )
}


# -----------------------------
# Sample-aware cross-validation
# -----------------------------

n_groups = len(np.unique(groups))
n_splits = min(5, n_groups)

cv = GroupKFold(n_splits=n_splits)

results = []

for model_name, X in models.items():

    print(f"\nEvaluating: {model_name}")
    print("Input shape:", X.shape)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, targets, groups),
        start=1
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = targets[train_idx]
        y_test = targets[test_idx]

        # Scaling and model fitting occur inside each fold.
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0))
            ]
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        test_samples = np.unique(groups[test_idx])

        results.append(
            {
                "model": model_name,
                "fold": fold,
                "MAE": mae,
                "R2": r2,
                "test_spots": len(test_idx),
                "test_samples": ", ".join(test_samples)
            }
        )

        print(
            f"Fold {fold}: "
            f"MAE={mae:.4f}, "
            f"R2={r2:.4f}, "
            f"Test spots={len(test_idx)}"
        )


# -----------------------------
# Save fold-level results
# -----------------------------

results_df = pd.DataFrame(results)

fold_results_path = OUTPUT_DIR / "fold_results.csv"
results_df.to_csv(fold_results_path, index=False)


# -----------------------------
# Calculate summary statistics
# -----------------------------

summary_df = (
    results_df
    .groupby("model")
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        folds=("fold", "count")
    )
    .reset_index()
)

summary_path = OUTPUT_DIR / "summary_results.csv"
summary_df.to_csv(summary_path, index=False)


# -----------------------------
# Display results
# -----------------------------

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

print(summary_df.to_string(index=False))

print("\nSaved fold results:")
print(fold_results_path)

print("\nSaved summary results:")
print(summary_path)

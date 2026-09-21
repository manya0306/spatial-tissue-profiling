
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
# Paths
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = ROOT / "outputs" / "image_features" / "resnet18_features.npy"
NEIGHBOR_PATH = ROOT / "outputs" / "spatial_features" / "neighbor_image_features.npy"
METADATA_PATH = ROOT / "outputs" / "image_features" / "resnet18_metadata.csv"

OUTPUT_DIR = ROOT / "outputs" / "pca_model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Configuration
# -----------------------------

N_COMPONENTS = 64
N_SPLITS = 5
RIDGE_ALPHA = 10.0


# -----------------------------
# Load data
# -----------------------------

image_features = np.load(IMAGE_PATH)
neighbor_features = np.load(NEIGHBOR_PATH)
metadata = pd.read_csv(METADATA_PATH)

assert len(image_features) == len(neighbor_features)
assert len(metadata) == len(image_features)

y = metadata["normalized_immune_score"].to_numpy()
groups = metadata["sample"].to_numpy()

combined_features = np.concatenate(
    [image_features, neighbor_features],
    axis=1
)

models = {
    "Image only": image_features,
    "Image + spatial context": combined_features
}

cv = GroupKFold(n_splits=N_SPLITS)

results = []


# -----------------------------
# Cross-validation
# -----------------------------

for model_name, X in models.items():

    print(f"\nEvaluating: {model_name}")
    print("Original shape:", X.shape)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # PCA is fitted only on the training fold.
        n_components = min(
            N_COMPONENTS,
            X_train.shape[0] - 1,
            X_train.shape[1]
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(
                    n_components=n_components,
                    random_state=42
                )),
                ("ridge", Ridge(alpha=RIDGE_ALPHA))
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results.append(
            {
                "model": model_name,
                "fold": fold,
                "MAE": mae,
                "R2": r2,
                "test_spots": len(test_idx),
                "test_samples": ", ".join(
                    np.unique(groups[test_idx])
                )
            }
        )

        print(
            f"Fold {fold}: "
            f"MAE={mae:.4f}, "
            f"R2={r2:.4f}, "
            f"Components={n_components}"
        )


# -----------------------------
# Save results
# -----------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "fold_results.csv",
    index=False
)

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

summary_df.to_csv(
    OUTPUT_DIR / "summary_results.csv",
    index=False
)


# -----------------------------
# Display summary
# -----------------------------

print("\n" + "=" * 60)
print("PCA MODEL COMPARISON SUMMARY")
print("=" * 60)

print(summary_df.to_string(index=False))

print("\nResults saved to:")
print(OUTPUT_DIR)

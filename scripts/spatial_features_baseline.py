
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "AD_1_LS"

FEATURES_PATH = OUTPUT_DIR / "aligned_features.npy"
EXPRESSION_PATH = OUTPUT_DIR / "aligned_expression.npz"
COORDS_PATH = OUTPUT_DIR / "aligned_coords.npy"
GENES_PATH = OUTPUT_DIR / "gene_names.txt"


# ============================================================
# 2. Load data
# ============================================================

image_features = np.load(FEATURES_PATH)

expression = load_npz(
    EXPRESSION_PATH
).toarray()

coords = np.load(COORDS_PATH)

with open(GENES_PATH, encoding="utf-8") as file:
    gene_names = [line.strip() for line in file]


TARGET_GENES = [
    "KRT1",
    "KRT10",
    "KRT14",
    "LOR",
    "DSG1",
    "FLG",
    "S100A7",
    "S100A8",
    "S100A9",
    "CLDN1",
]


gene_to_index = {
    gene: i
    for i, gene in enumerate(gene_names)
}

target_indices = [
    gene_to_index[gene]
    for gene in TARGET_GENES
]


# Log-transformed target expression
y = np.log1p(
    expression[:, target_indices]
)


# ============================================================
# 3. Add spatial coordinates to image features
# ============================================================

# Normalize coordinates to approximately [0, 1].
# coords[:, 0] = spatial Y
# coords[:, 1] = spatial X

normalized_coords = (
    coords - coords.min(axis=0)
) / (
    coords.max(axis=0) - coords.min(axis=0) + 1e-8
)


# Combine image features and spatial coordinates
X = np.hstack([
    image_features,
    normalized_coords,
])


print("Image feature shape:", image_features.shape)
print("Coordinate shape:", normalized_coords.shape)
print("Combined feature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 4. Create the same 5 spatial blocks
# ============================================================

x_coordinates = coords[:, 1]

sorted_indices = np.argsort(x_coordinates)

spatial_blocks = np.array_split(
    sorted_indices,
    5,
)


# ============================================================
# 5. Spatial cross-validation
# ============================================================

all_results = []


for fold, test_indices in enumerate(
    spatial_blocks,
    start=1,
):

    train_indices = np.concatenate([
        block
        for i, block in enumerate(spatial_blocks)
        if i != fold - 1
    ])

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]


    model = ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_features=1.0,
    )


    model.fit(
        X_train,
        y_train,
    )


    predictions = model.predict(X_test)


    print(f"\nFold {fold}")
    print("=" * 55)
    print("Training spots:", len(train_indices))
    print("Testing spots:", len(test_indices))


    for i, gene in enumerate(TARGET_GENES):

        mae = mean_absolute_error(
            y_test[:, i],
            predictions[:, i],
        )

        r2 = r2_score(
            y_test[:, i],
            predictions[:, i],
        )


        all_results.append({
            "fold": fold,
            "gene": gene,
            "MAE": mae,
            "R2": r2,
        })


        print(
            f"{gene:8s} | "
            f"MAE: {mae:.4f} | "
            f"R²: {r2:.4f}"
        )


# ============================================================
# 6. Save results
# ============================================================

results_df = pd.DataFrame(all_results)


results_df.to_csv(
    OUTPUT_DIR / "spatial_features_cv_results.csv",
    index=False,
)


summary = (
    results_df
    .groupby("gene")[["MAE", "R2"]]
    .agg(["mean", "std"])
)


summary.to_csv(
    OUTPUT_DIR / "spatial_features_cv_summary.csv"
)


print("\nSummary Across Spatial Folds")
print("=" * 55)
print(summary)


print("\nResults saved:")
print(
    OUTPUT_DIR / "spatial_features_cv_results.csv"
)
print(
    OUTPUT_DIR / "spatial_features_cv_summary.csv"
)


from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "AD_1_LS"

X = np.load(OUTPUT_DIR / "aligned_features.npy")
expression = load_npz(
    OUTPUT_DIR / "aligned_expression.npz"
).toarray()
coords = np.load(OUTPUT_DIR / "aligned_coords.npy")

with open(
    OUTPUT_DIR / "gene_names.txt",
    encoding="utf-8",
) as file:
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
    gene: i for i, gene in enumerate(gene_names)
}

target_indices = [
    gene_to_index[gene]
    for gene in TARGET_GENES
]

y = np.log1p(expression[:, target_indices])


# Create spatial blocks using X-coordinate ordering
x_coordinates = coords[:, 1]
sorted_indices = np.argsort(x_coordinates)

spatial_blocks = np.array_split(sorted_indices, 5)


all_results = []

print("Total spots:", len(X))
print("Number of spatial blocks:", len(spatial_blocks))


for fold, test_indices in enumerate(spatial_blocks, start=1):

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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\nFold {fold}")
    print("-" * 55)
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


# Save fold-level results
results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_DIR / "spatial_block_cv_results.csv",
    index=False,
)


# Aggregate results
summary = (
    results_df
    .groupby("gene")[["MAE", "R2"]]
    .agg(["mean", "std"])
)

summary.to_csv(
    OUTPUT_DIR / "spatial_block_cv_summary.csv"
)

print("\nSummary Across Spatial Folds")
print("=" * 55)
print(summary)

print("\nResults saved.")

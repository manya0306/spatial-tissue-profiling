
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score

OUTPUT_DIR = Path("outputs/AD_1_LS")

# Load data
coords = np.load(OUTPUT_DIR / "aligned_coords.npy")
image_features = np.load(OUTPUT_DIR / "aligned_features.npy")

expression = load_npz(
    OUTPUT_DIR / "aligned_expression.npz"
).toarray()

gene_names = np.loadtxt(
    OUTPUT_DIR / "gene_names.txt",
    dtype=str
)

target_genes = [
    "KRT1", "KRT10", "KRT14", "LOR", "DSG1",
    "FLG", "S100A7", "S100A8", "S100A9", "CLDN1"
]

# Create neighborhood features
knn = NearestNeighbors(n_neighbors=6)
knn.fit(coords)

_, indices = knn.kneighbors(coords)

neighbor_features = np.zeros_like(image_features)

for i in range(len(coords)):
    neighbor_features[i] = np.mean(
        image_features[indices[i, 1:]],
        axis=0
    )

X = np.hstack([
    image_features,
    neighbor_features
])

# Spatial holdout
threshold = np.percentile(coords[:, 1], 80)

train_mask = coords[:, 1] <= threshold
test_mask = coords[:, 1] > threshold

X_train = X[train_mask]
X_test = X[test_mask]

results = []

for target_gene in target_genes:

    gene_index = np.where(
        gene_names == target_gene
    )[0][0]

    y = np.log1p(
        expression[:, gene_index].astype(float)
    )

    y_train = y[train_mask]
    y_test = y[test_mask]

    model = ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_features=1.0
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "gene": target_gene,
        "MAE": mae,
        "R2": r2
    })

    print(
        f"{target_gene:8s} | "
        f"MAE: {mae:.4f} | "
        f"R²: {r2:.4f}"
    )

results_df = pd.DataFrame(results)

output_path = OUTPUT_DIR / "multi_gene_holdout_results.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\nMULTI-GENE SPATIAL HOLDOUT RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))

print(f"\nSaved results to: {output_path}")

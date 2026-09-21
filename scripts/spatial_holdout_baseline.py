
from pathlib import Path

import numpy as np
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


# Spatial holdout:
# Hold out the rightmost 20% of spots by spatial X.
x_coordinates = coords[:, 1]
threshold = np.percentile(x_coordinates, 80)

test_mask = x_coordinates >= threshold
train_mask = ~test_mask

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]


print("Training spots:", len(X_train))
print("Testing spots:", len(X_test))
print("Spatial holdout threshold:", threshold)


# Train model
model = ExtraTreesRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features=1.0,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# Evaluate
print("\nSpatial Holdout Results")
print("=" * 55)

for i, gene in enumerate(TARGET_GENES):

    mae = mean_absolute_error(
        y_test[:, i],
        predictions[:, i],
    )

    r2 = r2_score(
        y_test[:, i],
        predictions[:, i],
    )

    print(
        f"{gene:8s} | "
        f"MAE: {mae:.4f} | "
        f"R²: {r2:.4f}"
    )

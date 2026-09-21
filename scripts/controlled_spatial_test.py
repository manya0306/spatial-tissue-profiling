
from pathlib import Path
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "outputs" / "AD_1_LS"

FEATURES_PATH = DATA_DIR / "aligned_features.npy"
EXPRESSION_PATH = DATA_DIR / "aligned_expression.npz"
COORDS_PATH = DATA_DIR / "aligned_coords.npy"
GENES_PATH = DATA_DIR / "gene_names.txt"

OUTPUT_PATH = DATA_DIR / "controlled_spatial_test_results.csv"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

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

N_NEIGHBORS = 5
N_FOLDS = 5
N_REPEATS = 5


# --------------------------------------------------
# Load data
# --------------------------------------------------

features = np.load(FEATURES_PATH)
expression_data = np.load(EXPRESSION_PATH)
coords = np.load(COORDS_PATH)
gene_names = np.loadtxt(GENES_PATH, dtype=str)

expression = csr_matrix(
    (
        expression_data["data"],
        expression_data["indices"],
        expression_data["indptr"],
    ),
    shape=expression_data["shape"],
).toarray()

expression = np.asarray(expression)

# Expression is stored as spots x genes
if expression.shape[0] != features.shape[0]:
    raise ValueError(
        f"Expression and feature spot counts do not match: "
        f"{expression.shape} vs {features.shape}"
    )

gene_to_index = {
    gene: index for index, gene in enumerate(gene_names)
}

missing_genes = [
    gene for gene in TARGET_GENES
    if gene not in gene_to_index
]

if missing_genes:
    raise ValueError(f"Missing target genes: {missing_genes}")

target_indices = [
    gene_to_index[gene] for gene in TARGET_GENES
]

y = np.log1p(expression[:, target_indices])

n_spots = features.shape[0]


# --------------------------------------------------
# Construct spatial neighbors
# --------------------------------------------------

nearest_neighbors = NearestNeighbors(
    n_neighbors=N_NEIGHBORS + 1
)

nearest_neighbors.fit(coords)

neighbor_indices = nearest_neighbors.kneighbors(
    return_distance=False
)

# Remove each spot itself
neighbor_indices = neighbor_indices[:, 1:]

neighbor_features = np.mean(
    features[neighbor_indices],
    axis=1
)

print(f"Number of spots: {n_spots}")
print(f"Image feature dimension: {features.shape[1]}")
print(f"Neighbor feature dimension: {neighbor_features.shape[1]}")


# --------------------------------------------------
# Spatial blocks
# --------------------------------------------------

# Use the same spatial ordering for every model
order = np.argsort(coords[:, 1])
folds = np.array_split(order, N_FOLDS)


# --------------------------------------------------
# Evaluate models
# --------------------------------------------------

results = []

for repeat in range(N_REPEATS):

    rng = np.random.default_rng(1000 + repeat)

    # Shuffle complete neighbor representations
    # while keeping their feature distribution unchanged
    shuffled_neighbor_features = neighbor_features[
        rng.permutation(n_spots)
    ]

    model_inputs = {
        "Image only": features,

        "Real neighbors": np.concatenate(
            [features, neighbor_features],
            axis=1,
        ),

        "Shuffled neighbors": np.concatenate(
            [features, shuffled_neighbor_features],
            axis=1,
        ),
    }

    for fold_number, test_indices in enumerate(folds, start=1):

        train_mask = np.ones(n_spots, dtype=bool)
        train_mask[test_indices] = False

        train_indices = np.where(train_mask)[0]

        for model_name, X in model_inputs.items():

            X_train = X[train_indices]
            X_test = X[test_indices]

            for gene_number, gene in enumerate(TARGET_GENES):

                model = ExtraTreesRegressor(
                    n_estimators=200,
                    max_features=1.0,
                    random_state=2000 + repeat,
                    n_jobs=-1,
                )

                model.fit(
                    X_train,
                    y[train_indices, gene_number],
                )

                predictions = model.predict(X_test)

                mae = mean_absolute_error(
                    y[test_indices, gene_number],
                    predictions,
                )

                r2 = r2_score(
                    y[test_indices, gene_number],
                    predictions,
                )

                results.append({
                    "repeat": repeat + 1,
                    "fold": fold_number,
                    "model": model_name,
                    "gene": gene,
                    "MAE": mae,
                    "R2": r2,
                })

                print(
                    f"Repeat {repeat + 1}/{N_REPEATS} | "
                    f"Fold {fold_number}/{N_FOLDS} | "
                    f"{model_name} | "
                    f"{gene} | "
                    f"MAE: {mae:.4f} | "
                    f"R2: {r2:.4f}"
                )


# --------------------------------------------------
# Save detailed results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

summary = (
    results_df
    .groupby("model")[["MAE", "R2"]]
    .mean()
    .reset_index()
)

print("\n")
print("=" * 70)
print("CONTROLLED SPATIAL TEST SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))

print("\n")
print(f"Saved results to: {OUTPUT_PATH}")

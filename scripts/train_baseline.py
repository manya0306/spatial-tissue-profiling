
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "AD_1_LS"

FEATURES_PATH = OUTPUT_DIR / "aligned_features.npy"
EXPRESSION_PATH = OUTPUT_DIR / "aligned_expression.npz"
GENES_PATH = OUTPUT_DIR / "gene_names.txt"


# Selected genes
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


# Load data
features = np.load(FEATURES_PATH)
expression = load_npz(EXPRESSION_PATH).toarray()
gene_names = [
    line.strip()
    for line in open(GENES_PATH, encoding="utf-8")
]

gene_to_index = {
    gene: i
    for i, gene in enumerate(gene_names)
}

target_indices = [
    gene_to_index[gene]
    for gene in TARGET_GENES
]

X = features
y = np.log1p(expression[:, target_indices])


print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# Train baseline model
model = ExtraTreesRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features=1.0,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# Evaluate each gene
print("\nBaseline Results")
print("=" * 50)

results = []

for i, gene in enumerate(TARGET_GENES):

    mae = mean_absolute_error(
        y_test[:, i],
        predictions[:, i],
    )

    r2 = r2_score(
        y_test[:, i],
        predictions[:, i],
    )

    results.append({
        "gene": gene,
        "MAE": mae,
        "R2": r2,
    })

    print(
        f"{gene:8s} | "
        f"MAE: {mae:.4f} | "
        f"R²: {r2:.4f}"
    )


# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "baseline_results.csv",
    index=False,
)

print("\nResults saved to:")
print(OUTPUT_DIR / "baseline_results.csv")

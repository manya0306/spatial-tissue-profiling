
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor

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

# Select target gene
target_gene = "KRT1"
gene_index = np.where(gene_names == target_gene)[0][0]

y = np.log1p(
    expression[:, gene_index].astype(float)
)

# Create spatial neighborhood features
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

# Spatial holdout: rightmost 20% of tissue
threshold = np.percentile(coords[:, 1], 80)

train_mask = coords[:, 1] <= threshold
test_mask = coords[:, 1] > threshold

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

# Train model
model = ExtraTreesRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features=1.0
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Calculate errors
errors = np.abs(y_test - predictions)

print(f"Target gene: {target_gene}")
print(f"Training spots: {len(y_train)}")
print(f"Testing spots: {len(y_test)}")
print(f"Mean absolute error: {np.mean(errors):.4f}")

# Plot actual, predicted, and error
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 7)
)

plot_data = [
    (y_test, "Actual Expression", "viridis"),
    (predictions, "Predicted Expression", "viridis"),
    (errors, "Absolute Error", "magma")
]

for ax, (values, title, cmap) in zip(axes, plot_data):
    scatter = ax.scatter(
        coords[test_mask, 1],
        coords[test_mask, 0],
        c=values,
        s=55,
        cmap=cmap,
        alpha=0.95
    )

    ax.set_title(f"{target_gene}: {title}")
    ax.set_xlabel("Spatial X Coordinate")
    ax.set_ylabel("Spatial Y Coordinate")
    ax.invert_yaxis()
    ax.set_aspect("equal")

    plt.colorbar(scatter, ax=ax)

plt.suptitle(
    f"Spatial Holdout Prediction Analysis — {target_gene}",
    fontsize=14
)

plt.tight_layout()

output_path = OUTPUT_DIR / (
    f"{target_gene}_spatial_holdout_predictions.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved visualization to: {output_path}")

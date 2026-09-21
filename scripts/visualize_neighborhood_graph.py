from pathlib import Path
from scipy.sparse import load_npz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

OUTPUT_DIR = Path("outputs/AD_1_LS")

# Load data
coords = np.load(OUTPUT_DIR / "aligned_coords.npy")
expression = load_npz(
    OUTPUT_DIR / "aligned_expression.npz"
).toarray()

gene_names = np.loadtxt(
    OUTPUT_DIR / "gene_names.txt",
    dtype=str
)

# Select gene to visualize
target_gene = "KRT1"
gene_index = np.where(gene_names == target_gene)[0][0]

# Log-transform expression
gene_expression = np.log1p(
    expression[:, gene_index].astype(float)
)

# Find 5 nearest neighbors
knn = NearestNeighbors(n_neighbors=6)
knn.fit(coords)

_, indices = knn.kneighbors(coords)

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Draw spatial connections
for i in range(len(coords)):
    for j in indices[i, 1:]:
        ax.plot(
            [coords[i, 1], coords[j, 1]],
            [coords[i, 0], coords[j, 0]],
            linewidth=0.35,
            alpha=0.2
        )

# Plot expression intensity
scatter = ax.scatter(
    coords[:, 1],
    coords[:, 0],
    c=gene_expression,
    s=35,
    alpha=0.95
)

colorbar = plt.colorbar(scatter, ax=ax)
colorbar.set_label("log1p Gene Expression")

ax.set_title(
    f"Spatial Neighborhood Graph with {target_gene} Expression"
)

ax.set_xlabel("Spatial X Coordinate")
ax.set_ylabel("Spatial Y Coordinate")
ax.invert_yaxis()
ax.set_aspect("equal")

plt.tight_layout()

output_path = OUTPUT_DIR / (
    f"neighborhood_{target_gene}_expression.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Gene: {target_gene}")
print(f"Spots: {len(coords)}")
print(f"Saved figure to: {output_path}")

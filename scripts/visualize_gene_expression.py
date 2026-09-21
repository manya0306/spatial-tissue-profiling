from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "AD_1_LS"

EXPRESSION_PATH = OUTPUT_DIR / "aligned_expression.npz"
COORDS_PATH = OUTPUT_DIR / "aligned_coords.npy"
GENE_FILE = OUTPUT_DIR / "gene_names.txt"


selected_genes = [
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


expression = load_npz(EXPRESSION_PATH).toarray()
coords = np.load(COORDS_PATH)

gene_names = pd.read_csv(
    GENE_FILE,
    header=None
)[0].astype(str).tolist()

gene_to_index = {
    gene.upper(): i
    for i, gene in enumerate(gene_names)
}


fig, axes = plt.subplots(
    2, 5,
    figsize=(20, 8)
)

axes = axes.flatten()

for ax, gene in zip(axes, selected_genes):

    gene_index = gene_to_index[gene]
    values = expression[:, gene_index]

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.log1p(values),
        cmap="viridis",
        s=35,
        edgecolors="none"
    )

    ax.set_title(gene)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    ax.invert_yaxis()

    plt.colorbar(scatter, ax=ax, label="log1p expression")


plt.suptitle(
    "Spatial Expression Patterns — AD_1_LS",
    fontsize=16
)

plt.tight_layout()

output_path = OUTPUT_DIR / "selected_gene_spatial_maps.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved visualization to: {output_path}")
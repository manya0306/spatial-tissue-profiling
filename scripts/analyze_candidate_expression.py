from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPRESSION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "AD_1_LS"
    / "aligned_expression.npz"
)

GENE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "AD_1_LS"
    / "gene_names.txt"
)


candidate_genes = [
    "FLG", "LOR", "IVL", "KRT1", "KRT10", "KRT14",
    "CLDN1", "DSG1", "IL4", "IL13", "IL17A", "IL22",
    "TNF", "IL1B", "IL6", "CCL17", "CCL22", "CXCL10",
    "IFNG", "DEFB4A", "S100A7", "S100A8", "S100A9",
    "S100A10",
]


expression = load_npz(EXPRESSION_PATH).toarray()
gene_names = pd.read_csv(GENE_FILE, header=None)[0].astype(str).tolist()

gene_to_index = {
    gene.upper(): i
    for i, gene in enumerate(gene_names)
}


results = []

for gene in candidate_genes:

    index = gene_to_index[gene]

    values = expression[:, index]

    results.append({
        "gene": gene,
        "nonzero_spots": int(np.count_nonzero(values)),
        "mean_expression": float(values.mean()),
        "max_expression": float(values.max()),
        "variance": float(values.var()),
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "nonzero_spots",
    ascending=False,
)

print(results_df.to_string(index=False))

results_df.to_csv(
    PROJECT_ROOT
    / "outputs"
    / "AD_1_LS"
    / "candidate_gene_expression_stats.csv",
    index=False,
)
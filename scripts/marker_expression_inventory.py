
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from scipy.io import mmread


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "extracted"
OUTPUT_DIR = BASE_DIR / "outputs"

TARGET_GENES = [
    "PTPRC", "CD3D", "CD3E", "CD4", "CD8A",
    "LST1", "TYROBP",
    "FLG", "LOR", "IVL", "CLDN1",
    "KRT1", "KRT10", "KRT14",
    "S100A7", "S100A8", "S100A9"
]


def read_genes(path):
    genes = []

    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            genes.append(parts[1])

    return genes


def inspect_sample(sample_dir):
    matrix_path = sample_dir / "matrix.mtx.gz"
    features_path = sample_dir / "features.tsv.gz"

    if not matrix_path.exists() or not features_path.exists():
        return []

    genes = read_genes(features_path)
    gene_to_index = {gene: i for i, gene in enumerate(genes)}

    selected = {
        gene: gene_to_index[gene]
        for gene in TARGET_GENES
        if gene in gene_to_index
    }

    with gzip.open(matrix_path, "rb") as f:
        matrix = mmread(f).tocsr()

    results = []

    for gene, index in selected.items():
        values = np.asarray(matrix[index, :].todense()).ravel()

        results.append({
            "sample": sample_dir.name,
            "gene": gene,
            "spots": len(values),
            "nonzero_spots": int(np.count_nonzero(values)),
            "nonzero_fraction": float(np.mean(values > 0)),
            "mean_expression": float(values.mean()),
            "total_expression": float(values.sum()),
        })

    return results


def main():
    all_results = []

    for sample_dir in sorted(DATA_DIR.iterdir()):
        if sample_dir.is_dir():
            all_results.extend(inspect_sample(sample_dir))

    if not all_results:
        print("No results found.")
        return

    df = pd.DataFrame(all_results)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "marker_expression_inventory.csv"
    df.to_csv(output_path, index=False)

    print("\nMARKER EXPRESSION INVENTORY")
    print("=" * 100)
    print(df.to_string(index=False))

    print("\nAVERAGE EXPRESSION FREQUENCY BY GENE")
    print("=" * 100)

    summary = (
        df.groupby("gene")["nonzero_fraction"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )

    print(summary.to_string())

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()

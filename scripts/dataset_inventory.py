
from pathlib import Path
import gzip
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "extracted"


CANDIDATE_GENES = [
    "PTPRC",
    "CD3D",
    "CD3E",
    "CD4",
    "CD8A",
    "LST1",
    "TYROBP",
    "FLG",
    "LOR",
    "IVL",
    "CLDN1",
    "KRT1",
    "KRT10",
    "KRT14",
    "S100A7",
    "S100A8",
    "S100A9",
]


def read_genes(features_path):
    genes = []

    with gzip.open(features_path, "rt") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                genes.append(parts[1])

    return genes


def count_barcodes(barcodes_path):
    with gzip.open(barcodes_path, "rt") as file:
        return sum(1 for _ in file)


def count_tissue_spots(positions_path):
    df = pd.read_csv(
        positions_path,
        header=None
    )

    # Standard Visium positions format:
    # barcode, in_tissue, array_row, array_col, pixel_row, pixel_col

    return int((df.iloc[:, 1] == 1).sum())


def inspect_sample(sample_dir):
    features_path = sample_dir / "features.tsv.gz"
    barcodes_path = sample_dir / "barcodes.tsv.gz"
    positions_path = sample_dir / "spatial" / "tissue_positions_list.csv"

    if not features_path.exists():
        return None

    genes = read_genes(features_path)
    barcode_count = count_barcodes(barcodes_path)

    if positions_path.exists():
        tissue_spots = count_tissue_spots(positions_path)
    else:
        tissue_spots = "Unavailable"

    gene_set = set(genes)

    result = {
        "sample": sample_dir.name,
        "genes": len(genes),
        "barcodes": barcode_count,
        "tissue_spots": tissue_spots,
    }

    for gene in CANDIDATE_GENES:
        result[gene] = gene in gene_set

    return result


def main():
    results = []

    for sample_dir in sorted(DATA_DIR.iterdir()):
        if sample_dir.is_dir():
            result = inspect_sample(sample_dir)

            if result is not None:
                results.append(result)

    if not results:
        print("No extracted samples found.")
        return

    df = pd.DataFrame(results)

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "dataset_inventory.csv"
    df.to_csv(output_path, index=False)

    print("\nDATASET INVENTORY")
    print("=" * 80)
    print(df.iloc[:, :4].to_string(index=False))

    print("\nCANDIDATE GENE AVAILABILITY")
    print("=" * 80)
    print(df.iloc[:, 4:].sum().sort_values(ascending=False).to_string())

    print(f"\nSaved inventory to: {output_path}")


if __name__ == "__main__":
    main()

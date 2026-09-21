import gzip
import tarfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "GSM5907077_AD_1_LS.tar.gz"
)

SAMPLE_NAME = "AD_1_LS"


with tarfile.open(DATA_PATH, "r:gz") as tar:

    gene_file = tar.extractfile(
        f"{SAMPLE_NAME}/features.tsv.gz"
    )

    with gzip.GzipFile(fileobj=gene_file) as decompressed:
        genes = pd.read_csv(
            decompressed,
            header=None,
            sep="\t",
        )


print("Number of genes:", len(genes))
print("\nFirst 20 genes:")

print(genes.head(20).to_string(index=False, header=False))


# Save gene names for easier inspection
gene_names = genes.iloc[:, 1].astype(str)

output_path = (
    PROJECT_ROOT
    / "outputs"
    / "AD_1_LS"
    / "gene_names.txt"
)

gene_names.to_csv(
    output_path,
    index=False,
    header=False,
)

print(f"\nSaved gene names to: {output_path}")
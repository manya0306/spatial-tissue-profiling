from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "AD_1_LS"
    / "gene_names.txt"
)


genes = set(
    pd.read_csv(
        GENE_FILE,
        header=None,
    )[0].astype(str).str.upper()
)


candidate_genes = [
    # Skin barrier
    "FLG",
    "LOR",
    "IVL",
    "KRT1",
    "KRT10",
    "KRT14",
    "CLDN1",
    "DSG1",

    # Inflammation / immune activity
    "IL4",
    "IL13",
    "IL17A",
    "IL22",
    "TNF",
    "IL1B",
    "IL6",
    "CCL17",
    "CCL22",
    "CXCL10",
    "IFNG",

    # Antimicrobial response
    "DEFB4A",
    "S100A7",
    "S100A8",
    "S100A9",
    "S100A10",
]


present = [
    gene for gene in candidate_genes
    if gene in genes
]

missing = [
    gene for gene in candidate_genes
    if gene not in genes
]


print("Candidate genes present:")
print(present)

print("\nCandidate genes missing:")
print(missing)

print(f"\nPresent: {len(present)} / {len(candidate_genes)}")
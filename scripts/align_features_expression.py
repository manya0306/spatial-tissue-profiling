import tarfile
import gzip
from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
from scipy.io import mmread


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "GSM5907077_AD_1_LS.tar.gz"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "AD_1_LS"
    / "resnet18_features.npy"
)

FEATURE_BARCODES_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "AD_1_LS"
    / "feature_barcodes.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "AD_1_LS"


SAMPLE_NAME = "AD_1_LS"


# Load extracted image features
features = np.load(FEATURE_PATH)

feature_barcodes = pd.read_csv(
    FEATURE_BARCODES_PATH
)["barcode"].astype(str).tolist()


# Read gene-expression files from archive
with tarfile.open(DATA_PATH, "r:gz") as tar:

    matrix_file = tar.extractfile(
        f"{SAMPLE_NAME}/matrix.mtx.gz"
    )

    with gzip.GzipFile(fileobj=matrix_file) as decompressed:
        matrix = mmread(decompressed).tocsr()

    barcode_file = tar.extractfile(
        f"{SAMPLE_NAME}/barcodes.tsv.gz"
    )

    with gzip.GzipFile(fileobj=barcode_file) as decompressed:
        expression_barcodes = [
            line.decode("utf-8").strip()
            for line in decompressed
        ]

    gene_file = tar.extractfile(
        f"{SAMPLE_NAME}/features.tsv.gz"
    )

    with gzip.GzipFile(fileobj=gene_file) as decompressed:
        genes = pd.read_csv(
            decompressed,
            header=None,
            sep="\t",
        )


# Matrix is generally genes × barcodes
print("Original expression matrix:", matrix.shape)
print("Expression barcodes:", len(expression_barcodes))
print("Image features:", features.shape)
print("Image barcodes:", len(feature_barcodes))


# Create barcode-to-expression-column mapping
expression_index = {
    barcode: i
    for i, barcode in enumerate(expression_barcodes)
}


# Align expression columns to feature barcode order
missing_barcodes = [
    barcode
    for barcode in feature_barcodes
    if barcode not in expression_index
]

if missing_barcodes:
    raise ValueError(
        f"Missing {len(missing_barcodes)} feature barcodes "
        "in expression data."
    )


expression_indices = [
    expression_index[barcode]
    for barcode in feature_barcodes
]


aligned_expression = matrix[:, expression_indices].T.tocsr()

aligned_features = features
# Read spatial coordinates from archive
with tarfile.open(DATA_PATH, "r:gz") as tar:

    positions_file = tar.extractfile(
        f"{SAMPLE_NAME}/spatial/tissue_positions_list.csv"
    )

    positions = pd.read_csv(
        positions_file,
        header=None,
    )

# Assign column names
positions.columns = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pixel_row",
    "pixel_col",
]

# Create barcode-to-coordinate mapping
coordinate_index = positions.set_index("barcode")

# Align coordinates to feature barcode order
aligned_coords = np.array([
    [
        coordinate_index.loc[barcode, "pixel_row"],
        coordinate_index.loc[barcode, "pixel_col"],
    ]
    for barcode in feature_barcodes
])

# Save aligned data
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from scipy.sparse import save_npz

save_npz(
    OUTPUT_DIR / "aligned_expression.npz",
    aligned_expression,
)

np.save(
    OUTPUT_DIR / "aligned_features.npy",
    aligned_features,
)

np.save(
    OUTPUT_DIR / "aligned_coords.npy",
    aligned_coords,
)

pd.DataFrame({
    "barcode": feature_barcodes
}).to_csv(
    OUTPUT_DIR / "aligned_barcodes.csv",
    index=False,
)


# Validation
print("\nAligned expression shape:", aligned_expression.shape)
print("Aligned features shape:", aligned_features.shape)
print(
    "Number of aligned spots:",
    len(feature_barcodes),
)
print(
    "Barcode alignment verified:",
    aligned_expression.shape[0] == aligned_features.shape[0],
)
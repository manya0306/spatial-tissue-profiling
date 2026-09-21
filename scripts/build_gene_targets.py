"""Build a per-spot, multi-gene expression target matrix for all samples.

This is the component the project was missing. The existing pipeline collapsed
expression into a single averaged immune score; this script instead produces a
wide matrix of individual genes, normalised per spot, and aligned row-for-row
to ``outputs/image_features/resnet18_metadata.csv`` by an explicit
``(sample, barcode)`` join rather than by assumed row order.

It also computes objective per-gene statistics - detection rate, variance,
cross-sample consistency and spatial autocorrelation (Moran's I) - which are
used by ``select_target_genes.py`` to choose modelling targets on stated
criteria rather than on downstream performance.

Normalisation matches the existing pipeline exactly: counts per spot are scaled
to a fixed library size, then log1p transformed. No target information is
written into any feature file.

Run from the project root:

    python scripts\\build_gene_targets.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data" / "extracted"
METADATA_PATH: Path = (
    PROJECT_ROOT / "outputs" / "image_features" / "resnet18_metadata.csv"
)
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "gene_targets"

TARGET_SUM: float = 10_000.0
N_SPATIAL_NEIGHBORS: int = 6

# A deliberately wide panel spanning several tissue compartments. Casting a
# wide net here is what makes the later selection objective: we measure many
# candidates and then filter on fixed criteria, rather than hand-picking a few
# genes we already expect to work.
CANDIDATE_PANEL: Dict[str, List[str]] = {
    "epidermal_keratin": [
        "KRT1", "KRT2", "KRT5", "KRT6A", "KRT10", "KRT14", "KRT16", "KRT17",
    ],
    "cornified_envelope": [
        "FLG", "LOR", "IVL", "TGM1", "CDSN", "SPINK5", "CASP14", "DSG1",
        "DSP", "CLDN1",
    ],
    "antimicrobial_inflammatory": [
        "S100A7", "S100A8", "S100A9", "S100A10", "DEFB4A", "LCN2", "PI3",
        "SERPINB3", "SERPINB4",
    ],
    "immune": [
        "PTPRC", "LST1", "TYROBP", "CD3D", "CD3E", "CD4", "CD8A", "CD68",
        "CD14", "HLA-DRA", "CXCL9", "CXCL10", "CCL19", "IL32",
    ],
    "dermal_stromal": [
        "COL1A1", "COL1A2", "COL3A1", "COL6A2", "DCN", "LUM", "MGP", "SPARC",
        "VIM", "ACTA2",
    ],
    "vascular_other": [
        "PECAM1", "AQP1", "MKI67", "KRT15", "MLANA",
    ],
}

ALL_CANDIDATES: List[str] = sorted(
    {gene for genes in CANDIDATE_PANEL.values() for gene in genes}
)

GENE_TO_GROUP: Dict[str, str] = {
    gene: group for group, genes in CANDIDATE_PANEL.items() for gene in genes
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_gzip_table(path: Path) -> List[List[str]]:
    """Read a gzipped tab-separated file into a list of split rows."""
    with gzip.open(path, "rt") as handle:
        return [line.rstrip("\n").split("\t") for line in handle]


def load_sample_matrix(sample_dir: Path) -> Tuple[csr_matrix, List[str], List[str]]:
    """Load one Visium sample as a spots x genes CSR matrix.

    Returns
    -------
    matrix
        Sparse counts, shape (n_spots, n_genes).
    gene_symbols
        Gene symbol for each column.
    barcodes
        Spot barcode for each row.
    """
    matrix = mmread(sample_dir / "matrix.mtx.gz").tocsr().T.tocsr()

    features = read_gzip_table(sample_dir / "features.tsv.gz")
    gene_symbols = [row[1] if len(row) > 1 else row[0] for row in features]

    barcodes = [row[0] for row in read_gzip_table(sample_dir / "barcodes.tsv.gz")]

    if matrix.shape[0] != len(barcodes) or matrix.shape[1] != len(gene_symbols):
        raise ValueError(
            f"{sample_dir.name}: matrix shape {matrix.shape} does not match "
            f"{len(barcodes)} barcodes x {len(gene_symbols)} genes"
        )

    return matrix, gene_symbols, barcodes


def extract_panel(
    matrix: csr_matrix,
    gene_symbols: List[str],
    panel: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return log-normalised expression for the panel genes, plus library sizes.

    Duplicate gene symbols (the same symbol on several rows of features.tsv)
    are summed, which is the conventional handling.
    """
    library_sizes = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    safe_sizes = np.where(library_sizes > 0, library_sizes, 1.0)

    symbol_to_columns: Dict[str, List[int]] = {}
    for index, symbol in enumerate(gene_symbols):
        symbol_to_columns.setdefault(symbol.upper(), []).append(index)

    present = [gene for gene in panel if gene.upper() in symbol_to_columns]

    columns = []
    for gene in present:
        indices = symbol_to_columns[gene.upper()]
        counts = np.asarray(matrix[:, indices].sum(axis=1)).ravel().astype(float)
        columns.append(counts)

    if not columns:
        raise ValueError("No panel genes found in this sample")

    counts_matrix = np.column_stack(columns)

    normalised = np.log1p(counts_matrix / safe_sizes[:, None] * TARGET_SUM)

    return normalised, library_sizes, present


# ---------------------------------------------------------------------------
# Spatial autocorrelation
# ---------------------------------------------------------------------------

def morans_i(values: np.ndarray, neighbor_indices: np.ndarray) -> float:
    """Moran's I using a binary k-nearest-neighbour weight matrix.

    A gene with high Moran's I varies smoothly across the tissue, which means
    there is a spatial pattern for an image model to recover. A gene near zero
    is spatially unstructured noise and is not a reasonable prediction target
    regardless of how a model scores on it.
    """
    centred = values - values.mean()
    denominator = float(np.sum(centred ** 2))

    if denominator == 0.0:
        return float("nan")

    neighbor_sum = centred[neighbor_indices].sum(axis=1)
    numerator = float(np.sum(centred * neighbor_sum))

    n_spots = len(values)
    total_weight = float(neighbor_indices.size)

    return (n_spots / total_weight) * (numerator / denominator)


def neighbor_index_matrix(coordinates: np.ndarray, k: int) -> np.ndarray:
    """k nearest neighbours of each spot, excluding the spot itself."""
    k_effective = min(k + 1, len(coordinates))
    model = NearestNeighbors(n_neighbors=k_effective, metric="euclidean")
    model.fit(coordinates)
    _, indices = model.kneighbors(coordinates)
    return indices[:, 1:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("\nBUILDING MULTI-GENE EXPRESSION TARGETS")
    print("=" * 78)

    if not METADATA_PATH.exists():
        print(f"ERROR: missing {METADATA_PATH}")
        print("Run scripts/extract_resnet_features.py first.")
        return 1

    if not DATA_DIR.exists():
        print(f"ERROR: missing {DATA_DIR}")
        return 1

    metadata = pd.read_csv(METADATA_PATH)

    required = {"sample", "barcode", "x_coordinate", "y_coordinate"}
    missing = required - set(metadata.columns)
    if missing:
        print(f"ERROR: resnet18_metadata.csv is missing columns: {sorted(missing)}")
        print(f"Columns found: {list(metadata.columns)}")
        return 1

    metadata["barcode"] = metadata["barcode"].astype(str)
    samples = sorted(metadata["sample"].unique())

    print(f"Anchor rows (patches) : {len(metadata)}")
    print(f"Samples               : {len(samples)}")
    print(f"Candidate panel genes : {len(ALL_CANDIDATES)}")

    per_sample_frames: List[pd.DataFrame] = []
    per_sample_stats: List[Dict[str, object]] = []
    genes_present_everywhere: List[set] = []

    for sample in samples:
        sample_dir = DATA_DIR / sample

        if not (sample_dir / "matrix.mtx.gz").exists():
            print(f"  {sample:10s} SKIPPED - no matrix.mtx.gz")
            continue

        matrix, gene_symbols, barcodes = load_sample_matrix(sample_dir)
        expression, library_sizes, present = extract_panel(
            matrix, gene_symbols, ALL_CANDIDATES
        )

        frame = pd.DataFrame(expression, columns=present)
        frame.insert(0, "barcode", [str(b) for b in barcodes])
        frame.insert(0, "sample", sample)
        frame["library_size"] = library_sizes

        per_sample_frames.append(frame)
        genes_present_everywhere.append(set(present))

        # Spatial statistics are computed on the spots that actually have
        # image patches, since those are the ones we will model.
        sample_meta = metadata[metadata["sample"] == sample]
        merged = sample_meta.merge(
            frame, on=["sample", "barcode"], how="inner", suffixes=("", "_expr")
        )

        if len(merged) >= N_SPATIAL_NEIGHBORS + 1:
            coordinates = merged[["x_coordinate", "y_coordinate"]].to_numpy(float)
            neighbors = neighbor_index_matrix(coordinates, N_SPATIAL_NEIGHBORS)
        else:
            neighbors = None

        for gene in present:
            values = merged[gene].to_numpy(float)
            per_sample_stats.append(
                {
                    "sample": sample,
                    "gene": gene,
                    "n_spots": len(values),
                    "detection_rate": float((values > 0).mean()),
                    "mean": float(values.mean()),
                    "variance": float(values.var()),
                    "morans_i": (
                        morans_i(values, neighbors)
                        if neighbors is not None
                        else float("nan")
                    ),
                }
            )

        print(
            f"  {sample:10s} spots={matrix.shape[0]:5d}  "
            f"matched={len(merged):5d}  panel_genes={len(present)}"
        )

    if not per_sample_frames:
        print("\nERROR: no samples processed.")
        return 1

    # ---------------------------------------------------------------- join ---

    expression_table = pd.concat(per_sample_frames, ignore_index=True)

    common_genes = sorted(set.intersection(*genes_present_everywhere))
    print(f"\nGenes present in every sample: {len(common_genes)}")

    dropped = sorted(set(ALL_CANDIDATES) - set(common_genes))
    if dropped:
        print(f"Dropped (absent from at least one sample): {dropped}")

    keep_columns = ["sample", "barcode", "library_size"] + common_genes

    try:
        targets = metadata[["sample", "barcode", "x_coordinate", "y_coordinate"]].merge(
            expression_table[keep_columns],
            on=["sample", "barcode"],
            how="left",
            validate="one_to_one",
        )
    except pd.errors.MergeError as error:
        print(f"\nERROR: the (sample, barcode) join is not one-to-one: {error}")
        print("This means duplicate barcodes exist in the image metadata or in a")
        print("count matrix. Run scripts/validate_dataset.py to locate them.")
        return 1

    if len(targets) != len(metadata):
        print(f"ERROR: join changed row count {len(metadata)} -> {len(targets)}")
        return 1

    unmatched = int(targets[common_genes[0]].isna().sum())
    if unmatched:
        print(f"ERROR: {unmatched} patches had no matching expression barcode.")
        print("The image metadata and the count matrices disagree.")
        return 1

    order_ok = bool(
        (targets["sample"].to_numpy() == metadata["sample"].to_numpy()).all()
        and (targets["barcode"].to_numpy() == metadata["barcode"].to_numpy()).all()
    )
    print(f"Row order preserved against resnet18_metadata.csv: {order_ok}")
    if not order_ok:
        print("ERROR: row order diverged; features and targets would be misaligned.")
        return 1

    # --------------------------------------------------------- statistics ---

    stats = pd.DataFrame(per_sample_stats)

    summary = (
        stats.groupby("gene")
        .agg(
            detection_rate_mean=("detection_rate", "mean"),
            detection_rate_min=("detection_rate", "min"),
            variance_mean=("variance", "mean"),
            variance_min=("variance", "min"),
            morans_i_mean=("morans_i", "mean"),
            morans_i_min=("morans_i", "min"),
            sample_mean_spread=("mean", "std"),
            n_samples=("sample", "nunique"),
        )
        .reset_index()
    )

    summary["group"] = summary["gene"].map(GENE_TO_GROUP)
    summary = summary.sort_values("morans_i_mean", ascending=False)

    # ------------------------------------------------------------- saving ---

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets_path = OUTPUT_DIR / "gene_expression_targets.csv"
    targets.to_csv(targets_path, index=False)

    matrix_path = OUTPUT_DIR / "gene_expression_targets.npy"
    np.save(matrix_path, targets[common_genes].to_numpy(dtype=np.float32))

    stats_path = OUTPUT_DIR / "gene_statistics_per_sample.csv"
    stats.to_csv(stats_path, index=False)

    summary_path = OUTPUT_DIR / "gene_statistics_summary.csv"
    summary.to_csv(summary_path, index=False)

    config_path = OUTPUT_DIR / "target_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "normalisation": "counts per spot scaled to TARGET_SUM, then log1p",
                "target_sum": TARGET_SUM,
                "n_spatial_neighbors": N_SPATIAL_NEIGHBORS,
                "genes": common_genes,
                "gene_groups": {g: GENE_TO_GROUP[g] for g in common_genes},
                "n_rows": int(len(targets)),
                "aligned_to": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
            },
            handle,
            indent=2,
        )

    # ------------------------------------------------------------ display ---

    print("\n" + "=" * 78)
    print("TOP 20 GENES BY MEAN SPATIAL AUTOCORRELATION (Moran's I)")
    print("=" * 78)

    display = summary.head(20)[
        [
            "gene",
            "group",
            "detection_rate_mean",
            "variance_mean",
            "morans_i_mean",
            "morans_i_min",
        ]
    ].round(4)
    print(display.to_string(index=False))

    print("\n" + "=" * 78)
    print("IMMUNE-GROUP GENES (for comparison with the existing immune score)")
    print("=" * 78)
    immune = summary[summary["group"] == "immune"][
        ["gene", "detection_rate_mean", "variance_mean", "morans_i_mean"]
    ].round(4)
    print(immune.to_string(index=False))

    print("\nSaved:")
    for path in [targets_path, matrix_path, stats_path, summary_path, config_path]:
        print(f"  {path.relative_to(PROJECT_ROOT)}")

    print(f"\nTarget matrix: {len(targets)} rows x {len(common_genes)} genes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

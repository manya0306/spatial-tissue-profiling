
from pathlib import Path
import gzip
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATA_DIR = Path("data/extracted")
OUTPUT_DIR = Path("outputs/spatial_immune_signal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMMUNE_GENES = [
    "PTPRC",
    "CD3D",
    "CD3E",
    "CD4",
    "CD8A",
    "LST1",
    "TYROBP",
]

BARRIER_GENES = [
    "FLG",
    "LOR",
    "IVL",
    "CLDN1",
]

K_NEIGHBORS = 6


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def read_gzip_lines(path):
    with gzip.open(path, "rt") as file:
        return [line.strip().split("\t") for line in file]


def load_expression(sample_dir):
    """
    Loads the sparse expression matrix and gene/barcode names.
    """

    matrix_path = sample_dir / "matrix.mtx.gz"
    features_path = sample_dir / "features.tsv.gz"
    barcodes_path = sample_dir / "barcodes.tsv.gz"

    matrix = mmread(matrix_path).tocsr().T

    features = read_gzip_lines(features_path)
    genes = [row[1] if len(row) > 1 else row[0] for row in features]

    barcodes = [
        row[0]
        for row in read_gzip_lines(barcodes_path)
    ]

    expression = matrix.toarray()

    gene_to_index = {
        gene: index
        for index, gene in enumerate(genes)
    }

    return expression, gene_to_index, barcodes


def load_coordinates(sample_dir, barcodes):
    """
    Loads Visium tissue coordinates and aligns them
    with the expression matrix barcodes.
    """

    position_path = (
        sample_dir
        / "spatial"
        / "tissue_positions_list.csv"
    )

    positions = pd.read_csv(
        position_path,
        header=None
    )

    positions.columns = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]

    positions = positions[
        positions["in_tissue"] == 1
    ].copy()

    positions = positions.set_index("barcode")

    valid_barcodes = [
        barcode
        for barcode in barcodes
        if barcode in positions.index
    ]

    indices = [
        barcodes.index(barcode)
        for barcode in valid_barcodes
    ]

    expression_indices = np.array(indices)

    coordinates = positions.loc[
        valid_barcodes,
        [
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ],
    ].values

    return expression_indices, coordinates, valid_barcodes


def calculate_spatial_autocorrelation(values, coordinates):
    """
    Calculates a simple distance-weighted spatial
    autocorrelation score.

    Positive values suggest nearby spots have
    similar expression levels.
    """

    values = np.asarray(values, dtype=float)

    if np.std(values) == 0:
        return np.nan

    distances = cdist(coordinates, coordinates)

    nonzero_distances = distances[distances > 0]

    if len(nonzero_distances) == 0:
        return np.nan

    bandwidth = np.median(nonzero_distances)

    weights = np.exp(
        -(distances ** 2) / (2 * bandwidth ** 2)
    )

    np.fill_diagonal(weights, 0)

    centered_values = values - np.mean(values)

    numerator = np.sum(
        weights
        * np.outer(centered_values, centered_values)
    )

    denominator = np.sum(centered_values ** 2)

    weight_sum = np.sum(weights)

    if denominator == 0 or weight_sum == 0:
        return np.nan

    score = (
        len(values)
        / weight_sum
        * numerator
        / denominator
    )

    return score


def calculate_neighborhood_correlation(values, coordinates):
    """
    Calculates the correlation between each spot's
    expression and the average expression of its
    nearest spatial neighbors.
    """

    values = np.asarray(values, dtype=float)

    if len(values) <= K_NEIGHBORS:
        return np.nan

    distances = cdist(coordinates, coordinates)
    np.fill_diagonal(distances, np.inf)

    neighbor_indices = np.argsort(
        distances,
        axis=1
    )[:, :K_NEIGHBORS]

    neighbor_values = values[neighbor_indices]
    neighbor_means = np.mean(neighbor_values, axis=1)

    if np.std(values) == 0 or np.std(neighbor_means) == 0:
        return np.nan

    correlation = np.corrcoef(
        values,
        neighbor_means
    )[0, 1]

    return correlation


def create_spatial_plot(
    values,
    coordinates,
    gene,
    sample_name,
):
    """
    Creates a spatial expression scatter plot.
    """

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=values,
        s=18,
        cmap="viridis",
    )

    plt.gca().invert_yaxis()
    plt.colorbar(scatter, label="Expression")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title(
        f"{sample_name} - Spatial expression of {gene}"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{sample_name}_{gene}_spatial.png"
    )

    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------

def main():

    results = []

    sample_dirs = sorted(
        [
            path
            for path in DATA_DIR.iterdir()
            if path.is_dir()
        ]
    )

    print("\nSPATIAL IMMUNE SIGNAL INSPECTION")
    print("=" * 60)

    for sample_dir in sample_dirs:

        sample_name = sample_dir.name

        required_files = [
            sample_dir / "matrix.mtx.gz",
            sample_dir / "features.tsv.gz",
            sample_dir / "barcodes.tsv.gz",
            sample_dir
            / "spatial"
            / "tissue_positions_list.csv",
        ]

        if not all(
            file.exists()
            for file in required_files
        ):
            print(
                f"Skipping {sample_name}: "
                "required files missing"
            )
            continue

        print(f"\nProcessing: {sample_name}")

        try:
            expression, gene_to_index, barcodes = (
                load_expression(sample_dir)
            )

            expression_indices, coordinates, valid_barcodes = (
                load_coordinates(
                    sample_dir,
                    barcodes
                )
            )

            expression = expression[
                expression_indices
            ]

            all_genes = IMMUNE_GENES + BARRIER_GENES

            for gene in all_genes:

                if gene not in gene_to_index:
                    continue

                gene_index = gene_to_index[gene]

                values = expression[:, gene_index]

                nonzero_fraction = np.mean(
                    values > 0
                )

                mean_expression = np.mean(values)
                total_expression = np.sum(values)

                autocorrelation = (
                    calculate_spatial_autocorrelation(
                        values,
                        coordinates
                    )
                )

                neighborhood_correlation = (
                    calculate_neighborhood_correlation(
                        values,
                        coordinates
                    )
                )

                results.append({
                    "sample": sample_name,
                    "gene": gene,
                    "spots": len(values),
                    "nonzero_spots": int(
                        np.sum(values > 0)
                    ),
                    "nonzero_fraction": (
                        nonzero_fraction
                    ),
                    "mean_expression": (
                        mean_expression
                    ),
                    "total_expression": (
                        total_expression
                    ),
                    "spatial_autocorrelation": (
                        autocorrelation
                    ),
                    "neighborhood_correlation": (
                        neighborhood_correlation
                    ),
                })

                # Plot only genes with measurable expression
                if nonzero_fraction > 0.01:
                    create_spatial_plot(
                        values,
                        coordinates,
                        gene,
                        sample_name,
                    )

            print(
                f"Spots analyzed: {len(coordinates)}"
            )

        except Exception as error:
            print(
                f"Error processing {sample_name}: "
                f"{error}"
            )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("\nNo results were generated.")
        return

    results_path = (
        OUTPUT_DIR
        / "spatial_immune_signal_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    summary = (
        results_df
        .groupby("gene")
        .agg({
            "nonzero_fraction": "mean",
            "mean_expression": "mean",
            "spatial_autocorrelation": "mean",
            "neighborhood_correlation": "mean",
        })
        .sort_values(
            "neighborhood_correlation",
            ascending=False
        )
    )

    summary_path = (
        OUTPUT_DIR
        / "gene_spatial_summary.csv"
    )

    summary.to_csv(summary_path)

    print("\n")
    print("=" * 60)
    print("GENE-LEVEL SPATIAL SUMMARY")
    print("=" * 60)

    print(
        summary.round(4).to_string()
    )

    print("\nOutput files saved to:")
    print(OUTPUT_DIR)

    print("\nInterpretation:")
    print(
        "- Positive spatial autocorrelation suggests "
        "spatially clustered expression."
    )
    print(
        "- Positive neighborhood correlation suggests "
        "nearby spots have similar expression."
    )
    print(
        "- These results indicate spatial transcriptomic "
        "patterns, not confirmed cell-type infiltration."
    )


if __name__ == "__main__":
    main()


from pathlib import Path
import gzip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import mmread


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATA_DIR = Path("data/extracted")

OUTPUT_DIR = Path(
    "outputs/composite_immune_score"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMMUNE_GENES = [
    "PTPRC",
    "LST1",
    "TYROBP",
    "CD3D",
    "CD3E",
    "CD4",
]


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def read_gzip_lines(path):

    with gzip.open(path, "rt") as file:
        return [
            line.strip().split("\t")
            for line in file
        ]


def load_sample(sample_dir):

    matrix_path = sample_dir / "matrix.mtx.gz"
    features_path = sample_dir / "features.tsv.gz"
    barcodes_path = sample_dir / "barcodes.tsv.gz"

    matrix = mmread(matrix_path).tocsr().T

    features = read_gzip_lines(features_path)

    genes = [
        row[1] if len(row) > 1 else row[0]
        for row in features
    ]

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

    expression_indices = [
        barcodes.index(barcode)
        for barcode in valid_barcodes
    ]

    coordinates = positions.loc[
        valid_barcodes,
        [
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ],
    ].values

    return (
        np.array(expression_indices),
        coordinates,
        valid_barcodes,
    )


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

def normalize_gene(values):

    """
    Log-normalizes and scales each gene within
    an individual sample.

    The 95th percentile is used as a robust upper
    scaling limit to reduce the influence of outliers.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    values = np.log1p(values)

    upper_limit = np.percentile(
        values,
        95
    )

    if upper_limit <= 0:
        return np.zeros_like(values)

    normalized = values / upper_limit

    normalized = np.clip(
        normalized,
        0,
        1
    )

    return normalized


# ---------------------------------------------------------
# SPATIAL PLOT
# ---------------------------------------------------------

def create_plot(
    coordinates,
    composite_score,
    sample_name,
):

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=composite_score,
        s=20,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plt.gca().invert_yaxis()

    plt.colorbar(
        scatter,
        label="Composite immune-associated score"
    )

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    plt.title(
        f"{sample_name} - Composite immune-associated score"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{sample_name}_composite_score.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    all_results = []

    sample_dirs = sorted(
        [
            path
            for path in DATA_DIR.iterdir()
            if path.is_dir()
        ]
    )

    print("\nCOMPOSITE IMMUNE-ASSOCIATED SCORE")
    print("=" * 65)

    for sample_dir in sample_dirs:

        sample_name = sample_dir.name

        required_files = [
            sample_dir / "matrix.mtx.gz",
            sample_dir / "features.tsv.gz",
            sample_dir / "barcodes.tsv.gz",
            (
                sample_dir
                / "spatial"
                / "tissue_positions_list.csv"
            ),
        ]

        if not all(
            file.exists()
            for file in required_files
        ):
            print(
                f"Skipping {sample_name}: files missing"
            )
            continue

        print(f"\nProcessing: {sample_name}")

        try:

            expression, gene_to_index, barcodes = (
                load_sample(sample_dir)
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

            normalized_genes = []

            gene_statistics = {}

            for gene in IMMUNE_GENES:

                if gene not in gene_to_index:
                    continue

                gene_index = gene_to_index[gene]

                values = expression[
                    :,
                    gene_index
                ]

                normalized_values = normalize_gene(
                    values
                )

                normalized_genes.append(
                    normalized_values
                )

                gene_statistics[
                    f"{gene}_nonzero_fraction"
                ] = np.mean(values > 0)

            if not normalized_genes:
                print(
                    f"No target genes found for {sample_name}"
                )
                continue

            normalized_matrix = np.column_stack(
                normalized_genes
            )

            composite_score = np.mean(
                normalized_matrix,
                axis=1
            )

            create_plot(
                coordinates,
                composite_score,
                sample_name
            )

            sample_result = pd.DataFrame({
                "sample": sample_name,
                "barcode": valid_barcodes,
                "x_coordinate": coordinates[:, 0],
                "y_coordinate": coordinates[:, 1],
                "composite_immune_score": composite_score,
            })

            all_results.append(sample_result)

            print(
                f"Spots analyzed: {len(composite_score)}"
            )

            print(
                "Mean composite score: "
                f"{np.mean(composite_score):.4f}"
            )

            print(
                "Maximum composite score: "
                f"{np.max(composite_score):.4f}"
            )

            print("Gene detection fractions:")

            for key, value in gene_statistics.items():

                print(
                    f"  {key}: {value:.4f}"
                )

        except Exception as error:

            print(
                f"Error processing {sample_name}: "
                f"{error}"
            )

    if not all_results:

        print("\nNo results generated.")
        return

    combined_results = pd.concat(
        all_results,
        ignore_index=True
    )

    output_path = (
        OUTPUT_DIR
        / "composite_immune_scores.csv"
    )

    combined_results.to_csv(
        output_path,
        index=False
    )

    sample_summary = (
        combined_results
        .groupby("sample")
        ["composite_immune_score"]
        .agg([
            "count",
            "mean",
            "std",
            "min",
            "max",
        ])
        .reset_index()
    )

    summary_path = (
        OUTPUT_DIR
        / "sample_composite_summary.csv"
    )

    sample_summary.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 65)
    print("SAMPLE-LEVEL COMPOSITE SCORE SUMMARY")
    print("=" * 65)

    print(
        sample_summary.round(4).to_string(
            index=False
        )
    )

    print("\nOutput files saved to:")
    print(OUTPUT_DIR)

    print(
        "\nNote: This score represents a combined "
        "immune-associated transcriptomic signal."
    )

    print(
        "It is not a validated measure of leukocyte "
        "infiltration or cell-type abundance."
    )


if __name__ == "__main__":

    main()

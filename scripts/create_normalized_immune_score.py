
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
    "outputs/normalized_immune_score"
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

TARGET_SUM = 10000


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

    matrix = mmread(
        sample_dir / "matrix.mtx.gz"
    ).tocsr().T

    features = read_gzip_lines(
        sample_dir / "features.tsv.gz"
    )

    genes = [
        row[1] if len(row) > 1 else row[0]
        for row in features
    ]

    barcodes = [
        row[0]
        for row in read_gzip_lines(
            sample_dir / "barcodes.tsv.gz"
        )
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

    expression_indices = np.array([
        barcodes.index(barcode)
        for barcode in valid_barcodes
    ])

    coordinates = positions.loc[
        valid_barcodes,
        [
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ],
    ].values

    return (
        expression_indices,
        coordinates,
        valid_barcodes,
    )


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

def library_size_normalize(expression):

    """
    Performs per-spot library-size normalization.

    Each spot is scaled to TARGET_SUM total counts,
    followed by log1p transformation.
    """

    library_sizes = expression.sum(axis=1)

    safe_library_sizes = np.where(
        library_sizes > 0,
        library_sizes,
        1
    )

    normalized = (
        expression
        / safe_library_sizes[:, None]
        * TARGET_SUM
    )

    normalized = np.log1p(normalized)

    return normalized, library_sizes


# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------

def create_plot(
    coordinates,
    score,
    sample_name,
):

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=score,
        s=20,
        cmap="viridis",
    )

    plt.gca().invert_yaxis()

    plt.colorbar(
        scatter,
        label="Normalized immune-associated score"
    )

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    plt.title(
        f"{sample_name} - Normalized immune score"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{sample_name}_normalized_score.png",
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

    print("\nNORMALIZED IMMUNE-ASSOCIATED SCORE")
    print("=" * 70)

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

            normalized_expression, library_sizes = (
                library_size_normalize(
                    expression
                )
            )

            gene_values = []

            for gene in IMMUNE_GENES:

                if gene not in gene_to_index:
                    continue

                gene_index = gene_to_index[gene]

                values = normalized_expression[
                    :,
                    gene_index
                ]

                gene_values.append(values)

            if not gene_values:

                print(
                    f"No target genes found: {sample_name}"
                )
                continue

            gene_matrix = np.column_stack(
                gene_values
            )

            composite_score = np.mean(
                gene_matrix,
                axis=1
            )

            create_plot(
                coordinates,
                composite_score,
                sample_name
            )

            result = pd.DataFrame({
                "sample": sample_name,
                "barcode": valid_barcodes,
                "x_coordinate": coordinates[:, 0],
                "y_coordinate": coordinates[:, 1],
                "library_size": library_sizes,
                "normalized_immune_score": composite_score,
            })

            all_results.append(result)

            print(
                f"Spots analyzed: {len(composite_score)}"
            )

            print(
                "Mean normalized score: "
                f"{np.mean(composite_score):.4f}"
            )

            print(
                "Median normalized score: "
                f"{np.median(composite_score):.4f}"
            )

            print(
                "Mean library size: "
                f"{np.mean(library_sizes):.2f}"
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

    combined_results.to_csv(
        OUTPUT_DIR
        / "normalized_immune_scores.csv",
        index=False
    )

    summary = (
        combined_results
        .groupby("sample")
        ["normalized_immune_score"]
        .agg([
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max",
        ])
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "normalized_sample_summary.csv",
        index=False
    )

    print("\n")
    print("=" * 70)
    print("NORMALIZED SAMPLE SUMMARY")
    print("=" * 70)

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    print("\nOutput files saved to:")
    print(OUTPUT_DIR)

    print(
        "\nNote: This is a normalized combined "
        "immune-associated transcriptomic score."
    )

    print(
        "It is not a validated measure of cell abundance "
        "or confirmed leukocyte infiltration."
    )


if __name__ == "__main__":

    main()


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = Path(
    "outputs/spatial_immune_signal/"
    "spatial_immune_signal_results.csv"
)

OUTPUT_DIR = Path("outputs/ls_nl_comparison")
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


def classify_sample(sample):
    if "_LS" in sample:
        return "Lesional"
    elif "_NL" in sample:
        return "Nonlesional"
    elif sample.startswith("HE"):
        return "Healthy"
    return "Unknown"


def main():

    df = pd.read_csv(INPUT_PATH)

    df["condition"] = df["sample"].apply(
        classify_sample
    )

    df = df[
        df["condition"].isin(
            ["Lesional", "Nonlesional", "Healthy"]
        )
    ].copy()

    selected_genes = IMMUNE_GENES + BARRIER_GENES

    df = df[
        df["gene"].isin(selected_genes)
    ].copy()

    summary = (
        df.groupby(
            ["condition", "gene"]
        )[
            [
                "nonzero_fraction",
                "mean_expression",
                "spatial_autocorrelation",
                "neighborhood_correlation",
            ]
        ]
        .mean()
        .reset_index()
    )

    summary = summary.sort_values(
        [
            "gene",
            "condition",
        ]
    )

    output_path = (
        OUTPUT_DIR / "ls_nl_gene_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False
    )

    print("\nLESIONAL VS NONLESIONAL COMPARISON")
    print("=" * 70)

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    # Plot average nonzero fraction
    fraction_pivot = summary.pivot(
        index="gene",
        columns="condition",
        values="nonzero_fraction"
    )

    fraction_pivot.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title(
        "Gene Detection Fraction by Tissue Condition"
    )

    plt.xlabel("Gene")
    plt.ylabel("Fraction of Spots with Expression")
    plt.xticks(rotation=45)
    plt.legend(title="Condition")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gene_detection_fraction.png",
        dpi=200
    )

    plt.close()

    # Plot neighborhood correlation
    correlation_pivot = summary.pivot(
        index="gene",
        columns="condition",
        values="neighborhood_correlation"
    )

    correlation_pivot.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title(
        "Spatial Neighborhood Correlation by Condition"
    )

    plt.xlabel("Gene")
    plt.ylabel("Neighborhood Correlation")
    plt.xticks(rotation=45)
    plt.legend(title="Condition")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "neighborhood_correlation.png",
        dpi=200
    )

    plt.close()

    print("\nFiles saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()

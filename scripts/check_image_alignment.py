
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


DATA_DIR = Path("data/extracted")
OUTPUT_DIR = Path("outputs/image_alignment_check")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLES_TO_CHECK = [
    "AD_2_LS",
    "AD_2_NL",
    "AD_6_LS",
]


def main():

    for sample_name in SAMPLES_TO_CHECK:

        sample_dir = DATA_DIR / sample_name

        image_path = (
            sample_dir
            / "spatial"
            / "tissue_hires_image.png"
        )

        coordinates_path = (
            sample_dir
            / "spatial"
            / "tissue_positions_list.csv"
        )

        scale_path = (
            sample_dir
            / "spatial"
            / "scalefactors_json.json"
        )

        if not image_path.exists():
            print(
                f"Missing image for {sample_name}"
            )
            continue

        if not coordinates_path.exists():
            print(
                f"Missing coordinates for {sample_name}"
            )
            continue

        if not scale_path.exists():
            print(
                f"Missing scale factors for {sample_name}"
            )
            continue

        image = Image.open(image_path)

        positions = pd.read_csv(
            coordinates_path,
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

        with open(scale_path, "r") as file:
            scale_factors = json.load(file)

        scale = scale_factors.get(
            "tissue_hires_scalef",
            None
        )

        if scale is None:
            print(
                f"Missing hires scale for {sample_name}"
            )
            continue

        x_coordinates = (
            positions["pxl_col_in_fullres"]
            * scale
        )

        y_coordinates = (
            positions["pxl_row_in_fullres"]
            * scale
        )

        print(f"\nSample: {sample_name}")
        print(f"Image size: {image.size}")
        print(f"Scale factor: {scale}")
        print(
            f"Number of tissue spots: "
            f"{len(positions)}"
        )

        print(
            f"X range: "
            f"{x_coordinates.min():.2f} - "
            f"{x_coordinates.max():.2f}"
        )

        print(
            f"Y range: "
            f"{y_coordinates.min():.2f} - "
            f"{y_coordinates.max():.2f}"
        )

        plt.figure(figsize=(10, 8))

        plt.imshow(image)

        plt.scatter(
            x_coordinates,
            y_coordinates,
            s=8,
            alpha=0.65,
        )

        plt.title(
            f"{sample_name} - Coordinate alignment"
        )

        plt.xlabel("Image X coordinate")
        plt.ylabel("Image Y coordinate")

        plt.xlim(0, image.width)
        plt.ylim(image.height, 0)

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR
            / f"{sample_name}_alignment.png"
        )

        plt.savefig(
            output_path,
            dpi=200
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

    print("\nAlignment checks completed.")


if __name__ == "__main__":
    main()

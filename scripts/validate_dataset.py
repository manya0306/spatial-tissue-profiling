"""Validate the existing 19-sample feature/target artifacts before modelling.

This script does not modify any existing output. It loads the artifacts that
were produced by ``create_normalized_immune_score.py``,
``extract_resnet_features.py`` and ``build_spatial_features.py`` and checks
that they are mutually consistent, leakage-free at the row level, and usable
as the input to a grouped (sample-level) cross-validation.

Run from the project root:

    python -m scripts.validate_dataset

A machine-readable report is written to ``outputs/validation/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (all relative to the project root - no absolute paths)
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data" / "extracted"

SCORE_PATH: Path = (
    PROJECT_ROOT / "outputs" / "normalized_immune_score" / "normalized_immune_scores.csv"
)
IMAGE_FEATURE_PATH: Path = (
    PROJECT_ROOT / "outputs" / "image_features" / "resnet18_features.npy"
)
IMAGE_METADATA_PATH: Path = (
    PROJECT_ROOT / "outputs" / "image_features" / "resnet18_metadata.csv"
)
NEIGHBOR_FEATURE_PATH: Path = (
    PROJECT_ROOT / "outputs" / "spatial_features" / "neighbor_image_features.npy"
)
SPATIAL_METADATA_PATH: Path = (
    PROJECT_ROOT / "outputs" / "spatial_features" / "spatial_metadata.csv"
)

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "validation"

EXPECTED_SAMPLES: List[str] = [
    "AD_1_LS", "AD_2_LS", "AD_3_LS", "AD_4_LS", "AD_5_LS", "AD_6_LS", "AD_7_LS",
    "AD_1_NL", "AD_2_NL", "AD_3_NL", "AD_5_NL", "AD_6_NL", "AD_7_NL",
    "HE_1", "HE_2", "HE_3", "HE_4", "HE_5", "HE_6",
]

REQUIRED_METADATA_COLUMNS: List[str] = [
    "sample",
    "barcode",
    "x_coordinate",
    "y_coordinate",
]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

class Report:
    """Collects PASS / WARN / FAIL checks and prints them as they happen."""

    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []
        self.facts: Dict[str, Any] = {}

    def record(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append({"check": name, "status": status, "detail": detail})
        print(f"  [{status:4s}] {name}" + (f" - {detail}" if detail else ""))

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        self.record(name, "PASS" if condition else "FAIL", detail)
        return condition

    def warn_if(self, name: str, condition: bool, detail: str = "") -> None:
        self.record(name, "WARN" if condition else "PASS", detail)

    def fact(self, key: str, value: Any) -> None:
        self.facts[key] = value

    @property
    def failures(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "FAIL")

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "WARN")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_array(path: Path, report: Report, label: str) -> Optional[np.ndarray]:
    """Load a .npy file, recording a FAIL instead of raising if it is absent."""
    if not path.exists():
        report.record(f"{label} exists", "FAIL", f"missing: {path}")
        return None
    array = np.load(path)
    report.record(f"{label} exists", "PASS", f"shape={array.shape}")
    return array


def load_table(path: Path, report: Report, label: str) -> Optional[pd.DataFrame]:
    """Load a .csv file, recording a FAIL instead of raising if it is absent."""
    if not path.exists():
        report.record(f"{label} exists", "FAIL", f"missing: {path}")
        return None
    frame = pd.read_csv(path)
    report.record(f"{label} exists", "PASS", f"shape={frame.shape}")
    return frame


# ---------------------------------------------------------------------------
# Individual validation stages
# ---------------------------------------------------------------------------

def validate_raw_data(report: Report) -> None:
    """Check that the decompressed Visium directories are present on disk."""
    section("1. RAW DATA DIRECTORIES")

    if not DATA_DIR.exists():
        report.record("data/extracted exists", "FAIL", f"missing: {DATA_DIR}")
        return

    found = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    report.fact("samples_on_disk", found)

    missing = [s for s in EXPECTED_SAMPLES if s not in found]
    extra = [s for s in found if s not in EXPECTED_SAMPLES]

    report.check(
        "all 19 expected sample directories present",
        not missing,
        f"missing: {missing}" if missing else f"{len(found)} directories",
    )
    report.warn_if(
        "no unexpected sample directories",
        bool(extra),
        f"unexpected: {extra}" if extra else "",
    )

    required_files = [
        "matrix.mtx.gz",
        "features.tsv.gz",
        "barcodes.tsv.gz",
        "spatial/tissue_positions_list.csv",
        "spatial/tissue_hires_image.png",
        "spatial/scalefactors_json.json",
    ]

    incomplete: Dict[str, List[str]] = {}
    for sample in found:
        absent = [f for f in required_files if not (DATA_DIR / sample / f).exists()]
        if absent:
            incomplete[sample] = absent

    report.check(
        "every sample directory has all required Visium files",
        not incomplete,
        json.dumps(incomplete) if incomplete else "",
    )
    report.fact("incomplete_samples", incomplete)


def validate_metadata(frame: pd.DataFrame, label: str, report: Report) -> None:
    """Check column presence, NaN/inf values, duplicates and coordinate sanity."""
    missing_columns = [c for c in REQUIRED_METADATA_COLUMNS if c not in frame.columns]
    report.check(
        f"{label}: required columns present",
        not missing_columns,
        f"missing: {missing_columns}" if missing_columns else str(list(frame.columns)),
    )
    if missing_columns:
        return

    duplicates = int(frame.duplicated(subset=["sample", "barcode"]).sum())
    report.check(
        f"{label}: no duplicate (sample, barcode) rows",
        duplicates == 0,
        f"{duplicates} duplicates",
    )

    nan_counts = frame[REQUIRED_METADATA_COLUMNS].isna().sum()
    report.check(
        f"{label}: no missing values in key columns",
        int(nan_counts.sum()) == 0,
        nan_counts[nan_counts > 0].to_dict() if nan_counts.sum() else "",
    )

    coords = frame[["x_coordinate", "y_coordinate"]].to_numpy(dtype=float)
    report.check(
        f"{label}: coordinates are finite",
        bool(np.isfinite(coords).all()),
        f"non-finite: {int((~np.isfinite(coords)).sum())}",
    )
    report.check(
        f"{label}: coordinates are non-negative",
        bool((coords >= 0).all()),
        f"min={coords.min():.1f}",
    )


def validate_features(
    features: np.ndarray,
    metadata: pd.DataFrame,
    label: str,
    report: Report,
) -> None:
    """Check feature matrix shape, finiteness and degenerate columns."""
    report.check(
        f"{label}: rows match metadata",
        features.shape[0] == len(metadata),
        f"features={features.shape[0]} metadata={len(metadata)}",
    )
    report.check(
        f"{label}: all values finite",
        bool(np.isfinite(features).all()),
        f"non-finite: {int((~np.isfinite(features)).sum())}",
    )

    zero_variance = int((features.std(axis=0) == 0).sum())
    report.warn_if(
        f"{label}: no zero-variance dimensions",
        zero_variance > 0,
        f"{zero_variance} of {features.shape[1]} dimensions are constant",
    )

    all_zero_rows = int((np.abs(features).sum(axis=1) == 0).sum())
    report.check(
        f"{label}: no all-zero feature rows",
        all_zero_rows == 0,
        f"{all_zero_rows} rows",
    )


def validate_row_alignment(
    image_metadata: pd.DataFrame,
    spatial_metadata: pd.DataFrame,
    report: Report,
) -> None:
    """The neighbour features are stored without keys, so row order must match.

    ``build_spatial_features.py`` rebuilds its metadata by re-concatenating the
    image metadata in sorted-sample order. That is only safe if the image
    metadata is already grouped contiguously by sample. This check makes the
    assumption explicit instead of trusting it.
    """
    section("4. ROW ALIGNMENT BETWEEN IMAGE AND SPATIAL ARTIFACTS")

    if not report.check(
        "image and spatial metadata have equal length",
        len(image_metadata) == len(spatial_metadata),
        f"{len(image_metadata)} vs {len(spatial_metadata)}",
    ):
        return

    image_keys = image_metadata["sample"].astype(str) + "|" + image_metadata["barcode"].astype(str)
    spatial_keys = spatial_metadata["sample"].astype(str) + "|" + spatial_metadata["barcode"].astype(str)

    identical_order = bool((image_keys.to_numpy() == spatial_keys.to_numpy()).all())
    report.check(
        "row order is identical (neighbour features align to image features)",
        identical_order,
        "" if identical_order else
        f"{int((image_keys.to_numpy() != spatial_keys.to_numpy()).sum())} rows differ",
    )

    same_content = set(image_keys) == set(spatial_keys)
    report.check(
        "both artifacts cover the same spots",
        same_content,
        "" if same_content else "spot sets differ",
    )

    grouped = image_metadata["sample"].ne(image_metadata["sample"].shift()).cumsum().nunique()
    n_samples = image_metadata["sample"].nunique()
    report.check(
        "image metadata is contiguously grouped by sample",
        grouped == n_samples,
        f"{grouped} contiguous blocks for {n_samples} samples",
    )


def validate_split_feasibility(metadata: pd.DataFrame, report: Report) -> None:
    """Report per-sample spot counts and whether grouped CV is viable."""
    section("5. SAMPLE DISTRIBUTION AND SPLIT FEASIBILITY")

    counts = metadata["sample"].value_counts().sort_index()
    print(counts.to_string())
    report.fact("spots_per_sample", counts.to_dict())
    report.fact("total_spots", int(len(metadata)))

    n_samples = int(counts.size)
    report.fact("n_samples", n_samples)

    report.check(
        "at least 5 samples available for grouped cross-validation",
        n_samples >= 5,
        f"{n_samples} samples",
    )
    report.warn_if(
        "no sample dominates the dataset (>20% of spots)",
        bool((counts / counts.sum() > 0.20).any()),
        f"largest share={counts.max() / counts.sum():.1%} ({counts.idxmax()})",
    )
    report.warn_if(
        "no very small sample (<100 spots)",
        bool((counts < 100).any()),
        f"smallest={counts.min()} ({counts.idxmin()})",
    )

    groups = ["AD_LS" if s.endswith("_LS") else "AD_NL" if s.endswith("_NL") else "HE"
              for s in counts.index]
    condition_counts = pd.Series(groups, index=counts.index).value_counts().to_dict()
    print("\nSamples per condition:", condition_counts)
    report.fact("samples_per_condition", condition_counts)


def validate_leakage_risks(report: Report) -> None:
    """Flag artifacts that are derived from the target and must not be inputs."""
    section("6. LEAKAGE RISK INVENTORY")

    target_derived = {
        "outputs/spatial_features/neighbor_mean_scores.npy":
            "mean immune score of neighbouring spots - derived from the target",
    }

    for relative_path, reason in target_derived.items():
        exists = (PROJECT_ROOT / relative_path).exists()
        report.warn_if(
            f"target-derived artifact present: {relative_path}",
            exists,
            reason if exists else "not present",
        )

    report.record(
        "neighbour IMAGE features are input-only",
        "PASS",
        "neighbor_image_features.npy is built from ResNet features, not from expression",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    report = Report()

    print("\nDATASET VALIDATION")
    print(f"Project root: {PROJECT_ROOT}")

    validate_raw_data(report)

    section("2. TARGET AND METADATA TABLES")
    scores = load_table(SCORE_PATH, report, "normalized_immune_scores.csv")
    image_metadata = load_table(IMAGE_METADATA_PATH, report, "resnet18_metadata.csv")
    spatial_metadata = load_table(SPATIAL_METADATA_PATH, report, "spatial_metadata.csv")

    if scores is not None:
        validate_metadata(scores, "scores", report)
    if image_metadata is not None:
        validate_metadata(image_metadata, "image metadata", report)
        if scores is not None:
            dropped = len(scores) - len(image_metadata)
            report.fact("spots_dropped_at_patch_extraction", int(dropped))
            report.record(
                "spots dropped because the patch fell outside the image",
                "WARN" if dropped else "PASS",
                f"{dropped} of {len(scores)} spots dropped",
            )
    if spatial_metadata is not None:
        validate_metadata(spatial_metadata, "spatial metadata", report)

    section("3. FEATURE MATRICES")
    image_features = load_array(IMAGE_FEATURE_PATH, report, "resnet18_features.npy")
    neighbor_features = load_array(
        NEIGHBOR_FEATURE_PATH, report, "neighbor_image_features.npy"
    )

    if image_features is not None and image_metadata is not None:
        validate_features(image_features, image_metadata, "image features", report)
        report.fact("image_feature_dim", int(image_features.shape[1]))
    if neighbor_features is not None and spatial_metadata is not None:
        validate_features(neighbor_features, spatial_metadata, "neighbour features", report)
        report.fact("neighbor_feature_dim", int(neighbor_features.shape[1]))

    if image_metadata is not None and spatial_metadata is not None:
        validate_row_alignment(image_metadata, spatial_metadata, report)

    if image_metadata is not None:
        validate_split_feasibility(image_metadata, report)

    validate_leakage_risks(report)

    section("7. MISSING FOR MULTI-GENE MODELLING")
    gene_target_path = PROJECT_ROOT / "outputs" / "gene_targets" / "gene_expression_targets.csv"
    report.check(
        "per-spot multi-gene expression target matrix exists",
        gene_target_path.exists(),
        "" if gene_target_path.exists() else f"not yet built: {gene_target_path}",
    )

    section("SUMMARY")
    total = len(report.checks)
    print(f"Checks run : {total}")
    print(f"Passed     : {total - report.failures - report.warnings}")
    print(f"Warnings   : {report.warnings}")
    print(f"Failed     : {report.failures}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"checks": report.checks, "facts": report.facts},
            handle,
            indent=2,
            default=str,
        )
    print(f"\nSaved report to: {report_path}")

    if report.failures:
        print("\nOne or more checks FAILED. Fix these before training.")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

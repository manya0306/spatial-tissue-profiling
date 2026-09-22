# Scripts

This directory contains the reproducible pipeline used for the virtual spatial transcriptomics experiments.

## Main Pipeline

### `dataset_inventory.py`
Summarizes the available spatial transcriptomics samples, image files, spot counts, and dataset structure.

### `extract_patches.py`
Extracts H&E image patches corresponding to spatial transcriptomics spots.

### `check_patch_quality.py`
Performs basic image-patch quality checks to identify unusable or low-quality patches.

### `check_image_alignment.py`
Generates visual checks for spatial alignment between H&E image patches and transcriptomic spots.

### `extract_resnet_features.py`
Extracts frozen ImageNet-pretrained ResNet18 image embeddings from H&E patches.

### `build_spatial_features.py`
Constructs spatial-neighborhood features from nearby spots.

### `build_gene_targets.py`
Selects and preprocesses gene-expression targets for supervised prediction and generates target statistics.

### `validate_dataset.py`
Validates the final image features, spatial features, gene targets, sample metadata, and dimensions before model training.

### `run_experiments.py`
Runs the main cross-sample evaluation pipeline.

The experiments include:
- Mean-expression baseline
- Image-only Ridge regression
- Image + spatial-neighborhood Ridge regression
- PCA + Ridge variants
- Multimodal MLP fusion

Evaluation uses grouped cross-validation by sample so spots from the same tissue section are not split between training and test sets.

### `make_figures.py`
Generates the final experiment plots, including:
- Model comparison
- Per-gene R²
- Per-gene MAE
- Gene/model heatmaps
- Predicted-vs-measured expression
- Spatial prediction maps

## Reproducibility

The intended workflow is:

```text
Dataset
   ↓
Dataset inventory / validation
   ↓
H&E patch extraction
   ↓
Patch quality + image alignment checks
   ↓
ResNet18 image features
   ↓
Spatial neighborhood features
   ↓
Gene-expression targets
   ↓
Dataset validation
   ↓
Cross-sample experiments
   ↓
Evaluation metrics + predictions
   ↓
Figures and spatial maps


# Virtual Spatial Transcriptomics from Histopathology

**AI-Based Virtual Spatial Transcriptomics: Predicting Spatial Gene Expression from H&E Histopathology Using Multimodal Deep Learning**

A research-inspired educational prototype that predicts spot-level expression of a 56-gene panel from H&E image patches, neighbouring-patch appearance, and relative spatial position, evaluated with leakage-free sample-grouped cross-validation.

> **Disclaimers.** This is not a clinical or diagnostic system and must not be used for any medical purpose. It is not a reproduction of OmiCLIP or STPath, and no foundation model was trained here. It does not reconstruct a full transcriptome — the target is a fixed 56-gene panel. All biological interpretation is exploratory.

---

## 1. Objective

Predict spatial gene-expression patterns from H&E histopathology using:

1. Frozen ImageNet ResNet18 features from each tissue-spot patch
2. Mean image features of the 6 nearest neighbouring patches (spatial context)
3. Within-sample standardised spatial coordinates

and evaluate the models under splits that hold out **whole tissue samples**.

Inspired by, but not reproducing:

[OmiCLIP](https://www.nature.com/articles/s41592-025-02707-1) · [STPath](https://www.nature.com/articles/s41746-025-02020-3)

---

## 2. Dataset

Public 10x Visium spatial transcriptomics of human skin (GSE197023).

| Group | Samples | Identifiers |
|---|---:|---|
| Atopic dermatitis, lesional | 7 | `AD_1_LS` … `AD_7_LS` |
| Atopic dermatitis, non-lesional | 6 | `AD_1_NL`, `AD_2_NL`, `AD_3_NL`, `AD_5_NL`, `AD_6_NL`, `AD_7_NL` |
| Healthy | 6 | `HE_1` … `HE_6` |

- **19 samples**, **13,742 tissue spots with valid image patches**
- `AD_4_NL` is absent from the dataset
- Spots whose 96 px patch fell outside the tissue image were dropped during feature extraction

---

## 3. Pipeline

```text
Visium sample (H&E image + counts + spot coordinates)
        |
        +-- tissue_hires_image.png --> 96 px patch per spot
        |                                     |
        |                          frozen ResNet18 (ImageNet)
        |                                     |
        |                              (13742, 512) image features
        |                                     |
        |                          6-NN mean within sample
        |                                     |
        |                              (13742, 512) neighbour features
        |
        +-- spot coordinates --> standardised within each sample --> (13742, 2)
        |
        +-- matrix.mtx.gz --> library-size normalise to 10k --> log1p
                                              |
                                       (13742, 56) gene targets
                                              |
        image + neighbour + coordinates ------+
                                              |
                               sample-grouped cross-validation
                                              |
                      mean / ridge / PCA+ridge / multimodal fusion net
                                              |
                           per-gene metrics + spatial prediction maps
```

---

## 4. Preprocessing

- **Counts**: per-spot library-size normalisation to 10,000, followed by `log1p`.
- **Duplicate gene symbols** in `features.tsv.gz` are summed.
- **Coordinates** are standardised within each sample. Raw pixel coordinates are not comparable between slides; spatial position is used independently of expression.
- **Feature scaling, PCA, and target scaling are fitted on training folds only.**

---

## 5. Gene Targets

All **56 genes present in every sample** are modelled and reported. No gene is dropped, and none was selected based on model performance.

The panel is defined in `scripts/build_gene_targets.py` before model training and spans epidermal keratins, cornified-envelope, antimicrobial/inflammatory, immune, dermal/stromal, and vascular compartments.

`build_gene_targets.py` also reports per-gene detection rate, variance, and **Moran's I** (spatial autocorrelation).

Moran's I provides a descriptive measure of whether expression varies spatially across the tissue. These statistics do not filter the modelled gene panel.

---

## 6. Feature Extraction

| Feature | Dim | Source | Target-derived? |
|---|---:|---|---|
| Patch appearance | 512 | Frozen ResNet18, ImageNet weights, 96 px crop | No |
| Neighbourhood appearance | 512 | Mean ResNet18 features of 6 nearest spots, within sample | No |
| Relative position | 2 | x/y standardised within sample | No |

`outputs/spatial_features/neighbor_mean_scores.npy` is **target-derived** and is **never used as a model input**. It is retained only as an exploratory diagnostic.

---

## 7. Models

| Name | Inputs | Notes |
|---|---|---|
| `mean_baseline` | — | Training-fold per-gene mean |
| `ridge_image` | 512 | Multi-output Ridge |
| `ridge_image_neighbor` | 1024 | Adds neighbourhood appearance |
| `pca_ridge_image` | 512 → 128 PC | PCA fitted per training fold |
| `pca_ridge_image_neighbor_coords` | 1026 → 128 PC | Adds spatial coordinates |
| `mlp_fusion` | 512 + 512 + 2 | PyTorch multimodal fusion network |

The fusion network uses three modality-specific projections for patch appearance, neighbourhood appearance, and position. These are concatenated and passed through a shared trunk with dropout and a 56-output prediction head.

Ridge hyperparameters are selected using training data only. The fusion network uses early stopping with whole training samples held out inside each fold. Neither model uses the test fold during model selection.

---

## 8. Evaluation

Primary evaluation uses `GroupKFold` grouped by sample:

- **5 folds**
- Seed: `42`
- Every sample appears in the test fold exactly once

`--cv loso` additionally supports 19-fold leave-one-sample-out evaluation.

Random spot-level splitting is **not** used. Neighbouring Visium spots share local tissue context, so a random spot split could place highly related spots on both sides of the train/test boundary and produce overly optimistic estimates.

### Metrics

The pipeline reports:

- MAE
- RMSE
- R²
- Pearson correlation
- Spearman correlation

Metrics are reported per gene, per fold, and in aggregate, including fold mean and standard deviation.

**Within-sample Pearson correlation** is also reported to assess whether spatial expression patterns within an individual tissue section are recovered.

---

## 9. Running the Pipeline

```bash
rpvenv\Scripts\activate
pip install -r requirements.txt

python scripts/validate_dataset.py       # artifact integrity checks
python scripts/build_gene_targets.py     # construct the 56-gene target matrix
python scripts/run_experiments.py        # run all models with grouped CV
python scripts/make_figures.py           # generate plots and spatial maps
```

Optional experiment settings:

```text
--cv loso
```

Runs 19-fold leave-one-sample-out evaluation.

```text
--skip-mlp
```

Runs the linear models without the multimodal MLP.

```text
--verbose
```

Enables additional experiment logging.

---

## 10. Outputs

```text
outputs/
├── gene_targets/
│   ├── gene_expression_targets.csv
│   ├── gene_expression_targets.npy
│   ├── gene_statistics_per_sample.csv
│   ├── gene_statistics_summary.csv
│   └── target_config.json
│
├── experiments/
│   ├── metrics_per_gene.csv
│   ├── metrics_per_gene_per_fold.csv
│   ├── metrics_aggregate.csv
│   ├── fold_assignments.csv
│   ├── fold_summary.csv
│   ├── predictions/
│   │   ├── ground_truth.csv
│   │   └── model predictions
│   ├── mlp_training_history.json
│   └── run_config.json
│
├── figures/
│   └── experiments/
│       ├── model_comparison.png
│       ├── per_gene_mae.png
│       ├── per_gene_r2.png
│       ├── gene_model_heatmap.png
│       ├── predicted_vs_measured.png
│       ├── spatial_maps/
│       └── figure_manifest.json
│
└── validation/
    └── validation_report.json

RESULTS.md
```

`RESULTS.md` is generated directly from the computed experiment metric tables by `run_experiments.py`.

---

## 11. Results

See [`RESULTS.md`](RESULTS.md), which is generated directly from the computed metric tables.

The results file reports the performance of every model across the complete 56-gene panel, including genes with negative R².

The main evaluation uses sample-grouped cross-validation, so the reported performance reflects prediction on held-out tissue samples rather than random spot-level splits.

The final model comparison and spatial prediction maps are generated by `scripts/make_figures.py`.

---

## 12. Limitations

- The evaluation contains 19 sections from one public dataset. Sample-grouped cross-validation controls within-dataset leakage but does not constitute independent external validation.
- The image encoder is a frozen ImageNet-pretrained ResNet18 rather than a pathology-specific encoder.
- The image backbone is not fine-tuned end-to-end.
- The model predicts a fixed panel of 56 genes rather than the full transcriptome.
- Visium spots are approximately 55 µm and contain multiple cells; predictions are therefore spot-level rather than single-cell.
- Image features are extracted from `tissue_hires_image.png` rather than full-resolution whole-slide images.
- Image alignment was visually checked for `AD_2_LS`, `AD_2_NL`, and `AD_6_LS`.
- Biological interpretation remains exploratory and should not be treated as clinical or diagnostic evidence.

---

## 13. Future Work

1. Pathology-pretrained encoders such as UNI, CONCH, or Phikon instead of ImageNet ResNet18.
2. Graph neural networks over the spatial spot adjacency graph.
3. Fine-tuning the image backbone end-to-end.
4. Full-resolution patches and multi-scale image context.
5. Independent external cohort validation.
6. Calibration and uncertainty estimates for individual predictions.

---

## 14. Team Contributions

### Manya Aggarwal — Image Processing, Spatial Features & Pipeline Integration

- Designed and implemented the H&E image-processing workflow and spot-level patch extraction.
- Implemented image-patch quality checks and spatial image-alignment validation.
- Developed the frozen ResNet18 feature-extraction pipeline for histopathology patches.
- Implemented spatial-neighbourhood feature construction from nearby tissue spots.
- Contributed to dataset validation, preprocessing, and leakage-aware feature construction.
- Integrated the image, spatial, and transcriptomic components into the end-to-end experimental pipeline.
- Developed experiment and visualization workflows for model evaluation and spatial prediction maps.
- Contributed to interpretation of quantitative and spatial results.

### Aashna — Gene Targets, Predictive Modelling & Evaluation

- Designed and implemented the gene-expression target construction and preprocessing workflow.
- Performed gene-level statistical analysis, including detection, variance, and spatial autocorrelation analysis.
- Developed the supervised prediction and baseline modelling workflow.
- Implemented Ridge and PCA-based regression experiments and contributed to multimodal fusion modelling.
- Designed the sample-grouped evaluation strategy to assess generalisation across tissue sections.
- Implemented and evaluated quantitative metrics including MAE, RMSE, R², Pearson, and Spearman correlation.
- Contributed to model comparison, per-gene analysis, and interpretation of predictive performance.
- Contributed to reproducibility, result generation, and documentation of the final experiments.

### Joint Work

Both members contributed to:

- Problem formulation and project methodology.
- Literature review and adaptation of ideas from OmiCLIP and STPath.
- Dataset and experimental design decisions.
- Leakage prevention and evaluation methodology.
- Analysis of limitations and future improvements.
- Final integration, testing, visualization, and presentation of the project.

---

## 15. Repository Layout

```text
spatial-tissue-profiling/
│
├── scripts/
│   ├── build_gene_targets.py
│   ├── build_spatial_features.py
│   ├── check_image_alignment.py
│   ├── check_patch_quality.py
│   ├── dataset_inventory.py
│   ├── extract_patches.py
│   ├── extract_resnet_features.py
│   ├── make_figures.py
│   ├── run_experiments.py
│   ├── validate_dataset.py
│   └── README_SCRIPTS.md
│
├── notebooks/
├── src/
├── tests/
├── data/                       # gitignored
├── outputs/                    # gitignored generated artifacts
├── requirements.txt
├── RESULTS.md
└── README.md
```

`scripts/` contains the current reproducible pipeline. See [`scripts/README_SCRIPTS.md`](scripts/README_SCRIPTS.md) for the role of each script.

`notebooks/` contains exploratory and development work.

`data/` contains the downloaded dataset and is excluded from Git because of its size.

`outputs/` contains generated features, predictions, metrics, and figures and is also excluded from Git. These artifacts can be regenerated using the pipeline commands in Section 9.

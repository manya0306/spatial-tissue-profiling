
# Spatially Guided Virtual Tissue Profiling from H&E Histopathology

## 1. Project Overview

This project explores whether spatial context and H&E image features can support the prediction of selected gene-expression patterns from spatial transcriptomics data.

The project is inspired by research on integrating histopathology images with spatial transcriptomics, including OmiCLIP and STPath.

**Application focus:** Atopic dermatitis (AD)

**Current implementation:** Research-inspired proof-of-concept pipeline.

---

## 2. Current Pipeline

```text
H&E Histopathology Image
          |
          v
Tissue Spot Detection
          |
          v
Image Patch Extraction
          |
          v
ResNet18 Image Features
          |
          v
Spatial Coordinate Alignment
          |
          v
Spatial Neighborhood Construction
          |
          v
Feature Combination
(Image Features + Neighbor Features)
          |
          v
ExtraTrees Regression
          |
          v
Gene Expression Prediction
          |
          v
Spatial Holdout Evaluation
```

---

## 3. Dataset and Preprocessing

- Dataset: GSE197023
- Sample: AD_1_LS
- Number of tissue spots: 188
- Image patch size: 224 × 224 pixels
- Image feature extractor: ResNet18 pretrained on ImageNet
- Image feature dimension: 512
- Number of spatial neighbors: 5
- Expression transformation: log1p
- Target genes: 10

### Target Genes

```text
KRT1
KRT10
KRT14
LOR
DSG1
FLG
S100A7
S100A8
S100A9
CLDN1
```

---

## 4. Model Comparison

Spatial block cross-validation was used to compare three feature configurations.

| Model | Mean MAE | Mean R² |
|---|---:|---:|
| Image only | 0.7331 | -1.4258 |
| Image + coordinates | 0.7116 | -1.3039 |
| Image + neighbors | 0.6373 | -0.8697 |

### Observation

The image + neighbors configuration produced lower mean MAE and improved mean R² compared with the image-only baseline on this sample.

However, the average R² remained negative. Therefore, the results do not establish reliable generalization across unseen spatial regions.

---

## 5. Spatial Holdout Evaluation

A rightmost 20% spatial region was held out for testing.

- Training spots: 150
- Testing spots: 38
- Model: ExtraTrees Regression
- Input: Image features + spatial-neighborhood image features

### Results

| Gene | MAE | R² |
|---|---:|---:|
| KRT1 | 0.8039 | -2.8196 |
| KRT10 | 0.9347 | -3.2805 |
| KRT14 | 0.5029 | -0.0048 |
| LOR | 0.9985 | -1.0008 |
| DSG1 | 0.9507 | -3.9593 |
| FLG | 0.9766 | -0.5265 |
| S100A7 | 0.9975 | -1.0827 |
| S100A8 | 0.5551 | 0.4490 |
| S100A9 | 0.4659 | 0.5581 |
| CLDN1 | 1.0506 | -2.7360 |

### Observation

S100A8 and S100A9 achieved positive R² on this particular spatial holdout. Most other genes showed negative R², indicating that predictive performance varied substantially between genes.

These results are exploratory and are not sufficient to establish biological or clinical validity.

---

## 6. Visualizations Generated

The following visualizations were generated:

1. Tissue spot overlay
2. Spatial neighborhood graph
3. Spatial gene-expression maps
4. Model comparison using MAE and R²
5. KRT1 actual versus predicted expression
6. KRT1 spatial prediction error
7. Multi-gene spatial holdout metrics

---

## 7. Limitations

- Evaluation was performed on one AD sample.
- The dataset contains 188 tissue spots.
- The spatial holdout contains only 38 testing spots.
- Spatial distribution shift affects performance.
- The model uses a general-purpose ImageNet-pretrained feature extractor.
- The current implementation does not reproduce the full OmiCLIP or STPath foundation models.
- The current results do not establish clinical utility.
- The results do not prove a biological relationship between spatial context and disease mechanisms.
- Additional samples and independent validation are required.

---

## 8. Current Contribution

The current project demonstrates an end-to-end experimental framework for:

- Extracting image features from H&E tissue patches.
- Aligning image features with spatial transcriptomics measurements.
- Constructing spatial-neighborhood representations.
- Comparing image-only and spatially enhanced features.
- Evaluating gene-expression prediction under spatial holdout conditions.

The project provides a foundation for future improvements involving stronger pathology encoders, spatially aware neural architectures, additional samples, and more rigorous validation.

---

## 9. Future Work

1. Evaluate additional AD samples.
2. Use patient-level train/test separation.
3. Compare against simple expression baselines.
4. Investigate stronger pretrained pathology encoders.
5. Explore graph-based spatial encoders.
6. Evaluate biological pathway-level predictions.
7. Compare performance across different spatial regions.
8. Investigate the influence of tissue quality and spot coverage.

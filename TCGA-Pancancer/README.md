# TCGA-Pancancer

This folder contains code and Jupyter notebooks for experiments using the **TCGA Pan-Cancer Atlas** dataset.  
The analyses focus on evaluating **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)** classifiers, combined with **Conformal Prediction** methods for multi-class cancer type classification.
---

## Contents

- **Data Preprocessing Scripts:** Load, clean, normalise, and transform TCGA gene expression data.
- **Feature Reduction:** Dimensionality reduction (e.g., PCA, Truncated SVD) for high-dimensional RNA-Seq data.
- **Model Training:** LDA and QDA classifiers with/without Conformal Prediction.
- **Evaluation:** Coverage, sharpness, accuracy, and proper scoring rule metrics (log-loss, Brier, spherical).
- **Notebooks:** Step-by-step workflows for data exploration, preprocessing, model training, and evaluation.

---

## Data Requirements

> The TCGA dataset is **not included** in this repository due to size restrictions (>100MB).  
> Please download it from official sources or the Kaggle repository referenced in the MSc thesis.

---

## Usage

1. Place the TCGA data in the appropriate `data/` directory inside this folder.
2. Run preprocessing scripts to prepare the dataset.
3. Execute notebooks or Python scripts to train models and evaluate results.

---

## Related Work

These experiments are part of the MSc Individual Research Report on evaluating LDA and QDA with Conformal Prediction.

---

## Attribution

Some preprocessing and modeling code in the **TCGA-Pancancer** folder were adapted from  
[AndreCNF/tcga-cancer-classification](https://github.com/AndreCNF/tcga-cancer-classification),  
which implements deep learning and traditional models for multi-class cancer type classification  
using TCGA RNA-Seq data. Further details and adaptations are documented in the MSc thesis.


# TCGA-Pancancer

This folder contains code and Jupyter notebooks for experiments using the **TCGA Pan-Cancer Atlas** dataset.  
The analyses focus on evaluating **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)** classifiers, combined with **Conformal Prediction** methods for multi-class cancer type classification.

---

## Directory Layout

### Main Notebooks
- `tcga-data-exploration.ipynb` — Initial exploration of the TCGA Pan-Cancer RNA-Seq dataset, including descriptive statistics, sample distribution, and preliminary visualisations.
- `tcga-data-preprocessing.ipynb` — Data cleaning, normalisation, and dimensionality reduction steps (PCA/SVD) for RNA-Seq features, with filtering based on minimum sample size.
- `tcga-data-joining.ipynb` — Merges RNA expression data with sample metadata (cancer type, clinical variables) using a SQLite database.
- `tcga-model-training.ipynb` — Trains and evaluates LDA and QDA classifiers (with and without Conformal Prediction) on the processed TCGA data, computing coverage, sharpness, and proper scoring rules.

### Data Directory
- `cleaned/` — Contains processed TCGA datasets in CSV format and encoding metadata.
  - `encod_dict.yaml` — Mapping of categorical variable encodings used in preprocessing.
- `normalized/` — Scaled datasets used for model training and evaluation.
  - `clinical_outcome.csv` — Processed clinical outcome variables.
  - `copy_number_ratio.csv` — Normalised copy number variation data.
  - `rna.csv` — Normalised RNA-Seq expression matrix.
  - `tcga.csv` — Combined dataset integrating multiple modalities.
- `unnormalized/` — Raw processed datasets without scaling.
  - `clinical_outcome.csv` — Unscaled clinical outcome variables.
  - `copy_number_ratio.csv` — Unscaled copy number variation data.
  - `rna.csv` — Unscaled RNA-Seq expression matrix.

---

## How to Prepare the Data

1. **Download the data**  
   Head to the Pan-Cancer Atlas page and download all the supplemental data.  
   The provided files have already gone through some preprocessing, such as reducing batch effects.  
   In this project, we primarily use:
   - RNA expression data
   - Copy number data (ABSOLUTE-annotated seg file)
   - Clinical outcome data (CDR)

   You are welcome to download additional modalities if you'd like to experiment further, especially in the data exploration and preprocessing notebooks.

2. **Place the TCGA data** in the appropriate `data/` directory inside this folder.

3. **Run preprocessing scripts** to prepare the dataset.

4. **Execute notebooks or Python scripts** to train models and evaluate results.

---

## Attribution

Parts of the preprocessing and data handling code in this folder were adapted from  
[`AndreCNF/tcga-cancer-classification`](https://github.com/AndreCNF/tcga-cancer-classification),  
with modifications to integrate LDA, QDA, and Conformal Prediction methods.

---

## Usage

1. Place the TCGA data in the appropriate `data/` directory inside this folder.
2. Run preprocessing scripts to prepare the dataset.
3. Execute notebooks or Python scripts to train models and evaluate results.

---

## Related Work

These experiments are part of the MSc Individual Research Report on evaluating LDA and QDA with Conformal Prediction.

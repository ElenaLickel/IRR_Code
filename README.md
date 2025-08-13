# Individual Research Report Code

**Imperial College London MSc Individual Research Report**  
*Conformal Prediction with LDA & QDA Classifiers*

---

This repository contains the full codebase for evaluating and implementing **Conformal Prediction analysis** as part of my MSc Individual Research Report.  
The project explores the performance of **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)** classifiers under various data conditions, including synthetic simulations and real RNA-Seq datasets.

---

## Repository Structure

```
IRR_Code/
│
├── TCGA-Pancancer/    # TCGA Pan-Cancer experiments: notebooks & code
├── simulation/        # Synthetic data simulation experiments for LDA & QDA
├── .gitignore         # Files/folders excluded from Git tracking
└── README.md          # This document
```

---

## Key Features

- **Synthetic simulation framework:** Generate high-dimensional classification tasks for robust benchmarking.
- **Conformal Prediction (MapieClassifier):** Implementation with multiple scoring rules (Brier, log-loss, spherical).
- **Comprehensive Evaluation:** Coverage, sharpness, and probabilistic calibration for LDA/QDA.
- **TCGA Pan-Cancer Pipeline:** RNA-Seq multi-class classification with preprocessing, dimensionality reduction, and CP evaluation.

---

## Directory Layout

### Main Notebooks (Simulation)
- `Data_simulation.ipynb` — Generates synthetic datasets for LDA/QDA experiments, including custom covariance structures and class arrangements.
- `Data_simulation_new_plots.ipynb` — Extended version of the simulation notebook with additional visualisations and updated plotting functions.
- `Data_simularion_new_data.ipynb` — Different synthetic data generator with alternative parameter settings and data output for downstream experiments.
- `Evaluate_Classifier.ipynb` — Compares classifier performance (LDA, QDA, and variants with Conformal Prediction) across multiple metrics and experimental setups.
- `Conformal_prediction-1.ipynb` — Initial conformal prediction implementation applied to synthetic data with coverage, sharpness, and scoring rule evaluation.
- `Conformal_prediction_new_data.ipynb` — Conformal prediction implementation applied to differently generated synthetic data with coverage, sharpness, and scoring rule evaluation.

### Python Scripts (Simulation)
- `simulation_utils.py` — Helper functions for synthetic data generation, covariance blending, and class layout configuration.
- `scoring_utils.py` — Utility functions for computing proper scoring rules (Brier, log-loss, spherical) and other performance metrics.

---

## Directory Layout — TCGA-Pancancer

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

## ⚙️ Requirements

- **Python** `3.9+`
- Install dependencies via `requirements.txt`:
  - `numpy`, `pandas`, `scikit-learn`, `mapie`, `matplotlib`, `seaborn`, etc.

---

## 📚 How to Get Started

1. Clone the repo:
   ```bash
   git clone https://github.com/ElenaLickel/IRR_Code.git
   cd IRR_Code
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Explore the experiments:
   - For synthetic simulations: see [`simulation/`](simulation/)
   - For TCGA Pan-Cancer: see [`TCGA-Pancancer/`](TCGA-Pancancer/)

---

## 📊 Data Preparation (TCGA)

1. **Download the data**  
   Head to the Pan-Cancer Atlas page and download all the supplemental data.  
   The provided files have already gone through some preprocessing, such as reducing batch effects.  
   In this project, we primarily use:
   - RNA expression data
   - Copy number data (ABSOLUTE-annotated seg file)
   - Clinical outcome data (CDR)

   You are welcome to download additional modalities if you'd like to experiment further, especially in the data exploration and preprocessing notebooks.

2. **Place the TCGA data** in the appropriate `data/` directory inside the `TCGA-Pancancer/` folder.

3. **Run preprocessing scripts** to prepare the dataset.

4. **Execute notebooks or Python scripts** to train models and evaluate results.
5. 
---

## Citation

Please cite this repository or related publications if you use this code for your research!

---

## Contact

For questions or collaborations, please contact:  
**Elena Lickel**  
[github.com/ElenaLickel](https://github.com/ElenaLickel)

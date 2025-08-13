# Simulation Experiments

This folder contains code for **synthetic data experiments** designed to compare  
**Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)** classifiers  
under varying levels of class heterogeneity and dataset complexity.

---

## Contents

- **Data Generation:** Custom Python scripts to create synthetic datasets with configurable:
  - Number of classes
  - Covariance blending parameter (η)
  - Dimensionality
  - Sample size
- **Model Evaluation:** LDA and QDA models evaluated with Conformal Prediction.
- **Metrics:** Coverage, sharpness, and proper scoring rules (log-loss, Brier, spherical).
- **Simulation Framework:** Automated loop to run multiple experiments across seeds and parameter ranges.

---

## Directory Layout

### Main Notebooks
- `Data_simulation.ipynb` — Generates synthetic datasets for LDA/QDA experiments, including custom covariance structures and class arrangements.
- `Data_simulation_new_plots.ipynb` — Extended version of the simulation notebook with additional visualisations and updated plotting functions.
- `Data_simularion_new_data.ipynb` — Different synthetic data generator with alternative parameter settings and data output for downstream experiments.
- `Evaluate_Classifier.ipynb` — Compares classifier performance (LDA, QDA, and variants with Conformal Prediction) across multiple metrics and experimental setups.
- `Conformal_prediction-1.ipynb` — Initial conformal prediction implementation applied to synthetic data with coverage, sharpness, and scoring rule evaluation.
- `Conformal_prediction_new_data.ipynb` — Conformal prediction implementation applied to differently generated synthetic data with coverage, sharpness, and scoring rule evaluation.

### Python Scripts
- `simulation_utils.py` — Helper functions for synthetic data generation, covariance blending, and class layout configuration.
- `scoring_utils.py` — Utility functions for computing proper scoring rules (Brier, log-loss, spherical) and other performance metrics.

---

## Usage

1. Adjust simulation parameters in the configuration section of the scripts.
2. Run the simulation code to generate datasets and train models.
3. Collect results for plotting and statistical comparison.

---

## Purpose

The synthetic experiments provide a controlled environment to understand  
when and why LDA or QDA performs better, and how Conformal Prediction behaves  
with changes in dataset size, dimensionality, and class heterogeneity.

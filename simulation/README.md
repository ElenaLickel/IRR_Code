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

## Usage

1. Adjust simulation parameters in the configuration section of the scripts.
2. Run the simulation code to generate datasets and train models.
3. Collect results for plotting and statistical comparison.

---

## Purpose

The synthetic experiments provide a controlled environment to understand  
when and why LDA or QDA performs better, and how Conformal Prediction behaves  
with changes in dataset size, dimensionality, and class heterogeneity.

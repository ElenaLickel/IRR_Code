# IRR_Code
This repository contains the full codebase for evaluating and implementing Conformal Prediction analysis as part of my MSc Individual Research Report at Imperial College London.
The project investigates the performance of Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) classifiers under various data conditions, both on synthetic simulations and on real-world multi-class cancer classification using the TCGA Pan-Cancer RNA-Seq dataset.

## Repository Structure

```
IRR_Code/
│
├── TCGA-Pancancer/    # TCGA Pan-Cancer experiments: notebooks & code
├── simulation/        # Synthetic data simulation experiments for LDA & QDA
├── .gitignore         # Files/folders excluded from Git tracking
└── README.md          # This document
```

## Key Features
- Synthetic simulation framework for generating high-dimensional classification tasks.
- Implementation of Conformal Prediction (MapieClassifier) with multiple scoring rules (Brier, log-loss, spherical).
- Evaluation of coverage, sharpness, and probabilistic calibration for LDA and QDA.
- TCGA Pan-Cancer RNA-Seq multi-class classification pipeline with preprocessing, dimensionality reduction, and CP evaluation.

## Requirements
Python 3.9+
Dependencies listed in requirements.txt (NumPy, pandas, scikit-learn, MAPIE, matplotlib, seaborn, etc.).

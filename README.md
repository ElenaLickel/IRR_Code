# IRR_Code

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

## Requirements

- **Python** `3.9+`
- Install dependencies:
  - `numpy`, `pandas`, `scikit-learn`, `mapie`, `matplotlib`, `seaborn`, etc.

---

## Citation

Please cite this repository or related publications if you use this code for your research!

---

## Contact

For questions or collaborations, please contact:  
**Elena Lickel**  
[github.com/ElenaLickel](https://github.com/ElenaLickel)

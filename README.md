# Car Price Prediction
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikitlearn)
![PyCaret](https://img.shields.io/badge/PyCaret-AutoML-1f77b4?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red?style=flat-square)
![LIME](https://img.shields.io/badge/LIME-Local_Interpretability-green?style=flat-square)
![Databricks](https://img.shields.io/badge/Databricks-Cloud_Ecosystem-EF3E42?style=flat-square&logo=databricks)

### A regression study on a heterogeneous automotive dataset  
_From cleaning to model interpretation._

> “Predicting car prices is not a regression problem.  
> It’s a stress test.”

---

## Project Overview

This project explores multiple machine-learning approaches for vehicle price prediction using a highly heterogeneous automotive dataset spanning from economy vehicles to multi-million-dollar hypercars.

The goal is not only to predict vehicle prices, but also to:

- compare different model families
- analyze behavior under extreme variance
- evaluate log vs raw USD targets
- apply explainable AI techniques (SHAP & LIME)
- study the limits of predictive modeling on sparse automotive data

The project was developed in Databricks using sklearn, PyCaret, MLflow, SHAP, and LIME.

---

# Project Structure

```text
├── notebooks/
│   ├── 01_load_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_linear_regression.ipynb
│   ├── 06_tree_models.ipynb
│   ├── 07_pycaret.ipynb
│   ├── 08_final_comparison.ipynb
│   └── 09_shap_lime.ipynb
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── models/
│
├── presentation/
│   └── car-price-prediction.pdf
│
└── README.md
```


# Dataset

- Source: Kaggle Cars Dataset 2025
- ~1,218 vehicles
- 11 raw columns
- Price range: ~$2K → $18M
- Heavy right-skewed distribution
- Nearly one unique model per row

### Main Challenges

- Extreme variance heterogeneity
- Sparse configurations
- Luxury outliers
- Missing structural variables
- Mixed EV / Hybrid / ICE powertrains

---

# Methodology

The project follows a complete end-to-end ML pipeline:

1. Load raw CSV
2. Cleaning & parsing
3. Exploratory Data Analysis
4. Feature engineering
5. Preprocessing pipeline
6. Linear & tree-based models
7. AutoML benchmarking with PyCaret
8. Explainability with SHAP & LIME
9. Experiment tracking with MLflow

---

# Feature Engineering

The feature engineering strategy focuses on:

- performance signals over raw specs
- EV / ICE structural differences
- market segmentation
- premium / luxury positioning
- variance stabilization using log(price)

Key engineered features include:

- performance_score
- horsepower classes
- acceleration classes
- market segment categories
- premium / luxury brand indicators
- EV / Hybrid / ICE flags

---

# Models Evaluated

## Linear Models
- Linear Regression
- Ridge
- Lasso
- Bayesian Ridge
- Huber Regressor

## Tree Models
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- LightGBM

## AutoML
- PyCaret benchmarking
- PyCaret Blender ensemble

---

# Dual Target Strategy

Two prediction targets were evaluated throughout the project:

| Target | Purpose |
|---|---|
| `log_price` | Statistical stability & proportional learning |
| `price_usd` | Direct business interpretability |

This dual-target approach revealed major differences in model behavior under variance heterogeneity.

---

# Key Results

## LOG Scale
- Random Forest achieved the best overall performance
- R² ≈ 0.92
- Strong generalization across market segments

## USD Scale
- Raw USD prediction proved significantly harder
- Extreme-value vehicles dominated RMSE behavior
- PyCaret Blender achieved the strongest direct-USD performance

### Key Finding

Blended ensembles partially recover predictive stability under extreme variance heterogeneity.

---

# Explainability

The project integrates both global and local interpretability techniques.

## SHAP
Used for:
- global feature importance
- feature impact analysis
- local waterfall explanations

## LIME
Used for:
- segment-level local explanations
- budget vs mid-range vs luxury interpretation

### Main Explainability Insight

Performance-related variables such as:
- horsepower
- torque
- top speed

consistently dominate vehicle pricing behavior.

---

# MLflow Tracking

MLflow was used for:

- experiment tracking
- metric logging
- parameter management
- model versioning
- reproducible workflows

All experiments were logged under standardized preprocessing and evaluation conditions.

---

# ⚠️ Project Limitations

The project intentionally highlights several structural limitations:

- ~1 unique vehicle configuration per row
- Hypercar outliers dominate variance-sensitive metrics
- Tree models easily overfit sparse data
- Missing critical market variables:
  - mileage
  - condition
  - location

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- PyCaret
- SHAP
- LIME
- MLflow
- Databricks
- Matplotlib

---

# Closing Thought

> “The right model on the wrong dataset is still the wrong answer.”

---

# Author

**Maria Petralia**  
MSIT — AI Enhanced Productivity Project  
May 2026

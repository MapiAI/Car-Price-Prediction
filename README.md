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
│   ├── 01_Data_Gathering.ipynb
│   ├── 02_Cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_1_Preprocessing_LinearRegression.ipynb
│   ├── 05_2_Preprocessing_DecisionTree.ipynb
│   ├── 06_1_Training_LinearRegression.ipynb
│   ├── 06_2_Training_DecisionTree.ipynb
│   ├── 07_PyCaret.ipynb
│   └── 08_Evaluation.ipynb
│
├── data/
│   ├── raw/
│   │   └── original Kaggle dataset
│   │
│   ├── processed/
│   │   └── cleaned & transformed datasets
│   │
│   ├── features/
│   │   └── feature-engineered datasets
│   │
│   ├── metrics/
│   │   └── exported model metrics & evaluation CSVs
│   │
│   └── train_test/
│       └── saved train/test splits & serialized datasets
│
├── models/
│   └── trained .pkl models
│
│
├── presentation/
│   └── car-price-prediction-maria-petralia.pdf
│
└── README.md
```

## 📌 Note

Model artifacts and intermediate files were intentionally excluded from the repository to keep the project lightweight and focused on reproducible workflows and analysis.


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

# Project Limitations

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

## Configuration

The project uses a local configuration file for dataset, model, and artifact paths.

Before running the notebooks, update the paths according to your local environment or Databricks workspace structure.

---

## Environment Notes

Most notebooks were developed and executed in Databricks.

PyCaret-related notebooks were executed locally in VS Code using:

- Python 3.10
- PyCaret
- Compatible NumPy / Pandas versions

This setup was required to ensure compatibility with the PyCaret ecosystem.

---

# Closing Thought

> “The right model on the wrong dataset is still the wrong answer.”

---

# Author

**Maria Petralia**  
MSIT — AI Enhanced Productivity Project  
May 2026

# Energy Consumption Predictor

A production-grade machine learning system that predicts building heating load
using Multiple Linear Regression with full statistical analysis including
R², F-statistic, p-values, VIF and residual diagnostics, served via a REST API.

---

## Project Overview

This project demonstrates an end-to-end ML engineering pipeline built on the
UCI Energy Efficiency dataset. It covers the full stack from statistical
feature selection through to a containerised, tested, and automatically
verified API service.

The project addresses a real engineering problem — predicting how much energy
a building needs for heating based on its physical characteristics, enabling
architects and engineers to design more energy efficient buildings.

---

## Results

| Metric | Score |
|---|---|
| Cross-validation R² | 0.9122 ± 0.0028 |
| Test set R² | 0.9086 |
| Test MAE | 2.31 kWh |
| Test RMSE | 3.09 kWh |
| Features used | 5 (after dropping 3) |
| Actual vs Predicted correlation | 0.9536 |

---

## Statistical Analysis

| Feature | Coefficient | p-value | Decision |
|---|---|---|---|
| Height | 9.658 | 0.000 | Keep |
| Glazing_Area | 2.714 | 0.000 | Keep |
| Wall_Area | 1.591 | 0.000 | Keep |
| Compactness | -1.394 | 0.000 | Keep |
| Glazing_Distribution | 0.336 | 0.004 | Keep |
| Orientation | -0.023 | 0.809 | Dropped |
| Roof_Area | — | — | Dropped (VIF=21) |
| Surface_Area | — | — | Dropped (VIF=24) |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Machine Learning | scikit-learn, Linear Regression |
| Statistical Analysis | statsmodels, OLS, VIF |
| Data Analysis | pandas, numpy, seaborn, matplotlib |
| Experiment Tracking | MLflow |
| API | FastAPI, uvicorn, pydantic |
| Testing | pytest |
| Containerisation | Docker |
| CI/CD | GitHub Actions |

---

## Project Structure

```
mlr-energy-predictor/
├── data/
├── notebooks/
│   └── 01_eda_and_regression.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
├── api/
│   └── main.py
├── models/
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Why Multiple Linear Regression?

Multiple Linear Regression was chosen because the relationship between
building characteristics and heating load is fundamentally linear —
larger buildings need more heating, more glass means more heat loss.
MLR provides full statistical interpretability through coefficients,
p-values and confidence intervals that more complex models cannot offer.

---

## Feature Selection Process

Three features were removed before training:

Roof_Area and Surface_Area were dropped due to severe multicollinearity
(VIF above 20) — they were almost perfectly correlated with Height and
Compactness respectively, carrying no additional information.

Orientation was dropped because its p-value of 0.809 confirmed it has
no statistically significant relationship with heating load — consistent
with the near-zero correlation of -0.003 observed in EDA.

---

## Regression Diagnostics

| Diagnostic | Result | Assessment |
|---|---|---|
| Residual mean | 0.289 | Near zero — no systematic bias |
| Residual-prediction correlation | -0.001 | Random scatter — linearity satisfied |
| Shapiro-Wilk p-value | 0.001 | Slight deviation — acceptable |
| Skewness | -0.027 | Near zero — symmetric errors |
| Durbin-Watson | 0.597 | Autocorrelation from repeated building types |

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/engrtayyab596-wq/mlr-energy-predictor.git
cd mlr-energy-predictor
```

### 2. Create conda environment

```bash
conda create -n ML_P python=3.11
conda activate ML_P
pip install -r requirements.txt
```

### 3. Add the dataset

Download the Energy Efficiency dataset from UCI and place it at:

```
data/energy_data.xlsx
```

### 4. Generate the model

Open and run all cells in notebooks/01_eda_and_regression.ipynb.
This trains the pipeline and saves it to models/energy_pipeline.pkl.

### 5. Start the API

```bash
uvicorn api.main:app --reload
```

### 6. Open the interactive docs

```
http://127.0.0.1:8000/docs
```

---

## How to Run with Docker

```bash
docker build -t energy-predictor .
docker run -p 8000:8000 energy-predictor
```

---

## How to Run Tests

```bash
pytest tests/ -v
```

---

## Experiment Tracking

This project uses MLflow to track two experiment runs:

Run 1 — Feature Selection: documents which features were dropped,
OLS results with 6 vs 5 features, VIF values before and after scaling.

Run 2 — Final Model: logs all pipeline parameters, R², MAE, RMSE,
CV scores, feature coefficients and diagnostic results.

To view experiments locally:

```bash
cd notebooks
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open: http://127.0.0.1:5000

---

## API Endpoints

### GET /health

```json
{
  "status": "ok",
  "model": "loaded",
  "description": "Energy Consumption Predictor API"
}
```

### POST /predict

Example request:

```json
{
  "Compactness": 0.98,
  "Wall_Area": 294.0,
  "Height": 7.0,
  "Glazing_Area": 0.0,
  "Glazing_Distribution": 0.0
}
```

Example response:

```json
{
  "predicted_heating_load": 15.55,
  "unit": "kWh",
  "efficiency_rating": "High Efficiency"
}
```

---

## Efficiency Rating

| Rating | Heating Load |
|---|---|
| High Efficiency | Below 15 kWh |
| Medium Efficiency | 15 to 30 kWh |
| Low Efficiency | Above 30 kWh |

Thresholds based on Q1 (12.99) and Q3 (31.67) of the training data
distribution — bottom 25%, middle 50%, and top 25% of energy consumers.

---

## Dataset

UCI Energy Efficiency Dataset

- 768 buildings
- 8 original features (5 used after feature selection)
- Target: Heating Load (kWh), range 6.01 to 43.10
- No missing values
- All numerical features

---

## Key Design Decisions

- OLS statistical analysis before ML pipeline — evidence based feature selection
- VIF analysis to detect and resolve multicollinearity
- p-value threshold of 0.05 used to drop statistically insignificant features
- StandardScaler inside Pipeline prevents data leakage during cross validation
- Residual diagnostics confirm linear regression assumptions are satisfied
- joblib used for model serialisation — Docker compatible
- MLflow tracks both feature selection decisions and final model metrics
- Efficiency ratings based on actual data distribution quartiles

---

## Author

Tayyab
ML/AI Engineering

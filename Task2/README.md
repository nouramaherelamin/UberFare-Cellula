# Uber Fare Prediction — Task 2

## Cellula Technologies | Machine Learning Track

This submission continues the Uber Fare Prediction project from Task 1 (EDA). It converts the cleaned Task 1 data into a leakage-safe regression workflow and saves a reusable scikit-learn pipeline.

### Included deliverables

- `Task2_Uber_Fare_Preprocessing_and_Regression.ipynb` — complete Task 2 notebook.
- `final_regression_pipeline.joblib` — single fitted `sklearn.pipeline.Pipeline` containing preprocessing, feature selection and the tuned regression model.
- `Task2_Uber_Fare_Presentation.html` — standalone HTML presentation.
- `data/uber_fare_eda_clean.csv` — cleaned Task 1 data used by the notebook.
- `figures/` — generated diagnostic/presentation figures.
- `requirements.txt` — Python dependencies.

### Dataset

The Task 2 notebook uses the cleaned Task 1 dataset containing 49,991 positive-fare ride records and 26 columns.

### Preprocessing decisions

- Identifier/name fields are excluded from modeling.
- The repeated `key` field is excluded rather than used as a predictive feature.
- Missing-value imputers are kept inside the pipeline as defensive preprocessing.
- Low-cardinality categorical features are one-hot encoded.
- Numeric features use Yeo-Johnson transformation with standardization.
- `hour_sin` and `hour_cos` are engineered from the existing hour feature.
- `SelectKBest(f_regression)` is placed inside the pipeline so feature selection is learned within each CV fold.
- Extreme observations are investigated rather than automatically deleted.

### Modeling

- Baseline: Ridge Regression.
- Final tuned model: HistGradientBoostingRegressor.
- Cross-validation: 5-fold shuffled K-Fold on the training set.
- Hyperparameter tuning: RandomizedSearchCV.
- Final test split: 20%, random_state=42.

### Final held-out test results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline Ridge | 4.0051 | 6.9963 | 0.4955 |
| Tuned HistGradientBoosting | 1.9893 | 4.2378 | 0.8149 |

Tuned search CV RMSE: **4.5741**

### How to run

```bash
pip install -r requirements.txt
jupyter notebook Task2_Uber_Fare_Preprocessing_and_Regression.ipynb
```

The notebook expects the CSV at:

```text
data/uber_fare_eda_clean.csv
```

### Presentation

Open `Task2_Uber_Fare_Presentation.html` directly in a browser. Use the arrow buttons or keyboard arrow keys to navigate.

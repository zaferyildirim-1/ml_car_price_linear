
# LINEAR REGRESSION MACHINE LEARNING MODEL for CAR PRICE PREDICTION

Predicting Used Car Prices with Linear Models

## Overview

This repository explores various linear regression techniques to predict car prices. We start with a baseline model (`lm1`) and progressively introduce:

- **Outlier removal**  
- **Regularization** (Ridge, Lasso, ElasticNet)  
- **Feature selection**

Our goal is to balance predictive performance (R²) against error metrics (MAE, RMSE, MAPE).

## Data & Preprocessing

1. **Original dataset**  
   - Baseline (`lm1`):  
     - Remove price outliers via 1.5 × IQR by model.  
2. **Filtered dataset**  
   - `linear_m2`:  
     - On top of `lm1` filtering, remove global price outliers outside [7 500 €, 70 000 €] in the first notebook 
   - **New filtering**:  
     - Similarly, remove global price outliers outside [3 500 €, 40 000 €] in the modified notebook

> **Note:** raw data is included here : sample_model_data.csv).

## Models & Results

| Model        | R²     | MAE      | RMSE     | MAPE   |
|--------------|-------:|---------:|---------:|-------:|
| **Original Dataset Metrics** |        |          |          |        |
| `lm1`        | 0.905  | 2435.6   | 3483.6   | 0.193  |
| `linear_m2`  | 0.903  | 2287.8   | 3215.4   | 0.123  |
| `ridge_m`    | 0.903  | 2287.8   | 3215.4   | 0.123  |
| `lasso_m`    | 0.903  | 2287.8   | 3215.4   | 0.123  |
| `elastic_m`  | 0.903  | 2288.7   | 3215.8   | 0.123  |
| `final_m`    | 0.866  | 2703.8   | 4254.3   | 0.205  |

| Model        | R²     | MAE      | RMSE     | MAPE   |
|--------------|-------:|---------:|---------:|-------:|
| **New Dataset Metrics**      |        |          |          |        |
| `lm1`        | 0.905  | 2435.6   | 3483.6   | 0.193  |
| `linear_m2`  | 0.896  | 1991.4   | 2654.3   | 0.138  |
| `ridge_m`    | 0.896  | 1991.4   | 2654.3   | 0.138  |
| `lasso_m`    | 0.896  | 1991.4   | 2654.3   | 0.138  |
| `elastic_m`  | 0.896  | 1992.7   | 2655.3   | 0.138  |
| `final_m`    | 0.866  | 2703.8   | 4254.3   | 0.205  |

### Model Descriptions

- **lm1**  
  - Model-by-model 1.5 × IQR outlier removal
- **linear_m2**  
  - lm1 filtering + global price filter     (!!!! 7 500 €–70 000 € on original;       !!!! 3 500 €–40 000 € on new/modified)
  - 
- **ridge_m / lasso_m / elastic_m**  
  - Regularized linear models tuned via cross-validation
- **final_m**  
  - A hand-crafted “final” pipeline (performed worst)

## How to Run

check requirements.txt file for relevant modules and libraries etc.






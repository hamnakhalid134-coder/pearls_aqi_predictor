# Pearls AQI Predictor

An end-to-end Machine Learning and MLOps project developed during the **10Pearls Data Science Internship** to forecast Lahore's US Air Quality Index (AQI) for the next **24, 48, and 72 hours**.

## Project Overview

Pearls AQI Predictor collects air-quality and weather data, performs feature engineering, stores features in Hopsworks, trains and evaluates multiple machine-learning models, registers the selected models, automates data pipelines using GitHub Actions, and presents forecasts through a professional Streamlit dashboard.

The repository also includes a FastAPI prediction service, SHAP-based model explainability, and a failure-safe prediction snapshot.

## Live Application

[Open Pearls AQI Predictor](https://pearls-aqi-predictor-js3vlytrcyp3cqqb8t9zbi.streamlit.app/)

## Data

- **Location:** Lahore, Pakistan
- **Coordinates:** 31.5204, 74.3587
- **Data Source:** Open-Meteo / CAMS
- **Final Model-Ready Rows:** 26,160
- **Model Features:** 51
- **Forecast Targets:**
  - AQI +24 hours
  - AQI +48 hours
  - AQI +72 hours

## Models

| Forecast Horizon | Selected Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| +24 h | XGBoost | 17.076 | 23.580 | 0.627 |
| +48 h | Ridge Regression | 23.323 | 31.637 | 0.329 |
| +72 h | Ridge Regression | 24.484 | 32.868 | 0.277 |

All selected models outperformed the persistence baseline in MAE.

## MLOps Architecture

The project includes:

- Hopsworks Feature Store
- Hopsworks Model Registry
- GitHub Actions hourly feature pipeline
- GitHub Actions daily model-training workflow
- Streamlit dashboard
- FastAPI prediction service
- SHAP model explainability
- JSON fallback prediction snapshot
- GitHub Codespaces development environment

## Hopsworks Integration

Two main feature groups are used:

- `aqi_historical_features`
- `aqi_live_features`

The selected production models were registered in the Hopsworks Model Registry:

- `aqi_24h_xgboost`
- `aqi_48h_ridge`
- `aqi_72h_ridge`

Hopsworks is used as the primary Feature Store and Model Registry for the project.

## FastAPI Prediction Service

The repository includes `api.py`, which provides the following endpoints:

- `/` — service information
- `/health` — API health check
- `/predict` — latest AQI predictions
- `/docs` — Swagger API documentation

The FastAPI service was successfully tested using Uvicorn in GitHub Codespaces.

Example prediction response:

```json
{
  "city": "Lahore",
  "current_aqi": 172,
  "forecast": {
    "24_hours": 169,
    "48_hours": 161,
    "72_hours": 149
  }
}
```

The FastAPI service is included in the repository but is not separately deployed as a public API endpoint.

## Failure-Safe Fallback

The latest repository version includes a fallback mechanism for cloud-backend failures.

If Hopsworks is temporarily unavailable, the application can use the last successful forecast stored in:

`latest_prediction_snapshot.json`

This allows the prediction interface to remain usable without depending completely on Hopsworks availability.

The snapshot contains:

- Current AQI
- +24 h forecast
- +48 h forecast
- +72 h forecast
- Atmospheric variables
- Forecast timestamp
- Selected model information

## Automation

### Hourly Feature Pipeline

Workflow:

`.github/workflows/hourly_features.yml`

The hourly workflow updates live AQI features using GitHub Actions.

### Daily Model Training

Workflow:

`.github/workflows/daily_training.yml`

A daily model-training workflow is configured to execute `train_and_register.py`.

A valid `HOPSWORKS_API_KEY` must be available through GitHub Actions Secrets for Hopsworks-dependent cloud workflows to run successfully.

## Model Explainability

SHAP is used to explain the +24-hour XGBoost forecast.

The Streamlit dashboard displays the most influential features contributing to the current AQI prediction when the live XGBoost model is available.

## Dashboard Features

The Streamlit dashboard includes:

- Current US AQI gauge
- US EPA AQI classification
- Health-condition interpretation
- Current atmospheric conditions
- +24 h, +48 h, and +72 h forecasts
- 72-hour forecast outlook
- Model-performance comparison
- MAE, RMSE, and R² visualizations
- Recent AQI analysis
- Pollutant concentration charts
- SHAP model explainability
- GitHub and project-resource links

## Technical Report

[View Technical Project Report](report/10Pearls_AQI_Predictor_Project%20Report.pdf)

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Hopsworks
- GitHub Actions
- GitHub Codespaces
- Streamlit
- FastAPI
- Uvicorn
- Plotly
- SHAP
- Open-Meteo / CAMS
- Git

## Repository Structure

```text
pearls_aqi_predictor/
│
├── app.py
├── api.py
├── predict_aqi.py
├── hourly_feature_pipeline.py
├── train_and_register.py
├── feature_engineering.py
├── create_feature_views.py
├── latest_prediction_snapshot.json
├── requirements.txt
├── README.md
│
├── .github/
│   └── workflows/
│       ├── hourly_features.yml
│       └── daily_training.yml
│
└── report/
    └── 10Pearls_AQI_Predictor_Project Report.pdf
```

## Author

**Hamna Khalid**  
Data Science Intern — 10Pearls  
Electrical Engineer

## Repository

[GitHub Repository](https://github.com/hamnakhalid134-coder/pearls_aqi_predictor)
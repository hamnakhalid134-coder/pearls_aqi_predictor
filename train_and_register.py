import os
import joblib
import numpy as np
import pandas as pd
import hopsworks

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ------------------------------------------------
# 1. Connect to Hopsworks
# ------------------------------------------------

project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    project="pearls_aqi_hk26",
    port=443,
    engine="python"
)

fs = project.get_feature_store()
mr = project.get_model_registry()


# ------------------------------------------------
# 2. Fixed chronological split
#    Same split as our final local experiment
# ------------------------------------------------

TRAIN_START = "2023-09-03 00:00:00"
TRAIN_END   = "2026-01-18 23:00:00"

# 72-hour purge gap:
# 2026-01-19, 20, 21

TEST_START  = "2026-01-22 00:00:00"
TEST_END    = "2026-08-27 23:00:00"


# ------------------------------------------------
# 3. Helper functions
# ------------------------------------------------

def clean_x(df):

    # Drop identifiers if returned by Hopsworks
    drop_cols = [
        col for col in ["city", "date_time"]
        if col in df.columns
    ]

    return df.drop(
        columns=drop_cols,
        errors="ignore"
    )


def get_y(y, target):

    if isinstance(y, pd.DataFrame):
        return y[target]

    if isinstance(y, pd.Series):
        return y

    return np.ravel(y)


def evaluate(y_true, predictions):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return mae, rmse, r2


# ------------------------------------------------
# 4. Models to train
# ------------------------------------------------

configs = [

    {
        "target": "aqi_24h",
        "view": "aqi_24h_view",
        "registry_name": "aqi_24h_xgboost",
        "file": "aqi_24h_xgboost.pkl",
        "model_type": "xgboost"
    },

    {
        "target": "aqi_48h",
        "view": "aqi_48h_view",
        "registry_name": "aqi_48h_ridge",
        "file": "aqi_48h_ridge.pkl",
        "model_type": "ridge"
    },

    {
        "target": "aqi_72h",
        "view": "aqi_72h_view",
        "registry_name": "aqi_72h_ridge",
        "file": "aqi_72h_ridge.pkl",
        "model_type": "ridge"
    }
]


# ------------------------------------------------
# 5. Train one independent model per horizon
# ------------------------------------------------

final_results = []


for config in configs:

    target = config["target"]

    print("\n====================================")
    print("TRAINING:", target)
    print("====================================")


    # Get corresponding Feature View
    fv = fs.get_feature_view(
        name=config["view"],
        version=1
    )


    # Fetch training/testing data FROM HOPSWORKS
    X_train, X_test, y_train, y_test = (
        fv.train_test_split(

            train_start=TRAIN_START,
            train_end=TRAIN_END,

            test_start=TEST_START,
            test_end=TEST_END
        )
    )


    X_train = clean_x(X_train)
    X_test = clean_x(X_test)

    y_train = get_y(y_train, target)
    y_test = get_y(y_test, target)


    print("Training Rows:", len(X_train))
    print("Testing Rows:", len(X_test))
    print("Features:", len(X_train.columns))


    # --------------------------------------------
    # Persistence baseline
    # --------------------------------------------

    baseline_predictions = X_test["aqi"]

    baseline_mae, baseline_rmse, baseline_r2 = (
        evaluate(
            y_test,
            baseline_predictions
        )
    )


    # --------------------------------------------
    # Train selected model
    # --------------------------------------------

    if config["model_type"] == "xgboost":

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )


    else:

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)
        )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    mae, rmse, r2 = evaluate(
        y_test,
        predictions
    )


    print("\nPersistence Baseline")
    print("MAE:", round(baseline_mae, 3))
    print("RMSE:", round(baseline_rmse, 3))
    print("R2:", round(baseline_r2, 3))

    print("\nSelected ML Model")
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))


    # --------------------------------------------
    # Save model locally
    # --------------------------------------------

    joblib.dump(
        model,
        config["file"]
    )


    # --------------------------------------------
    # Register in Hopsworks Model Registry
    # --------------------------------------------

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "baseline_mae": float(baseline_mae),
        "baseline_rmse": float(baseline_rmse),
        "baseline_r2": float(baseline_r2)
    }


    registry_model = mr.sklearn.create_model(

        name=config["registry_name"],

        metrics=metrics,

        description=(
            f"Direct {target} AQI forecasting model "
            "for Lahore. Trained using Hopsworks "
            "Feature Store historical features."
        ),

        input_example=X_train.head(1),

        feature_view=fv
    )


    registry_model.save(
        config["file"]
    )


    print(
        "\nRegistered:",
        config["registry_name"],
        "Version:",
        registry_model.version
    )


    final_results.append({

        "Target": target,

        "Model": config["registry_name"],

        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,

        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
        "Baseline_R2": baseline_r2
    })


# ------------------------------------------------
# 6. Final cloud results
# ------------------------------------------------

results = pd.DataFrame(
    final_results
)

results.to_csv(
    "Hopsworks_Model_Results.csv",
    index=False
)


print("\n\n====================================")
print("HOPSWORKS MODEL TRAINING COMPLETE")
print("====================================")

print(
    results.round(3).to_string(
        index=False
    )
)

print(
    "\nAll 3 models registered successfully!"
)
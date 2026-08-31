import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# --------------------------------
# 1. Load data
# --------------------------------

data = pd.read_csv("ML_Ready_Data.csv")

data["Date_Time"] = pd.to_datetime(data["Date_Time"])

data = data.sort_values(
    "Date_Time"
).reset_index(drop=True)


# --------------------------------
# 2. Targets


targets = [
    "AQI_24h",
    "AQI_48h",
    "AQI_72h"
]

# 3. Features

features = [
    col for col in data.columns
    if col not in [
        "Date_Time",
        "AQI_24h",
        "AQI_48h",
        "AQI_72h"
    ]
]


print("\nNumber of Features:")
print(len(features))


# --------------------------------
# 4. Chronological 80/20 split
# --------------------------------

split_index = int(len(data) * 0.80)

test_start_time = data.loc[
    split_index,
    "Date_Time"
]


# 72-hour purge gap
train_end_time = (
    test_start_time
    - pd.Timedelta(hours=72)
)


train = data[
    data["Date_Time"] < train_end_time
].copy()

test = data[
    data["Date_Time"] >= test_start_time
].copy()


print("\nTIME SPLIT")

print("\nTraining Rows:")
print(len(train))

print("\nTesting Rows:")
print(len(test))

print("\nTraining Period:")
print(
    train["Date_Time"].min(),
    "to",
    train["Date_Time"].max()
)

print("\nTesting Period:")
print(
    test["Date_Time"].min(),
    "to",
    test["Date_Time"].max()
)


# 5. X data

X_train = train[features]
X_test = test[features]


# --------------------------------
# 6. Results
# --------------------------------

results = []


# --------------------------------
# 7. Train separate model
#    for each horizon
# --------------------------------

for target in targets:

    print("\n==============================")
    print("FORECAST:", target)
    print("==============================")

    y_train = train[target]
    y_test = test[target]


    # ----------------------------
    # A. Persistence Baseline
    # ----------------------------

    baseline_pred = test["AQI"]

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_pred
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_pred
        )
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_pred
    )


    # ----------------------------
    # B. Ridge Regression
    # ----------------------------

    ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    ridge.fit(
        X_train,
        y_train
    )

    ridge_pred = ridge.predict(
        X_test
    )

    ridge_mae = mean_absolute_error(
        y_test,
        ridge_pred
    )

    ridge_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            ridge_pred
        )
    )

    ridge_r2 = r2_score(
        y_test,
        ridge_pred
    )


    # Save separate Ridge model
    joblib.dump(
        ridge,
        f"ridge_{target}.pkl"
    )


    # ----------------------------
    # C. Random Forest
    # ----------------------------

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_pred = rf.predict(
        X_test
    )

    rf_mae = mean_absolute_error(
        y_test,
        rf_pred
    )

    rf_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            rf_pred
        )
    )

    rf_r2 = r2_score(
        y_test,
        rf_pred
    )


    # Save separate RF model
    joblib.dump(
        rf,
        f"random_forest_{target}.pkl"
    )


    # ----------------------------
    # D. XGBoost
    # ----------------------------

    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(
        X_train,
        y_train
    )

    xgb_pred = xgb.predict(
        X_test
    )

    xgb_mae = mean_absolute_error(
        y_test,
        xgb_pred
    )

    xgb_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            xgb_pred
        )
    )

    xgb_r2 = r2_score(
        y_test,
        xgb_pred
    )


    # Save separate XGBoost model
    joblib.dump(
        xgb,
        f"xgboost_{target}.pkl"
    )


    # ----------------------------
    # Print results
    # ----------------------------

    print("\nNaive Baseline")
    print("MAE:", round(baseline_mae, 2))
    print("RMSE:", round(baseline_rmse, 2))
    print("R2:", round(baseline_r2, 3))

    print("\nRidge Regression")
    print("MAE:", round(ridge_mae, 2))
    print("RMSE:", round(ridge_rmse, 2))
    print("R2:", round(ridge_r2, 3))

    print("\nRandom Forest")
    print("MAE:", round(rf_mae, 2))
    print("RMSE:", round(rf_rmse, 2))
    print("R2:", round(rf_r2, 3))

    print("\nXGBoost")
    print("MAE:", round(xgb_mae, 2))
    print("RMSE:", round(xgb_rmse, 2))
    print("R2:", round(xgb_r2, 3))


    # ----------------------------
    # Store results
    # ----------------------------

    results.append({

        "Target": target,

        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
        "Baseline_R2": baseline_r2,

        "Ridge_MAE": ridge_mae,
        "Ridge_RMSE": ridge_rmse,
        "Ridge_R2": ridge_r2,

        "RF_MAE": rf_mae,
        "RF_RMSE": rf_rmse,
        "RF_R2": rf_r2,

        "XGBoost_MAE": xgb_mae,
        "XGBoost_RMSE": xgb_rmse,
        "XGBoost_R2": xgb_r2
    })


# --------------------------------
# 8. Final results table
# --------------------------------

results_df = pd.DataFrame(results)


# Find best actual ML model by MAE
ml_columns = {
    "Ridge_MAE": "Ridge Regression",
    "RF_MAE": "Random Forest",
    "XGBoost_MAE": "XGBoost"
}


best_models = []

for _, row in results_df.iterrows():

    best_column = min(
        ml_columns.keys(),
        key=lambda col: row[col]
    )

    best_models.append(
        ml_columns[best_column]
    )


results_df["Best_ML_Model"] = best_models


# --------------------------------
# 9. Save results
# --------------------------------

results_df.to_csv(
    "Final_Model_Results.csv",
    index=False
)


print("\n\n==============================")
print("FINAL MODEL RESULTS")
print("==============================")

print(
    results_df.round(3).to_string(
        index=False
    )
)


print("\nBEST ML MODEL PER HORIZON")

for _, row in results_df.iterrows():

    print(
        row["Target"],
        "->",
        row["Best_ML_Model"]
    )


print(
    "\nFinal_Model_Results.csv created successfully."
)
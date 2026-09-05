import os
import glob
import joblib
import pandas as pd
import hopsworks


# ------------------------------------------------
# 1. Connect to Hopsworks
# ------------------------------------------------

project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    project="pearls_aqi_hk26",
    port=443,
    engine="python",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

fs = project.get_feature_store()
mr = project.get_model_registry()


# ------------------------------------------------
# 2. Read latest LIVE features
# ------------------------------------------------

live_fg = fs.get_feature_group(
    name="aqi_live_features",
    version=1
)

live_data = live_fg.read(
    online=True,
    dataframe_type="pandas"
)

if live_data.empty:
    raise ValueError("No live features found in Hopsworks.")


# Since Lahore is our current city
latest = live_data[
    live_data["city"] == "lahore"
].copy()

if latest.empty:
    raise ValueError("No Lahore live features found.")


latest = latest.sort_values(
    "date_time"
).tail(1)


current_time = latest["date_time"].iloc[0]
current_aqi = latest["aqi"].iloc[0]


print("\nLATEST LIVE DATA")
print("Time:", current_time)
print("Current AQI:", current_aqi)


# ------------------------------------------------
# 3. Prepare model input
# ------------------------------------------------

X = latest.drop(
    columns=[
        "city",
        "date_time"
    ],
    errors="ignore"
)


# ------------------------------------------------
# 4. Helper to download and load model
# ------------------------------------------------

def load_registry_model(model_name):

    model_meta = mr.get_model(
        name=model_name,
        version=1
    )

    model_dir = model_meta.download()

    pkl_files = glob.glob(
        os.path.join(
            model_dir,
            "**",
            "*.pkl"
        ),
        recursive=True
    )

    if not pkl_files:
        raise FileNotFoundError(
            f"No .pkl file found for {model_name}"
        )

    model = joblib.load(
        pkl_files[0]
    )

    return model


# ------------------------------------------------
# 5. Load models from Model Registry
# ------------------------------------------------

print("\nLoading models from Hopsworks Model Registry...")


model_24h = load_registry_model(
    "aqi_24h_xgboost"
)

model_48h = load_registry_model(
    "aqi_48h_ridge"
)

model_72h = load_registry_model(
    "aqi_72h_ridge"
)


# ------------------------------------------------
# 6. Ensure exact feature order
# ------------------------------------------------

def prepare_features(model, dataframe):

    model_input = dataframe.copy()

    if hasattr(model, "feature_names_in_"):

        model_input = model_input[
            list(model.feature_names_in_)
        ]

    return model_input


X24 = prepare_features(
    model_24h,
    X
)

X48 = prepare_features(
    model_48h,
    X
)

X72 = prepare_features(
    model_72h,
    X
)


# ------------------------------------------------
# 7. Predict separately
# ------------------------------------------------

prediction_24h = model_24h.predict(
    X24
)[0]

prediction_48h = model_48h.predict(
    X48
)[0]

prediction_72h = model_72h.predict(
    X72
)[0]


# AQI shown as whole number
prediction_24h = int(round(prediction_24h))
prediction_48h = int(round(prediction_48h))
prediction_72h = int(round(prediction_72h))


# ------------------------------------------------
# 8. Forecast dates
# ------------------------------------------------

current_time = pd.to_datetime(
    current_time
)

day1_time = current_time + pd.Timedelta(hours=24)
day2_time = current_time + pd.Timedelta(hours=48)
day3_time = current_time + pd.Timedelta(hours=72)


# ------------------------------------------------
# 9. Print predictions
# ------------------------------------------------

print("\n================================")
print("PEARLS AQI PREDICTOR")
print("================================")

print(
    "\nCurrent AQI:",
    int(round(current_aqi))
)

print(
    "\nDay +1:",
    day1_time,
    "-> AQI",
    prediction_24h
)

print(
    "Day +2:",
    day2_time,
    "-> AQI",
    prediction_48h
)

print(
    "Day +3:",
    day3_time,
    "-> AQI",
    prediction_72h
)


# ------------------------------------------------
# 10. Save latest prediction result
# ------------------------------------------------

results = pd.DataFrame({

    "Forecast": [
        "Current",
        "Day +1",
        "Day +2",
        "Day +3"
    ],

    "Date_Time": [
        current_time,
        day1_time,
        day2_time,
        day3_time
    ],

    "AQI": [
        int(round(current_aqi)),
        prediction_24h,
        prediction_48h,
        prediction_72h
    ],

    "Model": [
        "Observed AQI",
        "XGBoost",
        "Ridge Regression",
        "Ridge Regression"
    ]
})


results.to_csv(
    "Latest_AQI_Predictions.csv",
    index=False
)


print(
    "\nLatest_AQI_Predictions.csv created successfully."
)
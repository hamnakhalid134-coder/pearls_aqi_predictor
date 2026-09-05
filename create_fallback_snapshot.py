import os
import glob
import json
import joblib
import pandas as pd
import hopsworks


# ------------------------------------------------
# HOPSWORKS CONNECTION
# ------------------------------------------------

api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    raise ValueError("HOPSWORKS_API_KEY is not set.")


project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    project="pearls_aqi_hk26",
    port=443,
    engine="python",
    api_key_value=api_key
)

fs = project.get_feature_store()
mr = project.get_model_registry()


# ------------------------------------------------
# GET LATEST LIVE FEATURES
# ------------------------------------------------

fg = fs.get_feature_group(
    name="aqi_live_features",
    version=1
)

data = fg.read(
    online=True,
    dataframe_type="pandas"
)

data = data[
    data["city"] == "lahore"
].copy()

if data.empty:
    raise ValueError("No Lahore live data found.")


latest = (
    data
    .sort_values("date_time")
    .tail(1)
)


# ------------------------------------------------
# LOAD MODEL FROM REGISTRY
# ------------------------------------------------

def load_model(model_name):

    model_meta = mr.get_model(
        name=model_name,
        version=1
    )

    model_dir = model_meta.download()

    files = glob.glob(
        os.path.join(
            model_dir,
            "**",
            "*.pkl"
        ),
        recursive=True
    )

    if not files:
        raise FileNotFoundError(
            f"No model file found for {model_name}"
        )

    return joblib.load(files[0])


model_24 = load_model("aqi_24h_xgboost")
model_48 = load_model("aqi_48h_ridge")
model_72 = load_model("aqi_72h_ridge")


# ------------------------------------------------
# PREPARE INPUT
# ------------------------------------------------

X = latest.drop(
    columns=[
        "city",
        "date_time"
    ],
    errors="ignore"
)


def prepare(model, dataframe):

    result = dataframe.copy()

    if hasattr(model, "feature_names_in_"):
        result = result[
            list(model.feature_names_in_)
        ]

    return result


# ------------------------------------------------
# PREDICTIONS
# ------------------------------------------------

pred_24 = int(
    round(
        model_24.predict(
            prepare(model_24, X)
        )[0]
    )
)

pred_48 = int(
    round(
        model_48.predict(
            prepare(model_48, X)
        )[0]
    )
)

pred_72 = int(
    round(
        model_72.predict(
            prepare(model_72, X)
        )[0]
    )
)


# ------------------------------------------------
# SAFE VALUE HELPER
# ------------------------------------------------

def value(column):

    if column not in latest.columns:
        return None

    v = latest[column].iloc[0]

    if pd.isna(v):
        return None

    return float(v)


# ------------------------------------------------
# CREATE FALLBACK SNAPSHOT
# ------------------------------------------------

snapshot = {

    "city": "Lahore",

    "source_timestamp":
        str(
            pd.to_datetime(
                latest["date_time"].iloc[0]
            )
        ),

    "current_aqi":
        int(round(latest["aqi"].iloc[0])),

    "aqi_24h":
        pred_24,

    "aqi_48h":
        pred_48,

    "aqi_72h":
        pred_72,

    "temperature":
        value("temperature"),

    "humidity":
        value("humidity"),

    "wind_speed":
        value("wind_speed"),

    "pm2_5":
        value("pm2_5"),

    "pm10":
        value("pm10"),

    "o3":
        value("o3"),

    "no2":
        value("no2"),

    "so2":
        value("so2"),

    "co":
        value("co"),

    "pressure":
        value("pressure"),

    "precipitation":
        value("precipitation"),

    "models": {
        "24h": "XGBoost",
        "48h": "Ridge Regression",
        "72h": "Ridge Regression"
    }
}


with open(
    "latest_prediction_snapshot.json",
    "w"
) as file:

    json.dump(
        snapshot,
        file,
        indent=4
    )


print("\nFallback snapshot created successfully.")
print("Current AQI:", snapshot["current_aqi"])
print("24h:", snapshot["aqi_24h"])
print("48h:", snapshot["aqi_48h"])
print("72h:", snapshot["aqi_72h"])
print(
    "Saved as: latest_prediction_snapshot.json"
)
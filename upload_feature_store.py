import pandas as pd
import hopsworks


# --------------------------------
# 1. Load ML-ready historical data
# --------------------------------

data = pd.read_csv("ML_Ready_Data.csv")

data["Date_Time"] = pd.to_datetime(
    data["Date_Time"]
)

data = data.sort_values(
    "Date_Time"
).reset_index(drop=True)


# --------------------------------
# 2. Clean column names for Hopsworks
# --------------------------------

data.columns = [
    col.lower()
    for col in data.columns
]

# Add location identifier
data["city"] = "lahore"


print("Rows:", len(data))
print("Columns:", len(data.columns))


# --------------------------------
# 3. Connect to Hopsworks
# --------------------------------

project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    project="pearls_aqi_hk26",
    port=443,
    engine="python"
)

fs = project.get_feature_store()


# --------------------------------
# 4. Create historical feature group
# --------------------------------

fg = fs.get_or_create_feature_group(

    name="aqi_historical_features",

    version=1,

    description=(
        "Historical hourly AQI, pollution, weather, "
        "engineered features and 24h/48h/72h AQI labels for Lahore."
    ),

    primary_key=["city"],

    event_time="date_time",

    online_enabled=False
)


# --------------------------------
# 5. Insert historical data
# --------------------------------

print("\nUploading data to Hopsworks Feature Store...")

fg.insert(data)

print("\nUpload complete.")

print("Feature Group:")
print(fg.name)

print("Version:")
print(fg.version)

print("Rows uploaded:")
print(len(data))

print("\nHopsworks Feature Store setup successful!")
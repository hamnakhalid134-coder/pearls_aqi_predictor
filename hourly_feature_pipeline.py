import os
import requests
import pandas as pd
import numpy as np
import hopsworks


LAT = 31.5204
LON = 74.3587


# =================================================
# 1. FETCH AIR QUALITY
# =================================================

air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

air_params = {
    "latitude": LAT,
    "longitude": LON,

    "hourly": [
        "us_aqi",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ],

    # We need enough history for 72h lag/rolling features
    "past_days": 4,
    "forecast_days": 1,

    "timezone": "Asia/Karachi"
}


air_response = requests.get(
    air_url,
    params=air_params,
    timeout=120
)

air_response.raise_for_status()

air = air_response.json()["hourly"]


air_df = pd.DataFrame({

    "Date_Time": air["time"],

    "AQI": air["us_aqi"],

    "PM2_5": air["pm2_5"],

    "PM10": air["pm10"],

    "CO": air["carbon_monoxide"],

    "NO2": air["nitrogen_dioxide"],

    "SO2": air["sulphur_dioxide"],

    "O3": air["ozone"]
})


# =================================================
# 2. FETCH WEATHER
# =================================================

weather_url = "https://api.open-meteo.com/v1/forecast"

weather_params = {
    "latitude": LAT,
    "longitude": LON,

    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "precipitation",
        "wind_speed_10m"
    ],

    "past_days": 4,
    "forecast_days": 1,

    "timezone": "Asia/Karachi"
}


weather_response = requests.get(
    weather_url,
    params=weather_params,
    timeout=120
)

weather_response.raise_for_status()

weather = weather_response.json()["hourly"]


weather_df = pd.DataFrame({

    "Date_Time": weather["time"],

    "Temperature": weather["temperature_2m"],

    "Humidity": weather["relative_humidity_2m"],

    "Pressure": weather["surface_pressure"],

    "Precipitation": weather["precipitation"],

    "Wind_Speed": weather["wind_speed_10m"]
})


# =================================================
# 3. PREPARE TIME
# =================================================

air_df["Date_Time"] = pd.to_datetime(
    air_df["Date_Time"]
)

weather_df["Date_Time"] = pd.to_datetime(
    weather_df["Date_Time"]
)


# =================================================
# 4. MERGE
# =================================================

data = pd.merge(
    air_df,
    weather_df,
    on="Date_Time",
    how="inner"
)

data = data.sort_values(
    "Date_Time"
).reset_index(drop=True)


# =================================================
# 5. REMOVE FUTURE HOURS
# =================================================

current_hour = (
    pd.Timestamp.now(
        tz="Asia/Karachi"
    )
    .tz_localize(None)
    .floor("h")
)


data = data[
    data["Date_Time"] <= current_hour
].copy()


# =================================================
# 6. CHECK HISTORY
# =================================================

if len(data) < 73:

    raise ValueError(
        "Not enough hourly history to calculate 72-hour features."
    )


# =================================================
# 7. TIME FEATURES
# =================================================

data["Hour"] = data["Date_Time"].dt.hour

data["Day_of_Week"] = (
    data["Date_Time"].dt.dayofweek
)

data["Month"] = data["Date_Time"].dt.month


data["Hour_sin"] = np.sin(
    2 * np.pi * data["Hour"] / 24
)

data["Hour_cos"] = np.cos(
    2 * np.pi * data["Hour"] / 24
)


data["DOW_sin"] = np.sin(
    2 * np.pi * data["Day_of_Week"] / 7
)

data["DOW_cos"] = np.cos(
    2 * np.pi * data["Day_of_Week"] / 7
)


data["Month_sin"] = np.sin(
    2 * np.pi * data["Month"] / 12
)

data["Month_cos"] = np.cos(
    2 * np.pi * data["Month"] / 12
)


# =================================================
# 8. AQI LAGS
# =================================================

data["AQI_Lag_1h"] = data["AQI"].shift(1)
data["AQI_Lag_6h"] = data["AQI"].shift(6)
data["AQI_Lag_24h"] = data["AQI"].shift(24)
data["AQI_Lag_48h"] = data["AQI"].shift(48)
data["AQI_Lag_72h"] = data["AQI"].shift(72)


# =================================================
# 9. POLLUTION LAGS
# =================================================

for col in [
    "PM2_5",
    "PM10",
    "NO2",
    "O3"
]:

    data[f"{col}_Lag_1h"] = (
        data[col].shift(1)
    )

    data[f"{col}_Lag_24h"] = (
        data[col].shift(24)
    )


# =================================================
# 10. WEATHER LAGS
# =================================================

for col in [
    "Temperature",
    "Humidity",
    "Wind_Speed"
]:

    data[f"{col}_Lag_1h"] = (
        data[col].shift(1)
    )

    data[f"{col}_Lag_24h"] = (
        data[col].shift(24)
    )


# =================================================
# 11. CHANGE FEATURES
# =================================================

data["AQI_Change_1h"] = (
    data["AQI"].diff(1)
)

data["PM2_5_Change_1h"] = (
    data["PM2_5"].diff(1)
)

data["PM10_Change_1h"] = (
    data["PM10"].diff(1)
)


# =================================================
# 12. ROLLING FEATURES
# =================================================

data["AQI_Rolling_6h"] = (
    data["AQI"].rolling(6).mean()
)

data["AQI_Rolling_24h"] = (
    data["AQI"].rolling(24).mean()
)

data["AQI_Rolling_72h"] = (
    data["AQI"].rolling(72).mean()
)


data["PM2_5_Rolling_24h"] = (
    data["PM2_5"].rolling(24).mean()
)

data["PM10_Rolling_24h"] = (
    data["PM10"].rolling(24).mean()
)


data["Temperature_Rolling_24h"] = (
    data["Temperature"].rolling(24).mean()
)

data["Humidity_Rolling_24h"] = (
    data["Humidity"].rolling(24).mean()
)

data["Wind_Speed_Rolling_24h"] = (
    data["Wind_Speed"].rolling(24).mean()
)


# =================================================
# 13. KEEP LATEST COMPLETE ROW
# =================================================

data = data.dropna().reset_index(drop=True)

if len(data) == 0:

    raise ValueError(
        "No complete feature row was generated."
    )


latest = data.tail(1).copy()


# =================================================
# 14. LOWERCASE FOR HOPSWORKS
# =================================================

latest.columns = [
    col.lower()
    for col in latest.columns
]


# Location identifier
latest["city"] = "lahore"


print("\nLATEST FEATURE ROW")

print(
    latest[
        [
            "date_time",
            "city",
            "aqi",
            "pm2_5",
            "temperature"
        ]
    ]
)


print("\nFeatures generated:", len(latest.columns))


# =================================================
# 15. CONNECT TO HOPSWORKS
# =================================================

project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    project="pearls_aqi_hk26",
    port=443,
    engine="python",
    api_key_value=os.getenv(
        "HOPSWORKS_API_KEY"
    )
)

fs = project.get_feature_store()


# =================================================
# 16. LIVE FEATURE GROUP
# =================================================

live_fg = fs.get_or_create_feature_group(

    name="aqi_live_features",

    version=1,

    description=(
        "Latest hourly AQI, pollution, weather "
        "and engineered features for Lahore."
    ),

    primary_key=["city"],

    event_time="date_time",

    online_enabled=True
)


# =================================================
# 17. INSERT LATEST FEATURES
# =================================================

print(
    "\nUploading latest features "
    "to Hopsworks..."
)


live_fg.insert(
    latest,
    wait=True
)


print("\nLIVE FEATURE PIPELINE SUCCESSFUL")

print(
    "Stored hour:",
    latest["date_time"].iloc[0]
)

print(
    "Current AQI:",
    latest["aqi"].iloc[0]
)
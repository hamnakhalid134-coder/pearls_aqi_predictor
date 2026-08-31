import requests
import pandas as pd


# Lahore coordinates
lat = 31.5204
lon = 74.3587

# Complete 3-year period
start_date = "2023-08-31"
end_date = "2026-08-30"


url = "https://air-quality-api.open-meteo.com/v1/air-quality"

params = {
    "latitude": lat,
    "longitude": lon,

    "start_date": start_date,
    "end_date": end_date,

    "hourly": [
        "us_aqi",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ],

    "timezone": "Asia/Karachi"
}


print("Downloading air-quality data...")

response = requests.get(
    url,
    params=params,
    timeout=120
)

print("Status Code:", response.status_code)

response.raise_for_status()

data = response.json()

hourly = data["hourly"]


df = pd.DataFrame({

    "Date_Time": hourly["time"],

    "AQI": hourly["us_aqi"],

    "PM2_5": hourly["pm2_5"],

    "PM10": hourly["pm10"],

    "CO": hourly["carbon_monoxide"],

    "NO2": hourly["nitrogen_dioxide"],

    "SO2": hourly["sulphur_dioxide"],

    "O3": hourly["ozone"]
})


# Convert time column
df["Date_Time"] = pd.to_datetime(
    df["Date_Time"]
)


# Sort data
df = df.sort_values(
    "Date_Time"
).reset_index(drop=True)


# Save
df.to_csv(
    "Air_Quality_Data.csv",
    index=False
)


print("\nDownload complete.")

print("\nTotal Rows:")
print(len(df))

print("\nDate Range:")
print(df["Date_Time"].min())
print(df["Date_Time"].max())

print("\nDuplicate Times:")
print(df["Date_Time"].duplicated().sum())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print(
    "\nAir_Quality_Data.csv created successfully."
)
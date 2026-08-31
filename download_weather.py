import requests
import pandas as pd


# Lahore coordinates
lat = 31.5204
lon = 74.3587

# Same dates as air-quality data
start_date = "2023-08-31"
end_date = "2026-08-30"


url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": lat,
    "longitude": lon,

    "start_date": start_date,
    "end_date": end_date,

    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "precipitation",
        "wind_speed_10m"
    ],

    "timezone": "Asia/Karachi"
}


print("Downloading weather data...")

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

    "Temperature": hourly["temperature_2m"],

    "Humidity": hourly["relative_humidity_2m"],

    "Pressure": hourly["surface_pressure"],

    "Precipitation": hourly["precipitation"],

    "Wind_Speed": hourly["wind_speed_10m"]
})


df["Date_Time"] = pd.to_datetime(
    df["Date_Time"]
)


df = df.sort_values(
    "Date_Time"
).reset_index(drop=True)


df.to_csv(
    "Weather_Data.csv",
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
    "\nWeather_Data.csv created successfully."
)
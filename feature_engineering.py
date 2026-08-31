import pandas as pd
import numpy as np


# --------------------------------
# 1. Load merged dataset
# --------------------------------

data = pd.read_csv("Complete_AQI_Dataset.csv")

data["Date_Time"] = pd.to_datetime(data["Date_Time"])

data = data.sort_values("Date_Time").reset_index(drop=True)


# --------------------------------
# 2. Time features
# --------------------------------

data["Hour"] = data["Date_Time"].dt.hour
data["Day_of_Week"] = data["Date_Time"].dt.dayofweek
data["Month"] = data["Date_Time"].dt.month


# Cyclic time features
data["Hour_sin"] = np.sin(2 * np.pi * data["Hour"] / 24)
data["Hour_cos"] = np.cos(2 * np.pi * data["Hour"] / 24)

data["DOW_sin"] = np.sin(2 * np.pi * data["Day_of_Week"] / 7)
data["DOW_cos"] = np.cos(2 * np.pi * data["Day_of_Week"] / 7)

data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)


# --------------------------------
# 3. AQI lag features
# --------------------------------

data["AQI_Lag_1h"] = data["AQI"].shift(1)
data["AQI_Lag_6h"] = data["AQI"].shift(6)
data["AQI_Lag_24h"] = data["AQI"].shift(24)
data["AQI_Lag_48h"] = data["AQI"].shift(48)
data["AQI_Lag_72h"] = data["AQI"].shift(72)


# --------------------------------
# 4. Pollution lag features
# --------------------------------

for col in ["PM2_5", "PM10", "NO2", "O3"]:
    data[f"{col}_Lag_1h"] = data[col].shift(1)
    data[f"{col}_Lag_24h"] = data[col].shift(24)


# --------------------------------
# 5. Weather lag features
# --------------------------------

for col in ["Temperature", "Humidity", "Wind_Speed"]:
    data[f"{col}_Lag_1h"] = data[col].shift(1)
    data[f"{col}_Lag_24h"] = data[col].shift(24)


# --------------------------------
# 6. Change features
# --------------------------------

data["AQI_Change_1h"] = data["AQI"].diff(1)
data["PM2_5_Change_1h"] = data["PM2_5"].diff(1)
data["PM10_Change_1h"] = data["PM10"].diff(1)


# --------------------------------
# 7. Rolling features
# --------------------------------

data["AQI_Rolling_6h"] = data["AQI"].rolling(6).mean()
data["AQI_Rolling_24h"] = data["AQI"].rolling(24).mean()
data["AQI_Rolling_72h"] = data["AQI"].rolling(72).mean()

data["PM2_5_Rolling_24h"] = data["PM2_5"].rolling(24).mean()
data["PM10_Rolling_24h"] = data["PM10"].rolling(24).mean()

data["Temperature_Rolling_24h"] = data["Temperature"].rolling(24).mean()
data["Humidity_Rolling_24h"] = data["Humidity"].rolling(24).mean()
data["Wind_Speed_Rolling_24h"] = data["Wind_Speed"].rolling(24).mean()


# --------------------------------
# 8. DIRECT AQI targets
# --------------------------------

# Mentor requirement:
# Predict AQI directly, not future PM2.5.

data["AQI_24h"] = data["AQI"].shift(-24)
data["AQI_48h"] = data["AQI"].shift(-48)
data["AQI_72h"] = data["AQI"].shift(-72)


# --------------------------------
# 9. Remove rows affected by
#    lags / rolling / future targets
# --------------------------------

data = data.dropna().reset_index(drop=True)


# --------------------------------
# 10. Save
# --------------------------------

data.to_csv(
    "ML_Ready_Data.csv",
    index=False
)


print("\nFEATURE ENGINEERING COMPLETE")

print("\nTotal Rows:")
print(len(data))

print("\nDate Range:")
print(data["Date_Time"].min())
print(data["Date_Time"].max())

print("\nMissing Values:")
print(data.isnull().sum().sum())

print("\nTargets:")
print(
    data[
        ["Date_Time", "AQI", "AQI_24h", "AQI_48h", "AQI_72h"]
    ].head()
)

print("\nTotal Columns:")
print(len(data.columns))

print("\nML_Ready_Data.csv created successfully.")
import pandas as pd


# Read files
air = pd.read_csv("Air_Quality_Data.csv")
weather = pd.read_csv("Weather_Data.csv")


# Convert Date_Time
air["Date_Time"] = pd.to_datetime(air["Date_Time"])
weather["Date_Time"] = pd.to_datetime(weather["Date_Time"])


# Merge on Date_Time
data = pd.merge(
    air,
    weather,
    on="Date_Time",
    how="inner"
)


# Sort
data = data.sort_values(
    "Date_Time"
).reset_index(drop=True)


# Save final merged dataset
data.to_csv(
    "Complete_AQI_Dataset.csv",
    index=False
)


print("\nMERGE COMPLETE")

print("\nTotal Rows:")
print(len(data))

print("\nDate Range:")
print(data["Date_Time"].min())
print(data["Date_Time"].max())

print("\nDuplicate Times:")
print(data["Date_Time"].duplicated().sum())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 Rows:")
print(data.head())

print("\nComplete_AQI_Dataset.csv created successfully.")
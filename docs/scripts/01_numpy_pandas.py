# %%
import os

import kaggle
import numpy as np
import pandas as pd

# Set up Kaggle API credentials
# Make sure you have a kaggle.json file in ~/.kaggle/ with your API token
kaggle.api.authenticate()

# %%
# Download datasets
kaggle.api.dataset_download_files("usgs/earthquake-database", path="./data", unzip=True)

# %%
earthquakes = pd.read_csv("./data/database.csv")
earthquakes.head()
# renmae columns Date to date, Time to time, Latitude to latitude, Longitude to longitude, Depth to depth, Magnitude to magnitude, Type to type
earthquakes.rename(
    columns={
        "Date": "date",
        "Time": "time",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Depth": "depth",
        "Magnitude": "magnitude",
        "Type": "type",
        "ID": "id",
        "Source": "source",
        "Status": "status",
    },
    inplace=True,
)
earthquakes["time"] = earthquakes["time"].str.replace("Z", "")
earthquakes["time"] = pd.to_datetime(
    earthquakes["date"] + " " + earthquakes["time"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
)
earthquakes = earthquakes[["time", "latitude", "longitude", "depth", "magnitude", "type", "id", "source", "status"]]
earthquakes.to_csv("./data/earthquakes.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")

# %%
import numpy as np

# Load earthquake data (magnitude and depth)
# the first coloumn is utc datetime
earthquakes = np.loadtxt("data/earthquakes.csv", delimiter=",", skiprows=1, usecols=(1, 2, 3, 4), dtype=float)

# Calculate average magnitude and depth
avg_depth = np.mean(earthquakes[:, 2])
avg_magnitude = np.mean(earthquakes[:, 3])
print(f"Average magnitude: M{avg_magnitude:.2f}")
print(f"Average depth: {avg_depth:.2f} km")

# Find the strongest earthquake
strongest_idx = np.argmax(earthquakes[:, 3])
strongest_magnitude = earthquakes[strongest_idx, 3]
strongest_depth = earthquakes[strongest_idx, 2]

print(f"Average magnitude: M{avg_magnitude:.2f}")
print(f"Average depth: {avg_depth:.2f} km")
print(f"Strongest earthquake: Magnitude {strongest_magnitude:.2f} at depth {strongest_depth:.2f} km")

# %%
import pandas as pd

# Load earthquake data
df = pd.read_csv("data/earthquakes.csv")

# Calculate average magnitude and depth
avg_depth = df["depth"].mean()
avg_magnitude = df["magnitude"].mean()

# Find the strongest earthquake
strongest_idx = df["magnitude"].idxmax()
strongest_magnitude = df.loc[strongest_idx, "magnitude"]
strongest_depth = df.loc[strongest_idx, "depth"]

print(f"Average magnitude: M{avg_magnitude:.2f}")
print(f"Average depth: {avg_depth:.2f} km")
print(f"Strongest earthquake: Magnitude {strongest_magnitude:.2f} at depth {strongest_depth:.2f} km")

# %%
kaggle.api.dataset_download_files(
    "berkeleyearth/climate-change-earth-surface-temperature-data", path="./data", unzip=True
)

# %%
import pandas as pd

df = pd.read_csv("data/GlobalTemperatures.csv")
df.head()

df.rename(
    columns={
        "dt": "date",
        "LandAverageTemperature": "temperature",
    },
    inplace=True,
)
df = df[["date", "temperature"]]
# filter out rows with missing values
df = df.dropna()
df = df[df["date"] >= "1900-01-01"]
df.to_csv("data/global_temperature.csv", index=False, date_format="%Y-%m-%d")

import matplotlib.pyplot as plt

# %%
import pandas as pd

# Load temperature data
df = pd.read_csv("data/global_temperature.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Set date as index
df.set_index("date", inplace=True)

# Find the hottest and coldest days
hottest_day = df["temperature"].idxmax()
coldest_day = df["temperature"].idxmin()

print(f"Hottest day: {hottest_day.date()} ({df.loc[hottest_day, 'temperature']:.1f}°C)")
print(f"Coldest day: {coldest_day.date()} ({df.loc[coldest_day, 'temperature']:.1f}°C)")

# Calculate monthly average temperatures
yearly_avg = df.resample("Y").mean()

# Plot monthly average temperatures
yearly_avg["temperature"].plot(figsize=(12, 6))

plt.title("Yearly Average Temperatures")
plt.ylabel("Temperature (°C)")
plt.show()

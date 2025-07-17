import pandas as pd
import pickle
import numpy as np
from pathlib import Path

data=pd.read_excel("/workspaces/Time-series-prediction-for-pollution-data/air_pollution_data.xlsx")

city_df = {}
for city in data["city"].unique():
    city_df[city] = data[data["city"] == city].copy()


def prepare_xy(city,use_lags=True, scale=True):
    """
    Processes df into x, y with optional lag and scaling.
    Returns x, y, and optionally scaler used.

    """
    df=data[data["city"] == city].copy()

    x = df.drop(columns=["city", "aqi","date"])
    y = df["aqi"]    

    if use_lags==True:
        for lag in range(1, 5):  # lag-1, lag-2, lag-3, lag-4
            x[f"aqi_lag_{lag}"] = y.shift(lag)
        x = x.dropna()
        y = y[x.index]  # Align y with x after lag

    if scale==True:
        with open("/workspaces/Time-series-prediction-for-pollution-data/models/minmaxscaling.pkl", "rb") as f:
            scaler = pickle.load(f)
            x_scaled = scaler.transform(x) 
        return x_scaled, y.values, scaler

    return x.values, y.values

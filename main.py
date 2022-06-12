import pandas as pd
import numpy as np
import datetime
from dataset_creation import *
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import (
    plot_cross_validation_metric,
    plot_plotly,
    plot_components_plotly,
)

pd.options.display.float_format = "{:,.3f}".format

# Creating Dymmt Dataset
numberofrowsperproduct = 120000
start = pd.to_datetime("2021-01-01")
end = pd.to_datetime("2022-06-15")
df = create_df(start, end, numberofrowsperproduct)
df["Date"] = pd.to_datetime(df["Datetime"]).dt.date

##Checking min max dates
print(df.groupby(["Product"]).agg({"Date": [np.min, np.max]}))
print(df.groupby(["Manufacturer"]).agg({"Date": [np.min, np.max]}))

##Groupby to day level all data
df = df.groupby(["Product", "Manufacturer", "Date"]).sum().reset_index()
df
###Apply forecast for each product and Manufacturer
for product in df["Product"].unique():
    for manufacturer in df["Manufacturer"].unique():
        dftemp = df[(df["Manufacturer"] == manufacturer) & (df["Product"] == product)]
        
        m = Prophet()
        m.fit(dftemp.rename(columns={"Date": "ds", "Revenue": "y"}))
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        plot_components_plotly(m, forecast)
        plot_plotly(m, forecast)

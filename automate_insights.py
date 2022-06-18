import pandas as pd
import numpy as np
import datetime

from sqlalchemy import asc
from dataset_creation import *
from calculate_prophet import *
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import (
    plot_cross_validation_metric,
    plot_plotly,
    plot_components_plotly,
)

pd.options.display.float_format = "{:,.3f}".format

# Creating dummy Dataset
numberofrowsperproduct = 120000
start = pd.to_datetime("2021-01-01")
end = pd.to_datetime("2022-06-15")
df = create_df(start, end, numberofrowsperproduct)


##Checking min max dates
print(df.groupby(["Product"]).agg({"Date": [np.min, np.max]}))
print(df.groupby(["Manufacturer"]).agg({"Date": [np.min, np.max]}))

##Groupby to day level all data
df = df.groupby(["Product", "Manufacturer", "Date"]).sum().reset_index()


def automatedInsights(
    data,
    categories,
):
    """
    On compare there are 4 options MoM,WoW,YTD,MTD
    Must not contain column with the name Date_val
    dates, value, compare
    """
    # data["Date_val"] = pd.to_datetime(data[dates]).dt.date
    # data["Date_val"] = pd.to_datetime(data["Date_val"])

    # Creating keys to filter the dataframe

    dfKeys = keysCreation(data, categories)
    
    return dfKeys


def keysCreation(data, categories, dfKeys):
    dfKeys = pd.DataFrame(data={"key": [1]})
    dfKeys["key"] = 1
    for category_ in categories:
        dfCategory = pd.DataFrame()
        dfCategory[category_] = data[category_].unique()
        dfCategory["key"] = 1
        dfKeys = pd.merge(dfCategory, dfKeys, on="key", how="inner")
    dfKeys = dfKeys.drop(columns="key")
    return dfKeys


cat = automatedInsights(data=df, categories=["Product", "Manufacturer"])

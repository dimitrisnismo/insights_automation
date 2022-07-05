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

# Creating Dymmt Dataset
numberofrowsperproduct = 120000
start = pd.to_datetime("2021-01-01")
end = pd.to_datetime("2022-06-15")
df = create_df(start, end, numberofrowsperproduct)
df["Date"] = pd.to_datetime(df["Datetime"]).dt.date
df["Date"] = pd.to_datetime(df["Date"])

##Checking min max dates
print(df.groupby(["Product"]).agg({"Date": [np.min, np.max]}))
print(df.groupby(["Manufacturer"]).agg({"Date": [np.min, np.max]}))

##Groupby to day level all data
df = df.groupby(["Product", "Manufacturer", "Date"]).sum().reset_index()

###Apply forecast for each product and Manufacturer and concatenate resuls into initial dataframe
for product in df["Product"].unique():
    manufacturerlist = df[df["Product"] == product]["Manufacturer"].unique()
    for manufacturer in manufacturerlist:
        dftemp = df[(df["Manufacturer"] == manufacturer) & (df["Product"] == product)]
        dfProjections = projections(project_prophet, product, manufacturer, dftemp)
        dfProjections["data"] = "Projection"

        df = pd.concat([df, dfProjections])

# Create a column with the last day of each month
df["data"] = df["data"].fillna("Actual")
df["EOMDate"] = (
    (df["Date"] + pd.DateOffset(months=1)).to_numpy().astype("datetime64[M]")
)
df["EOMDate"] = df["EOMDate"] - pd.DateOffset(days=1)

# Filter data to include only end of current month forecasted
df = df[df["Date"] <= df[df["data"] == "Actual"]["EOMDate"].max()]

# creating columns for WoW,QoQ,MoM,YoY,YTDvsYTD,MTDvsMTD
df["year"] = df["Date"].dt.year
df["quarter"] = df["Date"].dt.quarter
df["month"] = df["Date"].dt.month
df["week"] = df["Date"].dt.week
df = df.sort_values(by="Date")



#############################################################################################
# MOM
#############################################################################################
# MoM Calculations per product and manufacturer
mom = pd.DataFrame()
for product in df["Product"].unique():
    manufacturerlist = df[df["Product"] == product]["Manufacturer"].unique()
    for manufacturer in manufacturerlist:
        dfmom_temp = df[
            (df["Manufacturer"] == manufacturer) & (df["Product"] == product)
        ][["year", "month", "Amount", "EOMDate"]]
        dfmom_temp = dfmom_temp[
            dfmom_temp["EOMDate"]
            >= dfmom_temp["EOMDate"].max() - pd.DateOffset(days=45)
        ]
        dfmom_temp = (
            dfmom_temp.groupby(["year", "month", "EOMDate"]).sum().reset_index()
        )
        dfmom_temp["MoM%"] = dfmom_temp["Amount"] / dfmom_temp["Amount"].shift(1) - 1
        dfmom_temp["Product"] = product
        dfmom_temp["Manufacturer"] = manufacturer
        mom_product_manufacturer = pd.concat([mom, dfmom_temp])

# MoM Calculations per product
mom_product = pd.DataFrame()
for product in df["Product"].unique():
    dfmom_temp = df[(df["Product"] == product)][["year", "month", "Amount", "EOMDate"]]
    dfmom_temp = dfmom_temp[
        dfmom_temp["EOMDate"] >= dfmom_temp["EOMDate"].max() - pd.DateOffset(days=45)
    ]
    dfmom_temp = dfmom_temp.groupby(["year", "month", "EOMDate"]).sum().reset_index()
    dfmom_temp["MoM%"] = dfmom_temp["Amount"] / dfmom_temp["Amount"].shift(1) - 1
    dfmom_temp["Product"] = product
    mom_product = pd.concat([mom_product, dfmom_temp])

#############################################################################################
# YoY
#############################################################################################
# YoY Calculations per product and manufacturer
yoy_product_manufacturer = pd.DataFrame()
for product in df["Product"].unique():
    manufacturerlist = df[df["Product"] == product]["Manufacturer"].unique()
    for manufacturer in manufacturerlist:
        dfmom_temp = df[
            (df["Product"] == product) & (df["Manufacturer"] == manufacturer)
        ][["year", "month", "Amount", "EOMDate"]]
        dfmom_temp = dfmom_temp[
            (dfmom_temp["EOMDate"] == dfmom_temp["EOMDate"].max())
            | (
                (dfmom_temp["year"] == dfmom_temp["year"].max() - 1)
                & (
                    dfmom_temp["month"]
                    == dfmom_temp[dfmom_temp["EOMDate"] == dfmom_temp["EOMDate"].max()][
                        "month"
                    ].max()
                )
            )
        ]
        dfmom_temp = (
            dfmom_temp.groupby(["year", "month", "EOMDate"]).sum().reset_index()
        )
        dfmom_temp["MoM%"] = dfmom_temp["Amount"] / dfmom_temp["Amount"].shift(1) - 1
        dfmom_temp["Product"] = product
        dfmom_temp["Manufacturer"] = manufacturer
        yoy_product_manufacturer = pd.concat([yoy_product_manufacturer, dfmom_temp])

# YoY Calculations per product
yoy_product = pd.DataFrame()
for product in df["Product"].unique():
    dfmom_temp = df[(df["Product"] == product)][["year", "month", "Amount", "EOMDate"]]
    dfmom_temp = dfmom_temp[
        (dfmom_temp["EOMDate"] == dfmom_temp["EOMDate"].max())
        | (
            (dfmom_temp["year"] == dfmom_temp["year"].max() - 1)
            & (
                dfmom_temp["month"]
                == dfmom_temp[dfmom_temp["EOMDate"] == dfmom_temp["EOMDate"].max()][
                    "month"
                ].max()
            )
        )
    ]
    dfmom_temp = dfmom_temp.groupby(["year", "month", "EOMDate"]).sum().reset_index()
    dfmom_temp["YoY%"] = dfmom_temp["Amount"] / dfmom_temp["Amount"].shift(1) - 1
    dfmom_temp["Product"] = product
    yoy_product = pd.concat([yoy_product, dfmom_temp])


#############################################################################################
# WoW
#############################################################################################
# WoW Calculations per product and manufacturer
wow_product_manufacturer = pd.DataFrame()
for product in df["Product"].unique():
    manufacturerlist = df[df["Product"] == product]["Manufacturer"].unique()
    for manufacturer in manufacturerlist:
        dfmom_temp = df[
            (df["Product"] == product) & (df["Manufacturer"] == manufacturer)
        ][["year", "month", "week", "Date", "Amount", "data", "EOMDate"]]
        dfmom_temp = dfmom_temp[
            (
                dfmom_temp["week"]
                == dfmom_temp[
                    (
                        dfmom_temp["Date"]
                        == dfmom_temp[dfmom_temp["data"] == "Actual"]["Date"].max()
                    )
                    & (dfmom_temp["data"] == "Actual")
                ]["week"].max()
            )
            | (
                dfmom_temp["week"]
                == dfmom_temp[
                    (
                        dfmom_temp["Date"]
                        == dfmom_temp[dfmom_temp["data"] == "Actual"]["Date"].max()
                    )
                    & (dfmom_temp["data"] == "Actual")
                ]["week"].max()
                - 1
            )
        ]
        dfmom_temp = dfmom_temp[dfmom_temp["year"] == dfmom_temp["year"].max()]
        dfmom_temp = dfmom_temp.groupby(["week", "EOMDate"]).sum().reset_index()
        dfmom_temp["WoW%"] = dfmom_temp["Amount"] / dfmom_temp["Amount"].shift(1) - 1
        dfmom_temp["Product"] = product
        dfmom_temp["Manufacturer"] = manufacturer
        wow_product_manufacturer = pd.concat([wow_product_manufacturer, dfmom_temp])

# WoW Calculations per product
wow_product = pd.DataFrame()
for product in df["Product"].unique():
    dfmom_temp = df[(df["Product"] == product)][
        ["year", "month", "week", "Date", "Amount", "data", "EOMDate"]
    ]
    dfmom_temp = dfmom_temp[
        (
            dfmom_temp["week"]
            == dfmom_temp[
                (
                    dfmom_temp["Date"]
                    == dfmom_temp[dfmom_temp["data"] == "Actual"]["Date"].max()
                )
                & (dfmom_temp["data"] == "Actual")
            ]["week"].max()
        )
        | (
            dfmom_temp["week"]
            == dfmom_temp[
                (
                    dfmom_temp["Date"]
                    == dfmom_temp[dfmom_temp["data"] == "Actual"]["Date"].max()
                )
                & (dfmom_temp["data"] == "Actual")
            ]["week"].max()
            - 1
        )
    ]
    dfmom_temp = dfmom_temp[dfmom_temp["year"] == dfmom_temp["year"].max()]
    dfmom_temp = dfmom_temp.groupby(["week", "EOMDate"]).sum().reset_index()
    dfmom_temp["YoY%"] = dfmom_temp["Amount"] / dfmom_temp["Amount"].shift(1) - 1
    dfmom_temp["Product"] = product
    wow_product = pd.concat([wow_product, dfmom_temp])

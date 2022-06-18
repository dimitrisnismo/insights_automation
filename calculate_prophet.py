import pandas as pd
import numpy as np
import datetime

from sqlalchemy import asc
from dataset_creation import *
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import (
    plot_cross_validation_metric,
    plot_plotly,
    plot_components_plotly,
)

def project_prophet(product, manufacturer, dftemp, value):
    # project the next 30 periods and return a dataframe for this next 30days
    m = Prophet()
    m.fit(dftemp.rename(columns={"Date": "ds", value: "y"}))
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    projections = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": value})
    projections = pd.merge(
        projections, dftemp[["Product", "Date"]], on="Date", how="left"
    )
    projections = projections[projections["Product"].isnull()]
    projections["Product"] = product
    projections["Manufacturer"] = manufacturer
    projections[value] = projections[value].round()
    return projections


def projections( product, manufacturer, dftemp):
    # implement projection for each revenue and amount and concatenate resuls
    # Project Amount
    proj_amounts = project_prophet(product, manufacturer, dftemp, value="Amount")
    # Project Revenues
    proj_revenue = project_prophet(product, manufacturer, dftemp, value="Revenue")
    projections = pd.concat([proj_amounts, proj_revenue])
    projections = projections.fillna(0)
    projections = (
        projections.groupby(["Date", "Product", "Manufacturer"]).sum().reset_index()
    )
    return projections
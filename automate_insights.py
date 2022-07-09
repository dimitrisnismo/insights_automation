import pandas as pd
import numpy as np
import datetime
import altair as alt
from dataset_creation import *
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import (
    plot_cross_validation_metric,
    plot_plotly,
    plot_components_plotly,
)
import logging
import warnings
import copy

warnings.filterwarnings("ignore")
logging.getLogger("fbprophet").setLevel(logging.WARNING)
pd.options.display.float_format = "{:,.3f}".format


def automatedInsights(df, categories, dates, value):
    tempcategories = copy.deepcopy(categories)
    """
    Must not contain column with the name Date_val
    """
    print("Creating keys to filter the dataframe")
    dfKeys = keysCreation(df, categories)
    print(" Transform column to date type")
    dfgrouped = dateTransform(df=df, date_column=dates)
    df = dateTransform(df=df, date_column=dates)
    print(" Creating a key for each Category")
    df = pd.merge(df, dfKeys, on=categories, how="left")
    print("Applying Forcast for each Category")
    df = projectionsCalculationPerCategory(df, categories, value)
    df = pd.merge(df, dfKeys, on=categories, how="left")
    print("Creating Date Utils Columns")
    df = datesUtilsColumns(df)
    maxdate = str(df[df["Data"] == "Actual"]["Date_val"].max())
    maxdate = str(df[df["Data"] == "Actual"]["Date_val"].max())
    print("Calculate MoM")
    momDf = mom_calculation(df, categories, value)
    print("Calculate WoW")
    wowdf = wow_calculation(df, categories, value)
    print("Calculate MTD")
    mtddf = mtd_calculation(df, categories, value)
    print("Calculate YTD")
    ytddf = ytd_calculation(df, categories, value)
    dateperiodscomparisons = create_comparisonsdf(
        categories, momDf, wowdf, mtddf, ytddf, value
    )
    df_compared = dateperiodscomparisons
    print(categories)
    vizualizeResults(modifycolumnstoviz, df_compared, maxdate, tempcategories, value)
    highandlow(df, tempcategories, value)
    return df, df_compared, dfgrouped


def create_comparisonsdf(categories, momDf, wowdf, mtddf, ytddf, value):
    dateperiodscomparisons = pd.concat([momDf, wowdf, mtddf, ytddf])
    categories.append("%_Difference")
    categories.append("Amount")
    categories.append("Type_period")
    categories.append("Data_max_date")
    x = "previous_" + value
    categories.append(x)
    dateperiodscomparisons = dateperiodscomparisons[categories]
    return dateperiodscomparisons


def mom_calculation(df, categories, value):
    momDf = df[
        df["SOMDate"].isin(
            [df["SOMDate"].max(), (df["SOMDate"].max() - pd.DateOffset(months=1))]
        )
    ]
    momDf = comparisons(categories, momDf, value, datesComparison="EOMDate")
    momDf["Type_period"] = "MoM"
    return momDf


def wow_calculation(df, categories, value):
    wowdf = df[
        df["maxdayweek"].isin(
            [
                df[df["Data"] == "Actual"]["maxdayweek"].max(),
                (
                    df[df["Data"] == "Actual"]["maxdayweek"].max()
                    - pd.DateOffset(days=7)
                ),
            ]
        )
    ]
    wowdf = comparisons(categories, wowdf, value, datesComparison="maxdayweek")
    wowdf["Type_period"] = "WoW"
    return wowdf


def ytd_calculation(df, categories, value):
    ytdyear = df[
        (df["year"] == df[df["Data"] == "Actual"].year.max()) & (df["Data"] == "Actual")
    ]
    ytdyearcurrent = ytdyear["Date_val"].drop_duplicates()
    ytdyearprevious = ytdyear["Date_val"].drop_duplicates() - pd.DateOffset(years=1)
    totaldatesytd = ytdyearcurrent.tolist() + ytdyearprevious.tolist()
    ytddf = df[df["Date_val"].isin(totaldatesytd)]
    ytddf = comparisons(categories, ytddf, value, datesComparison="year")
    ytddf["Type_period"] = "YTD"
    return ytddf


def mtd_calculation(df, categories, value):
    maxday = df[df["Data"] == "Actual"]
    maxday = maxday[maxday["Date_val"] == maxday["Date_val"].max()]
    maxday = maxday.day.max()
    mtddf = df[
        df["SOMDate"].isin(
            [df["SOMDate"].max(), (df["SOMDate"].max() - pd.DateOffset(months=1))]
        )
        & (df["day"] <= maxday)
    ]
    mtddf = comparisons(categories, mtddf, value, datesComparison="EOMDate")
    mtddf["Type_period"] = "MTD"
    return mtddf


def datesUtilsColumns(df):
    df["year"] = df["Date_val"].dt.year
    df["year"] = df["year"].astype("int")
    df["quarter"] = df["Date_val"].dt.quarter
    df["month"] = df["Date_val"].dt.month
    df["day"] = df["Date_val"].dt.day
    df["week"] = df["Date_val"].dt.isocalendar().week
    df = df.sort_values(by="Date_val")
    df["SOMDate"] = df["EOMDate"] = (df["Date_val"]).to_numpy().astype("datetime64[M]")
    df["EOMDate"] = df["SOMDate"] + pd.DateOffset(months=1)
    df["EOMDate"] = df["EOMDate"] - pd.DateOffset(days=1)
    df = df[df["Date_val"] <= df[df["Data"] == "Actual"]["EOMDate"].max()]
    weeksdf = (
        df[["Date_val", "year", "week"]]
        .groupby(["year", "week"])
        .max()
        .reset_index()
        .rename(columns={"Date_val": "maxdayweek"})
    )
    df = pd.merge(df, weeksdf, on=["year", "week"], how="left")
    return df


def projectionsCalculationPerCategory(
    df,
    categories,
    value,
):
    groups = groupsColumns(categories)
    requiredcolumns = filterDatasetColumns(value, groups)
    dfTotal = pd.DataFrame()
    for i in range(0, df.cat_number.max() + 1):
        temporaryDf = df[df["cat_number"] == i]
        temporaryDf = temporaryDf.groupby(groups).sum().reset_index()
        temporaryDf = temporaryDf[requiredcolumns]
        temporaryDf["Data"] = "Actual"
        projections = calculate_prophet(df=temporaryDf, value=value)
        projections = projections[
            ~projections["Date_val"].isin(temporaryDf.Date_val.tolist())
        ]
        for category in categories:
            projections[category] = temporaryDf[category][0]
        projections["Data"] = "Projected"
        dfTotal = pd.concat([dfTotal, temporaryDf, projections])
    return dfTotal


def filterDatasetColumns(value, groups):
    requiredcolumns = groups.copy()
    requiredcolumns.append(value)
    return requiredcolumns


def groupsColumns(categories):
    groups = categories.copy()
    groups.append("Date_val")
    return groups


def dateTransform(df, date_column):
    df["Date_val"] = pd.to_datetime(df[date_column]).dt.date
    df["Date_val"] = pd.to_datetime(df["Date_val"])
    return df


def keysCreation(data, categories):
    """
    Creating based on the given categories (columns) the keys for the dataset
    """
    df = data.groupby(categories).sum().reset_index()[categories]
    df["cat_number"] = df.index
    return df


def calculate_prophet(df, value):
    # project the next 30 periods and return a dataframe for this next 30days
    m = Prophet()
    m.fit(df.rename(columns={"Date_val": "ds", value: "y"}))
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    projections = forecast[["ds", "yhat"]].rename(
        columns={"ds": "Date_val", "yhat": value}
    )
    return projections


def comparisons(categories, input_df, value, datesComparison):
    filterColumns = [datesComparison, value]
    groupsColumns = [datesComparison]
    compareColumns = []
    sortColumns = [datesComparison]
    category_val = ""
    tdf = pd.DataFrame()
    for category in categories:
        filterColumns.append(category)
        compareColumns.append(category)
        groupsColumns.append(category)
        sortColumns.insert(0, category)
        tempDf = input_df[filterColumns].groupby(groupsColumns).sum().reset_index()
        tempDf = tempDf.sort_values(by=sortColumns)
        tempDf["%_Difference"] = np.where(
            (tempDf[compareColumns] == tempDf[compareColumns].shift(1)).all(axis=1),
            tempDf[value] / tempDf[value].shift(1) - 1,
            np.nan,
        )

        tempDf["previous_" + value] = tempDf[value].shift(1)
        tempDf = tempDf[~tempDf["%_Difference"].isnull()]
        category_val = category_val + category
        tempDf["category_comparison"] = category_val
        tempDf["Data_max_date"] = input_df[input_df["Data"] == "Actual"][
            "Date_val"
        ].max()
        tdf = pd.concat([tdf, tempDf])
    return tdf


def modifycolumnstoviz(t2com):
    tt2com = t2com.copy().dropna()
    tt2com["%_Difference"] = tt2com["%_Difference"] * 100
    tt2com["%_Difference"] = tt2com["%_Difference"].round(1)
    tt2com["Col_label"] = np.where(
        tt2com["%_Difference"] == tt2com["%_Difference"].max(),
        "max",
        np.where(
            tt2com["%_Difference"] == tt2com["%_Difference"].min(),
            "min",
            "other",
        ),
    )

    return tt2com


def vizualizeResults(modifycolumnstoviz, df_compared, maxdate, categories, value):
    lista = copy.deepcopy(categories)
    lista.append("%_Difference")
    lista.append(value)
    lista.append("previous_" + value)

    print("Below are the key metrics as of " + maxdate + " for the given dataset")
    for periods in df_compared["Type_period"].unique():
        tcom = df_compared[df_compared["Type_period"] == periods]
        for category in categories[:-1]:
            print(categories)
            for item in tcom[category].unique():
                print("item is :" + item)
                print("category is :" + category)
                t2com = tcom[tcom[category] == item]
                t22com = t2com[t2com.isna().any(axis=1)]
                print(
                    """  
        The"""
                    + periods
                    + "  for the "
                    + category
                    + ": "
                    + item
                    + " is "
                    + str(round(t22com["%_Difference"].sum(), 5) * 100)
                    + "%"
                )
                tt2com = modifycolumnstoviz(t2com)
                chart = (
                    alt.Chart(tt2com)
                    .mark_bar()
                    .encode(
                        alt.X(categories[-1], scale=alt.Scale(zero=False)),
                        alt.Y("%_Difference"),
                        color=alt.Color(
                            "Col_label",
                            scale=alt.Scale(
                                domain=["max", "min", "other"],
                                range=[
                                    "#00891E",
                                    "#890000",
                                    "#AFAFAF",
                                ],
                            ),
                        ),
                    )
                )
                rule = (
                    alt.Chart(tt2com)
                    .mark_rule(color="black")
                    .encode(y="mean(%_Difference)")
                )
                (
                    (chart + rule)
                    .properties(title=category + " by " + periods + "  % ", width=600)
                    .configure_title(fontSize=20, anchor="middle", color="#404040")
                    .configure_view(strokeWidth=0)
                    .display()
                )
                print(tt2com[tt2com["Col_label"] != "other"][lista])
    return lista


def highandlow(df, categories, value):
    lista = []
    for category in categories:
        lista.append(category)
        maxmonthdate = df["EOMDate"].max()
        outliers = df.groupby(lista).agg({value: [min, max]}).reset_index()
        outliers.columns = ["".join(col).strip() for col in outliers.columns.values]
        min_outliers = outliers.rename(columns={value + "min": value})
        minlista = copy.deepcopy(lista)
        minlista.append(value)
        min_outliers = pd.merge(min_outliers, df, on=minlista, how="left")
        min_outliers["Count"] = 1
        min_outliers = min_outliers.groupby(minlista).count().reset_index()
        min_outliers = min_outliers[(min_outliers["Count"] == 1)]
        min_outliers = pd.merge(min_outliers[minlista], df, on=minlista, how="left")
        min_outliers = min_outliers[(min_outliers["EOMDate"] == maxmonthdate)]
        if len(min_outliers) == 0:
            print("There are no new lowest values to show for the last month")
        else:
            print("New  Lows!")
            print(min_outliers)
        max_outliers = outliers.rename(columns={value + "max": value})
        maxlista = copy.deepcopy(lista)
        maxlista.append(value)
        max_outliers = pd.merge(max_outliers, df, on=maxlista, how="left")
        max_outliers["Count"] = 1
        max_outliers = max_outliers.groupby(maxlista).count().reset_index()
        max_outliers = max_outliers[(max_outliers["Count"] == 1)]
        max_outliers = pd.merge(max_outliers[maxlista], df, on=maxlista, how="left")
        max_outliers = max_outliers[(max_outliers["EOMDate"] == maxmonthdate)]
        print("About: " + category)
        if len(max_outliers) == 0:
            print("There are no new highest values to show for the last month")
        else:
            print("New Highs!")
            print(max_outliers)


# Creating dummy Dataset
numberofrowsperproduct = 15000
start = pd.to_datetime("2021-01-01")
end = pd.to_datetime("2022-06-15")
df = create_df(start, end, numberofrowsperproduct)


##Checking min max dates
print(df.groupby(["Product"]).agg({"Datetime": [np.min, np.max]}))
print(df.groupby(["Manufacturer"]).agg({"Datetime": [np.min, np.max]}))


##Groupby to day level all data
df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.date
df = df.groupby(["Product", "Manufacturer", "Datetime"]).sum().reset_index()


df, df_compared, dfgrouped = automatedInsights(
    df=df, categories=["Product", "Manufacturer"], dates="Datetime", value="Amount"
)

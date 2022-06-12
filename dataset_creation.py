import pandas as pd
import numpy as np


##Creating datadates
def createdatetime_dataframe(start, end, n):
    start_u = start.value // 10**9
    end_u = end.value // 10**9
    return (
        pd.to_datetime(np.random.randint(start_u, end_u, n), unit="s")
        .to_frame(index=False)
        .rename(columns={0: "Datetime"})
    )


def create_df(start, end, numberofrowsperproduct):
    df_temp = createdatetime_dataframe(start, end, numberofrowsperproduct)
    products = ["DSLR", "Laptop", "Smartphones", "TVs", "Gaming"]
    # DSLR Fake Dataset
    df_DSLR = df_temp.copy()
    df_DSLR["Product"] = "DSLR"
    df_DSLR["Manufacturer"] = np.random.choice(
        ["Nikon", "Canon", "GoPro"], len(df_DSLR)
    )
    df_DSLR["Revenue"] = np.random.randint(350, 700, len(df_DSLR))
    df_DSLR["Amount"] = np.where(np.random.randint(1, 10, len(df_DSLR)) <= 9, 1, 2)

    # Smartphones Fake Dataset
    df_smart = df_temp.copy()
    df_smart["Product"] = "Smartphones"
    df_smart["Manufacturer"] = np.random.choice(
        ["Apple", "Samsung", "Xiaomi"], len(df_smart)
    )
    df_smart["Revenue"] = np.random.randint(200, 1500, len(df_smart))
    df_smart["Amount"] = np.where(np.random.randint(1, 10, len(df_smart)) <= 9, 1, 2)
    col = "Amount"
    conditions = [df_smart[col] == "Apple", df_smart[col] == "Xiaomi"]
    choices = [df_smart["Amount"] * 1.2, df_smart["Amount"] * 0.8]
    df_smart["Amount"] = np.select(conditions, choices, default=df_smart["Amount"])

    # TVs Fake Dataset
    df_TV = df_temp.copy()
    df_TV["Product"] = "TV"
    df_TV["Manufacturer"] = np.random.choice(["Samsung", "LG", "Sony"], len(df_TV))
    df_TV["Revenue"] = np.random.randint(700, 1900, len(df_TV))
    df_TV["Amount"] = np.where(np.random.randint(1, 10, len(df_TV)) <= 9, 1, 2)

    # Laptop Fake Dataset
    df_Laptop = df_temp.copy()
    df_Laptop["Product"] = "Laptop"
    df_Laptop["Manufacturer"] = np.random.choice(
        ["Lenovo", "Dell", "HP"], len(df_Laptop)
    )
    df_Laptop["Revenue"] = np.random.randint(500, 1500, len(df_Laptop))
    df_Laptop["Amount"] = np.where(np.random.randint(1, 10, len(df_Laptop)) <= 9, 1, 2)

    # Gaming Fake Dataset
    df_Gaming = df_temp.copy()
    df_Gaming["Product"] = "Gaming"
    df_Gaming["Manufacturer"] = np.random.choice(
        ["Sony", "Microsoft", "Nintendo"], len(df_Gaming)
    )
    df_Gaming["Revenue"] = np.random.randint(250, 600, len(df_Gaming))
    df_Gaming["Amount"] = np.where(np.random.randint(1, 10, len(df_Gaming)) <= 9, 1, 2)

    df = pd.concat([df_DSLR, df_smart, df_Gaming, df_TV, df_Laptop])
    return df

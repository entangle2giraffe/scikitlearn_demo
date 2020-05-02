import pandas as pd

df = pd.read_csv("UNdata_Export_20181006_090917280.csv")


# Function for slicing specific country
def country(a):
    print(f"Found country name {a} in the dataset")

    con = df["Country_or_Area"] == str(a.upper())
    cdf = df[con]
    cdf = cdf[["Year", "Value"]]
    cdf.loc[:, "Value"] = cdf["Value"].div(10 ** 6)
    # Return dataframe
    return cdf


def des(coun, degree):
    print(f"Government Consumption Expense of {coun}")
    print(f"f(x^{degree}) prediction")
    print("===============================================")


def instance(d):
    if isinstance(d, int):
        return True

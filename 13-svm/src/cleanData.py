import pandas as pd

df = pd.read_csv("../data/data.csv")

df = df.drop(columns=['Unnamed: 32'])

map = {
    "M": 1,
    "B": 0
}

df["diagnosis"] = df["diagnosis"].map(map)

df.drop(columns=["id"], inplace=True)

df.to_csv("../data/cleanData.csv", index=False)
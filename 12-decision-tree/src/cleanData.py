# here's what we gonna do

# sex => one hot encoding
# BP, cholestrol => label encoding
# drug => label encoding (during prediction convert back)
import numpy as np
import pandas as pd

df = pd.read_csv("../data/drug200.csv")

print(df.info())

print(df["Drug"].unique()) # ['DrugY' 'drugC' 'drugX' 'drugA' 'drugB']
print(df["BP"].unique()) # ['HIGH' 'LOW' 'NORMAL']
print(df["Cholesterol"].unique()) # ['HIGH' 'NORMAL']

mappings = {
    "Drug": {
        "DrugY": 1,
        "drugC": 2,
        "drugX": 3,
        "drugA": 4,
        "drugB": 5
    },
    "BP": {
        "LOW": 1,
        "NORMAL": 2,
        "HIGH": 3
    },
    "Cholesterol": {
        "NORMAL": 1,
        "HIGH": 2
    }
}

df["Drug"] = df["Drug"].map(mappings["Drug"])
df["BP"] = df["BP"].map(mappings["BP"])
df["Cholesterol"] = df["Cholesterol"].map(mappings["Cholesterol"])

df = pd.get_dummies(df, columns=["Sex"], dtype=np.uint8)

print(df.info())

df.to_csv("../data/clean.csv", index=False)
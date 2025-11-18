import pandas as pd

df = pd.read_csv("../data/train.csv")

print(df.info()) # there aren't any null values

# we will just normalise the data
# and move it's range from [0,255] -> [0,1]
# will help our model to converge fast
pd.set_option('display.max_columns', None)

feature_cols = [col for col in df.columns if col != "label"]
df[feature_cols] = df[feature_cols] / 255.0

# print(df.describe())

df.to_csv("../data/cleanTrain.csv", index=False)
import pandas as pd

df = pd.read_csv("../data/train.csv")

print(df.info()) 

pd.set_option('display.max_columns', None)

feature_cols = [col for col in df.columns if col != "label"]
df[feature_cols] = df[feature_cols] / 255.0

df.to_csv("../data/cleanTrain.csv", index=False)
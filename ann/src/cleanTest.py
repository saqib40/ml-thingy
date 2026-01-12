import pandas as pd

df = pd.read_csv("../data/test.csv")

pd.set_option('display.max_columns', None)

df[df.columns] = df[df.columns] / 255.0

df.index = df.index + 1
df.index.name = "ImageId"

df.to_csv("../data/cleanTest.csv", index=True)
import pandas as pd

df = pd.read_csv("../data/test.csv")

# print(df.info()) # there aren't any null values

# we will just normalise the data again
# and move it's range from [0,255] -> [0,1]
# will help our model to converge fast
pd.set_option('display.max_columns', None)

df[df.columns] = df[df.columns] / 255.0

df.index = df.index + 1
df.index.name = "ImageId"

# print(df.describe())

df.to_csv("../data/cleanTest.csv", index=True)
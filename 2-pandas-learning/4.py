import pandas as pd

df = pd.read_json("./data/data.json")

# print(df.to_string())
print(df.head()) # or head(10)
print('\n')
print(df.tail())
print("\n")
print(df.info())
print("\n")
print(df.describe())
print("\n")
print(df.shape)
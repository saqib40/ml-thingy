# cleaning the data
# aka fixing the bad data, which could be:
# Empty cells
# Data in wrong format
# Wrong data
# Duplicates

# 1- handling empty cells with dropna() and fillna()

import pandas as pd
df = pd.read_csv('./data/data.csv')
# df.dropna(inplace=True) -> to update the original df
df2 = df.dropna()

print(df.info())
print('\n')
print(df2.info())
print('\n')
print(df['Pulse'].fillna(244))
# fillna(k, inplace = True) to update original
# df.fillna({"Pulse": 244}, inplace=True)

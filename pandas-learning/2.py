# basics about dataframe
# 2D labeled array
# Table with both row index and column label

import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 28],
    'City': ['New York', 'London', 'Paris']
}
df = pd.DataFrame(data)
print(df)
print('\n')

df2 = pd.DataFrame(data, index=['a', 'b', 'c'])
print(df2)
print('\n')

print(df.loc[0]) # -> returns a panda series
print('\n')

print(df.loc[[0,1]])
print('\n')

print(df2.loc['a'])
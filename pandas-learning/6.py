# fixing wrong data
# some logic

import pandas as pd
df = pd.read_csv('./data/data.csv')

print(df.loc[3, 'Duration'])

# Method 1 -> Replacing a single value
df.loc[3, 'Duration'] = 55


# Method 2 -> Replacing Values
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.loc[x, "Duration"] = 120

# Method 3 -> Removing Rows
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.drop(x, inplace = True)

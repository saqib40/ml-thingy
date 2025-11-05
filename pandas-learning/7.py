# handling duplicates

import pandas as pd

data = {
    'Pulse': [100, 110, 115, 100],
    'Age': [25, 30, 28, 25],
    'Calories': [405, 408, 383, 405]
}

df = pd.DataFrame(data)

print(df.to_string())
print('\n')

print(df.duplicated())
print('\n')

df.drop_duplicates(inplace=True)
print(df.to_string())
print('\n')
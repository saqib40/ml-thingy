# for nominal data

import pandas as pd

# Our data
df = pd.DataFrame({'city': ['London', 'Paris', 'London']})

# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=['city'])

#print(df.info())
print(df.to_string())
#print(df_encoded.info())
print(df_encoded.to_string())
# for ordinal data

import pandas as pd

df = pd.DataFrame({'rating': ['Poor', 'Good', 'Excellent', 'Good']})

# Create the mapping
rating_map = {'Poor': 0, 'Good': 1, 'Excellent': 2}

# Apply the mapping
df['rating_encoded'] = df['rating'].map(rating_map)

print(df.info())

print(df.to_string())
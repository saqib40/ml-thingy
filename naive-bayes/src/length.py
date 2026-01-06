# did it on kaggle notebook
# that's why filepath is like that

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]

# Add message length
df['length'] = df['v2'].apply(len)

# Plot histogram
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='length', hue='v1', bins=50, kde=True)
plt.title("SMS Length Distribution: Spam vs Ham")
plt.xlabel("Message Length")
plt.ylabel("Count")
plt.show()



# conclusion is that ham tend to be shorter
# when compared to spam texts
# so length is actually a good parameter to consider
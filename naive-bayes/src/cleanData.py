import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')

df = pd.read_csv("../data/spam.csv", encoding='latin-1')

#print(df.info())

df = df[['v1', 'v2']]

#print(df.info())
#print(df.head())

# lowercase
df['v2'] = df['v2'].str.lower()
# remove punctuation
# string.punctuation -> RegExp that handles all the punctuation
df['v2'] = df['v2'].str.translate(str.maketrans('', '', string.punctuation))
# add length as property to df
df['length'] = df['v2'].apply(len)
# remove stop words
stop_words = set(stopwords.words('english'))
df['v2'] = df['v2'].apply(
    lambda x: " ".join(word for word in x.split() if word not in stop_words)
)
# label encoding of v1 using map
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
# vectorisation => when training the model

print(df.info())
# save
# df.to_csv("../data/cleanData.csv", index=False)
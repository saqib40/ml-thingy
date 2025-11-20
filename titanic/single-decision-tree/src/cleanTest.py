import pandas as pd

df = pd.read_csv("../data/test.csv")

print(df.info())

# 1- fill the missing values

# for the age and fare columns
# median is safe because data is skewed
df["Age"] = df["Age"].fillna(df["Age"].median())

df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# for the Cabin column
# i think meaningfull absence
# don't drop only fill
df["Cabin"] = df["Cabin"].fillna("Unknown")

# 2- drop use less columns
# name, ticket, passengerId
# but we need to extract titles from names
df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
df.drop(["Name", "Ticket"], axis=1, inplace=True)

# 3- fix the types via encoding and all
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

df["Cabin"] = df["Cabin"].str[0] # firsr letter denotes deck
df = pd.get_dummies(df, columns=["Cabin"], drop_first=True)

df["Title"] = df["Title"].replace({
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Royalty", "Countess": "Royalty", "Sir": "Royalty",
    "Jonkheer": "Royalty", "Don": "Royalty",
    "Dr": "Officer", "Rev": "Officer", "Col": "Officer",
    "Major": "Officer", "Capt": "Officer"
})
title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Officer": 5, "Royalty": 6}
df["Title"] = df["Title"].map(title_map)
df["Title"] = df["Title"].fillna(0).astype(int) # fallback

bool_cols = df.select_dtypes(include="bool").columns
if len(bool_cols):
    df[bool_cols] = df[bool_cols].astype(int)

print(df.info())
df.to_csv("../data/cleanTest.csv", index=False)
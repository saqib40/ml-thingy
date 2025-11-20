import pandas as pd

df = pd.read_csv("../data/train.csv") # test.csv, train.csv

print(df.info())

# 1- fill the missing values

# for the age column
# median is safe because data is skewed
df["Age"] = df["Age"].fillna(df["Age"].median())

# for the embarked column
# mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# for the Cabin column
# i think meaningfull absence
# don't drop only fill
df["Cabin"] = df["Cabin"].fillna("Unknown")

# 2- drop use less columns
# name, ticket, passengerId
# but we need to extract titles from names
df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True) # for training add -> "PassengerId", 

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

bool_cols = df.select_dtypes(include="bool").columns
if len(bool_cols):
    df[bool_cols] = df[bool_cols].astype(int)

print(df.info())
df.to_csv("../data/cleanTrain.csv") # cleanTest.csv, cleanTrain.csv

# 3- do the standardisation for proper convergence 
# (since scikit-learn uses penalised regularisation internally
# where standardisation is needed)
# should technically do for all models relying on gradient-based optimization
# we will do that in the training example
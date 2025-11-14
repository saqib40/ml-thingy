import pandas as pd

# load
data = pd.read_csv("./data/train.csv")

# inspect by getting some info about the data
print(data.info())
print("*" * 80)

# fixing the missing values
# 1- dropping Cabin, since majorly null
data.drop("Cabin", axis=1, inplace=True)
# 2- mean for missing values of Age
data.fillna({"Age": data["Age"].mean()}, inplace=True)
# 3- Embarked?? only two are null so we will just drop them
data.dropna(subset=['Embarked'], inplace=True)

# inspect again
print(data.info())
print("*" * 80)

# some analysis
print("Percentage of people who survived:")
print(data["Survived"].mean()*100)

print("Survival rate of males vs females:")
print(data.groupby("Sex")["Survived"].mean()*100)

print("Survival rate of each passenger class")
print(data.groupby("Pclass")["Survived"].mean()*100)

print("Average age of those who survived vs. those who did not")
print(data.groupby("Survived")["Age"].mean())
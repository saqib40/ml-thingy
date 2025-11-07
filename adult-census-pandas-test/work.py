import pandas as pd

data = pd.read_csv("./data/adult.csv", na_values='?')

# droping null rows
data.dropna(inplace=True)

# education column -> ordinal data
print(data['education'].unique())
print("*" * 40)
# workclass column -> nominal data
print(data['workclass'].unique())
print("*" * 40)
# income column
print(data['income'].unique())
print("*" * 40)

# encoding
# 1- income column
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})

# 2- ordinal data
education_map = {
    'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3,
    '9th': 4, '10th': 5, '11th': 6, '12th': 7, 'HS-grad': 8,
    'Some-college': 9, 'Assoc-voc': 10, 'Assoc-acdm': 11,
    'Bachelors': 12, 'Masters': 13, 'Prof-school': 14, 'Doctorate': 15
}
data['education'] = data['education'].map(education_map)

# 3- nominal data
nominal_cols = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
data = pd.get_dummies(data, columns=nominal_cols)

print(data.info())
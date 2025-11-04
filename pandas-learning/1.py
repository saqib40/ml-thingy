# basics about series
# same as numpy array with associated index/label
# aka -> 1D Labeled Array

import pandas as pd

s1 = pd.Series([10, 20, 30, 40])
print(s1)

s2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s2)

s3 = pd.Series({"day1": 420, "day2": 380, "day3": 390})
print(s3)
# you can access elements of series either by label or by index
print(s3['day1'])
print(s3.iloc[0])
# basic aggregation and axis parameter

import numpy as np

arr = np.random.randint(0, 11, (4, 4))
print(arr)

# column
sumAggCol = arr.sum(axis=0)
print(sumAggCol)

# row
sumAggRow = arr.sum(axis=1)
print(sumAggRow)
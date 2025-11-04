# basic slicing and indexing with numpy

import numpy as np

arr1 = np.arange(25).reshape(5,5)

print(arr1)
print(arr1[2][3])
print(arr1[2, :])
print(arr1[:, 1])
print(arr1[0:2, 0:2])
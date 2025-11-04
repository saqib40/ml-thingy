# Basic Arrays with numpy

# np.array & np.ndarray
# np.ndarray is the fundamental N-dimensional array class, 
# while np.array is a function that constructs an instance 
# of the ndarray class

import numpy as np

arr1 = np.array([1,2,3,4,5])
arr2 = np.zeros(5, dtype=int)
arr3 = np.ones(5, dtype=int)
arr4 = np.arange(6)
print(arr1)
print(arr2)
print(arr3)
print(arr4)


arr5 = np.array([
    [1,2,3],
    [4,5,6]
])
arr6 = np.zeros((2,3), dtype=int)
arr7 = np.ones((2,3), dtype=int)
arr8 = np.random.rand(2,3)
# y=mx+b -> m = (upper bound - lower bound) & b = lower bound
arr8 = arr8 * (10-5) + 5
print(arr5)
print(arr6)
print(arr7)
print(arr8)
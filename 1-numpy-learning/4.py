# basic boolean indexing
import numpy as np

arr = np.random.randint(0,100,20)
print(arr)
print(arr[arr>50])
print(arr[arr%2 == 0])

# random 2d array
# replace all values greater than 10 with number 10
arr2 = np.random.randint(0, 21, (4, 4))
print(arr2)
mask = arr2 > 10 # mask same shape as of arr2 with boolean values
arr2[mask] = 10
print(arr2)
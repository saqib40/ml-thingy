# a very basic implementation of 
# signle feature linear regression
# as i understood it

import pandas as pd

sampleData = pd.DataFrame({
    "x" : [1, 2, 3],
    "y" : [2, 4, 5]
})

m, c = 0.0, 0.0
alpha = 0.1
epochs = 20
n = len(sampleData)

for t in range(epochs):
    sXE = 0.0
    sE  = 0.0
    for i in range(n):
        x = sampleData.loc[i, "x"]
        y = sampleData.loc[i, "y"]
        y_hat = m*x + c
        err = y - y_hat
        sXE += x * err
        sE  += err

    dj_dm = -(2/n) * sXE
    dj_dc = -(2/n) * sE

    # parameter update
    m -= alpha * dj_dm
    c -= alpha * dj_dc

print("m =", m, "c =", c)

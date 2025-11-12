from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv("../data/clean_train.csv")

model = LinearRegression()

# dependent columns
feature_cols = [col for col in data.columns if (col != "Id") and (col != "SalePrice")]
X = data[feature_cols]

# independent columns
y = np.log(data["SalePrice"])

# training the model
model.fit(X, y)

# save at ../output
out = {"model": model, "columns": feature_cols}
joblib.dump(out, "../output/linear_regression_model.pkl")

print("Model saved successfully")
import pandas as pd
import joblib
import numpy as np

# load test data
test = pd.read_csv("../data/clean_test.csv")
ids = test['Id'].copy()
X_test = test.drop(columns=['Id'])

# load the model
model_bundle = joblib.load("../output/linear_regression_model.pkl")
model = model_bundle['model']
train_columns = model_bundle['columns']

# Align columns: add missing columns with 0, drop extra columns not seen in training
X_test = X_test.reindex(columns=train_columns, fill_value=0)

# make predictions and save to ../data/result.csv
# format of result.csv -> Id,SalePrice
logPredictions = model.predict(X_test.values)
finalPredictions = np.exp(logPredictions)
finalDf = pd.DataFrame({
    "Id": ids,
    "SalePrice": finalPredictions
})
finalDf.to_csv("../data/prediction.csv", index=False)
print("Saved predictions to ../data/prediction.csv")
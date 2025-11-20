import pandas as pd
import joblib

# data loaded
test = pd.read_csv("../data/cleanTest.csv")
ids = test['PassengerId'].copy()
X_test = test.drop(columns=['PassengerId'])

# load model
model_bundle = joblib.load("../output/logistic_regression_model.pkl")
model = model_bundle['model']
train_columns = model_bundle['columns']

# Align columns: add missing columns with 0, drop extra columns not seen in training
X_test = X_test.reindex(columns=train_columns, fill_value=0)

# make predictions and save to ../data/prediction.csv
# format of result.csv -> PassengerId,SalePrice
predictions = model.predict(X_test)
finalDf = pd.DataFrame({
    "PassengerId": ids,
    "Survived": predictions
})
finalDf.to_csv("../data/prediction.csv", index=False)
print("Saved predictions to ../data/prediction.csv")
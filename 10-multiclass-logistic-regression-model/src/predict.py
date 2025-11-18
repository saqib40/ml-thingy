import pandas as pd
import joblib

test = pd.read_csv("../data/cleanTest.csv")
ids = test['ImageId'].copy()
X_test = test.drop(columns=['ImageId'])

model_bundle = joblib.load("../output/model.pkl")
model = model_bundle['model']
train_columns = model_bundle['columns']

X_test = X_test.reindex(columns=train_columns, fill_value=0)

predictions = model.predict(X_test)
finalDf = pd.DataFrame({
    "ImageId": ids,
    "Label": predictions
})
finalDf.to_csv("../data/prediction.csv", index=False)
print("Saved predictions to ../data/prediction.csv")
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("../data/cleanTrain.csv")
feature_cols = [col for col in df.columns if col != "label"]
X = df[feature_cols]
y = df["label"]

model_softmax = LogisticRegression(
    multi_class='multinomial', 
    solver='saga',
    max_iter=5000,
    n_jobs=-1
)

model_softmax.fit(X,y)

# save the model
out = {"model": model_softmax, "columns": feature_cols}
joblib.dump(out, "../output/model.pkl")

print("Model saved successfully")
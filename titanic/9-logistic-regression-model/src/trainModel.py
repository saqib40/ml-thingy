from sklearn.linear_model import LogisticRegression # for model
from sklearn.preprocessing import StandardScaler # for standardization
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

df = pd.read_csv("../data/cleanTrain.csv")
feature_cols = [col for col in df.columns if col != "Survived"]
X = df[feature_cols]
y = df["Survived"]

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight=None
    ))
])

pipe.fit(X, y) # model trained

# save the model
out = {"model": pipe, "columns": feature_cols}
joblib.dump(out, "../output/logistic_regression_model.pkl")

print("Model saved successfully")
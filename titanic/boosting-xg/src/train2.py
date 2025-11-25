from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("../data/cleanTrain.csv")
feature_cols = [col for col in df.columns if col != "Survived"]
y = df["Survived"]
X = df.drop("Survived", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    # Step 1: Tune Depth and Weight first (Most impact)
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    
    # Step 2: Tune Sampling (Secondary impact)
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
    
    # Keep these fixed for now to save time
    'n_estimators': [100],
    'learning_rate': [0.1]
}

xgb = XGBClassifier(
    objective='binary:logistic', 
    random_state=42,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=1.0,
    colsample_bytree=0.7,
    reg_lambda=5,
    reg_alpha=0.5,
    n_estimators=2000,          # high number on purpose
    tree_method='hist',
    eval_metric="auc",
    early_stopping_rounds=10,
)

model = xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    #early_stopping_rounds=10,
    verbose=False
)

print(f"Best Iteration: {xgb.best_iteration}")
print(f"Best Score: {xgb.best_score}")

# save the model
out = {"model": model, "columns": feature_cols}
joblib.dump(out, "../output/xgboost.pkl")

print("Model saved successfully")

"""
Best Iteration: 11
Best Score: 0.8913127413127413
"""
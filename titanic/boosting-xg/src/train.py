from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from xgboost import XGBClassifier

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
    'n_estimators': [11],
    'learning_rate': [0.1]
}

xgb = XGBClassifier(objective='binary:logistic', random_state=42)

clf = GridSearchCV(
    estimator=xgb, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy', 
    verbose=1, 
    n_jobs=-1
)

clf.fit(X_train, y_train)

print(f"Best Parameters: {clf.best_params_}")
print(f"Best Cross-Val Score: {clf.best_score_:.4f}")

best_model = clf.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")

"""
Fitting 5 folds for each of 36 candidates, totalling 180 fits
Best Parameters: {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 3, 'n_estimators': 100, 'subsample': 1.0}
Best Cross-Val Score: 0.8441
Test Set Accuracy: 0.8268
"""
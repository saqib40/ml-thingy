import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("../data/cleanTrain.csv")
feature_cols = [col for col in df.columns if col != "Survived"]
y = df["Survived"]
X = df.drop("Survived", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 110, 120, 130, 140, 150],  
    'max_depth': [None, 5, 10, 15, 20], 
    'min_samples_leaf': [1, 2, 4] 
}

clf = GridSearchCV(
    estimator=clf, 
    param_grid=param_grid, 
    cv=5,
    n_jobs=-1,
    verbose=1, # Show progress logs
    scoring='accuracy' # Metric to optimize
)

print("Starting Grid Search...")
clf.fit(X_train, y_train)

# 6. Output Results
print("--------------------------------------------")
print(f"Best Parameters: {clf.best_params_}")
print(f"Best Cross-Val Score: {clf.best_score_:.4f}")

# 7. Validate on the Test Set (The data the grid never saw)
best_model = clf.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")


"""
Starting Grid Search...
Fitting 5 folds for each of 90 candidates, totalling 450 fits
--------------------------------------------
Best Parameters: {'max_depth': 10, 'min_samples_leaf': 4, 'n_estimators': 100}
Best Cross-Val Score: 0.8356
"""
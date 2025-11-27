from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("../data/cleanData.csv")
feature_cols = [col for col in df.columns if col != "diagnosis"]
y = df["diagnosis"]
X = df.drop("diagnosis", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = [
    {
      'clf__kernel': ['rbf'],
      'clf__C': [1e-2, 1e-1, 1, 10, 20, 30, 50, 100],
      'clf__gamma': ['scale', 'auto', 1e-3, 1e-2, 1e-1, 0.02, 0.05]
    },
    {
      'clf__kernel': ['linear'],
      'clf__C': [1e-3, 1e-2, 1e-1, 1, 10]
    },
    {
      'clf__kernel': ['poly'],
      'clf__C': [0.1, 1, 10],
      'clf__degree': [2,3,4]
    }
]

pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC())
])

gs = GridSearchCV(pipeline,
                  param_grid,
                  cv=cv,
                  scoring='accuracy',
                  n_jobs=-1,
                  verbose=2,
                  return_train_score=False)

gs.fit(X_train, y_train)

print(f"Best Parameters: {gs.best_params_}")
print(f"Best Cross-Val Score: {gs.best_score_:.4f}")

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("Train accuracy:", accuracy_score(y_train, best_model.predict(X_train)))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
"""
Best Parameters: {'clf__C': 10, 'clf__gamma': 0.01, 'clf__kernel': 'rbf'}
Best Cross-Val Score: 0.9780
Test Set Accuracy: 0.9825
"""
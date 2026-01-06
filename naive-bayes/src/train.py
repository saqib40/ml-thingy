import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import precision_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer

df = pd.read_csv("../data/cleanData.csv")

df['v2'] = df['v2'].fillna("").astype(str) # something was float somewhere

y = df["v1"]
X = df[["v2", "length"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('text', CountVectorizer(), 'v2'),     # text → bag of words
        ('length', 'passthrough', ['length']) # numeric → keep as-is
    ]
)

pipeline = Pipeline([
    ('features', preprocessor),
    ('clf', ComplementNB())
])
param_grid = {'clf__alpha': [0.1, 0.5, 1.0, 2.0, 10.0]}

gs = GridSearchCV(pipeline,
                  param_grid,
                  cv=cv,
                  scoring='precision',
                  n_jobs=-1,
                  verbose=2,
                  return_train_score=False)

gs.fit(X_train, y_train)

print(f"Best Parameters: {gs.best_params_}")
print(f"Best Cross-Val Score: {gs.best_score_:.4f}")

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test Set Precision: {precision_score(y_test, y_pred):.4f}")

print("Train Precision:", precision_score(y_train, best_model.predict(X_train)))
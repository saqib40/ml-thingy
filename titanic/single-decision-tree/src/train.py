import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

df = pd.read_csv("../data/cleanTrain.csv")
# print(df.info())
feature_cols = [col for col in df.columns if col != "Survived"]
y = df["Survived"]
X = df.drop("Survived", axis=1)

"""
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for depth in [1,3,4,10,20]:
    clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=42)
    acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"Depth {depth}: Accuracy {acc_scores.mean()}")
    f1_macro_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
    print(f"Depth {depth}: F1 macro {f1_macro_scores.mean()}")
    f1_weighted_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_weighted')
    print(f"Depth {depth}: F1 weighted {f1_weighted_scores.mean()}")
"""

# at depth 3 we get the best results
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model = clf.fit(X,y)

# save the model
out = {"model": model, "columns": feature_cols}
joblib.dump(out, "../output/single_decision_tree_model.pkl")

print("Model saved successfully")
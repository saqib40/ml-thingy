import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

df = pd.read_csv("../data/clean.csv")
y = df["Drug"]
X = df.drop("Drug", axis=1)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# workaround, use GridSearchCV for the hyperparameters
for depth in [1,3,4,10,20]:
    clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=42)
    acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"Depth {depth}: Accuracy {acc_scores.mean()}")
    f1_macro_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
    print(f"Depth {depth}: F1 macro {f1_macro_scores.mean()}")
    f1_weighted_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_weighted')
    print(f"Depth {depth}: F1 weighted {f1_weighted_scores.mean()}")


# so max_depth=4 is the game
# Depth 4: Accuracy 0.9899999999999999
# Depth 4: F1 macro 0.986462519936204
# Depth 4: F1 weighted 0.9895709728867624
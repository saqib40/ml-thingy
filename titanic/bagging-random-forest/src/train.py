import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/cleanTrain.csv")
feature_cols = [col for col in df.columns if col != "Survived"]
y = df["Survived"]
X = df.drop("Survived", axis=1)

# let's just figure out the best number of trees
# not the best way use gridsearchcv
"""
for n in [200, 300, 400, 500, 600, 700, 800]:
    clf = RandomForestClassifier(
        n_estimators=n,
        max_features='sqrt',
        max_depth=None,
        min_samples_leaf=1,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X, y)
    print(f"n: {n}: OOB score: {clf.oob_score_}")

n: 70: OOB score: 0.813692480359147
n: 80: OOB score: 0.813692480359147

n: 110: OOB score: 0.8159371492704826
n: 120: OOB score: 0.8159371492704826

n: 130: OOB score: 0.813692480359147

n: 150: OOB score: 0.813692480359147
n: 160: OOB score: 0.813692480359147
n: 170: OOB score: 0.813692480359147
"""

"""

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for n in [70, 80, 110, 120, 130, 150, 160, 170]:
    clf = RandomForestClassifier(
        n_estimators=n,
        max_features='sqrt',
        max_depth=None,
        min_samples_leaf=1,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"n {n}: Accuracy {acc_scores.mean()}")
    f1_macro_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
    print(f"n {n}: F1 macro {f1_macro_scores.mean()}")
    f1_weighted_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_weighted')
    print(f"n {n}: F1 weighted {f1_weighted_scores.mean()}")


n 80: Accuracy 0.8203495630461923
n 80: F1 macro 0.8081620524187535
n 80: F1 weighted 0.8192615990650861
n 110: Accuracy 0.8248564294631711
n 110: F1 macro 0.8121012664327335
n 110: F1 weighted 0.8233603266336449
n 120: Accuracy 0.824856429463171
n 120: F1 macro 0.8122788395189963
n 120: F1 weighted 0.8234353072200434
n 130: Accuracy 0.823732833957553
n 130: F1 macro 0.8110100218381323
n 130: F1 weighted 0.8222774622114523
n 150: Accuracy 0.8203745318352059
n 150: F1 macro 0.807805187096705
n 150: F1 weighted 0.8190892295284963
n 160: Accuracy 0.8203745318352059
n 160: F1 macro 0.8079606727710587
n 160: F1 weighted 0.819150316691555
n 170: Accuracy 0.8192384519350812
n 170: F1 macro 0.8067255074963544
n 170: F1 weighted 0.8180137079305465
"""

clf = RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        max_depth=10,
        min_samples_leaf=4,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )

model = clf.fit(X, y)

# save the model
out = {"model": model, "columns": feature_cols}
joblib.dump(out, "../output/random_forest.pkl")

print("Model saved successfully")
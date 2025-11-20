from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA

df = pd.read_csv("../data/train.csv")
X = df.drop("price_range", axis=1)
y = df["price_range"]

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ("pca", PCA(n_components=0.85)),   # 85% variance
    ('knn', KNeighborsClassifier(n_neighbors=40))
])

# stratified k-fold is preferred for classification
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# accuracy
acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Accuracy: {acc_scores.mean()*100:.2f}% ± {acc_scores.std()*100:.2f}%")

# f1 macro (treats all classes equally)
f1_macro_scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"F1 (macro): {f1_macro_scores.mean()*100:.2f}% ± {f1_macro_scores.std()*100:.2f}%")

# f1 weighted (accounts for class-support)
f1_weighted_scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
print(f"F1 (weighted): {f1_weighted_scores.mean()*100:.2f}% ± {f1_weighted_scores.std()*100:.2f}%")

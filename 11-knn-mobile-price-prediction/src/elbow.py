# figuring out the best k
# using elbow method
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv("../input/training/train.csv") # "../data/train.csv"
y = df["price_range"]
X = df.drop("price_range", axis=1)

error_rate = []
k_range = range(1, 40) 

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for i in k_range:
    #knn = KNeighborsClassifier(n_neighbors=i)
    #knn.fit(X_train, y_train)
    #pred_i = knn.predict(X_test)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ("pca", PCA(n_components=0.85)),
        ('knn', KNeighborsClassifier(n_neighbors=i))
    ])
    #pipe.fit(X_train, y_train)
    #pred_i = pipe.predict(X_test)
    #error = np.mean(pred_i != y_test)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    error = 1 - scores.mean() # error = 1 - accuracy
    error_rate.append(error)

plt.figure(figsize=(10,6))
plt.plot(k_range, error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.xticks(k_range, rotation=45, ha='right')
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
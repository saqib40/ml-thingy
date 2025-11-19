# figuring out the best k
# using elbow method

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("../data/train.csv")
y = df["price_range"]
feature_columns = [x for x in df.columns if x != "price_range"]
X= df[feature_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


error_rate = []
k_range = range(1, 40) 

for i in k_range:
    #knn = KNeighborsClassifier(n_neighbors=i)
    #knn.fit(X_train, y_train)
    #pred_i = knn.predict(X_test)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=i))
    ])
    pipe.fit(X_train, y_train)
    pred_i = pipe.predict(X_test)

    error = np.mean(pred_i != y_test)
    
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

# lowest on => 34

# Performance
#Accuracy: 61.75% ± 2.42%
#F1 (macro): 61.84% ± 2.35%
#F1 (weighted): 61.84% ± 2.35%


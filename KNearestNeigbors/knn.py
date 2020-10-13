from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv('./Social_Network_Ads.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25)


sc = StandardScaler()
x_tr = sc.fit_transform(x_tr)
x_te = sc.transform(x_te)


# metricminkowski + p=2 = euclidian distance
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_tr, y_tr)

y_pred = classifier.predict(x_te)


cm = confusion_matrix(y_te, y_pred)
print(cm)
print(accuracy_score(y_te, y_pred))

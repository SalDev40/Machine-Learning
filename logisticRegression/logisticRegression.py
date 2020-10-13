from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


df = pd.read_csv('Social_Network_Ads.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0)


sc = StandardScaler()
x_tr = sc.fit_transform(x_tr)
x_te = sc.transform(x_te)


classifier = LogisticRegression(random_state=0)
classifier.fit(x_tr, y_tr)

yPred = classifier.predict(x_te)


# Making the Confusion Matrix
cm = confusion_matrix(y_te, yPred)
print(cm)
print(accuracy_score(y_te, yPred))

print("accuracy: ", 1 - (sum(np.square(y_te - yPred)) / len(y_te)))
# print(np.concatenate((yPred.reshape(len(yPred), 1), y_te.reshape(len(yPred), 1)), 1))

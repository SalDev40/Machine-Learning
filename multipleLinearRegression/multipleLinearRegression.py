from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd


df = pd.read_csv('./50_Startups.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0)
# print(x_tr)

reg = LinearRegression()
reg.fit(x_tr, y_tr)
y_pred = reg.predict(x_te)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1),
                      y_te.reshape(len(y_te), 1)), 1))

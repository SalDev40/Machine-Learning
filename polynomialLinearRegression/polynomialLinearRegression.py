from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./Position_Salaries.csv')
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values


linReg = LinearRegression()
linReg.fit(x, y)


polyReg2 = PolynomialFeatures(degree=2)
linReg2 = LinearRegression()
linReg2.fit(polyReg2.fit_transform(x), y)

polyReg3 = PolynomialFeatures(degree=3)
linReg3 = LinearRegression()
linReg3.fit(polyReg3.fit_transform(x), y)


plt.scatter(x, y, color = "red")
plt.plot(x, linReg.predict(x), color="blue")
plt.xlabel('position (degree 1)')
plt.ylabel('salary')
# plt.show()


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, linReg2.predict(polyReg2.fit_transform(X_grid)), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
# plt.show()


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, linReg3.predict(polyReg3.fit_transform(X_grid)), color = 'blue')
plt.xlabel('Position level (degree 3)')
plt.ylabel('Salary')
# plt.show()


# Predicting a new result with Linear Regression
# [[]] -> array
print(linReg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(linReg2.predict(polyReg2.fit_transform([[6.5]])))
print(linReg3.predict(polyReg3.fit_transform([[6.5]])))
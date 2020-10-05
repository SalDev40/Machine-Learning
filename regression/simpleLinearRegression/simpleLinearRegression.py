import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""## Importing the dataset"""

df = pd.read_csv('Salary_Data.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)

"""## Training the Simple Linear Regression model on the Training set"""

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train, y_train)

"""## Predicting the Test set results"""

yPred = reg.predict(x_test)
print("Predicted y: ", yPred)
print("Actual y: ", y_test)
print("Difference: ", y_test - yPred)

"""## Visualising the Training set results"""

plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, reg.predict(x_train), color="blue")
plt.title("Salary vs YE (Train)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

"""## Visualising the Test set results"""

plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, reg.predict(x_train), color="blue")
plt.title("Salary vs YE (Train)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

import numpy as np
import pandas as pd


# Importing the dataset
df = pd.read_csv('Data.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(x)

# Taking care of missing data
from sklearn.impute import SimpleImputer
x[:,1:3] = SimpleImputer(missing_values=np.nan,strategy="mean").fit_transform(x[:,1:3])


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

# split data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])

#since test is like "newdata" we only transform data we can't fit it again
#with new scaler, we need the same scaler from training set to get accurate result
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)

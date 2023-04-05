import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
BostonTrain = pd.read_csv("boston_test.csv")
print(BostonTrain.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = BostonTrain[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'black', 'lstat']]
y = BostonTrain['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)

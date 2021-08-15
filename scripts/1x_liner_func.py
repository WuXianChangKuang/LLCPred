import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def stdError_func(y_test, y):
  return np.sqrt(np.mean((y_test - y) ** 2))


def R2_1_func(y_test, y):
  return 1 - ((y_test - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()


def R2_2_func(y_test, y):
  y_mean = np.array(y)
  y_mean[:] = y.mean()
  return 1 - stdError_func(y_test, y) / stdError_func(y_mean, y)

filename = "cycle_cpuuse.csv"
df= pd.read_csv(filename)
x = np.array(df.iloc[:,1].values)

y = np.array(df.iloc[:,0].values)

cft = linear_model.LinearRegression()
cft.fit(x[:,np.newaxis], y) 

print("model coefficients", cft.coef_)
print("model intercept", cft.intercept_)

predict_y =  cft.predict(x[:,np.newaxis])
strError = stdError_func(predict_y, y)
R2_1 = R2_1_func(predict_y, y)
R2_2 = R2_2_func(predict_y, y)
score = cft.score(x[:,np.newaxis], y)

print(' strError={:.2f}, R2_1={:.2f},  R2_2={:.2f}, clf.score={:.2f}'.format(
    strError,R2_1,R2_2,score))

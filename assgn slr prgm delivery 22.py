# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 01:04:25 2022

@author: aditya
"""
simple linear regression to predict delivery time
import pandas as pd
import numpy as np
import seaborn as sns
data=pd.read_csv("delivery_time.csv")
data.shape
list(data)

#splitting data into x and y variables
x=data["Sorting Time"] #x---independent variable
x=x[:,np.newaxis] #converting x from 1D to 2D
x.ndim
sns.displot(x)
data["Sorting Time"].skew()
data["Sorting Time"].kurtosis()

y=data["Delivery Time"] #y---dependent variable
sns.displot(y)
data["Delivery Time"].skew()
data["Delivery Time"].kurtosis()

import matplotlib.pyplot as plt
plt.scatter(y="Delivery Time",x="Sorting Time",data=data) #scatter plot
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.title("Scatter Plot")
plt.show()

#fitting the model using scikit learn package
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(x,y)
model.intercept_
model.coef_
y_pred=model.predict(x)

#plot
import matplotlib.pyplot as plt
plt.scatter(x,y,color="Blue") #acutal data points
plt.plot(x,y_pred,color="Red") #prediction line
plt.show()

#metrics
from sklearn.metrics import mean_squared_error,r2_score
MSE=mean_squared_error(y,y_pred)
r2=r2_score(y,y_pred)
RMSE=np.sqrt(MSE)
print(MSE)
print(RMSE)
print(r2)

#fitting the model using statsmodels package
import statsmodels.api as sma
model_1=sma.OLS(x,y).fit()
y_pred_1=model_1.predict(x)
model_1.summary()


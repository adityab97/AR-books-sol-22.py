# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:31:02 2022

@author: aditya
"""

#Simple linear regression to predict delivery time
import pandas as pd
import numpy as np
import seaborn as sns
data=pd.read_csv("F:\\ExcelR\\Assignments\\Simple Linear regression\\delivery_time.csv")
data.shape  
list(data)

#Splitting data into X and Y variables
X=data['Sorting Time'] #X-->independent variable
X=X[:,np.newaxis] #Converting X from 1D to 2D 
X.ndim
sns.distplot(X)
data['Sorting Time'].skew() #The skewness value is 0.047 it is under the acceptable range of -0.5 to +0.5
                            #So, it can be considered to follow normal distribution
data['Sorting Time'].kurtosis()

Y=data['Delivery Time'] #Y--> dependent variable
sns.distplot(Y)
data['Delivery Time'].skew() #The skewness value is 0.352 it is under the acceptable range of -0.5 to +0.5
                             #So, it can be considered to follow normal distribution
data['Delivery Time'].kurtosis()

import matplotlib.pyplot as plt
plt.scatter(y='Delivery Time',x='Sorting Time',data=data) #Scatter Plot
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Scatter Plot')
plt.show()

#Fitting the model using scikit learn package
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(X,Y)
model.intercept_
model.coef_
Y_pred=model.predict(X)

#Plot
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='Blue') #Actual data points
plt.plot(X,Y_pred,color='Red') #Prediction line
plt.show()

#Metrics
from sklearn.metrics import mean_squared_error,r2_score
MSE=mean_squared_error(Y,Y_pred)
r2=r2_score(Y,Y_pred)
RMSE=np.sqrt(MSE)
print(MSE)
print(RMSE)
print(r2)

#Fitting the using statsmodels package
import statsmodels.api as sma
model1=sma.OLS(X,Y).fit()
Y_pred1=model1.predict(X)
model1.summary()
'''
Inference: The Mean Square Error using the scikit learn package
           is 7.793 and Root Mean Square Error is 2.791 and r2 is around 68%
           The R2 score by using statsmodels package is around 95%
'''
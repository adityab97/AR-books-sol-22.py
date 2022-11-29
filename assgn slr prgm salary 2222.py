# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:48:39 2022

@author: aditya
"""
#simple linear regression to predict salary
import pandas as pd
import numpy as np
data=pd.read_csv("Salary_Data.csv")
data.shape
list(data)
data.corr()

#exploratory data analaysis
import seaborn as sns
x=data["YearsExperience"]
x=x[:,np.newaxis]
x.ndim
y=data["Salary"]
sns.displot(data["YearsExperience"])
data["YearsExperience"].hist()
data["YearsExperience"].skew()
data["YearsExperience"].describe()

sns.displot(data["Salary"])
data["Salary"].hist()
data["Salary"].skew()
data["Salary"].describe()

#scatter plot
data.plot.scatter(x="YearsExperience",y="Salary")

#fitting the model
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(x,y)
model.intercept_
model.coef_
y_pred=model.predict(x)

#To draw plots
import matplotlib.pyplot as plt
plt.scatter(x,y,color="Blue")
plt.scatter(x,y,color="Red")
plt.show()

#finding error
y_error=y-y_pred
sns.displot(y_error)

#metrics
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y,y_pred)
print(MSE)
from math import sqrt
RMSE=np.sqrt(MSE)
print(RMSE)
#####################################################################################
#applying transformations on x variable
#exploratory data analysis
data["Sq YE"]=np.sqrt(data["YearsExperience"])
x1=data["Sq YE"]
x1=x1[:,np.newaxis]
x1.ndim
sns.displot(x1)
data["Sq YE"].skew()
data["Sq YE"].describe()

data["lg Salary"]=np.log(data["Salary"])
Y1=data["lg Salary"]
sns.displot(Y1)
data["lg Salary"].skew()
data["lg Salary"].describe()

#scatter plot
data.plot.scatter(x="Sq YE",y="lg Salary")

#fitting the model
from sklearn.linear_model import LinearRegression
model1=LinearRegression().fit(x1,Y1)
model1.intercept_
model1.coef_
y_pred1=model.predict(x1)

sns.regplot(x=x1,y=Y1,color="Blue")

#finding the error
y_error1=y_pred-Y1
sns.displot(y_error1)

#plot
import matplotlib.pyplot as plt
plt.scatter(x1,Y1,color="Blue")
plt.plot(x1,Y1,color="Red")
plt.show()

#metrics
from sklearn.metrics import mean_squared_error
MSE1=mean_squared_error(Y1,y_pred1)
print(MSE1)
from math import sqrt
RMSE1=np.sqrt(MSE1)
print(RMSE1)

#fitting the using statsmodels package
import statsmodels.api as sma
model2=sma.OLS(x1,Y1).fit()
y_pred2=model2.predict(x1)
model2.summary()

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:14:52 2022

@author: aditya
"""
#multiple linear regression to predict profit
import pandas as pd
import numpy as np
import seaborn as sns
data=pd.read_csv("50_Startups.csv")
data.shape
list(data)
data.dtypes
#firstly trying to drop the state column from the data and then fitting the model
data.drop(["State"],axis=1)

data.corr()

#Splitting data into x and y variables
x=data.iloc[:,0:3]
list(x)
x.hist()
sns.displot(x["R&D Spend"])
x["R&D Spend"].skew() #skewness is 0.164(positively skewed),its under acceptable limit of -0.5 to +0.5
                       #the distribution can be accepted as normal.
sns.displot(x["Administration"]) #skewness is -0.489(negatively skewed),its under acceptable limit og -0.5 to +0.5
                                  #the distribution can be accepted as normal.
sns.displot(x["Marketing Spend"]) 
x["Marketing Spend"].skew() #skewness is -0.046(negatively skewed),its under acceptable limit of -0.5 to +0.5 
                             #the distribution can be accepted as normal.
y=data["Profit"] #target variable

#as per the correlation coefficients,considering the x variable and its combination
x1=data["R&D Spend"]
x1=x1[:,np.newaxis]
x1.ndim

#fitting the model
from sklearn.linear_model import LinearRegression
model1=LinearRegression().fit(x1,y)
y_pred1=model1.predict(x1)

y_error1=y-y_pred1
sns.distplot(y_error1)

from sklearn.metrics import r2_score
r2a=r2_score(y,y_pred1)
print(r2a) #R2 score considering 1 variable in x is 94.65
############################################################################
#2 variables in x
x2=data[["R&D Spend","Marketing Spend"]]

#fitting the model
from sklearn.linear_model import LinearRegression
model2=LinearRegression().fit(x2,y)
y_pred2=model2.predict(x2)

y_error2=y-y_pred2
sns.distplot(y_error2)

from sklearn.metrics import r2_score
r2b=r2_score(y,y_pred2)
print(r2b) #R2 score considering 1 variable in x is 95.04
###################################################################################
#3 variables in x
x3=data[["R&D Spend","Marketing Spend","Administration"]]

#fitting the model
from sklearn.linear_model import LinearRegression
model3=LinearRegression().fit(x3,y)
y_pred3=model3.predict(x3)

y_error3=y-y_pred3
sns.distplot(y_error3)

from sklearn.metrics import r2_score
r2c=r2_score(y,y_pred3)
print(r2c) #R2 score considering 1 variable in x is 95.07
####################################################################################
#2 variables in x
x4=data[["R&D Spend","Administration"]]

#fitting the model
from sklearn.linear_model import LinearRegression
model4=LinearRegression().fit(x4,y)
y_pred4=model4.predict(x4)

y_error4=y-y_pred4
sns.displot(y_error4)

from sklearn.metrics import r2_score
r2d=r2_score(y,y_pred4)
print(r2d) #R2 score considering 1 variable in x is 94.78

table={"x variables":pd.Series(["R&D Spend","R&D Spend and Marketing Spend","R&D Spend,Marketing Spend and Administration","R&D spend and Administration"]),"R2 score":pd.Series([(r2a*100),(r2b*100),(r2c*100),(r2d*100)])}
type(table)

r2_table=pd.DataFrame(table)
r2_table

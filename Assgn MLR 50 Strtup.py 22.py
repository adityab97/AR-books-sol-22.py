# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:24:39 2022

@author: aditya
"""


#Multiple Linear Regression to predict profit

import pandas as pd
import numpy as np
import seaborn as sns

data=pd.read_csv("F:\\ExcelR\\Assignments\\Multiple Linear regression\\50_Startups.csv")
data.shape
list(data)
data.dtypes
#Firstly trying to drop the State column from the data and then fitting the model
data.drop(['State'],axis=1)

data.corr()

#Splitting data into X and Y variables
X=data.iloc[:,0:3]
list(X)
X.hist()
sns.distplot(X['R&D Spend'])
X['R&D Spend'].skew() #Skewness is 0.164 (Positively skewed), Its under acceptable limit of -0.5 to +0.5
                      #The distribution can be accepted as Normal.

sns.distplot(X['Administration'])
X['Administration'].skew() #Skewness is -0.489 (Negatively skewed), Its under acceptable limit of -0.5 to +0.5
                      #The distribution can be accepted as Normal.

sns.distplot(X['Marketing Spend'])
X['Marketing Spend'].skew() #Skewness is -0.046 (Negatively skewed), Its under acceptable limit of -0.5 to +0.5
                      #The distribution can be accepted as Normal.

Y=data['Profit'] #Target variable

#As per the correlation coefficients, considering the X variable and its combination
X1=data['R&D Spend']
X1=X1[:,np.newaxis]
X1.ndim

#Fitting the model
from sklearn.linear_model import LinearRegression
model1=LinearRegression().fit(X1,Y)
Y_pred1=model1.predict(X1)

Y_err1=Y-Y_pred1
sns.distplot(Y_err1)

from sklearn.metrics import r2_score
r2a=r2_score(Y,Y_pred1)
print(r2a) #R2 score considering 1 variable in X is 94.65
###############################################################################################
#2 variables in X
X2=data[['R&D Spend','Marketing Spend']]

#Fitting the model
from sklearn.linear_model import LinearRegression
model2=LinearRegression().fit(X2,Y)
Y_pred2=model2.predict(X2)

Y_err2=Y-Y_pred2
sns.distplot(Y_err2)

from sklearn.metrics import r2_score
r2b=r2_score(Y,Y_pred2)
print(r2b) #R2 score considering 1 variable in X is 95.04
###############################################################################################
#3 variables in X
X3=data[['R&D Spend','Marketing Spend','Administration']]

#Fitting the model
from sklearn.linear_model import LinearRegression
model3=LinearRegression().fit(X3,Y)
Y_pred3=model3.predict(X3)

Y_err3=Y-Y_pred3
sns.distplot(Y_err3)

from sklearn.metrics import r2_score
r2c=r2_score(Y,Y_pred3)
print(r2c) #R2 score considering 1 variable in X is 95.07
###############################################################################################
#2 variables in X
X4=data[['R&D Spend','Administration']]

#Fitting the model
from sklearn.linear_model import LinearRegression
model4=LinearRegression().fit(X4,Y)
Y_pred4=model4.predict(X4)

Y_err4=Y-Y_pred4
sns.distplot(Y_err4)

from sklearn.metrics import r2_score
r2d=r2_score(Y,Y_pred4)
print(r2d) #R2 score considering 1 variable in X is 94.78

table={'X Variables':pd.Series(['R&D Spend','R&D Spend and Marketing Spend','R&D Spend, Marketing Spend and Administration','R&D Spend and Administration']),'R2 score':pd.Series([(r2a*100),(r2b*100),(r2c*100),(r2d*100)])}
type(table)

r2_table=pd.DataFrame(table)
r2_table
'''
Inference: The variables of X i.e., R&D Spend and Marketing Spend are giving out good r2 score
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:26:15 2022

@author: aditya
"""

#Multiple Linear Regression to predict price

import pandas as pd
import numpy as np
import seaborn as sns

data=pd.read_csv("F:\\ExcelR\\Assignments\\Multiple Linear regression\\ToyotaCorolla.csv",encoding='latin1')
data.shape
list(data)
data.corr().Price
'''
As mentioned in the prob statement we are considering only the variables given 
According to the correlation coefficients, the important parameter
among the given X variables are noted:
1) Age_08_04         -0.876590  
2) Weight            0.581198
3) KM                -0.569960  
4) HP                 0.314990 
5) Quarterly_Tax      0.219197 
6) Doors              0.185326  
7) cc                 0.126389  
8) Gears              0.063104  
'''
#Splitting the data into X and Y
#X=data[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
Y=data['Price']
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#1 variable in X
X1=data['Age_08_04']
X1=X1[:,np.newaxis]
X1.ndim
data['Age_08_04'].hist()

#Fitting the model
model1=LinearRegression().fit(X1,Y)
Y_pred1=model1.predict(X1)

Y_err1=Y-Y_pred1
sns.distplot(Y_err1) #FErrors are following Normal Distribution

r2a=r2_score(Y,Y_pred1)
print(r2a)
######################################################################################
#2 variables in X-->Age, Weight
X2=data[['Age_08_04','Weight']]
X2.hist()
X2.skew()

model2=LinearRegression().fit(X2,Y)
Y_pred2=model2.predict(X2)

Y_err2=Y-Y_pred2
sns.distplot(Y_err2)

r2b=r2_score(Y,Y_pred2)
print(r2b)
######################################################################################
#3 variables in X-->Age, Weight, Km
X3=data[['Age_08_04','Weight','KM']]
X3.hist()
X3.skew()

model3=LinearRegression().fit(X3,Y)
Y_pred3=model3.predict(X3)

Y_err3=Y-Y_pred3
sns.distplot(Y_err3)

r2c=r2_score(Y,Y_pred3)
print(r2c)
######################################################################################
#4 variables in X-->Age, Weight, Km, HP
X4=data[['Age_08_04','Weight','KM','HP']]
X4.hist()
X4.skew()

model4=LinearRegression().fit(X4,Y)
Y_pred4=model4.predict(X4)

Y_err4=Y-Y_pred4
sns.distplot(Y_err4)

r2d=r2_score(Y,Y_pred4)
print(r2d)
######################################################################################
#5 variables in X-->Age, Weight, Km, HP, Quarterly_Tax
X5=data[['Age_08_04','Weight','KM','HP','Quarterly_Tax']]
X5.hist()
X5.skew()

model5=LinearRegression().fit(X5,Y)
Y_pred5=model5.predict(X5)

Y_err5=Y-Y_pred5
sns.distplot(Y_err5)

r2e=r2_score(Y,Y_pred5)
print(r2e)
######################################################################################
#6 variables in X-->Age, Weight, Km, HP, Quarterly_Tax,Doors
X6=data[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors']]
X6.hist()
X6.skew()

model6=LinearRegression().fit(X6,Y)
Y_pred6=model6.predict(X6)

Y_err6=Y-Y_pred6
sns.distplot(Y_err6)

r2f=r2_score(Y,Y_pred6)
print(r2f)
######################################################################################
#7 variables in X-->Age, Weight, Km, HP, Quarterly_Tax,Doors,cc
X7=data[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors','cc']]
X7.hist()
X7.skew()

model7=LinearRegression().fit(X7,Y)
Y_pred7=model7.predict(X7)

Y_err7=Y-Y_pred7
sns.distplot(Y_err7)

r2g=r2_score(Y,Y_pred7)
print(r2g)
######################################################################################
#8 variables in X-->Age, Weight, Km, HP, Quarterly_Tax,Doors,cc,Gears
X8=data[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors','cc','Gears']]
X8.hist()
X8.skew()

model8=LinearRegression().fit(X8,Y)
Y_pred8=model8.predict(X8)

Y_err8=Y-Y_pred8
sns.distplot(Y_err8)

r2h=r2_score(Y,Y_pred8)
print(r2h)
######################################################################################
#9 combination of variables in X-->Age, Weight, Km, HP,Quarterly_Tax,Gears
X9=data[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Gears']]
X9.hist()
X9.skew()

model9=LinearRegression().fit(X9,Y)
Y_pred9=model9.predict(X9)

Y_err9=Y-Y_pred9
sns.distplot(Y_err9)

r2i=r2_score(Y,Y_pred9)
print(r2i)
######################################################################################
table={'X Variables':pd.Series(['Age_08_04','Age_08_04,Weight','Age_08_04,Weight,KM','Age_08_04,Weight,KM,HP','Age_08_04,Weight,KM,HP,Quarterly_Tax','Age_08_04,Weight,KM,HP,Quarterly_Tax,Doors','Age_08_04,Weight,KM,HP,Quarterly_Tax,Doors,cc','Age_08_04,Weight,KM,HP,Quarterly_Tax,Doors,cc,Gears','Age_08_04,Weight,KM,HP,Quarterly_Tax,Gears']),'R2 score':pd.Series([(r2a*100),(r2b*100),(r2c*100),(r2d*100),(r2e*100),(r2f*100),(r2g*100),(r2h*100),(r2i*100)])}
type(table)

r2_table=pd.DataFrame(table)
r2_table
'''
Inference: The 9th combination of X variables is giving out the best r2 score is 86.35
            by considering 6 variables.
'''
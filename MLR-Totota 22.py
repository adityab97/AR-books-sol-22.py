# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 01:29:39 2022

@author: aditya
"""
#multiple linear Regression to pedict price
import pandas as pd
import numpy as np
import seaborn as sns
data=pd.read_csv("ToyotaCorolla.csv",encoding="latin1")
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
y=data["Price"]
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#1 variable in x
x1=data["Age_08_04"]
x1=x1[:,np.newaxis]
x1.ndim
data["Age_08_04"].hist()

#fitting the model
model1=LinearRegression().fit(x1,y)
y_pred1=model1.predict(x1)

y_error1=y-y_pred1
sns.distplot(y_error1) #errors are following normal distribution
 r2a=r2_score(y,y_pred1)
print(r2a)
###########################################################################
#2 variables in x-->Age,Weight
x2=data[["Age_08_04","Weight"]]
x2.hist()
x2.skew()

model2=LinearRegression().fit(x2,y)
y_pred2=model2.predict(x2)

y_error2=y-y_pred2
sns.distplot(y_error2)

r2b=r2_score(y,y_pred2)
print(r2b)
####################################################################################
#3 variable in x-->Age,Weight,km
x3=data[["Age_08_04","Weight","KM"]]
x3.hist()
x3.skew()

model3=LinearRegression().fit(x3,y)
y_pred3=model3.predict(x3)

y_error3=y-y_pred3
sns.distplot(y_error3)

r2c=r2_score(y,y_pred3)
print(r2c)
##############################################
#4 variables in x-->Age,Weight,km,Hp
x4=data[["Age_08_04","Weight","KM","HP"]]
x4.hist()
x4.skew()

model4=LinearRegression().fit(x4,y)
y_pred4=model4.predict(x4)

y_error4=y-y_pred4
sns.distplot(y_error4)

r2d=r2_score(y,y_pred4)
print(r2d)
#############################################################
#5 variables in X-->Age, Weight, Km, HP, Quarterly_Tax
x5=data[['Age_08_04','Weight','KM','HP','Quarterly_Tax']]
x5.hist()
x5.skew()

model5=LinearRegression().fit(x5,y)
y_pred5=model5.predict(x5)

y_err5=y-y_pred5
sns.distplot(y_err5)

r2e=r2_score(y,y_pred5)
print(r2e)
####################################################################
#6 variables in X-->Age, Weight, Km, HP, Quarterly_Tax,Doors
x6=data[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors']]
x6.hist()
x6.skew()

model6=LinearRegression().fit(x6,y)
y_pred6=model6.predict(x6)

y_err6=y-y_pred6
sns.distplot(y_err6)

r2f=r2_score(y,y_pred6)
print(r2f)
#######################################################################
#7 variables in X-->Age, Weight, Km, HP, Quarterly_Tax,Doors,cc
x7=data[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors','cc']]
x7.hist()
x7.skew()

model7=LinearRegression().fit(x7,y)
y_pred7=model7.predict(x7)

y_err7=y-y_pred7
sns.distplot(y_err7)

r2g=r2_score(y,y_pred7)
print(r2g)
###########################################################################
#8 variables in X-->Age, Weight, Km, HP, Quarterly_Tax,Doors,cc,Gears
x8=data[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Doors','cc','Gears']]
x8.hist()
x8.skew()

model8=LinearRegression().fit(x8,y)
y_pred8=model8.predict(x8)

y_err8=y-y_pred8
sns.distplot(y_err8)

r2h=r2_score(y,y_pred8)
print(r2h)
######################################################################################
#9 combination of variables in X-->Age, Weight, Km, HP,Quarterly_Tax,Gears
x9=data[['Age_08_04','Weight','KM','HP','Quarterly_Tax','Gears']]
x9.hist()
x9.skew()

model9=LinearRegression().fit(x9,y)
y_pred9=model9.predict(x9)

y_err9=y-y_pred9
sns.distplot(y_err9)

r2i=r2_score(y,y_pred9)
print(r2i)
###############################################################################
table={"x variables":pd.Series(['Age_08_04','Age_08_04,Weight','Age_08_04,Weight,KM','Age_08_04,Weight,KM,HP','Age_08_04,Weight,KM,HP,Quarterly_Tax','Age_08_04,Weight,KM,HP,Quarterly_Tax,Doors','Age_08_04,Weight,KM,HP,Quarterly_Tax,Doors,cc','Age_08_04,Weight,KM,HP,Quarterly_Tax,Doors,cc,Gears','Age_08_04,Weight,KM,HP,Quarterly_Tax,Gears']),'R2 score':pd.Series([(r2a*100),(r2b*100),(r2c*100),(r2d*100),(r2e*100),(r2f*100),(r2g*100),(r2h*100),(r2i*100)])}
type(table)

r2_table=pd.DataFrame(table)
r2_table

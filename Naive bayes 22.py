# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:50:22 2022

@author: aditya
"""

#taking the data
import pandas as pd
data_train=pd.read_csv("SalaryData_Train.csv")
data_test=pd.read_csv("SalaryData_Test.csv")
data_test.shape
data_train.shape

list(data_train)
list(data_test)

type(data_train)
type(data_test)

data_train.head()
data_test.head()

#finding the missing values
data_train.isnull().sum()
list(data_train)
data_test.isnull().sum()
list(data_test)

#drop the values
data_train.drop(["educationno"],axis=1,inplace=True)
data_test.drop(["educationno"],axis=1,inplace=True)

#label encode for the train data 
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
data_train["workclass"]=LE.fit_transform(data_train["workclass"])
data_train["workclass"]
data_train["education"]=LE.fit_transform(data_train["education"])
data_train["education"]
data_train["maritalstatus"]=LE.fit_transform(data_train["maritalstatus"])
data_train["maritalstatus"]
data_train["occupation"]=LE.fit_transform(data_train["occupation"])
data_train["occupation"]
data_train['relationship']=LE.fit_transform(data_train["relationship"])
data_train["relationship"]
data_train["race"]=LE.fit_transform(data_train["race"])
data_train["race"]
data_train["sex"]=LE.fit_transform(data_train["sex"])
data_train["sex"]
data_train["capitalgain"]=LE.fit_transform(data_train["capitalgain"])
data_train["capitalgain"]
data_train["hoursperweek"]=LE.fit_transform(data_train["hoursperweek"])
data_train["hoursperweek"]
data_train["native"]=LE.fit_transform(data_train["native"])
data_train["native"]

#label encode for test data
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
data_test["workclass"]=LE.fit_transform(data_test["workclass"])
data_test["workclass"]
data_test["education"]=LE.fit_transform(data_test["education"])
data_test["education"]
data_test["maritalstatus"]=LE.fit_transform(data_test["maritalstatus"])
data_test["maritalstatus"]
data_test["occupation"]=LE.fit_transform(data_test["occupation"])
data_test["occupation"]
data_test["relationship"]=LE.fit_transform(data_test["relationship"])
data_test["relationship"]
data_test["race"]=LE.fit_transform(data_test["race"])
data_test["race"]
data_test["sex"]=LE.fit_transform(data_test["sex"])
data_test["sex"]
data_test["native"]=LE.fit_transform(data_test["native"])
data_test["native"]

#split x and y variables 
x_train=data_train.iloc[:,1:12]
x_train
y_train=data_train["Salary"]

x_test=data_test.iloc[:,1:12]
x_test
Y_test=data_test["Salary"]

#model development
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB
MNB.fit(x_train,y_train)

Y_pred = MNB.predict(x_test)

#confusion matrix and accuracy score 
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(Y_test,Y_pred)
acc=accuracy_score(Y_test,Y_pred).round(2)

print("naive bayes model accuracy score:",acc)

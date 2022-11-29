# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 18:04:12 2022

@author: aditya
"""
import pandas as pd
data_train=pd.read_csv("SalaryData_train(1).csv")
data_test=pd.read_csv("SalaryData_test(1).csv")
data_train.shape
data_test.shape

data_train.head()
data_test.head()
list(data_train)
list(data_test)
data_train.corr()

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
data_train["Salary_code"]=LE.fit_transform(data_train["Salary"])
data_train[["Salary","Salary_code"]].head(14)
pd.crosstab(data_train.Salary,data_train.Salary_code)

data_train["native_code"]=LE.fit_transform(data_train["native"])
data_train[["native","native_code"]].head(14)
pd.crosstab(data_train.native,data_train.native_code)

data_train["workclass_code"]=LE.fit_transform(data_train["education"])
data_train[["workclass","workclass_code"]].head(14)
pd.crosstab(data_train.workclass,data_train.workclass_code)

data_train["education_code"]=LE.fit_transform(data_train["education"])
data_train[["education","education_code"]].head(14)
pd.crosstab(data_train.education,data_train.education_code)

data_train["maritalstatus_code"]=LE.fit_transform(data_train["maritalstatus"])
data_train[["maritalstatus","maritalstatus_code"]].head(14)
pd.crosstab(data_train.maritalstatus,data_train.maritalstatus_code)

data_train["occupation_code"]=LE.fit_transform(data_train["occupation"])
data_train[["occupation","occupation_code"]].head(14)
pd.crosstab(data_train.occupation,data_train.occupation_code)

data_train["relationship_code"]=LE.fit_transform(data_train["relationship"])
data_train[["relationship","relationship_code"]].head(14)
pd.crosstab(data_train.relationship,data_train.relationship_code)

data_train["race_code"]=LE.fit_transform(data_train["race"])
data_train[["race","race_code"]].head(14)
pd.crosstab(data_train.race,data_train.race_code)

data_train["sex_code"]=LE.fit_transform(data_train["sex"])
data_train[["sex","sex_code"]].head(14)
pd.crosstab(data_train.sex,data_train.sex_code)

data_train.drop(["workclass","education","educationno","maritalstatus","occupation","relationship","race","sex","native"],axis=1,inplace=True)

x_train=data_train.drop(["Salary_code"],axis=1)
y_train=data_train["Salary_code"]

LE=LabelEncoder()
data_test["native_code"]=LE.fit_transform(data_test["native"])
data_test[["native","native_code"]].head(14)
pd.crosstab(data_test.native,data_test.native_code)

data_test["workclass_code"]=LE.fit_transform(data_test["workclass"])
data_test[["workclass","workclass_code"]].head(14)
pd.crosstab(data_test.workclass,data_test.workclass_code)

data_test["education_code"]=LE.fit_transform(data_test["education"])
data_test[["education","education_code"]].head(14)
pd.crosstab(data_test.education,data_test.education_code)

data_test["maritalstatus_code"]=LE.fit_transform(data_test["maritalstatus"])
data_test[["maritalstatus","maritalstatus_code"]].head(14)
pd.crosstab(data_test.maritalstatus,data_test.maritalstatus_code)

data_test["occupation_code"]=LE.fit_transform(data_test["occupation"])
data_test[["occupation","occupation_code"]].head(14)
pd.crosstab(data_test.occupation,data_test.occupation_code)

data_test["relationship_code"]=LE.fit_transform(data_test["relationship"])
data_test[["relationship","relationship_code"]].head(14)
pd.crosstab(data_test.relationship,data_test.relationship_code)

data_test["race_code"]=LE.fit_transform(data_test["race"])
data_test[["race","race_code"]].head(14)
pd.crosstab(data_test.race,data_test.race_code)

data_test["sex_code"]=LE.fit_transform(data_test["sex"])
data_test[["sex","sex_code"]].head(14)
pd.crosstab(data_test.sex,data_test.sex_code)

data_test["Salary_code"]=LE.fit_transform(data_test["Salary"])
data_test[["Salary","Salary_code"]].head(14)
pd.crosstab(data_test.Salary,data_test.Salary_code)

data_test.drop(["workclass","education","educationno","maritalstatus","occupation","relationship","race","sex","native"],axis=1,inplace=True)

x_test=data_train.drop(["Salary_code"],axis=1)
y_test=data_train["Salary_code"]
list(x_test)

loading svc
training a classifier-kernel='rbf'
from sklearn.svm import SVC
SVC()
#clf=SVC(kernel='linear')
#clf=SVC(kernel='poly',degree=3)
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
y_pred_train=clf.predict(X_train)


#import the metrics class
from sklearn import metrics
cm=metrics.confusion_matrix(y_test,y_pred)
print(cm)

print("Training Accuracy:",metrics.accuracy_score(y_train,y_pred_train).round(2))
print("Testing Accuracy:",metrics.accuracy_score(y_test,y_pred).round(2))

cm=metrics.confusion_matrix(y_train,y_pred_train)
print(cm)

'''
training acc : 0.78 #poly
testing acc : 0.78
training acc : 0.8 #rbf
testing acc : 0.8
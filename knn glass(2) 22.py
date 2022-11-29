# -*- coding: utf-8 -*-
"""
Created on Tue May 17 21:06:51 2022

@author: aditya
"""
import pandas as pd
data=pd.read_csv("glass.csv")
data.shape
type(data)
list(data)
data.head()

#split as x and y
y=data["Type"]
x=data.iloc[:,1:9]
list(x)

#standardization
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
x_scale=SS.fit_transform(x)
print(x_scale)
x_scale=pd.DataFrame(x_scale)

#splitting data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scale,y,test_size=0.25,random_state=95)
x_train.shape
y_train.shape

#KNN classifier 
from sklearn.neighbors import KNeighborsClassifier
Knn=KNeighborsClassifier(n_neighbors=7,p=2)
Knn.fit(x_train,y_train)
y_pred=Knn.predict(x_test)

#metrics
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print("Accuracy of KNN with K=7 is:",(acc*100).round(3))


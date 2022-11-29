# -*- coding: utf-8 -*-
"""
Created on Sat May  7 16:25:48 2022

@author: aditya
"""
import pandas as pd
data=pd.read_csv("zoo.csv")
data.shape
data.head()

#split as x and y
y=data["type"]
x=data.iloc[:,1:17]
list(x)

#standardization
from sklearn.preprocessing import StandardScaler
x_scale=StandardScaler().fit_transform(x)
x_scale
type(x_scale)

pd.crosstab(y,y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scale,y,stratify=y,random_state=42)

#install knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

#prediction
y_pred=knn.predict(x_test)

#compute confusion matrix
from sklearn import metrics
cm=metrics.confusion_matrix(y_test,y_pred)
print(cm)

import numpy as np
print(np.mean(y_pred==y_test).round(2))
print('accuracy of Knn with k=5,on the test set:{:.3f}'.format(knn.score(x_test,y_test)))
#value 1.0

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
#1.0

knn.score(x_test,y_test)
#1.0







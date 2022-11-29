"""
Created on Fri Apr 22 15:29:42 2022

@author: aditya
"""

#logistic regression
import pandas as pd
data=pd.read_csv("bank-full.csv",sep=";")
data.shape
data.head()
list(data)
X=data.iloc[:,0:16]
X.head()
list(X)
X.dtypes
X1=data[['job','marital','education','default','housing','loan','contact','month','poutcome']]
X1.head()
X1.shape
type(X1)

#Converting to Numeric format
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in range(0,9,1):
    X1.iloc[:,i]=LE.fit_transform(X1.iloc[:,i])
print(X1)

X2=data[['age','balance','day','duration','campaign','pdays','previous']]
X2.head()

from sklearn.preprocessing import StandardScaler
X2_new=StandardScaler().fit_transform(X2)
X2_new=pd.DataFrame(X2_new)

X_new=pd.concat([X1,X2_new],axis=1)
X_new.head()
Y=data['y']
Y

#Converting Y to numeric format
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
Y=LE.fit_transform(Y)

#fitting the model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression().fit(X2_new,Y)
Y_pred=model.predict(X2_new)

#metrics 
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
cm=confusion_matrix(Y,Y_pred)
acc=accuracy_score(Y,Y_pred)
recs=recall_score(Y,Y_pred)
f1=f1_score(Y,Y_pred)

print('Confusion matrix:',cm)
print('Accuracy Score:',(acc*100).round(3))
print('Recall Score:',(recs*100).round(3))
print('F1 Score:',(f1*100).round(3))




























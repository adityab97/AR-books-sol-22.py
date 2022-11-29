# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 00:10:43 2022


@author: aditya
"""
#performing Clustering on crime data
#(a)kMeans Clustering

import pandas as pd
import numpy as np
data=pd.read_csv("crime_data.csv")
data.shape
list(data)
data.dtypes
x=data.iloc[:,1:5].values
x
list(x)

#standardizing the data
from sklearn.preprocessing import StandardScaler
x_scale=StandardScaler().fit_transform(x)
x_scale

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(16,9)

%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x_scale[:,0],x_scale[:,1],x_scale[:,2],x_scale[:,3])
plt.show()

#initializing kMeans
from sklearn.cluster import KMeans
km=KMeans(n_clusters=5).fit(x_scale)
lab=km.predict(x_scale)
type(lab)

c=km.cluster_centers_
km.inertia_

%matplotlib qt
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x_scale[:,0],x_scale[:,1],x_scale[:,2],x_scale[:,3])
ax.scatter(c[:,0],c[:,1],c[:,2],c[:,3],marker='*',c='Red',s=1000)

clust=[]
for i in range(1,9,1):
    km=KMeans(n_clusters=i).fit(x_scale)
    km.inertia_
    clust.append(km.inertia_)

#Elbow plot
plt.plot(range(1,9),clust)
plt.title("Elbow Plot")
plt.xlabel("No of Clusters")
plt.ylabel("Cluster Inertia values")
plt.show()
###############################################################################
#(b)Hierarchial Clustering
#Agglomerative Clustering 
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
ac.fit_predict(x_scale)

plt.figure(figsize=(16,9))
plt.scatter(x_scale[:,0],x_scale[:,1],x_scale[:,2],c=ac.labels_,cmap='rainbow')

#dendogram
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(16,9))
plt.title("Dendogram")
dend=shc.dendrogram(shc.linkage(x,method='complete'))
###################################################################################################
#(c)DBScan
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=2,min_samples=3).fit(x_scale)
db.labels_

cl=pd.DataFrame(db.labels_,columns=['Cluster'])
cl
cl['Cluster'].value_counts()

data_new=pd.concat([pd.DataFrame(x_scale),cl],axis=1)

#noise data
nd=data_new[data_new['Cluster']==-1]
nd

#final data without outliers
fd=data_new[data_new['Cluster']==0]
fd
data_new.mean()
fd.mean()

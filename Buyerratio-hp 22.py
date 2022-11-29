# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:44:53 2022

@author: aditya
"""
import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/HP/Downloads/excelr/assigments/Hypothesis testing/BuyerRatio.csv")
data.shape
list(data)
#a=50+142+131+70
#b=435+1523+1356+750
A=393
B=4064

#EAST
AE1=50/A
AE2=435/B
print(AE1,AE2)

nump=np.array([AE1,AE2])
count=np.array([A,B])
from statsmodels.stats.proportion import proportions_ztest
stat,BE = proportions_ztest(count, nump)

alpha=0.5
#H0 : males and females are same 
#H1 : males and females are not same 

#west
AE1=142/A
AE2=1523/B
print(AE1,AE2)

numpy=np.array([AE1,AE2])    
count=np.array([A,B])
from statsmodels.stats.proportion import proportions_ztest
stat,BE=proportions_ztest(count,numpy)

alpha=0.5
#H0=males and females are equal
#H1=males and females are not equal
if BE < alpha:
    print("H0 is accepted and H1 is rejected")
else:
    print("H1 is accepted and H0 is rejected")
    
#north
AE1=131/A
AE2=1356/B
print(AE1,AE2)

numpy=np.array([AE1,AE2])
count=np.array([A,B])
from statsmodels.stats.proportion import proportions_ztest
stat,BE=proportions_ztest(count,numpy)

alpha=0.05
#H0=males and females are equal
#H1=males and females are not equal
if BE < alpha:
    print("H0 is accepted and H1 is rejected")
else:
    print("H1 is accepted and H0 is rejected")
    
#south
AE1=70
AE2=750
print(AE1,AE2)

numpy=np.array([AE1,AE2])    
count=np.array([A,B])
from statsmodels.stats.proportion import proportions_ztest
stat,BE=proportions_ztest(count,numpy)

alpha=0.05
#H0=males and females are equal
#H1=males and females are not equal
if BE < alpha:
    print("H0 is accepted and H1 is rejected")
else:
    print("H1 is accepted and H0 is rejected")


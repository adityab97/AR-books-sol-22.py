# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:26:15 2022

@author: aditya
"""
import pandas as pd
data=pd.read_csv("C://Users/HP/Downloads/excelr/assigments/Hypothesis testing/LabTAT.csv")
data.shape
list(data)

data["Laboratory 1"].hist()
data["Laboratory 2"].hist()
data["Laboratory 3"].hist()
data["Laboratory 4"].hist()

#test of hypothesis
'''
Ho: L1=L2=L3=L4----->All laboratories avg TAT is same
H1: L1!=L2!=L3!=L4!=----->Any one of these laboratories avg TAT among the 4 is not same
'''
L1= data["Laboratory 1"]
L2= data["Laboratory 2"]
L3= data["Laboratory 3"]
L4 =data["Laboratory 4"]
import scipy.stats as stats
z,p=stats.f_oneway(L1,L2,L3,L4)
print(z,p)
alpha=0.05

if p>0.05:
    print("Accept Ho and reject H1")
else:
        print("Accept H1 and Reject Ho")
        
#H1 is accepted
        
        
        
        
        
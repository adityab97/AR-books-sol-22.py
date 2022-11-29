# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 12:34:58 2022

@author: aditya
"""
import pandas as pd
data=pd.read_csv("C://Users/HP/Downloads/excelr/assigments/Hypothesis testing/Cutlets.csv")
data.shape
list(data)
'''
#test of hypothesis
Ho: UnitA= UnitB---->No siginificance difference in diameter of cutlets of two units
H1: UnitA= UnitB----> Significance difference in diameter of cutlets of two units
'''
A=data["Unit A"] 
B= data["Unit B"]
alpha=0.05  #level of significance 
from scipy.stats import ttest_ind
z,p=ttest_ind(A,B)
print(z,p)

if p>alpha:
    print("Accept Ho and Reject H1")
else:
    print("Accept H1 and Reject Ho")
    
    
    
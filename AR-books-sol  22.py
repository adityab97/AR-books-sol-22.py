# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:28:06 2022


@author: aditya
"""
import pandas as pd
data=pd.read_csv("book.csv")
data.shape
list(data)
data.head()
type(data)

data_new=pd.get_dummies(data)
data_new.head()

#apriori algorithm
from mlxtend.frequent_patterns import apriori,association_rules
frequent_books=apriori(data_new,min_support=0.1,use_colnames=True)
frequent_books

rules=association_rules(frequent_books,metric="lift",min_threshold=0.7)
rules

rules.sort_values("lift",ascending=False)

rules.sort_values("lift",ascending=False)[0:20]

rules[rules.lift>1]

rules[["support","confidence"]].hist()

rules[["support","confidence","lift"]].hist()

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')

plt.show()

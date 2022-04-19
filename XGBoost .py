#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier


# In[3]:


dataframe = pd.read_csv('pima-indians-diabetes.csv')


# In[4]:


array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 30


# In[6]:


kfold = model_selection.KFold(n_splits = 10, random_state = seed)
model = AdaBoostClassifier(n_estimators = num_trees, random_state = seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[7]:


from sklearn import svm
from xgboost import XGBClassifier
clf = XGBClassifier()


# In[8]:


seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits = 10, random_state = seed)
model =XGBClassifier(n_estimators = num_trees, random_state = seed)
results = model_selection.cross_val_score(model, X, Y, cv = kfold)
print(results.mean())


# In[ ]:





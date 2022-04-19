#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install matplotlib ')


# In[40]:


import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


# In[3]:


train = pd.read_csv('bigmart_train.csv')


# In[4]:


train


# In[5]:


train.head(10)


# In[6]:


train.shape


# In[7]:


train.isnull().sum()


# In[8]:


train['Item_Fat_Content'].unique()


# In[9]:


train['Outlet_Establishment_Year'].unique()


# In[10]:


train['Outlet_Age'] = 2018 -train['Outlet_Establishment_Year']
train.head()


# In[11]:


train['Outlet_Size'].unique()


# In[12]:


train.describe()


# In[13]:


train['Item_Fat_Content'].value_counts()


# In[14]:


train['Item_Visibility'].hist(bins=20)


# In[15]:


Q1 = train['Item_Visibility'].quantile(0.25)
Q3 = train['Item_Visibility'].quantile(0.75)
IQR = Q3-Q1


# In[16]:


IQR


# In[18]:


filt_train = train.query('(@Q1-1.5*@IQR)<=Item_Visibility<=(@Q3-1.5*@IQR)')


# In[22]:


train.shape


# In[23]:


train['Item_Visibility_bins'] = pd.cut(train['Item_Visibility'],[0.000,0.006,0.13,0.2], labels = ['Low Viz','Viz','High Viz'])


# In[30]:


train['Item_Visibility_bins'].value_counts()


# In[31]:


train['Item_Visibility_bins'] = train['Item_Visibility_bins'].replace(np.nan,'Low Viz', regex = True)


# In[36]:


train['Item_Fat_Content'] = train['Item_Fat_Content'].replace('LF', 'Low Fat')
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace('reg', 'Regular')


# In[37]:


train.head(10)


# In[45]:


le = LabelEncoder()
train['Item_Fat_Content'].unique()
train['Item_Fat_Content'] = le.fit_transform(train['Item_Fat_Content'])
train['Item_Visibility_bins'] = le.fit_transform(train['Item_Visibility_bins'])
train['Outlet_Size'] = le.fit_transform(train['Outlet_Size'])
train['Outlet_Location_Type'] = le.fit_transform(train['Outlet_Location_Type'])


# In[49]:


dummy = pd.get_dummies(train['Outlet_Size'])
train = train.drop(['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Type','Outlet_Establishment_Year'], axis = 1)


# In[50]:


train.columns


# In[51]:


train.head()


# In[52]:


X = train.drop('Item_Outlet_Sales', axis = 1)
y = train.Item_Outlet_Sales


# In[55]:


test = pd.read_csv('bigmart_test.csv')
test['Outlet_Size'] = test['Outlet_Size'].fillna('Medium')
test['Item_Visibility_bins'] = pd.cut(test['Item_Visibility'],[0.000,0.006,0.13,0.2], labels = ['Low Viz','Viz','High Viz'])
test['Item_Visibility_bins'] = test['Item_Visibility'].fillna('Low Viz')
test['Item_Weight'] = test['Item_Weight'].fillna(test['Item_Weight'].mean())


# In[58]:


test['Outlet_Size']
test['Item_Visibility_bins'] 
test['Item_Weight']


# In[ ]:





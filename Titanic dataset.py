#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


titanic = pd.read_csv('titanic.csv')


# In[3]:


titanic.shape


# In[4]:


titanic.columns


# In[5]:


titanic


# In[6]:


titanic.drop(columns=['PassengerId', 'Name','Ticket','Cabin'], inplace=True)


# In[7]:


titanic= titanic.dropna()


# In[8]:


titanic['Sex'].replace({'male':1,'female':0},inplace=True)


# In[9]:


titanic['Embarked'].value_counts()


# In[10]:


titanic['Embarked'].replace({'S':1,'C':0},inplace =True)


# In[11]:


titanic.info()


# In[16]:


titanic.corr()


# In[17]:


import matplotlib.pyplot as plt #for data visualization
import seaborn as sns # for data data visualization


# In[18]:


sns.barplot(x="Sex", y ="Survived", data = titanic)
plt.show()


# In[19]:


sns.barplot(x="Pclass", y="Survived", data = titanic)
plt.show()


# In[20]:


sns.boxplot(x="Sex",y="Age",hue="Survived",data=titanic)
plt.show()


# In[ ]:





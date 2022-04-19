#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system(' pip install seaborn')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


sns.set()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12,6)


# In[11]:


df = pd.read_csv('driver-data.csv')


# In[19]:


df.head()


# In[20]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
df_analyse = df.drop('id', axis = 1)


# In[21]:


df_analyse


# In[25]:


kmeans.fit(df_analyse)
kmeans.Cluster_clusters_


# In[ ]:





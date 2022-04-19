#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


data = pd.read_csv('zoo.csv')
data.info()


# In[21]:


labels = data['class_type']


# In[22]:


print(np.unique(labels.values))


# In[23]:


fig, axes = plt.subplots()
(labels.value_counts()).plot(axes=axes, kind = 'bar')
features = data.values[:, 1:-1]
features.shape


# In[24]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances


# In[25]:


model = AgglomerativeClustering(n_clusters = 7, linkage= "average", affinity = "cosine")
model.fit(features)
model.labels_
print(np.unique(model.labels_))
labels= labels-1


# In[18]:


from sklearn.metrics import mean_squared_error
score = mean_squared_error(labels, model.labels_)
abs_error = np.sqrt(score)
print(abs_error)


# In[ ]:





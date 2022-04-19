#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings ('ignore')


# In[3]:


df =pd.read_csv('voice-classification.csv')
df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[8]:


df.isnull().sum()


# In[14]:


print("Shape of Data:",df.shape)
print("Total no. of Labels:{}".format(df.shape[0]))
print("Total no. of Males:{}".format(df[df.label=='male'].shape[0]))
print("Total no. of Females:{}".format(df[df.label=='female'].shape[0]))


# In[15]:


X=df.iloc[:,:-1]
print(df.shape)
print(X.shape)


# In[16]:


from sklearn.preprocessing import LabelEncoder


# In[17]:


y = df.iloc[:,-1]

gender_encoder =LabelEncoder()
y = gender_encoder.fit_transform(y)
y


# In[18]:


from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
scaler.fit(X)
X= scaler.transform(X)


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=100)


# In[20]:


from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix


# In[21]:


svc_model=SVC()
svc_model.fit(X_train,y_train)
y_pred =svc_model.predict(X_test)


# In[22]:


print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# In[23]:


print(confusion_matrix(y_test,y_pred))


# In[25]:


from sklearn.model_selection import GridSearchCV


# In[26]:


param_grid={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}


# In[27]:


grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


# In[28]:


grid_predictions = grid.predict(X_test)


# In[29]:


print('Accuracy Score:')
print(metrics.accuracy_score(y_test,grid_predictions))


# In[30]:


print(confusion_matrix(y_test,grid_predictions))


# In[31]:


print(classification_report(y_test,grid_predictions))


# In[ ]:





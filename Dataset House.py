#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings


# In[51]:


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[3]:


dataset = pd.read_excel('1553768847_housing.xlsx')


# In[6]:


dataset.head(10)


# In[5]:


dataset.describe()


# In[7]:


dataset.shape


# In[8]:


dataset.isnull().sum()


# In[9]:


dataset['total_bedrooms'].unique()


# In[10]:


from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[11]:


dataset.hist(figsize =(25,25),bins=50);


# In[14]:


datasetcorr = dataset.corr()
datasetcorr.style.background_gradient()


# In[1]:


plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()


# In[22]:


totalnotNullBedroom = dataset[dataset['total_bedrooms'].notnull()]
totalnotNullBedroom.hist(figsize=(20,10),bins=50)


# In[24]:


print(dataset.iloc[:,4:5].head())


# In[27]:


imputer = Imputer(np.nan,strategy ="median")
imputer.fit(dataset.iloc[:,4:5])
dataset.iloc[:,4:5] = imputer.transform(dataset.iloc[:,4:5])
dataset.isnull().sum()


# In[29]:


labelEncoder = LabelEncoder()
print(dataset["ocean_proximity"].value_counts())
dataset["ocean_proximity"] = labelEncoder.fit_transform(dataset["ocean_proximity"])
dataset["ocean_proximity"].value_counts()
dataset.describe()


# In[30]:


dataset_ind = dataset.drop("median_house_value",axis=1)
print(dataset_ind.head())
dataset_dep = dataset["median_house_value"]
print("Medain Housing Values")
print(dataset_dep.head())


# In[32]:


X_train,X_test,y_train,y_test = train_test_split(dataset_ind,dataset_dep,test_size=0.2,random_state=0)


# In[33]:


print(X_train.head())
print(X_test.head())


# In[36]:


print(y_train.head())
print(y_test.head())


# In[35]:


X_train.head()


# In[37]:


independent_scaler = StandardScaler()
X_train = independent_scaler.fit_transform(X_train)
X_test = independent_scaler.transform(X_test)
print(X_train[0:5,:])
print("test data")
print(X_test[0:5,:])


# In[38]:


linearRegModel = LinearRegression(n_jobs=-1)
linearRegModel.fit(X_train,y_train)


# In[39]:


print("Intercept is "+str(linearRegModel.intercept_))
print("coefficients  is "+str(linearRegModel.coef_))


# In[40]:


y_pred = linearRegModel.predict(X_test)


# In[41]:


print(len(y_pred))
print(len(y_test))
print(y_pred[0:5])
print(y_test[0:5])


# In[52]:


print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print(np.sqrt(metrics.mean_squared_error(y_train,linearRegModel.predict(X_train))))


# In[53]:


dtReg = DecisionTreeRegressor(max_depth=9)
dtReg.fit(X_train,y_train)


# In[54]:


dtReg_y_pred = dtReg.predict(X_test)
dtReg_y_pred


# In[55]:


print(len(dtReg_y_pred))
print(len(y_test))
print(dtReg_y_pred[0:5])
print(y_test[0:5])


# In[56]:


print(np.sqrt(metrics.mean_squared_error(y_test,dtReg_y_pred)))


# In[57]:


rfReg = RandomForestRegressor(30)
rfReg.fit(X_train,y_train)


# In[58]:


rfReg_y_pred = rfReg.predict(X_test)
print(len(rfReg_y_pred))
print(len(y_test))
print(rfReg_y_pred[0:5])
print(y_test[0:5])


# In[59]:


print(np.sqrt(metrics.mean_squared_error(y_test,rfReg_y_pred)))


# In[61]:


dropcol = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","ocean_proximity"]
print(dropcol)
dataset_med = dataset_ind.drop(dropcol,axis=1)
print(type(dataset_med))


# In[63]:


X_train2,X_test2,y_train2,y_test2 = train_test_split(dataset_med,dataset_dep,test_size=0.2,random_state=0)


# In[64]:


linReg2 = LinearRegression()
linReg2.fit(X_train2,y_train2)


# In[65]:


y_pred2 = linReg2.predict(X_test2)
print(len(y_pred2))
print(len(y_test2))
print(y_pred2[0:5])
print(y_test2[0:5])


# In[66]:


fig = plt.figure(figsize=(25,8))
plt.scatter(y_test2,y_pred2,marker="o",edgecolors ="r",s=60)
plt.scatter(y_train2,linReg2.predict(X_train2),marker="+",s=50,alpha=0.5)
plt.xlabel(" Actual median_house_value")
plt.ylabel(" Predicted median_house_value")


# In[ ]:





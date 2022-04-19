#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


iris_data = load_iris()
print(iris_data)


# In[3]:


data_input = iris_data.data
data_output = iris_data.target


# In[4]:


print(data_output)


# In[5]:


from sklearn.model_selection import KFold
kfold = KFold(n_splits=6,shuffle=True)


# In[6]:


print("Train Set        Test Set       ")
for train_set,test_set in kfold.split(data_input):
    print(train_set,test_set)


# In[7]:


from sklearn.ensemble import RandomForestClassifier
rf_class = RandomForestClassifier(n_estimators=10)


# In[8]:


from sklearn.model_selection import cross_val_score
print(cross_val_score(rf_class,data_input,data_output,scoring='accuracy',cv=10))


# In[10]:


accuracy = cross_val_score(rf_class,data_input,data_output,scoring='accuracy',cv=10).mean()*100
print('Accuracy of Random Forests is:', accuracy)


# In[ ]:





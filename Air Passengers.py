#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from datetime import datetime as dt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] =15,6

import warnings
warnings.filterwarnings('ignore')


# In[5]:


data = pd.read_csv('AirPassengers.csv')


# In[6]:


data['Month'].head()


# In[10]:


data['Month']=data['Month'].apply(lambda x: dt(int(x[:4]),int(x[5:]),15))
data = data.set_index('Month')
data.head()


# In[11]:


ts = data['#Passengers']


# In[12]:


plt.plot(ts)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing library
import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


web_customers = [123,645,950,1290,1009,3245,8745,5679,3456,1240,9787]
time_hrs =[7,8,9,10,11,12,13,14,15,16,17]


# In[3]:


#select the style of the plot
style.use('ggplot')
plt.plot(time_hrs,web_customers)
plt.title('People on Website')
plt.xlabel('Hours')
plt.ylabel('Number of Users')
plt.show()


# In[4]:


#select the style of the plot
style.use('ggplot')
plt.plot(time_hrs,web_customers,color='b',linestyle='--',linewidth=2.5)
plt.title('People on Website')
plt.xlabel('Hours')
plt.ylabel('Number of Users')
plt.show()


# In[8]:


peopleon_monday = [123,432,423,121,322,743]
peopleon_tuesday = [233,321,123,332,421,444]
peopleon_wednesday = [111,222,333,444,555,212]

time_hrs = [8,9,10,11,12,13]


# In[10]:


style.use('ggplot')
plt.plot(time_hrs,peopleon_monday,'r',label='monday',linewidth=1)
plt.plot(time_hrs,peopleon_tuesday,'g',label='tuesday',linewidth=1.5)
plt.plot(time_hrs,peopleon_wednesday,'b',label='monday',linewidth=2)
plt.axis([6.5,17.5,50,2000])
plt.title('Website traffic')
plt.xlabel('Hrs')
plt.ylabel('Number of users')
plt.legend()
plt.show()


# In[ ]:





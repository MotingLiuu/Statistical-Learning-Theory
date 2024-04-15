#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import random


# In[10]:

#A linear function used to generate data which is linear speratable
def function(x):
    return 0.5*x+2


# In[11]:


#This function is used to generate training data
def generate_data(num_points):
    data_list_1=[]
    data_list_2=[]
    for i in range(num_points):
        tem1=np.random.rand()*100
        tem2=np.random.rand()*100
        data1=[tem1,function(tem1)+np.random.rand()*20+1,1]
        data2=[tem2,function(tem2)-np.random.rand()*20-1,-1]
        data_list_1.append(data1)
        data_list_2.append(data2)
    return data_list_1,data_list_2


# In[12]:


data1,data2=generate_data(20)
data1=np.array(data1)
data2=np.array(data2)
data=np.concatenate((data1,data2),axis=0)
print(data[2][0:2])


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data1[:,0],data1[:,1])
plt.scatter(data2[:,0],data2[:,1])
plt.show()


# In[14]:


#This function is used to train a linear plane and return its weight and bias
def learning_alg(data, learning_rate):
    w=np.zeros(2)
    b=0
    pos=0
    
    while True:
        if data[pos][2]*(np.inner(w,data[pos][0:2])+b)<=0:
            w=w+learning_rate*data[pos][2]*data[pos][0:2]
            b=b+learning_rate*data[pos][2]
            pos=0
        elif pos==len(data)-1:
            break
        else:
            pos=pos+1
    
    return w,b


# In[15]:


w,b=learning_alg(data,0.1)


# In[16]:


x=np.linspace(0,100,200)
y=-w[0]/w[1]*x-b/w[1]
plt.plot(x,y,color='blue')
plt.scatter(data1[:,0],data1[:,1])
plt.scatter(data2[:,0],data2[:,1])
plt.show()


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


ds = pd.read_csv('dataset.csv')


# In[3]:


y = ds['admitted']
m = len(y)
X = ds.drop('admitted', axis=1)


# In[4]:


X = (X-X.mean())/X.std()


# In[5]:


X.insert(0, "", [1 for _ in range(m)])


# In[6]:


theta = np.zeros(4, dtype=np.float32)
iterations = 1500
alpha = 0.01


# In[7]:


def cost_function(X, y, theta):
    h = 1/(1 + np.exp(-X.dot(theta)))
    cost_1 = np.log(h)
    cost_2 = np.array([np.log(1-i) for i in h])
    summ = y.dot(cost_1) + np.array([1-i for i in y]).dot(cost_2)
    return -summ/m


# In[8]:


print(cost_function(X, y, theta))


# In[9]:


j_history = np.array([0 for _ in range(iterations)], dtype=np.float32)
for i in range(iterations):    
    temp = [0 for _ in range(4)]
    for j in range(4):
        h = 1/(1 + np.exp(-X.dot(theta)))
        temp[j] = theta[j] - (alpha/m) * np.sum((h-y) * np.array(X.iloc[:, j]))
    theta = temp
    j_history[i] = cost_function(X, y, theta)


# In[10]:


print(theta)


# In[14]:


plt.plot(np.arange(0, iterations), j_history)
plt.ylabel('J (cost function)')
plt.xlabel('Iterations')
plt.show()


# In[12]:


df = pd.read_csv('train.csv')
df = (df-df.mean())/df.std()
df.insert(0, "", [1 for _ in range(len(df))])
df


# In[13]:


df.dot(theta)


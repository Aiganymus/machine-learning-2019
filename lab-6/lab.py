#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


ds = pd.read_csv('glass.csv')


# In[ ]:


y = ds['Type']
m = len(y)
X = ds.drop('Type', axis=1)


# In[ ]:


X = (X-X.mean())/(X.max()-X.min())


# In[ ]:


X.insert(0, "", [1 for _ in range(m)])
X.head()


# In[ ]:


def cost_function(X, y, theta):
    h = 1/(1 + np.exp(-X.dot(theta)))
    cost_1 = np.log(h)
    cost_2 = np.array([np.log(1-i) for i in h])
    summ = y.dot(cost_1) + np.array([1-i for i in y]).dot(cost_2)
    return -summ/m + (l/2*m)*np.sum(np.power(theta[1:], 2))


# In[ ]:


def gradient_descent(X, y, theta):
#     j_history = np.array([0 for _ in range(iterations)], dtype=np.float32)
    iterations = 1500
    alpha = 0.01
    l = 0.1
    for i in range(iterations):    
        temp = [0 for _ in range(10)]
        for j in range(10):
            h = 1/(1 + np.exp(-X.dot(theta)))
            if j == 0:
                l = 0
            else:
                l = 0.1
            temp[j] = theta[j] - ((alpha/m) * np.sum((h-y) * np.array(X.iloc[:, j]))) - theta[j]*alpha*l/m
        theta = temp
#         j_history[i] = cost_function(X, y, theta)
    return theta


# In[ ]:


k = 7
theta = np.zeros([k, 10], dtype=np.float32)


# In[ ]:


for j in range(k):
    new_y = [int(y[i] == (j+1)) for i in range(m)]
    theta[j] = gradient_descent(X, new_y, theta[j])


# In[ ]:


theta


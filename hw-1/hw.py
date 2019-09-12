import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from numpy.linalg import inv

data = genfromtxt('train_data.csv', delimiter=',')
data = np.delete(data, 0, 0)
y = [data[i][4] for i in range(len(data))]
X = np.delete(data, 4, 1)

theta = np.zeros(5)
alpha = 0.01
iterations = 1500
m = len(y)

# normalize variables
y = [(i-np.mean(y))/(np.max(y)-np.min(y)) for i in y]
X = np.array([[(X[i][j]-np.mean(X[:, j]))/(np.max(X[:, j])-np.min(X[:, j])) for j in range(len(X[i]))] for i in range(len(X))])
X = np.array([np.insert(row, 0, 1) for row in X])

def cost_function(X, y, theta):
    h = X.dot(theta)
    return np.sum(np.square(h-y))/(2*m)

print(cost_function(X, y, theta))

j_history = np.array([0.0 for _ in range(iterations)])
for i in range(iterations):
    temp = [0.0 for _ in range(5)]
    for j in range(5):
        temp[j] = theta[j] - (alpha/m) * np.sum((X.dot(theta) - y) * np.array(X[:, j]))
    theta = temp
    j_history[i] = cost_function(X, y, theta)

print(theta)

theta_norm = np.matmul(np.matmul(inv(np.matmul(X.transpose(), X)), X.transpose()), y)
print(theta_norm)

plt.plot(np.arange(0, iterations), j_history, 'rx')
plt.ylabel('J (cost function)')
plt.xlabel('Iterations')
plt.show()
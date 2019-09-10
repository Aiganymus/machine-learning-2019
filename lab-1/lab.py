import numpy as np
import matplotlib.pyplot as plt

x, y = np.loadtxt("ex1data1.txt", dtype=np.float32, comments="#", delimiter=",", unpack=True)
m = len(y)

plt.plot(x, y, 'rx')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
# plt.show()

X = np.array([[1, i] for i in x])
theta = np.zeros(2)
iterations = 1500
alpha = 0.01

def cost_function(X, y, theta):
    h = X.dot(theta)
    return np.sum(np.square(h-y))/(2*m)

print(cost_function(X, y, theta))
j_history = np.array([0 for _ in range(iterations)], dtype=np.float)
for i in range(iterations):    
    temp0 = theta[0] - (alpha/m) * np.sum((X.dot(theta) - y) * np.array(X[:, 0]))
    temp1 = theta[1] - (alpha/m) * np.sum((X.dot(theta) - y) * np.array(X[:, 1]))
    theta[0] = temp0
    theta[1] = temp1
    j_history[i] = cost_function(X, y, theta)

# print(j_history[len(j_history)-1])
print(theta)



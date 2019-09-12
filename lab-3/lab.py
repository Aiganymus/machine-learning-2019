import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x0, x1, y = np.loadtxt("ex1data2.txt", dtype=np.float32, comments="#", delimiter=",", unpack=True)

m = len(y)

x0 = [[i/(np.amax(x0)-np.amin(x0))] for i in x0]
x1 = [[i/(np.amax(x1)-np.amin(x1))] for i in x1]
y = [i/(np.amax(y)-np.amin(y)) for i in y]

X = np.concatenate(([[1] for _ in range(m)], x0, x1), axis=1)
theta = np.zeros(3, dtype=np.float32)
iterations = 1500
alpha = 0.01

def cost_function(X, y, theta):
    h = X.dot(theta)
    return np.sum(np.square(h-y))/(2*m)

print(cost_function(X, y, theta))

j_history = np.array([0 for _ in range(iterations)], dtype=np.float32)
for i in range(iterations):    
    temp = [0 for _ in range(3)]
    for j in range(3):
        temp[j] = theta[j] - (alpha/m) * np.sum((X.dot(theta) - y) * np.array(X[:, j]))
    theta = temp
    j_history[i] = cost_function(X, y, theta)

print(theta)

plt.plot(np.arange(0, iterations), j_history, 'rx')
plt.ylabel('J (cost function)')
plt.xlabel('Iterations')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x0, x1, y)

x0, x1 = np.meshgrid(x0, x1)
z = theta[0] + theta[1]*x0 + theta[2]*x1
ax.plot_surface(x0, x1, z)

plt.ylabel('No. of rooms')
plt.xlabel('Size')
plt.show()

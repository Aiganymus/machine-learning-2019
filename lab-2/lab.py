import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y = np.loadtxt("ex1data1.txt", dtype=np.float32, comments="#", delimiter=",", unpack=True)
m = len(y)

plt.plot(x, y, 'rx')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')

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

print(theta)

predict1 = theta.dot([1, 3.5])
predict2 = theta.dot([1, 7])

print(predict1, predict2)

plt.plot(x, X.dot(theta), '-b')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
j_vals = np.array([[0.0 for _ in range(len(theta1_vals))] for _ in range(len(theta0_vals))])
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = [theta0_vals[i], theta1_vals[j]]
        j_vals[i][j] = cost_function(X, y, t)

j_vals = j_vals.transpose()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(theta0_vals, theta1_vals, j_vals)
plt.xlabel('Intercept')
plt.ylabel('Slope')

fig1, ax1 = plt.subplots()
ax1.contour(theta0_vals, theta1_vals, j_vals, levels=30)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0], theta[1], 'rx')

plt.show()
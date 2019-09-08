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

def cost_function(h, y):
    sum = 0
    for i in range(m):
        sum += (h[i] - y[i])**2
    return sum/(2*m)


h = [0.0 for x in range(m)]
print(cost_function(h, y))

for i in range(iterations):
    for i in range(m):
        h[i] = theta.dot(X[i])
    
    temp0 = theta[0] - (alpha/m) * cost_function(h, y)
    temp1 = theta[0] - (alpha/m) * cost_function(h, y)
    theta[0] = temp0
    theta[1] = temp1

print(theta)



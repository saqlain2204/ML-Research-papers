import numpy as np
import matplotlib.pyplot as plt

# alpha is usually a small constant like 0.01
def leaky_relu(x, alpha = 0.1):
    return np.where(x > 0, x, alpha * x)

x = np.array([x for x in range(-5, 6)])
y = leaky_relu(x)

plt.plot(x, y)
plt.title('Leaky ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('Leaky ReLU(x)')
plt.grid(True)
plt.show()

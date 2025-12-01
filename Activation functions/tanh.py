import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


x = np.linspace(-10, 10, 100)
y = tanh(x)
plt.plot(x, y)
plt.title('Tanh Activation Function')
plt.xlabel('x')
plt.ylabel('Tanh(x)')
plt.grid(True)
plt.show()
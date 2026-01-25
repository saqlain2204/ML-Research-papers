import numpy as np
import matplotlib.pyplot as plt


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x * _sigmoid(x)

x = np.linspace(-10, 10, 100)
y = swish(x)
plt.plot(x, y, label='Swish Activation Function')
plt.title('Swish Activation Function')
plt.legend()
plt.show()
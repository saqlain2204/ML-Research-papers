import numpy as np
import matplotlib.pyplot as plt


def _sigmoid(x):
    return 1/ (1 + np.exp(-x))

# SiLU activation function, also know as Swish
def silu(x):
    return x*_sigmoid(x)

X = np.linspace(-10, 10, 100)
Y = silu(X)

plt.figure(figsize=(10, 6))
plt.plot(X, Y, label='SiLU / Swish', color='blue', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('SiLU(x)')
plt.title('SiLU (Sigmoid Linear Unit) Activation Function')
plt.legend()
plt.show()
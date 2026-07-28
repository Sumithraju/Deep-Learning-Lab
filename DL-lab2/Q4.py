# For the network 1, there is:

# 4 input neurons (x1, x2, x3, x4)
# 1 output neuron
# No hidden layer
# 1 bias
# ReLU activation

import numpy as np

# ReLU Activation Function
def relu(z):
    if z > 0:
        return z
    else:
        return 0

# Random Inputs
x1 = np.random.randn()
x2 = np.random.randn()
x3 = np.random.randn()
x4 = np.random.randn()

# Random Weights
w11 = np.random.randn()
w12 = np.random.randn()
w13 = np.random.randn()
w14 = np.random.randn()

# Random Bias
b = np.random.randn()

# Forward Pass
z = (w11 * x1) + (w12 * x2) + (w13 * x3) + (w14 * x4) + b
a = relu(z)
y_hat = a

# Output
print("Inputs")
print(x1, x2, x3, x4)

print("\nWeights")
print(w11, w12, w13, w14)

print("\nBias")
print(b)

print("\nz =", z)

print("\na =", a)

print("\nPrediction =", y_hat)

#using FUn
import numpy as np

def relu(z):
    return np.maximum(0, z)

def neuron(W, x, b):

    z = W @ x + b

    a = relu(z)

    return z, a

# Random Input
x = np.random.randn(4,1)

# Random Weight
W = np.random.randn(1,4)

# Random Bias
b = np.random.randn(1,1)

z, y_hat = neuron(W, x, b)

print("Input")
print(x)

print("\nWeight")
print(W)

print("\nBias")
print(b)

print("\nz")
print(z)

print("\nPrediction")
print(y_hat)

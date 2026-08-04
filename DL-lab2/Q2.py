#3. Implement the forward pass using vectorized operations, i.e. W should be a matrix, x, z and a are vectors. The implementation should not contain any loops

import numpy as np

# Generate the same random values every time
np.random.seed(42)

# ReLU Activation Function
def relu(z):
    return np.maximum(0, z)

# Forward Pass Function
def forward(W, x, b):
    z = W @ x + b
    a = relu(z)
    return z, a


# Input Layer (4 neurons)
x = np.random.randn(4, 1)

# Hidden Layer 1 (3 neurons)
# W1 = (3 x 4)

W1 = np.random.randn(3, 4)
b1 = np.random.randn(3, 1)

# Hidden Layer 2 (2 neurons)
# W2 = (2 x 3)

W2 = np.random.randn(2, 3)
b2 = np.random.randn(2, 1)

# Output Layer (1 neuron)
# W3 = (1 x 2)
W3 = np.random.randn(1, 2)
b3 = np.random.randn(1, 1)


# Forward Propagation
z1, a1 = forward(W1, x, b1)
z2, a2 = forward(W2, a1, b2)
z3, y_hat = forward(W3, a2, b3)


# Print Results
print(x)

print("Weight Matrix W1")
print(W1)

print("Bias b1")
print(b1)

print("Hidden Layer 1")
print("z1 =")
print(z1)
print("a1 =")
print(a1)

print("Weight Matrix W2")
print(W2)

print("Bias b2")
print(b2)

print("Hidden Layer 2")
print("z2 =")
print(z2)
print("a2 =")
print(a2)

print("Weight Matrix W3")
print(W3)

print("Bias b3")
print(b3)

print("\nOutput Layer")
print("z3 =")
print(z3)

print("Final Prediction (y_hat)")
print(y_hat)
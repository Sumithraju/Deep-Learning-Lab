#
# By the end of this lab, you should be able to
# Appreciate how the input data flows into the network, how they get transformed and finally leads to computing the target value or the output.
# Understand various parameters and hyperparameters used in the neural network, their dimensions and how they interact with each other.
# Become familiar with writing loops and vector operations for computing neuron activations.
# How activation functions help in introducing non-linear operations and overall how a single neuron works.


# Exercises

# 1.Consider the following two networks.  W is a matrix, x is a vector, z is a vector, and a is a vector. y^ is a scalar and a final prediction.
# Initialize x, w randomly, z is a dot product of x and w, a is ReLU(z).  Initialize X and W randomly. Every neuron has a bias term.

import numpy as np
def ReLU(x):
    return np.maximum(0, x)
#input
x=np.random.randn(4,1) #network has 4 input neurons
#HL-1
w1=np.random.randn(3,4)
bias1=np.random.randn(3,1)
z1=w1@x+bias1
a1=ReLU(z1)
#HL-2
w2=np.random.randn(2,3)
bias2=np.random.randn(2,1)
z2=w2@a1+bias2
a2=ReLU(z2)
#output
w3=np.random.randn(1,2)
bias3=np.random.randn(1,1)
z3=w3@a2+bias3
y_hat=ReLU(z3)

print(y_hat)
#print all
print("Input")
print(x)

print("\nHidden Layer 1")
print("z1")
print(z1)

print("a1")
print(a1)

print("\nHidden Layer 2")
print("z2")
print(z2)

print("a2")
print(a2)

print("\nOutput Layer")

print("z3")
print(z3)

print("Prediction")
print(y_hat)




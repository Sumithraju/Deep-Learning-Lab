# Implement backward pass for the above two networks. Print the gradient values for each neuron in each layer.

import numpy as np
#types of activation funtion and its derivative func
def sigmoid(x):
	return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
	return x*(1-x)
def relu(x):
	return x if x > 0 else 0
def relu_derivative(x):
	return 1 if x > 0 else 0
def tanh(x):
	return np.tanh(x)
def tanh_derivative(x):
	return 1 if x > 0 else 0
X=np.array([[0.05],[0.1]])
W=np.random.rand(2,2)



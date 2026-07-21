#1.sigmoid function
import numpy as np
import matplotlib.pyplot as plt

z=np.linspace(-10,10,100)
def sigmoid(z):
	return 1/(1+np.exp(-z))
def sigmoid_derivative(z):
	return z*(1-z)
plt.plot(z,sigmoid(z),label='sigmoid')
plt.plot(z,sigmoid_derivative(z),label='sigmoid derivative')
plt.grid(True)
plt.legend()
plt.show()

#2.Tanh
# tanh activation function
def tanh(z):
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
# Derivative of Tanh Activation Function
def tanh_prime(z):
	return 1 - np.power(tanh(z), 2)
plt.plot(z,tanh_prime(z),label='tanh')
plt.grid(True)
plt.legend()
plt.show()

#3. ReLU (Rectified Linear Unit):
def relu(z):
	return np.maximum(0,z)
plt.plot(z,relu(z),label='relu')
plt.grid(True)
plt.legend()
plt.show()
def relu_prime(z):
	return 1 - np.power(relu(z), 2)
plt.figure()
plt.plot(z,relu(z),label="ReLU")
plt.plot(z,relu_prime(z),label="Derivative")
plt.legend()
plt.grid(True)
plt.legend()
plt.show()

#4.Leaky relu
def leakyrelu(z,alpha=0.01):
    return np.maximum(alpha*z,z)
def leakyrelu_prime(z,alpha=0.01):
    return np.where(z>0,1,alpha)

plt.figure()
plt.plot(z,leakyrelu(z),label="Leaky ReLU")
plt.plot(z,leakyrelu_prime(z),label="Derivative")
plt.legend()
plt.grid(True)
#5.softmax
def softmax(z):
    exp_values=np.exp(z-np.max(z))
    return exp_values/np.sum(exp_values)

print("Softmax Output:")
print(softmax(z))

plt.show()

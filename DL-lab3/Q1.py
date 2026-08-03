# Implement backward pass for the above two networks. Print the gradient values for each neuron in each layer.
#network1
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

x1,x2=2,-1
w1,w2=0.5,1.5
b=0.2
print("Single-Layer Gradients")
z=w1*x1+w2*x2+b
a=sigmoid(z)
print("Z=",z)
print(" Fwd output,A=",a)

#BWP
da=1
dz=sigmoid_derivative(a) #loc grad of sigmoid
print("DZ=",dz)

dw1=dz*x1
dw2=dz*x2
db=dz*1
dx1=dz*w1
dx2=dz*w2
print("DW1=",dx1)
print("DW2=",dx2)
print("DB=",db)
print("W1=",w1)
print("W2=",w2)
print("b=",b)



#network2

import numpy as np
X=np.array([0.05,0.1,2.0,1.5])
#l1: 4 input-3 neurons
W1=np.array([[0.1,0.3,-0.2],[0.5,-0.1,0.4],[0.3,0.2,0.1],[0.2,-0.4,0.6]])
b1=np.array([0.1,-0.2,0.0])

W2=np.array([[0.4,-0.1],[-0.2,-0.5],[0.3,0.2]])
b2=np.array([0.1,-0.2])

W3=np.array([0.6,-0.4])
b3=0.3

#FWD pass
z1=np.dot(X,W1)+b1
a1=np.maximum(0,z1)
z2=np.dot(a1,W2)+b2
a2=np.maximum(0,z2)
z3=np.dot(a2,W3)+b3
output=z3

print("Z1=",z1)
print("Z2=",z2)
print("Z3-FWD pass output=",z3)
#BWD pass
# Assume an upstream loss gradient of dL/d(output) = 1.0
dz3 = 1.0
# Layer 3 Gradients
dW3 = dz3 * a2
db3 = dz3 * 1.0
da2=dz3 *W3
# Layer 2 Gradients
dz2 = da2.copy()
dz2[z2 <= 0] = 0.0           # Local gradient of ReLU derivative
dW2 = np.outer(a1, dz2)
db2 = dz2 * 1.0
da1 = np.dot(dz2, W2.T)      # Gradient sent backward to Layer 1
# Layer 1 Gradients
dz1 = da1.copy()
dz1[z1 <= 0] = 0.0           # Local gradient of ReLU derivative
dW1 = np.outer(X, dz1)
db1 = dz1 * 1.0
# PRINT GRADIENT VALUES FOR EVERY NEURON
print("BACKWARD PASS GRADIENTS PER LAYER")
print("Layer 3 (Output Layer - 1 Neuron)")
print(f"Pre-activation Gradient (dz3): {dz3}")
print(f"Weight Gradients (dW3):        {W3}")
print(f"Bias Gradient (db3):           {db3}")

print("Layer 2 (Hidden Layer 2 - 2 Neurons)")
print(f"Upstream Activation Grad (da2): {da2}")
print(f"Pre-activation Grad (dz2):      {dz2}")
print(f"Bias Gradients (db2):           {db2}")
print(f"Weight Matrix Gradients (dW2):{dW2}")

print("\nLayer 1 (Hidden Layer 1 - 3 Neurons)")
print(f"Upstream Activation Grad (da1): {da1}")



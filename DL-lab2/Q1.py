#2. Implement forward pass for the above two networks. Print activation values for each neuron at each layer. Print the loss value (y^).

import numpy as np

def sigmoid(z):
	return 1/(1+np.exp(-z))

def forward(W,X,b):
	z = np.dot(W,X) + b     #weighted_sum
	A = sigmoid(z)          #activation func
	return A,z

X = np.array([[2],
              [3],
              [4],
              [5]])
weights = [
    np.array([[0.2, 0.4, 0.1, 0.5],
              [0.3, 0.2, 0.6, 0.4],
              [0.7, 0.5, 0.2, 0.1]]),

    np.array([[0.2, 0.3, 0.5],
              [0.6, 0.4, 0.7]]),

    np.array([[0.5, 0.8]])
]
biases = [
    np.array([[0.1],
              [0.2],
              [0.3]]),

    np.array([[0.1],
              [0.2]]),

    np.array([[0.3]])
]
A=X
a_list=[]
z_list=[]

for i in range(len(weights)):
	A,Z = forward(weights[i],A,biases[i])
	a_list.append(A)
	z_list.append(Z)
	
	print("Z=")
	print(Z)
	print("\nA=")
	print(A)
prediction =A
Y=np.array([[1]])
loss=0.5*np.sum((Y-prediction)** 2)
print("loss=",loss)
print("prediction=",prediction)
print("loss=",loss)
	
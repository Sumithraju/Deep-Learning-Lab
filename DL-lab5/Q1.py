# Input layer: 2 neurons (x1, x2)
# Hidden layer: 2 perceptrons (OR and NAND)
# Output layer: 1 perceptron (AND)

# Implement a 2-layer (input layer, hidden layer and output layer) neural network from scratch
# for the XOR operation. This includes implementing forward and backward passes from scratch.

import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
np.random.seed(45)
W1=np.random.randn(2,2)
b1=np.random.randn(1,2)
W2=np.random.randn(2,1)
b2=np.random.randn(1,1)

def sigmoid(x):
	return 1/(1+np.exp(-x))
def derivate(x):
	return x*(1-x)
#fwd prop
Z1=np.dot(X,W1)+b1
A1=sigmoid(Z1)
Z2=np.dot(A1,W2)+b2
y_hat=sigmoid(Z2)
#backwrd pass
#1.output-err
error=y_hat-y
#2.output_grad
d_output=error*derivate(y_hat)
#3.hidden-err
hidden_err=np.dot(d_output,W2.T)
#4.hidden-grad
d_hidden=hidden_err*derivate(A1)
#5.upd_wts and bias
learning_rate=0.1
W2 += np.dot(X.T,d_output)*learning_rate
b2 += np.sum (d_output,axis=0, keepdims=True)*learning_rate
W1 += np.dot(A1.T,d_hidden)*learning_rate
b1 += np.sum (d_hidden)*learning_rate

#Train
for epoch in range(1000):
	Z1=np.dot(X,W1)+b1
	A1=sigmoid(Z1)
	Z2=np.dot(A1,W2)+b2
	y_hat=sigmoid(Z2)
	error=y_hat-y
	d_output=error*derivate(y_hat)
	hidden_error=np.dot(d_output,W2.T)
	d_hidden=hidden_error*derivate(A1)
	W2 += 0.1*np.dot(A1.T,d_output)
	b2 += 0.1*np.sum (d_output,axis=0,keepdims=True)
	W1 += 0.1*np.dot(X.T,d_hidden)
	b1 += 0.1*np.sum (d_hidden,axis=0,keepdims=True)
#pred
print(np.round(y_hat))
	
# Forward Pass

# X
# Z1
# A1
# Z2
# Y_hat

# Backward Pass

# Loss
# Output Error
# dOutput
# dW2, db2
# Hidden Error
# dHidden
# dW1, db1
# Update Weights & Biases




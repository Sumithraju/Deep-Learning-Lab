# Implement a 1-layer (input - output layer) neural network from scratch for the following dataset. This includes
# implementing forward and backward passes from scratch. Print the training loss and plot it over 1000 iterations.
# Dataset
#  x1   x2   x3   y
#   0    0    1   0
#   1    1    1   1
#   1    0    1   1
#   0    1    1   0

# x1, x2, x3 = Input Features
# y = Target Output

import numpy as np
import matplotlib.pyplot as plt

X=np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
Y=np.array([[0],[1],[1],[0]])
def sigmoid(x):
	return 1/(1+np.exp(-x))
def sigmoidDeriv(x):
	return sigmoid(x)*(1-sigmoid(x))
np.random.seed(45)
W1=np.random.randn(3,1)
b1=np.random.randn(1,1)
learning_rate=0.1
epochs=1000
loss_history=[]
for epoch in range(epochs):
	#FP
	Z1=np.dot(X,W1)+b1
	y_hat=sigmoid(Z1)
	#loss
	loss=np.mean((y_hat-Y)**2)
	loss_history.append(loss)
	#BWP
	err=Y-y_hat
	d_output=err*sigmoidDeriv(y_hat)
	dW=np.dot(X.T,d_output)
	db=np.sum(d_output,keepdims=True)
	#upd
	W1 += learning_rate * dW
	b1 += learning_rate * db
	if epoch % 100 == 0:
		print(f"epoch:{epoch},Loss:{loss},W1:{W1},b1:{b1}")
#prediction
print("\nFinal Predictions:")

pred = sigmoid(np.dot(X, W1+b1))
print(np.round(pred))
# Plot Loss
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()



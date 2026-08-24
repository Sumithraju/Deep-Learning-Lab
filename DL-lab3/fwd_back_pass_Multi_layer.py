import numpy as np
np.random.seed(39)
#4-inputs,3HL
x=np.array([[0.5],[0.3,],[0.2],[0.7]])
y=np.array([[1.0]])
lr=0.5
#layersize:4->5->3>1
sizes = [4,5,4,3,1]

#Initialize weights and biases for the 4 connection blocks
# W[L] has shape (neurons _in_layer_L,neuron_in_layer_L-1)
W,b={},{}
for L in range(1,5):
	W[L]=np.random.randn(sizes[L],sizes[L-1]) *0.5
	b[L]=np.zeros((sizes[L],1))
def sigmoid(z):
	return 1/(1+np.exp(-z))
def sigmoid_d(a):
	return a*(1-a)

for epoch in range(1,2001):
	#fwd pass
	a={0:x}# a[0] is the input
	z={}
	for L in range(1,5):
		z[L]=W[L]@a[L-1]+b[L]
		a[L]=sigmoid(z[L])
	y_hat=a[4]
	loss=np.sum((y-y_hat)**2)
	
	#backpass
	delta={}
	delta[4]=2*(y_hat-y)*sigmoid_d(a[4])# output layer delta
	
	#Backprog through HL
	for L in range(3,0,-1):
		delta[L]=(W[L+1].T @ delta[L+1]) * sigmoid_d(a[L])
	#upd wts
	for L in range(1,5):
		W[L] -= lr*(delta[L] @ a[L-1].T)
		b[L] -= lr*delta[L]
		
	#Print progress every 200 epochs
	if epoch % 200 == 0:
		print(f"epoch{epoch} | Loss: {loss} | y_hat: {y_hat[0][0]} | b: {b}")
print(f"\n Final prediction : {y_hat[0][0]} (target: {y[0][0]})")

	
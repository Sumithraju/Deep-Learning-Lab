import numpy as np
class vanilaRNN:
	def __init__(self, vocab_size, hidden_size=50, learning_rate=0.1):
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.learning_rate = learning_rate
		#wt inialization
		self.wxh=np.random.rand(hidden_size,vocab_size)*0.01 #input->hidden
		self.whh=np.random.rand(hidden_size,hidden_size)*0.01#hidden->hidden
		self.why=np.random.rand(vocab_size,hidden_size)*0.01 #hidden->output
		#bias
		self.bh=np.zeros((hidden_size,1))
		self.by=np.zeros((vocab_size,1))
def softmax(x):
	exp_x = np.exp(x - np.max(x)) #numerical stability
	return exp_x / np.sum(exp_x)

		
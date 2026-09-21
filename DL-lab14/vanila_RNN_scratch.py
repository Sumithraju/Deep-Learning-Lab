import numpy as np
class vanilaRNN:
    def __init__(self, vocab_size, hidden_size=50, learning_rate=0.1):
       self.vocab_size = vocab_size
       self.hidden_size = hidden_size
       self.learning_rate = learning_rate
       #wt inialization
       self.wxh=np.random.rand(hidden_size,vocab_size)*0.01 #input->hidden
       self.whh= np.random.rand(hidden_size, hidden_size) * 0.01#hidden->hidden
       self.why=np.random.rand(vocab_size,hidden_size)*0.01 #hidden->output
       #bias
       self.bh=np.zeros((hidden_size,1))
       self.by=np.zeros((vocab_size,1))
    def softmax(self,x):
       exp_x = np.exp(x - np.max(x)) #numerical stability
       return exp_x / np.sum(exp_x)
    def fwd(self,inputs, h_prev):
       xs={}
       hs = {}
       ys = {}
       ps = {}
       hs[-1]=h_prev.copy()
       for t in range(len(inputs)):
          xs[t]=np.zeros((self.vocab_size,1))
          xs[t][inputs[t]]=1                  #one-hot input
          hs[t]=np.tanh(np.dot(self.wxh,xs[t])+np.dot(self.whh,hs[t-1])+self.bh)
          ys[t]=np.dot(self.why,hs[t])+self.by
          ps[t]=self.softmax(ys[t])
       return xs, hs, ys, ps
    def computeLoss(self,ps, targets):          #cross entropy loss
       loss=0
       for t in range(len(targets)):
          loss+= -np.log(ps[t][targets[t],0] +1e-12)
       return loss
    def bwd(self, xs, hs, ps,targets):
       dWxh = np.zeros_like(self.wxh)
       dWhh = np.zeros_like(self.whh)
       dWhy = np.zeros_like(self.why)
       dbh = np.zeros_like(self.bh)
       dby = np.zeros_like(self.by)
       dh_next=np.zeros_like(hs[0])
       for t in reversed(range(len(targets))):
          #softmax+ CE grad
          dy = ps[t].copy()
          dy[targets[t]]-=1
          #hidden->output wts
          dWhy += np.dot(dy,hs[t].T)
          dby += dy
          dh = (np.dot(self.why.T,dy)+dh_next)
          #tanh derivative
          dh_raw = (1 - hs[t]**2)*dh
          dbh += dh_raw
          #input -> hidden
          dWxh += np.dot(dh_raw,xs[t].T)
          #hidden -> hidden
          dWhh += np.dot(dh_raw,hs[t-1].T)
          dh_next = np.dot(self.whh.T,dh_raw)
       #Grad clipping
       for grad in [dWxh,dWhh,dWhy,dbh,dby]:
          np.clip(grad,-10,10,out=grad)
       return dWxh,dWhh,dWhy,dbh,dby
    def update(self,dWxh,dWhh,dWhy,dbh,dby):
       self.wxh -= self.learning_rate*dWxh
       self.whh -= self.learning_rate*dWhh
       self.why -= self.learning_rate*dWhy
       self.bh -= self.learning_rate*dbh
       self.by -= self.learning_rate*dby
    def train_step(self,inputs,targets,h_prev):
       xs,hs,ys,ps = self.fwd(inputs,h_prev)
       loss=self.computeLoss(ps,targets)
       grads = self.bwd(xs,hs,ps,targets)
       self.update(*grads)
       return loss,hs[len(inputs)-1]
text = "Hola Amigo! I love DL "
chars = sorted(list(set(text)))
vocab_size=len(chars)
chr_to_idx={
    ch: i
    for i,ch in enumerate(chars)
}
idx_to_char={
    i:ch
    for ch,i in chr_to_idx.items()
}
print(chars)
print(idx_to_char)
sequence="Hola Amigo!"
inputs = [
    chr_to_idx[ch]
    for ch in sequence
]
targets=[
    chr_to_idx[ch]
    for ch in reversed(sequence)
]
np.random.seed(45)
model=vanilaRNN(
    vocab_size,
    hidden_size=50,
    learning_rate=0.01
)
h_prev=np.zeros((50,1))
for epoch in range(1000):
    inputs=[
       chr_to_idx[ch]
       for ch in text[:-1]
    ]
    targets=[
       chr_to_idx[ch]
       for ch in text[1:]
    ]
    loss,h_prev=model.train_step(
       inputs,
       targets,
       h_prev
    )
    if epoch % 100 == 0:
       print(f'Epoch {epoch}, Loss={loss:.4f}')
       print("Final hidden first 5 values:")
       print(h_prev[:5])
    h_prev=np.zeros((50,1))
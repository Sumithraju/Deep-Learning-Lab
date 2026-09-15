import torch
import torch
print(torch.__version__)
print("CUDA available:",torch.cuda.is_available())
# Exercise 2: Tensors, Dataset, DataLoader, Transform, Model, Autograd, Optimization, Save and Load
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
# Tensor
x=torch.tensor([[1.,2.],[3.,4.]])
print("Tensor:",x)
print("Shape:",x.shape)
# Dataset
class MyDataset(Dataset):
    def __init__(self):
        self.x=torch.tensor([[1.,2.],[2.,3.],[3.,4.],[4.,5.]])
        self.y=torch.tensor([0,0,1,1])
    def __len__(self):
        return len(self.x)
    def __getitem__(self,i):
        return self.x[i],self.y[i]
dataset=MyDataset()
loader=DataLoader(dataset,batch_size=2,shuffle=True)
# Model
model=nn.Sequential(nn.Linear(2,8),nn.ReLU(),nn.Linear(8,2))
# Loss and Optimizer
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
# Training
for epoch in range(10):
    for X,y in loader:
        pred=model(X)
        loss=loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print("Training completed")
# Save model
torch.save(model.state_dict(),"model.pth")
# Load model
model.load_state_dict(torch.load("model.pth",weights_only=True))
print("Model saved and loaded successfully")
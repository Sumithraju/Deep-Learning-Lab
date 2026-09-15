import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
device="cuda" if torch.cuda.is_available() else "cpu"
print("Using:",device)
# Dataset
X=torch.randn(1000,20)
y=torch.randint(0,2,(1000,))
dataset=TensorDataset(X,y)
loader=DataLoader(dataset,batch_size=32,shuffle=True)
# Deep Neural Network
model=nn.Sequential(
    nn.Linear(20,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,2)
).to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
# Training
for epoch in range(10):
    model.train()
    total_loss=0
    for Xb,yb in loader:
        Xb,yb=Xb.to(device),yb.to(device)
        pred=model(Xb)
        loss=loss_fn(pred,yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
print("Training completed successfully")
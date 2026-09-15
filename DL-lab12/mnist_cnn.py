import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train_dataset=datasets.MNIST(root="/home/ibab/PycharmProjects/PythonProject2/DL/DL-lab12/data",train=True,download=False,transform=transform)
test_dataset=datasets.MNIST(root="/home/ibab/PycharmProjects/PythonProject2/DL/DL-lab12/data",train=False,download=False,transform=transform)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(64*7*7,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)))
        x=self.pool(self.relu(self.conv2(x)))
        x=x.view(x.size(0),-1)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNN().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=5
for epoch in range(epochs):
    model.train()
    running_loss=0
    correct=0
    total=0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print("Epoch:",epoch+1,"Loss:",round(running_loss/len(train_loader),4),"Training Accuracy:",round(100*correct/total,2),"%")
model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print("Test Accuracy:",round(100*correct/total,2),"%")
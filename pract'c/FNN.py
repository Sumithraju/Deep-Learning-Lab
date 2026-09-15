import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
#Convert FashionMNIST image to tensor
transform=transforms.ToTensor()
#Load 60,000 training and 10,000 test images
train_data=datasets.FashionMNIST(root="./data",train=True,download=True,transform=transform)
test_data=datasets.FashionMNIST(root="./data",train=False,download=True,transform=transform)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)
#Feed Forward Deep Neural Network
class FashionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(28*28,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        self.fc4=nn.Linear(64,10)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.relu(self.fc3(x))
        x=self.fc4(x)
        return x
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=FashionNet().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
#Train the model
epochs=5
for epoch in range(epochs):
    model.train()
    total_loss=0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print("Epoch:",epoch+1,"Loss:",total_loss/len(train_loader))
#Test the model
model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        probabilities=torch.softmax(outputs,dim=1)
        predicted=torch.argmax(probabilities,dim=1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print("Test Accuracy:",100*correct/total,"%")
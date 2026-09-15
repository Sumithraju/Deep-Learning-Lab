import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

#Device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Transform image to tensor and normalize
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

#Load FashionMNIST dataset
train_data=datasets.FashionMNIST(root="./data",train=True,download=False,transform=transform)
test_data=datasets.FashionMNIST(root="./data",train=False,download=False,transform=transform)

#Create DataLoaders
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)

print("Training dataset size:",len(train_data))
print("Test dataset size:",len(test_data))

#Feed Forward Neural Network with 3 fully connected layers
class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(28*28,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        self.fc4=nn.Linear(64,10)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=x.view(x.size(0),-1) #Flatten 28x28 image
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.relu(self.fc3(x))
        x=self.fc4(x) #10 class scores
        return x

#Create model
model=FNN().to(device)

#Loss function and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

#Train the model
epochs=5
for epoch in range(epochs):
    model.train()
    running_loss=0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print(f"Epoch {epoch+1}/{epochs},Loss:{running_loss/len(train_loader):.4f}")

#Test the model
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

accuracy=100*correct/total
print("Test Accuracy: {:.2f}%".format(accuracy))
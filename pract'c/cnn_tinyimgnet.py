import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
#Transform TinyImageNet images
transform=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
#Folder names are automatically used as class labels
train_data=datasets.ImageFolder("/home/ibab/PycharmProjects/PythonProject2/DL/tiny-imagenet-200/train",transform=transform)
test_data=datasets.ImageFolder("/home/ibab/PycharmProjects/PythonProject2/DL/tiny-imagenet-200/test_classified",transform=transform)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)
num_classes=len(train_data.classes)
print("Classes:",num_classes)
#CNN with three convolution layers
class CNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.conv1=nn.Conv2d(3,32,3,padding=1)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.conv3=nn.Conv2d(64,128,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(128*8*8,256)
        self.fc2=nn.Linear(256,num_classes)
    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)))
        x=self.pool(self.relu(self.conv2(x)))
        x=self.pool(self.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNN(num_classes).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
#Train CNN
epochs=10
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
        predicted=torch.argmax(outputs,dim=1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print("Epoch:",epoch+1,"Loss:",running_loss/len(train_loader),"Train Accuracy:",100*correct/total,"%")
#Calculate test accuracy
model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        predicted=torch.argmax(outputs,dim=1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print("Test Accuracy:",100*correct/total,"%")
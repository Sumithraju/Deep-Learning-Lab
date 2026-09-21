import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
from pathlib import Path
BASE=Path(__file__).resolve().parent
FASTA=BASE/"gene_accessibility"/"Basset"/"data"/"accessibility_10000.fa"
LABELS=BASE/"gene_accessibility"/"Basset"/"data"/"accessibility_10000_labels.txt"
def read_fasta(file):
    seqs=[]
    seq=""
    with open(file) as f:
        for line in f:
            line=line.strip()
            if line.startswith(">"):
                if seq!="":
                    seqs.append(seq)
                seq=""
            else:
                seq+=line
        if seq!="":
            seqs.append(seq)
    return seqs
def read_labels(file):
    labels=[]
    with open(file) as f:
        header=f.readline().strip().split()
        for line in f:
            parts=line.strip().split()
            if len(parts)<2:
                continue
            try:
                values=[float(x) for x in parts[1:]]
                labels.append(values)
            except ValueError:
                continue
    labels=np.array(labels,dtype=np.float32)
    if len(header)==labels.shape[1]:
        names=header
    elif len(header)==labels.shape[1]+1:
        names=header[1:]
    else:
        names=[f"celltype_{i}" for i in range(labels.shape[1])]
    return labels,names
def one_hot(seq,length=600):
    mapping={"A":[1,0,0,0],"C":[0,1,0,0],"G":[0,0,1,0],"T":[0,0,0,1],"N":[0,0,0,0]}
    seq=seq.upper()[:length]
    seq=seq+"N"*(length-len(seq))
    x=np.array([mapping.get(base,[0,0,0,0]) for base in seq],dtype=np.float32)
    return x.T
sequences=read_fasta(FASTA)
label_matrix,label_names=read_labels(LABELS)
print("FASTA sequences:",len(sequences))
print("Label rows:",label_matrix.shape[0])
print("Accessibility columns:",label_matrix.shape[1])
n=min(len(sequences),label_matrix.shape[0])
sequences=sequences[:n]
label_matrix=label_matrix[:n]
positive_counts=label_matrix.sum(axis=0)
negative_counts=n-positive_counts
valid=np.where((positive_counts>=2)&(negative_counts>=2))[0]
if len(valid)==0:
    raise ValueError("No accessibility column contains both classes")
target_col=valid[0]
labels=label_matrix[:,target_col].astype(np.float32)
print("Selected column:",target_col)
print("Cell type:",label_names[target_col])
print("Positive samples:",int(np.sum(labels==1)))
print("Negative samples:",int(np.sum(labels==0)))
print("First sequence length:",len(sequences[0]))
train_seq,test_seq,train_y,test_y=train_test_split(sequences,labels,test_size=0.2,random_state=42,stratify=labels)
class GenomicDataset(Dataset):
    def __init__(self,seqs,labels):
        self.seqs=seqs
        self.labels=labels
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self,idx):
        x=one_hot(self.seqs[idx])
        y=self.labels[idx]
        return torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
train_ds=GenomicDataset(train_seq,train_y)
test_ds=GenomicDataset(test_seq,test_y)
train_loader=DataLoader(train_ds,batch_size=32,shuffle=True)
test_loader=DataLoader(test_ds,batch_size=32,shuffle=False)
class AccessibilityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv1d(4,64,19)
        self.pool1=nn.MaxPool1d(3)
        self.conv2=nn.Conv1d(64,128,11)
        self.pool2=nn.MaxPool1d(4)
        self.conv3=nn.Conv1d(128,128,7)
        self.pool3=nn.MaxPool1d(4)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.5)
        self.fc1=nn.Linear(128*10,256)
        self.fc2=nn.Linear(256,1)
    def forward(self,x):
        x=self.pool1(self.relu(self.conv1(x)))
        x=self.pool2(self.relu(self.conv2(x)))
        x=self.pool3(self.relu(self.conv3(x)))
        x=torch.flatten(x,1)
        x=self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x).squeeze(1)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)
model=AccessibilityCNN().to(device)
criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
for epoch in range(10):
    model.train()
    total_loss=0
    for x,y in train_loader:
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        output=model(x)
        loss=criterion(output,y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1},Loss={total_loss/len(train_loader):.4f}")
model.eval()
true=[]
pred=[]
prob=[]
with torch.no_grad():
    for x,y in test_loader:
        x=x.to(device)
        output=model(x)
        p=torch.sigmoid(output)
        prediction=(p>=0.5).float()
        true.extend(y.numpy())
        pred.extend(prediction.cpu().numpy())
        prob.extend(p.cpu().numpy())
print("Accuracy:",accuracy_score(true,pred))
print("AUROC:",roc_auc_score(true,prob))
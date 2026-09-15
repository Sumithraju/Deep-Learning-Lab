import torch
from torch.utils.data import Dataset,DataLoader,random_split
from torch.nn.utils.rnn import pad_sequence
#Custom Dataset for FASTA file
class FastaDataset(Dataset):
    def __init__(self,file):
        self.sequences=[]
        seq=""
        with open(file,"r") as f:
            for line in f:
                line=line.strip()
                if line.startswith(">"):
                    if seq:
                        self.sequences.append(seq)
                        seq=""
                else:
                    seq+=line
            if seq:
                self.sequences.append(seq)
        self.map={'A':0,'C':1,'G':2,'T':3,'N':4}
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self,idx):
        seq=self.sequences[idx]
        tensor=torch.tensor([self.map.get(x,4) for x in seq],dtype=torch.long)
        return tensor,seq
#Function to pad variable-length DNA sequences in a batch
def collate_fn(batch):
    tensors,seqs=zip(*batch)
    tensors=pad_sequence(tensors,batch_first=True,padding_value=4)
    return tensors,seqs
#Load dataset
dataset=FastaDataset("sample.fasta")
#70% training and 30% testing
train_size=int(0.7*len(dataset))
test_size=len(dataset)-train_size
train_set,test_set=random_split(dataset,[train_size,test_size])
#Create DataLoaders with batch size 2
train_loader=DataLoader(train_set,batch_size=2,shuffle=True,collate_fn=collate_fn)
test_loader=DataLoader(test_set,batch_size=2,shuffle=False,collate_fn=collate_fn)
#Print loader sizes
print("Total sequences:",len(dataset))
print("Train samples:",len(train_set))
print("Test samples:",len(test_set))
print("Train loader batches:",len(train_loader))
print("Test loader batches:",len(test_loader))
#Print first training batch
sequence_tensor,sequences=next(iter(train_loader))
print("Sequence tensor:")
print(sequence_tensor)
print("Sequences:")
print(sequences)
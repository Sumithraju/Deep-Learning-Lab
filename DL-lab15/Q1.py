# Implement vanilla RNN using PyTorch for image captioning. You can use the PyTorch library.
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
vocab_size = 5000
images = torch.randn(100, 3, 224, 224)
# 100 captions, each containing 10 token IDs
captions = torch.randint(  1,  vocab_size, (100, 10))
train_dataset = TensorDataset( images,  captions)
train_loader = DataLoader( train_dataset, batch_size=32, shuffle=True)
#CNN encoder
class EncoderCNN(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18 (weights=models.ResNet18_Weights.DEFAULT)
        self.cnn=nn.Sequential(*list(resnet.children())[:-1])               #rmb final classification layer
        self.fc=nn.Linear(resnet.fc.in_features, hidden_size)               #convert cnn feature to rnn hidden size
    def forward(self,img):
	    with torch.no_grad():
		    feature = self.cnn(img)
		    feature=feature.flatten(1)
		    feature=self.fc(feature)
		    return feature
#vanilla RNN encoder
class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, vocab_size, embed_size):
		super(DecoderRNN, self).__init__()
	
		self.embedding = nn.Embedding(vocab_size,embed_size)
		self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
		self.fc = nn.Linear(hidden_size, vocab_size)
	def forward(self, img_features, caption):
		inputs=caption[:,:-1]
		embeddings = self.embedding(inputs)
		h0=img_features.unsqueeze(0)
		outputs,hidden=self.rnn(embeddings,h0)
		logits=self.fc(outputs)
		return logits
class ImageCaptionRNN(nn.Module):
	def __init__(self,  vocab_size, embed_size=256,hidden_size=512):
		super(ImageCaptionRNN, self).__init__()
		self.encoder = EncoderCNN(hidden_size)
		self.decoder = DecoderRNN(hidden_size, vocab_size, embed_size)
	def forward(self, imges, captions):
		image_features = self.encoder(imges)
		outputs = self.decoder(image_features, captions)
		return outputs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 5000
model = ImageCaptionRNN( vocab_size=vocab_size, embed_size=256,  hidden_size=512).to(device)
criterion = nn.CrossEntropyLoss( ignore_index=0)
optimizer = torch.optim.Adam(  model.parameters(), lr=0.001)
for epoch in range(10):
    model.train()
    for images, captions in train_loader:
        images = images.to(device)
        captions = captions.to(device)
        # Clear old gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(  images, captions  )
        # Expected next words
        targets = captions[:, 1:]
        # Calculate loss
        loss = criterion(  outputs.reshape(-1, vocab_size),targets.reshape(-1))
        # Backpropagation through time
        loss.backward()
        # Update weights
        optimizer.step()
    print( f"Epoch {epoch + 1}, " f"Loss = {loss.item():.4f}")
    
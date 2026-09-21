# ============================================
# A1 - UCI SPLICE JUNCTION CLASSIFICATION
# Feed Forward Neural Network
# ============================================

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# -------------------------------------------------
# 1. Set seed
# -------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------------------------
# 2. Load dataset
# -------------------------------------------------

df = pd.read_csv(
	"splice.data",
    header=None,
    names=["class", "name", "sequence"]
)

print("Dataset shape:", df.shape)

# remove extra spaces
df["class"] = df["class"].str.strip().str.upper()
df["sequence"] = df["sequence"].str.strip().str.upper()

print("\nClass distribution:")
print(df["class"].value_counts())

print("\nExample sequence:")
print(df["sequence"].iloc[0])

print("Sequence length:", len(df["sequence"].iloc[0]))


# -------------------------------------------------
# 3. Preprocess DNA sequence
# -------------------------------------------------

# We DO NOT use the instance-name column.

base_to_index = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}


def one_hot_encode(sequence):

    # 60 positions, 4 possible bases
    encoded = np.zeros((60, 4), dtype=np.float32)

    for i, base in enumerate(sequence):

        if base in base_to_index:
            encoded[i, base_to_index[base]] = 1.0

        # ambiguous bases such as N/D/R/S
        # remain [0,0,0,0]

    # convert 60x4 -> 240
    return encoded.flatten()


# Encode every DNA sequence
X = np.stack(
    df["sequence"].apply(one_hot_encode).values
)

print("\nEncoded X shape:", X.shape)


# -------------------------------------------------
# 4. Encode class labels
# -------------------------------------------------

label_map = {
    "EI": 0,
    "IE": 1,
    "N": 2
}

y = df["class"].map(label_map).values.astype(np.int64)

print("Label shape:", y.shape)
print("Classes:", np.unique(y))


# -------------------------------------------------
# 5. Train-test split
# 80:20, stratified, seed = 42
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# -------------------------------------------------
# 6. Convert to PyTorch tensors
# -------------------------------------------------

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# -------------------------------------------------
# 7. DataLoader
# -------------------------------------------------

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)


# -------------------------------------------------
# 8. Build Feed Forward Neural Network
# -------------------------------------------------

class SpliceFFN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # FC Layer 1
            nn.Linear(240, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # FC Layer 2
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)


model = SpliceFFN()

print("\nModel:")
print(model)


# -------------------------------------------------
# 9. Loss and optimizer
# -------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------------------------------------
# 10. Train model
# At least 5 epochs required
# -------------------------------------------------

epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        # clear gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(X_batch)

        # calculate loss
        loss = criterion(outputs, y_batch)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    average_loss = total_loss / len(train_loader.dataset)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Training Loss: {average_loss:.4f}"
    )


# -------------------------------------------------
# 11. Test the trained model
# -------------------------------------------------

model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch)

        predicted = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(predicted.numpy())
        all_targets.extend(y_batch.numpy())


# -------------------------------------------------
# 12. Accuracy and Macro F1
# -------------------------------------------------

accuracy = accuracy_score(
    all_targets,
    all_predictions
)

macro_f1 = f1_score(
    all_targets,
    all_predictions,
    average="macro"
)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(f"Test Accuracy : {accuracy:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
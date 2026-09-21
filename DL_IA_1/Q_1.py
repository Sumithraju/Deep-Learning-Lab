# ============================================
# A2 - PATHMNIST CLASSIFICATION
# Convolutional Neural Network
# ============================================

import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------
# 1. Set seed
# -------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------------------------
# 2. Check whether GPU is available
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------------------------------
# 3. Load NPZ dataset
# -------------------------------------------------

class PathMNISTDataset(Dataset):

    def __init__(self, filename):

        data = np.load(filename)

        self.images = data["images"]

        self.labels = data["labels"].astype(
            np.int64
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        image = self.images[index]

        label = self.labels[index]

        # numpy:
        # H x W x C

        image = torch.from_numpy(image)

        # PyTorch:
        # C x H x W

        image = image.permute(
            2,
            0,
            1
        )

        # uint8 -> float32
        # normalize 0-255 -> 0-1

        image = image.float() / 255.0

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# -------------------------------------------------
# 4. Use supplied train/validation/test splits
# DO NOT merge and resplit
# -------------------------------------------------

train_dataset = PathMNISTDataset(
	"train_20000.npz"
)

val_dataset = PathMNISTDataset(
	"val_3000.npz"
)

test_dataset = PathMNISTDataset(
	"test_3000.npz"
)


print("\nTraining images:",
      train_dataset.images.shape)

print("Validation images:",
      val_dataset.images.shape)

print("Testing images:",
      test_dataset.images.shape)

print(
    "Classes:",
    np.unique(train_dataset.labels)
)


# -------------------------------------------------
# 5. DataLoaders
# -------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)


# -------------------------------------------------
# 6. Build CNN
# -------------------------------------------------

class PathMNISTCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # -------------------------
            # CONVOLUTION LAYER 1
            # -------------------------

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),


            # -------------------------
            # CONVOLUTION LAYER 2
            # -------------------------

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),


            # -------------------------
            # CONVOLUTION LAYER 3
            # -------------------------

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),


            # Makes the architecture work
            # safely for different image sizes
            nn.AdaptiveAvgPool2d((4, 4))
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            # 9 PathMNIST classes
            nn.Linear(
                128,
                9
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = PathMNISTCNN().to(device)

print("\nCNN Model:")
print(model)


# -------------------------------------------------
# 7. Loss and optimizer
# -------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------------------------------------
# 8. Function for validation/test accuracy
# -------------------------------------------------

def evaluate(loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return accuracy


# -------------------------------------------------
# 9. Train CNN
# Minimum required = 3 epochs
# -------------------------------------------------

epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        # Clear previous gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update parameters
        optimizer.step()


        # -------------------------------
        # Calculate statistics
        # -------------------------------

        total_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    # Average training loss
    training_loss = (
        total_loss
        / len(train_loader.dataset)
    )


    # Training accuracy
    training_accuracy = (
        correct / total
    )


    # Validation accuracy
    validation_accuracy = evaluate(
        val_loader
    )


    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {training_loss:.4f} "
        f"Train Accuracy: "
        f"{training_accuracy:.4f} "
        f"Validation Accuracy: "
        f"{validation_accuracy:.4f}"
    )


# -------------------------------------------------
# 10. Final test accuracy
# -------------------------------------------------

test_accuracy = evaluate(
    test_loader
)


print("\n==============================")
print("FINAL TEST RESULT")
print("==============================")

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Accuracy (%): "
    f"{test_accuracy * 100:.2f}%"
)
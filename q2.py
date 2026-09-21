import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
os.environ["MKL_VISIBLE_DEVICES"]="8"
os.environ["MKL_interper_DEVICES"]="8"

device = "cuda" if torch.cuda.is_available() else "cpu"


class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def run_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_set=np.load("train_20000.npz")
    test_set=np.load("test_3000.npz")
    # train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = train_set
    test_loader = test_set
    # train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = MNIST_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        print(f"[MNIST] Epoch {epoch + 1}/{epochs} "
              f"| loss {running_loss / total:.4f} | train acc {correct / total:.4f}")

    # final test accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    print(f"[MNIST] Test accuracy: {correct / total:.4f}")


# ========================================================================
# PART 2 - CIFAR-10: train error vs. number of layers
# ========================================================================
class SimpleCNN(nn.Module):
  

    def __init__(self, num_conv_layers, num_classes=10):
        super().__init__()
        layers = []
        in_channels = 3
        out_channels = 16
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
            # downsample every couple of layers so spatial size shrinks
            if i % 2 == 1:
                layers.append(nn.MaxPool2d(2))

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)  # (N, C, 1, 1) regardless of depth/input size
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_one_config(num_layers, train_loader, epochs=8):
    model = SimpleCNN(num_conv_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    final_train_error = None
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total
        final_train_error = 1 - train_acc
        print(f"  [{num_layers} conv layers] epoch {epoch + 1}/{epochs} | train error {final_train_error:.4f}")

    return final_train_error


def run_cifar10_depth_study():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    layer_counts = [2, 4, 8, 16, 24]
    train_errors = []
    for n_layers in layer_counts:
        print(f"Training CNN with {n_layers} conv layers...")
        err = train_one_config(n_layers, train_loader, epochs=3)
        train_errors.append(err)



if __name__ == "__main__":
    print("=== Part 1: MNIST classification ===")
    run_mnist()

    print("\n=== Part 2: CIFAR-10 depth study ===")
    run_cifar10_depth_study()
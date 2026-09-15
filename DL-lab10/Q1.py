"""
Lab 10 - Deep Learning model for Target Gene Prediction
from a set of Landmark Genes
IMPORTANT - you must plug in the real dataset yourself:
    1. Read the paper linked in the lab sheet and follow it to the
       dataset it uses (GEO / LINCS L1000 gene expression data).
    2. Replace the `load_real_data()` function below with code that
       reads your downloaded landmark-gene matrix (X) and target-gene
       matrix (y) -- e.g. from a CSV / HDF5 / GCT file.

Until you plug in the real data, this script runs end-to-end on
SYNTHETIC data of the same shape, purely so the model + training loop
can be verified before you point it at the real dataset.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 1. Data loading
def load_real_data():
    """
    TODO: replace this with actual loading code, e.g.

        import pandas as pd
        df_X = pd.read_csv("landmark_genes.csv")   # (num_samples, num_landmark_genes)
        df_y = pd.read_csv("target_genes.csv")     # (num_samples, num_target_genes)
        return df_X.values.astype(np.float32), df_y.values.astype(np.float32)
    """
    raise NotImplementedError("Plug in the real landmark/target gene dataset here.")


def load_synthetic_data(num_samples=5000, num_landmark=943, num_target=500, seed=0):
    """
    Generates a synthetic dataset with roughly the same shape as the
    real problem, so the pipeline can be tested end-to-end before the
    real data is available. Target genes are simulated as a noisy
    non-linear combination of the landmark genes, which a neural net
    can actually learn to predict.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(num_samples, num_landmark).astype(np.float32)
    W_true = rng.randn(num_landmark, num_target).astype(np.float32) * 0.05
    y = np.tanh(X @ W_true) + 0.05 * rng.randn(num_samples, num_target).astype(np.float32)
    return X, y
# 2. Model
class GeneExpressionPredictor(nn.Module):
    """
    A simple feed-forward network: landmark genes in, target genes out.
    """

    def __init__(self, num_landmark_genes, num_target_genes, hidden_sizes=(1000, 1000)):
        super().__init__()
        layers = []
        in_size = num_landmark_genes
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_size = h
        layers.append(nn.Linear(in_size, num_target_genes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 3. Training / evaluation
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                running_val += loss_fn(pred, yb).item() * xb.size(0)
        val_loss = running_val / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1:3d}/{epochs} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")

    return train_losses, val_losses


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # swap this line for load_real_data() once you have the real dataset
    X, y = load_synthetic_data()

    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = GeneExpressionPredictor(num_landmark_genes=X.shape[1], num_target_genes=y.shape[1])
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=30, device=device)

    plt.plot(train_losses, label="train MSE")
    plt.plot(val_losses, label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Target gene prediction from landmark genes")
    plt.legend()
    plt.savefig("lab10_gene_prediction_loss.png")
    plt.show()
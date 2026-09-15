"""
Lab 8 - Dropouts, BatchNorm and Optimizations
implements:
1. Batch Normalization from scratch (forward + backward)
2. Layer Normalization from scratch (forward + backward)
3. Dropout from scratch (forward + backward)
4. Optimizers using PyTorch: SGD, SGD+Momentum, AdaGrad
"""

import numpy as np
import torch
import torch.nn as nn
# 1. Batch Normalization from scratch
class BatchNormScratch:
    """
    Normalizes across the BATCH dimension (per feature) -- same idea as
    nn.BatchNorm1d for inputs of shape (N, D).
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        # learnable parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        # running statistics, used at test/inference time
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        self.cache = None

    def forward(self, x, training=True):
        if training:
            batch_mean = x.mean(axis=0, keepdims=True)
            batch_var = x.var(axis=0, keepdims=True)

            x_hat = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            out = self.gamma * x_hat + self.beta

            # exponential moving average of batch statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            self.cache = (x, x_hat, batch_mean, batch_var)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta
        return out

    def backward(self, dout):
        x, x_hat, mean, var = self.cache
        N = x.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.eps)

        dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * std_inv ** 3, axis=0, keepdims=True)
        dmean = np.sum(dx_hat * -std_inv, axis=0, keepdims=True) + \
            dvar * np.mean(-2.0 * (x - mean), axis=0, keepdims=True)

        dx = dx_hat * std_inv + dvar * 2 * (x - mean) / N + dmean / N
        return dx, dgamma, dbeta

# 2. Layer Normalization from scratch
class LayerNormScratch:
    """
    Normalizes across the FEATURE dimension, independently per sample --
    same idea as nn.LayerNorm.
    """

    def __init__(self, num_features, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.cache = None

    def forward(self, x):
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta

        self.cache = (x, x_hat, mean, var)
        return out

    def backward(self, dout):
        x, x_hat, mean, var = self.cache
        D = x.shape[1]
        std_inv = 1.0 / np.sqrt(var + self.eps)

        dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * std_inv ** 3, axis=1, keepdims=True)
        dmean = np.sum(dx_hat * -std_inv, axis=1, keepdims=True) + \
            dvar * np.mean(-2.0 * (x - mean), axis=1, keepdims=True)

        dx = dx_hat * std_inv + dvar * 2 * (x - mean) / D + dmean / D
        return dx, dgamma, dbeta

# 3. Dropout from scratch (inverted dropout)
class DropoutScratch:
    def __init__(self, p=0.5):
        # p = probability of DROPPING a unit
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if training:
            # inverted dropout: scale kept units by 1/(1-p) so test-time
            # forward pass needs no change at all
            self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            return x * self.mask
        else:
            return x

    def backward(self, dout):
        return dout * self.mask

# sanity check for the scratch implementations
if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.randn(8, 5)  # batch of 8 samples, 5 features each

    bn = BatchNormScratch(num_features=5)
    out_bn = bn.forward(x, training=True)
    dx_bn, dgamma, dbeta = bn.backward(np.ones_like(out_bn))
    print("BatchNorm output -> mean per feature (~0):", out_bn.mean(axis=0))
    print("BatchNorm output -> var per feature (~1):", out_bn.var(axis=0))

    ln = LayerNormScratch(num_features=5)
    out_ln = ln.forward(x)
    dx_ln, dgamma_ln, dbeta_ln = ln.backward(np.ones_like(out_ln))
    print("\nLayerNorm output -> mean per sample (~0):", out_ln.mean(axis=1))
    print("LayerNorm output -> var per sample (~1):", out_ln.var(axis=1))

    do = DropoutScratch(p=0.5)
    out_do = do.forward(x, training=True)
    print("\nDropout output (roughly half the entries become exactly 0):")
    print(out_do)
    # 4. Optimizers (SGD, Momentum, AdaGrad) -- using PyTorch is allowed
    #    for this part of the exercise.
    print("\n--- Optimizer comparison on a toy linear-regression problem ---")

    torch.manual_seed(0)
    X = torch.randn(100, 3)
    true_w = torch.tensor([[2.0], [-1.0], [0.5]])
    y = X @ true_w + 0.1 * torch.randn(100, 1)

    def build_model():
        torch.manual_seed(42)  # identical init for a fair comparison
        return nn.Linear(3, 1)

    optimizers = {
        "SGD": lambda p: torch.optim.SGD(p, lr=0.05),
        "SGD+Momentum": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
        "AdaGrad": lambda p: torch.optim.Adagrad(p, lr=0.05),
    }

    loss_fn = nn.MSELoss()
    history = {}

    for name, make_opt in optimizers.items():
        model = build_model()
        opt = make_opt(model.parameters())
        losses = []
        for epoch in range(100):
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history[name] = losses
        print(f"{name:15s}: final MSE loss = {losses[-1]:.5f}")

    import matplotlib.pyplot as plt
    for name, losses in history.items():
        plt.plot(losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Optimizer comparison: SGD vs Momentum vs AdaGrad")
    plt.legend()
    plt.savefig("lab8_optimizer_comparison.png")
    plt.show()
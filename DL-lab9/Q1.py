"""
Lab 9 - Vanishing & Exploding Gradient Problem
1. VANISHING gradients  -> small weight init (sigmoid derivative <=0.25
   gets multiplied in at every layer, so the gradient shrinks fast the
   further back it travels).
2. EXPLODING gradients  -> large weight init (each backward step
   multiplies by a large weight matrix, so the gradient grows fast the
   further back it travels, even though the sigmoid keeps the forward
   activations bounded in (0, 1)).
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    # 'a' is sigmoid(z), already computed during the forward pass
    return a * (1 - a)


class DeepNet:
    """
    A plain fully-connected sigmoid network with `num_layers` layers,
    used purely to observe how the gradient magnitude changes with depth.
    """

    def __init__(self, num_layers, layer_size, input_size, weight_scale):
        self.num_layers = num_layers
        self.weights = []
        self.biases = []

        sizes = [input_size] + [layer_size] * num_layers
        for i in range(num_layers):
            W = np.random.randn(sizes[i], sizes[i + 1]) * weight_scale
            b = np.zeros((1, sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        activations = [x]
        for W, b in zip(self.weights, self.biases):
            z = activations[-1] @ W + b
            a = sigmoid(z)
            activations.append(a)
        return activations

    def backward(self, activations, dout):
        """
        Back-propagate `dout` (gradient of the loss w.r.t. the network
        output) through every layer and record the gradient NORM of
        dW at each layer, so we can plot how it changes with depth.
        """
        grad_norms = []
        delta = dout
        for i in reversed(range(self.num_layers)):
            a_out = activations[i + 1]
            a_in = activations[i]

            delta = delta * sigmoid_derivative(a_out)
            dW = a_in.T @ delta
            grad_norms.append(np.linalg.norm(dW))

            # propagate the gradient one layer further back
            delta = delta @ self.weights[i].T

        grad_norms.reverse()  # index 0 = layer closest to the input
        return grad_norms


if __name__ == "__main__":
    np.random.seed(0)
    NUM_LAYERS = 30
    LAYER_SIZE = 20
    INPUT_SIZE = 20
    BATCH = 16

    x = np.random.randn(BATCH, INPUT_SIZE)
    dout = np.random.randn(BATCH, LAYER_SIZE)  # pretend gradient coming from the loss

    # vanishing gradient setup: small weights
    net_vanish = DeepNet(NUM_LAYERS, LAYER_SIZE, INPUT_SIZE, weight_scale=0.5)
    acts_vanish = net_vanish.forward(x)
    grads_vanish = net_vanish.backward(acts_vanish, dout)

    # exploding gradient setup: large weights
    net_explode = DeepNet(NUM_LAYERS, LAYER_SIZE, INPUT_SIZE, weight_scale=3.0)
    acts_explode = net_explode.forward(x)
    grads_explode = net_explode.backward(acts_explode, dout)

    print("Gradient norms per layer, vanishing setup (small weights, sigmoid):")
    print(np.array(grads_vanish))
    print("\nGradient norms per layer, exploding setup (large weights, sigmoid):")
    print(np.array(grads_explode))

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(1, NUM_LAYERS + 1), grads_vanish, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_title("Vanishing gradients\n(sigmoid, small weight init)")
    axes[0].set_xlabel("Layer index (1 = closest to input)")
    axes[0].set_ylabel("Gradient norm (log scale)")

    axes[1].plot(range(1, NUM_LAYERS + 1), grads_explode, marker="o", color="red")
    axes[1].set_yscale("log")
    axes[1].set_title("Exploding gradients\n(sigmoid, large weight init)")
    axes[1].set_xlabel("Layer index (1 = closest to input)")
    axes[1].set_ylabel("Gradient norm (log scale)")

    plt.tight_layout()
    plt.savefig("lab9_vanishing_exploding_gradients.png")
    plt.show()
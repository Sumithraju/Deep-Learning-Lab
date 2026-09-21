"""
Lab 11 - Convolutional Neural Network (CNN) components from scratch
1. 2D convolution ("valid" mode, i.e. no padding by default) with a
   3x3 kernel applied on a 32x32 input image.
2. Max-pooling.
"""

import numpy as np
import matplotlib.pyplot as plt


def conv2d(image, kernel, stride=1, padding=0):
    """
    image   : 2D numpy array, shape (H, W)
    kernel  : 2D numpy array, shape (kH, kW)
    stride  : int, step size of the sliding window
    padding : int, number of zero-pixels added on each side of the image

    Returns the convolved output as a 2D numpy array.
    Output size = floor((H + 2*padding - kH) / stride) + 1  (same for W)
    """
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")

    H, W = image.shape
    kH, kW = kernel.shape

    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            row_start = i * stride
            col_start = j * stride
            patch = image[row_start:row_start + kH, col_start:col_start + kW]
            output[i, j] = np.sum(patch * kernel)

    return output


def maxpool2d(image, pool_size=2, stride=2):
    """
    image     : 2D numpy array, shape (H, W)
    pool_size : size of the (square) pooling window
    stride    : step size between windows

    Returns the max-pooled output as a 2D numpy array.
    Output size = floor((H - pool_size) / stride) + 1  (same for W)
    """
    H, W = image.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            row_start = i * stride
            col_start = j * stride
            patch = image[row_start:row_start + pool_size, col_start:col_start + pool_size]
            output[i, j] = np.max(patch)

    return output


if __name__ == "__main__":
    np.random.seed(0)

    # a 32x32 "image" (e.g. a single greyscale channel)
    image = np.random.rand(32, 32)

    # a few standard 3x3 kernels to try
    kernels = {
        "Edge detection": np.array([[-1, -1, -1],
                                     [-1,  8, -1],
                                     [-1, -1, -1]]),
        "Sharpen": np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]]),
        "Box blur": np.ones((3, 3)) / 9.0,
    }

    fig, axes = plt.subplots(1, len(kernels) + 1, figsize=(4 * (len(kernels) + 1), 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original (32x32)")
    axes[0].axis("off")

    for ax, (name, kernel) in zip(axes[1:], kernels.items()):
        conv_out = conv2d(image, kernel, stride=1, padding=0)
        print(f"{name}: input {image.shape} -> conv output {conv_out.shape}")
        ax.imshow(conv_out, cmap="gray")
        ax.set_title(f"{name}\n{conv_out.shape}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("lab11_convolution_outputs.png")
    plt.show()

    # maxpool demo, applied to the edge-detected output
    conv_out = conv2d(image, kernels["Edge detection"], stride=1, padding=0)
    pooled = maxpool2d(conv_out, pool_size=2, stride=2)
    print(f"\nMaxpool: conv output {conv_out.shape} -> pooled output {pooled.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(conv_out, cmap="gray")
    axes[0].set_title(f"Before maxpool {conv_out.shape}")
    axes[0].axis("off")
    axes[1].imshow(pooled, cmap="gray")
    axes[1].set_title(f"After maxpool {pooled.shape}")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("lab11_maxpool_output.png")
    plt.show()
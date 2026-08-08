import math

x1, x2 = 0.5, 0.3
w1, w2 = 0.4, 0.6
b = 0.1
y = 1.0
lr = 0.5   # bigger step so learning is visible faster

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

for epoch in range(1, 1001):
    # forward
    z = x1 * w1 + x2 * w2 + b
    y_hat = sigmoid(z)
    loss = (y - y_hat) ** 2

    # backward
    delta = 2 * (y_hat - y) * y_hat * (1 - y_hat)
    grad_w1, grad_w2, grad_b = delta * x1, delta * x2, delta

    # update
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
    b  -= lr * grad_b

    if epoch % 200 == 0 or epoch == 1:
        print(f"epoch {epoch:4d} | loss {loss:.5f} | y_hat {y_hat:.4f}")
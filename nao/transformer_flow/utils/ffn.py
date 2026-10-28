import numpy as np


def ffn(Z, W1, W2, b1, b2):
    H = Z @ W1 + b1
    A = np.maximum(H, 0)
    Y_FFN = A @ W2 + b2

    return Y_FFN
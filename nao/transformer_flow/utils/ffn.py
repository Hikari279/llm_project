import numpy as np


def ffn(Z_norm, W1, W2, b1, b2):
    H = Z_norm @ W1 + b1
    A = np.maximum(H, 0)
    Y_FFN = A @ W2 + b2

    return H, A, Y_FFN
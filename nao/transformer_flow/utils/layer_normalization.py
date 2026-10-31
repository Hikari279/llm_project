import numpy as np


def layer_norm(X, gamma, beta):
    mean = np.mean(X, axis=1, keepdims=True)
    diff = X - mean
    variance = np.mean(np.square(diff), axis=1, keepdims=True)
    epsilon = 1e-5


    norm = diff / np.sqrt(variance + epsilon)
    Y_LN = gamma * norm + beta

    return norm, Y_LN, variance

def layer_norm_backward(dY, norm, variance, gamma):
    d_norm = dY * gamma

    d = norm.shape[1]
    inv_std = 1 / np.sqrt(variance + 1e-5)

    dX = inv_std * (
        d_norm
        - np.mean(d_norm, axis=1, keepdims=True)
        - norm * np.mean(d_norm * norm, axis=1, keepdims=True)
    )

    return dX
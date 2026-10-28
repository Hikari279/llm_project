import numpy as np


def layer_norm(X, gamma, beta):
    mean = np.mean(X, axis=1, keepdims=True)
    diff = X - mean
    variance = np.mean(np.square(diff), axis=1, keepdims=True)
    epsilon = 1e-5


    norm = diff / np.sqrt(variance + epsilon)
    Y_LN = gamma * norm + beta

    return Y_LN
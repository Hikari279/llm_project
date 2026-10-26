import numpy as np

def layer_norm_backward(grad_y, x, gamma, epsilon=1e-5):
    D = x.shape[1]

    mean = np.mean(x, axis=1, keepdims=True)
    variance = np.var(x, axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(variance + epsilon)

    normalized = (x - mean) * inv_std

    grad_gamma = np.sum(grad_y * normalized, axis=0)
    grad_beta = np.sum(grad_y, axis=0)

    grad_normalized = grad_y * gamma

    grad_x = (inv_std / D) * (
        D * grad_normalized
        - np.sum(grad_normalized, axis=1, keepdims=True)
        - normalized * np.sum(
            grad_normalized * normalized,
            axis=1,
            keepdims=True
        )
    )

    return grad_x, grad_gamma, grad_beta

"""
# Test LayerNorm backward pass
N = 6
D = 8

x = np.random.randn(N, D)
gamma = np.ones(D)
grad_y = np.random.randn(N, D)

grad_x, grad_gamma, grad_beta = layer_norm_backward(
    grad_y, x, gamma
)

print("grad_x shape:", grad_x.shape)
print("grad_gamma shape:", grad_gamma.shape)
print("grad_beta shape:", grad_beta.shape)
"""
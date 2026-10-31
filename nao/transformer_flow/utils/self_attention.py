import numpy as np


def self_attention(X_norm, W_Q, W_K, W_V, W_O):
    t = X_norm.shape[0]


    Q = X_norm @ W_Q
    K = X_norm @ W_K
    V = X_norm @ W_V


    scores = Q @ np.transpose(K)
    scaled_scores = scores / np.sqrt(K.shape[1])

    causal_mask = np.triu(np.ones((t, t), dtype=bool), k=1)

    masked_scores = np.where(causal_mask, -np.inf, scaled_scores)

    exp_scores = np.exp(masked_scores)
    sum_exp = np.sum(exp_scores, axis=1, keepdims=True)

    attention_weights = exp_scores / sum_exp

    C = attention_weights @ V
    
    attention_output = C @ W_O

    cache = {
        "X_norm": X_norm,
        "Q": Q,
        "K": K,
        "V": V,
        "attention_weights": attention_weights,
        "C": C,
        "causal_mask": causal_mask
    }

    return attention_output, cache

def self_attention_backward(
    dAttention_output, cache,
    W_Q, W_K, W_V, W_O
):
    X_norm = cache["X_norm"]
    Q = cache["Q"]
    K = cache["K"]
    V = cache["V"]
    A = cache["attention_weights"]
    C = cache["C"]
    causal_mask = cache["causal_mask"]

    d_k = K.shape[1]

    # 1. Output projection: attention_output = C @ W_O
    dW_O = C.T @ dAttention_output
    dC = dAttention_output @ W_O.T

    # 2. Context: C = A @ V
    dA = dC @ V.T
    dV = A.T @ dC

    # 3. Backpropagation through softmax
    dScores = A * (
        dA - np.sum(dA * A, axis=1, keepdims=True)
    )

    # Causal mask: future positions cannot receive gradients
    dScores[causal_mask] = 0

    # 4. Backpropagation through scaled scores
    dScaledScores = dScores / np.sqrt(d_k)

    dQ = dScaledScores @ K
    dK = dScaledScores.T @ Q

    # 5. Gradients for Q, K, V projections
    dW_Q = X_norm.T @ dQ
    dW_K = X_norm.T @ dK
    dW_V = X_norm.T @ dV

    # 6. Gradient flowing back to LayerNorm 1
    dX_norm = (
        dQ @ W_Q.T
        + dK @ W_K.T
        + dV @ W_V.T
    )

    return dW_Q, dW_K, dW_V, dW_O, dX_norm
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

    attention_output = attention_weights @ V
    
    attention_output = attention_output @ W_O

    return attention_output
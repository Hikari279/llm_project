import numpy as np

def logits(transformer_output, W_vocab):
    logits = transformer_output @ W_vocab

    return logits


def probs(logits):
    # 1. Softmax
    row_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - row_max)
    
    probs_matrix = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


    return probs_matrix

def loss(probs_matrix, target_ids):
    rows = np.arange(len(target_ids))
    correct_probs = probs_matrix[rows, target_ids]

    epsilon = 1e-12
    losses = -np.log(np.clip(correct_probs, epsilon, 1.0))

    return np.mean(losses)

import numpy as np

def layer_norm(matrix):
    epsilon = 1e-5

    means = np.mean(matrix, axis=1, keepdims=True)
    variances = np.var(matrix, axis=1, keepdims=True)
    denominator = np.sqrt(variances + epsilon)
    normalized = (matrix - means) / denominator

    gamma = np.ones(matrix.shape[1])
    beta = np.zeros(matrix.shape[1])

    return gamma * normalized + beta

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention(matrix):
    num_rows, dim = matrix.shape

    normalized_matrix = layer_norm(matrix)

    # *0.1 to make the initial random weights smaller
    W_query = np.random.randn(dim, dim) * 0.1
    W_key = np.random.randn(dim, dim) * 0.1
    W_value = np.random.randn(dim, dim) * 0.1

    q = normalized_matrix @ W_query
    k = normalized_matrix @ W_key
    v = normalized_matrix @ W_value

    scores = q @ k.T / np.sqrt(dim)

    mask = np.triu(
    np.ones((num_rows, num_rows), dtype=bool),   k=1)
    masked_scores = np.where(mask, -np.inf, scores)

    attention_weights = softmax(masked_scores)

    W_output = np.random.randn(dim, dim) * 0.1

    attention_output = attention_weights @ v
    projected_output = attention_output @ W_output

    return matrix + projected_output

def feed_forward(matrix):
    dim = matrix.shape[1]

    normalized_matrix = layer_norm(matrix)

    W1 = np.random.randn(dim, 32) * 0.1
    W2 = np.random.randn(32, dim) * 0.1

    hidden = normalized_matrix @ W1
    hidden = np.maximum(0, hidden)
    ffn_output = hidden @ W2

    return matrix + ffn_output
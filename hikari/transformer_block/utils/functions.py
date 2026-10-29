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

def softmax_backward(grad_output, probabilities):
    weighted_sum = np.sum(
        grad_output * probabilities,
        axis=-1,
        keepdims=True
    )

    return probabilities * (grad_output - weighted_sum)

def check_gradient(block, input_matrix, grad_output, parameter_name,
                   index=(0, 0), epsilon=1e-5):

    parameter = getattr(block, parameter_name)
    original = parameter[index].copy()

    # Calculate analytical gradient
    block.forward(input_matrix)
    block.backward(grad_output)

    analytical_gradient = getattr(
        block, "grad_" + parameter_name
    )[index]

    # Calculate numerical gradient: loss at +epsilon
    parameter[index] = original + epsilon
    loss_plus = np.sum(block.forward(input_matrix) * grad_output)

    # Calculate numerical gradient: loss at -epsilon
    parameter[index] = original - epsilon
    loss_minus = np.sum(block.forward(input_matrix) * grad_output)

    # Restore original parameter
    parameter[index] = original

    numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)

    difference = abs(analytical_gradient - numerical_gradient)

    print(f"\nParameter: {parameter_name}{index}")
    print(f"Analytical: {analytical_gradient:.12f}")
    print(f"Numerical:  {numerical_gradient:.12f}")
    print(f"Difference: {difference:.3e}")

def check_input_gradient(block, input_matrix, grad_output, index=(0, 0),
                         epsilon=1e-5):

    original = input_matrix[index].copy()

    # Analytical gradient
    block.forward(input_matrix)
    analytical = block.backward(grad_output)[index]

    # Numerical gradient
    input_matrix[index] = original + epsilon
    loss_plus = np.sum(block.forward(input_matrix) * grad_output)

    input_matrix[index] = original - epsilon
    loss_minus = np.sum(block.forward(input_matrix) * grad_output)

    input_matrix[index] = original

    numerical = (loss_plus - loss_minus) / (2 * epsilon)

    print(f"Analytical: {analytical:.12f}")
    print(f"Numerical:  {numerical:.12f}")
    print(f"Difference: {abs(analytical - numerical):.3e}")

def apply_temperature(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")

    scaled = logits / temperature
    scaled = scaled - np.max(scaled)

    exp_values = np.exp(scaled)
    return exp_values / np.sum(exp_values)

def sample_token(probs):
    return np.random.choice(len(probs), p=probs)

def top_k_sample(probs, k=10):
    if not 1 <= k <= len(probs):
        raise ValueError("k must be between 1 and vocabulary size")

    top_indices = np.argpartition(probs, -k)[-k:]

    filtered = np.zeros_like(probs)
    filtered[top_indices] = probs[top_indices]
    filtered /= filtered.sum()

    return sample_token(filtered)

def top_p_sample(probs, p=0.9):
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    cumulative = np.cumsum(sorted_probs)
    remove = cumulative - sorted_probs >= p

    filtered = probs.copy()
    filtered[sorted_indices[remove]] = 0
    filtered /= filtered.sum()

    return sample_token(filtered)

#temp
"""next_token_logits = logits[-1]

for temp in [0.5, 1.0, 2.0]:
    probs = apply_temperature(next_token_logits, temp)
    print(temp, np.max(probs))"""

#random sampling
"""probs = apply_temperature(logits[-1], temperature=1.0)

next_token_id = sample_token(probs)
print("Chosen token ID:", next_token_id)
print("Token:", inverted_vocab[next_token_id])"""

#top k-sampling
"""probs = apply_temperature(logits[-1], temperature=1.0)

next_token_id = top_k_sample(probs, k=10)
print("Token:", inverted_vocab[next_token_id])"""

#top p-sampling
"""probs = apply_temperature(logits[-1], temperature=1.0)

next_token_id = top_p_sample(probs, p=0.9)
print("Token:", inverted_vocab[next_token_id])"""


import numpy as np

# # gradient
# probs = np.array([0.05, 0.05, 0.9])
# target = np.array([0, 0, 1])

# gradient = probs - target

# # weight
# transformer_output = np.array([2, 3])
# g = np.array([0.2, 0.3, -0.5])

# W_vocab = np.outer(transformer_output, g)


num_loop = 3
transformer_output = np.array([2, 3])
target = np.array([0, 0, 1])
learning_rate = 0.1
W_vocab = np.array([[0.1, -0.2, 0.05], [0.0, 0.1, -0.1]])


for i in range(num_loop):
    # forward pass: hitung logits
    logits = transformer_output @ W_vocab
    print(f"W_vocab ke - {i}: {W_vocab}")

    # softmax
    row_max = np.max(logits)
    exp_logits = np.exp(logits - row_max)
    probs_matrix = exp_logits / np.sum(exp_logits)


    # loss
    rows = np.arange(len(target))
    correct_probs = probs_matrix[2]
    losses = -np.log(correct_probs)
    loss = np.mean(losses)

    print("Loss ke-", i, ": ", loss )
    # gradient
    dZ = probs_matrix - target
    dW_vocab = np.outer(transformer_output, dZ)

    # update
    W_vocab = W_vocab - (learning_rate * dW_vocab)

print("Last Weight: ", W_vocab)
print("")



dH = dZ @ W_vocab.T
print(dH)
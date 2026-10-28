import numpy as np

from utils.bpe_functions import tokenize_bytes, build_vocab, encode
from transformer_block_class import TransformerBlock
from layerNorm_backward_gradient import layer_norm_backward

merge_rules = [(b'a', b't'), (b'h', b'e'), (b'he', b' '), (b'at', b' '), (b'T', b'he ')]

# Optional: makes initialization reproducible
np.random.seed(42)

# =========================
# 1. SETUP — RUN ONCE
# =========================

text = "The cat sat."
tokens = tokenize_bytes(text, merge_rules)

byte_vocab = build_vocab(merge_rules)
token_ids = encode(byte_vocab, tokens)

vocab_size = len(byte_vocab)
embedding_dim = 8
hidden_dim = 32
max_positions = 128

input_ids = token_ids[:-1]
target_ids = np.array(token_ids[1:])

# Embedding parameters
embedding_matrix = np.random.randn(vocab_size, embedding_dim)
positional_matrix = np.random.randn(max_positions, embedding_dim)

# Transformer blocks — initialize only once
block1 = TransformerBlock(dim=embedding_dim, hidden_dim=hidden_dim)
block2 = TransformerBlock(dim=embedding_dim, hidden_dim=hidden_dim)

# Final normalization parameters
gamma_final = np.ones(embedding_dim)
beta_final = np.zeros(embedding_dim)

# Language-model head — initialize only once
W_vocab = np.random.randn(embedding_dim, vocab_size) * 0.1

# =========================
# 2. FORWARD PASS
# =========================
def forward_pass(input_ids, target_ids):
    token_embeddings = embedding_matrix[input_ids]
    position_embeddings = positional_matrix[:len(input_ids)]

    combined_embeddings = token_embeddings + position_embeddings

    output1 = block1.forward(combined_embeddings)
    output2 = block2.forward(output1)

    gpt_output = block1.layer_norm(output2, gamma_final, beta_final)

    logits = gpt_output @ W_vocab
    probs = block1.softmax(logits)
    
    #gradient
    errors = probs.copy()
    errors[np.arange(len(target_ids)), target_ids] -= 1
    N = len(target_ids)
    grad_W_vocab = gpt_output.T @ errors / N
    grad_gpt_output = errors @ W_vocab.T

    grad_output2, grad_gamma_final, grad_beta_final = layer_norm_backward(
    grad_gpt_output,
    output2,
    gamma_final)

    #loss
    correct_probs = probs[np.arange(len(target_ids)), target_ids]
    loss = -np.mean(np.log(correct_probs))

    return loss, grad_W_vocab, probs


loss, grad_W_vocab, probs = forward_pass(input_ids, target_ids)



learning_rate = 0.1
for i in range(5):
    loss_before, grad_W_vocab, _ = forward_pass(input_ids, target_ids)
    W_vocab -= learning_rate * grad_W_vocab
    loss_after, _, _ = forward_pass(input_ids, target_ids)
    print(f"Loop {i}: {loss_before:.4f} -> {loss_after:.4f}")


current_loss, _, probs = forward_pass(input_ids, target_ids)
prob_after = probs[-1, 99]
print("prob after:", prob_after)

value = target_ids[-1]
print(value)


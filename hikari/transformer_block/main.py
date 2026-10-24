import numpy as np

from utils.bpe_functions import tokenize_bytes, build_vocab, encode
from transformer_block_class import TransformerBlock

merge_rules = [(b'a', b't'), (b'h', b'e'), (b'he', b' '), (b'at', b' '), (b'T', b'he ')]


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

    correct_probs = probs[np.arange(len(target_ids)), target_ids]
    loss = -np.mean(np.log(correct_probs))
    return loss


loss1 = forward_pass(input_ids, target_ids)
loss2 = forward_pass(input_ids, target_ids)

print("Loss 1:", loss1)
print("Loss 2:", loss2)

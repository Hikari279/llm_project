import sys
from pathlib import Path
import json
import numpy as np

# DIRECTORY SETTING
nao_dir = Path(__file__).resolve().parents[1]

if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

vocabulary_path = nao_dir / "bpe" / "data" / "vocabulary.json"


# IMPORT FUNCTION AND VARIABLES
from embeddings.embedding import COMBINED, embedding_dim, IDS
with vocabulary_path.open("r", encoding="utf-8") as f:
    vocabulary = json.load(f)

from transformer_flow.utils.layer_normalization import layer_norm
from transformer_flow.utils.self_attention import self_attention
from transformer_flow.utils.ffn import ffn
from transformer_flow.utils.loss import logits, probs, loss


# CONFIG
# Load vocabulary
with vocabulary_path.open("r", encoding="utf-8") as f:
    vocabulary = json.load(f)

# CONFIG
VOCABULARY = vocabulary
vocab_size = len(VOCABULARY)
ff_dim = 32

# flat ids
token_ids = np.array([
    token_id
    for group in IDS
    for token_id in group
])

X = COMBINED[:-1]
target_ids = token_ids[1:]

T = X.shape[0]

# LayerNorm
gamma = np.ones(embedding_dim)
beta = np.zeros(embedding_dim)

# FFN
W1 = np.random.randn(embedding_dim, ff_dim) * 0.02
b1 = np.zeros(ff_dim)
W2 = np.random.randn(ff_dim, embedding_dim) * 0.02
b2 = np.zeros(embedding_dim)

# Self-Attention
W_Q = np.random.randn(embedding_dim, embedding_dim) * 0.02
W_K = np.random.randn(embedding_dim, embedding_dim) * 0.02
W_V = np.random.randn(embedding_dim, embedding_dim) * 0.02
W_O = np.random.randn(embedding_dim, embedding_dim) * 0.02

# Vocabulary Head
W_vocab = np.random.randn(embedding_dim, vocab_size) * 0.02

# Loss
learning_rate = 0.01


# loop
num_loop = 20

# MAIN

# LayerNorm 1 + Self-Attention
X_norm = layer_norm(X, gamma, beta)
attention_output = self_attention(
    X_norm, W_Q, W_K, W_V, W_O
)

# Residual 1
Z = X + attention_output

# LayerNorm 2 + FFN
Z_norm = layer_norm(Z, gamma, beta)
ffn_output = ffn(Z_norm, W1, W2, b1, b2)

# Residual 2
output = Z + ffn_output


for i in range(num_loop):
    # Vocabulary Head
    logits_output = logits(output, W_vocab)
    probs_matrix = probs(logits_output)

    # Loss
    loss_output = loss(probs_matrix, target_ids)
    print(f"Loss ke {i+1} : {loss_output}")

    # Gradient
    dZ = probs_matrix.copy()
    dZ[np.arange(len(target_ids)), target_ids] -= 1

    dW_vocab = (output.T @ dZ) / T
    
    W_vocab = W_vocab - (learning_rate * dW_vocab)



# print("Input shape:", X.shape)
# print("Target shape:", target_ids.shape)
# print("Output shape:", output.shape)
# print("Logits shape:", logits_output.shape)
# print("Probabilities shape:", probs_matrix.shape)
# print("Loss:", loss_output)
# print("dZ shape:", dZ.shape)
# print("dW_vocab shape:", dW_vocab.shape)
# print("Shape compatible:", dW_vocab.shape == W_vocab.shape)
# print("All finite:", np.isfinite(dW_vocab).all())

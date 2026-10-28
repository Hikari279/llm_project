import json
import sys
from pathlib import Path

import numpy as np


nao_dir = Path(__file__).resolve().parents[1]

if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from embeddings.embedding import embedding_dim
from transformer_practice.transformer_block.transformer import output

vocabulary_path = nao_dir / "bpe" / "data" / "vocabulary.json"

with vocabulary_path.open("r", encoding="utf-8") as f:
    vocabulary = json.load(f)

vocab_size = len(vocabulary)


W_vocab = np.random.randn(embedding_dim, vocab_size)

def logits(transformer_output):
    logits = transformer_output @ W_vocab

    return logits

logits = logits(output)



def probs(logits):
    # 1. Softmax
    row_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - row_max)
    
    probs_matrix = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 2. Predict
    predicted_ids = np.argmax(probs_matrix, axis=1)

    return probs_matrix, predicted_ids


def loss(probs_matrix, token_ids):
    target_ids = token_ids

    rows = np.arange(len(target_ids))
    correct_probs = probs_matrix[rows, target_ids]


    losses = -np.log(correct_probs)
    loss = np.mean(losses)

    return loss


    

# # 1. Validasi shape
# print("Logits shape:", logits.shape)
# print("Probabilities shape:", probs_matrix.shape)
# print("Token IDs shape:", np.shape(token_ids))

# print(
#     "Jumlah target cocok:",
#     probs_matrix.shape[0] == len(token_ids)
# )

# # 2. Validasi rentang target IDs
# target_ids = np.asarray(token_ids, dtype=int)

# print(
#     "Semua target ID valid:",
#     np.all((target_ids >= 0) & (target_ids < probs_matrix.shape[1]))
# )

# # 3. Validasi probabilitas
# print(
#     "Semua probabilitas finite:",
#     np.all(np.isfinite(probs_matrix))
# )

# print(
#     "Probabilitas dalam rentang [0, 1]:",
#     np.all((probs_matrix >= 0) & (probs_matrix <= 1))
# )

# print(
#     "Jumlah probabilitas tiap baris mendekati 1:",
#     np.allclose(np.sum(probs_matrix, axis=1), 1.0)
# )

# # 4. Validasi loss
# print("Loss:", loss)
# print("Loss finite:", np.isfinite(loss))
# print("Loss non-negatif:", loss >= 0)

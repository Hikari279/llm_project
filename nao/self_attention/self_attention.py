# DIRECTORY SETTING
import sys
from pathlib import Path

import numpy as np


nao_dir = Path(__file__).resolve().parents[1]

if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from embeddings.embedding import COMBINED, embedding_dim

t = COMBINED.shape[0]


W_Q = np.random.randn(embedding_dim, embedding_dim)
W_K = np.random.randn(embedding_dim, embedding_dim)
W_V = np.random.randn(embedding_dim, embedding_dim)

Q = COMBINED @ W_Q
K = COMBINED @ W_K
V = COMBINED @ W_V


scores = Q @ np.transpose(K)
scaled_scores = scores / np.sqrt(K.shape[1])

causal_mask = np.triu(np.ones((t, t), dtype=bool), k=1)

masked_scores = np.where(causal_mask, -np.inf, scaled_scores)

exp_scores = np.exp(masked_scores)
sum_exp = np.sum(exp_scores, axis=1, keepdims=True)

attention_weights = exp_scores / sum_exp

attention_output = attention_weights @ V

W_O = np.random.randn(embedding_dim, embedding_dim)
Y = attention_output @ W_O


# print(Y.shape == COMBINED.shape)
# print(np.allclose(attention_weights.sum(axis=1), 1.0))
# print(np.allclose(attention_weights[causal_mask], 0.0))
# print(np.isfinite(Y).all())
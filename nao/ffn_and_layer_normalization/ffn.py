import sys
from pathlib import Path
import numpy as np

nao_dir = Path(__file__).resolve().parents[1]

if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from self_attention.self_attention import Y
from embeddings.embedding import embedding_dim


d_ff = 32


# TRANSFORMATION 1
W1 = np.random.randn(embedding_dim, d_ff)
b1 = np.random.randn(d_ff)

H = Y @ W1 + b1


# ReLU
A = np.maximum(H, 0)



# TRANSFORMATION 2
W2 = np.random.randn(d_ff, embedding_dim)
b2 = np.random.randn(embedding_dim)

Y_FFN = A @ W2 + b2


# print("Output shape:", Y_FFN.shape)
# print("Shape cocok:", Y_FFN.shape == Y.shape)
# print("Semua nilai finite:", np.isfinite(Y_FFN).all())
# print("ReLU non-negatif:", np.all(A >= 0))
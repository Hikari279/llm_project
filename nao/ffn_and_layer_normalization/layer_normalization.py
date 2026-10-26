import sys
from pathlib import Path
import numpy as np


nao_dir = Path(__file__).resolve().parents[1]

if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from embeddings.embedding import embedding_dim
from ffn_and_layer_normalization.ffn import Y_FFN

gamma = np.ones(embedding_dim)
beta = np.zeros(embedding_dim)

mean = np.mean(Y_FFN, axis=1, keepdims=True)
diff = Y_FFN - mean
variance = np.mean(np.square(diff), axis=1, keepdims=True)
epsilon = 1e-5


norm = diff / np.sqrt(variance + epsilon)
Y_LN = gamma * norm + beta

# print("Input shape:", Y_FFN.shape)
# print("Output shape:", Y_LN.shape)

# print("Shape compatible:", Y_LN.shape == Y_FFN.shape)
# print("All Finite:", np.isfinite(Y_LN).all())
# print("Mean close to 0:", np.allclose(Y_LN.mean(axis=1), 0.0, atol=1e-5))
# print("Variance close to 1:", np.allclose(Y_LN.var(axis=1), 1.0, atol=1e-4))
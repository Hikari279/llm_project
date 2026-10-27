import sys
from pathlib import Path
import numpy as np

nao_dir = Path(__file__).resolve().parents[1]

if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from embeddings.embedding import COMBINED, embedding_dim
from ffn_and_layer_normalization.layer_normalization import layer_norm
from ffn_and_layer_normalization.ffn import ffn
from self_attention.self_attention import self_attention

# Input
X = COMBINED
d_ff = 32

# LayerNorm 1
attention_output = self_attention(layer_norm(X))

# First Residual
Z =  X + attention_output

# LayerNorm 2
ffn_output = ffn(layer_norm(Z), d_ff)
output = Z + ffn_output

# print("Input shape:", X.shape)
# print("Output shape:", output.shape)
# print("Shape compatible:", X.shape == output.shape)
# print("All finite:", np.isfinite(output).all())




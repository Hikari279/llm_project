import numpy as np
from pathlib import Path
import sys

# DIRECTORY SETTING
nao_dir = Path(__file__).resolve().parents[2]

if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from bpe.utils.tokenize_utils import tokenize_text
from transformer_flow.utils.layer_normalization import layer_norm
from transformer_flow.utils.self_attention import self_attention
from transformer_flow.utils.ffn import ffn
from transformer_flow.utils.loss import logits, probs


def predict_next_token(
    text,
    embedding_table,
    positional_embedding,
    W_Q, W_K, W_V, W_O,
    W1, b1, W2, b2,
    W_vocab,
    gamma, beta
):
    # 1. Tokenization
    IDS = tokenize_text(text)

    token_ids = np.array(
        [token_id for group in IDS for token_id in group],
        dtype=np.int64
    )

    # 2. Positional IDs
    position_ids = np.arange(len(token_ids))

    # 3. Embeddings
    token_embeddings = embedding_table[token_ids]
    position_embeddings = positional_embedding[position_ids]

    COMBINED = token_embeddings + position_embeddings

    # 4. Layer Normalization
    norm1, X_norm, variance1 = layer_norm(
        COMBINED, gamma, beta
    )

    # 5. Self Attention
    attention_output, attention_cache = self_attention(
        X_norm, W_Q, W_K, W_V, W_O
    )

    Z = COMBINED + attention_output

    # 6. Layer Normalization
    norm2, Z_norm, variance2 = layer_norm(
        Z, gamma, beta
    )

    # 7. Feed Forward Network
    ffn_result = ffn(Z_norm, W1, W2, b1, b2)

    Y_FFN = ffn_result[2]

    output = Z + Y_FFN

    # 8. Vocabulary Head
    logits_output = logits(output, W_vocab)

    # 9. Softmax
    probs_matrix = probs(logits_output)

    # 10. Ambil posisi terakhir
    next_token_probs = probs_matrix[-1]

    predicted_id = np.argmax(next_token_probs)

    return predicted_id, next_token_probs
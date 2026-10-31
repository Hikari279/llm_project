import sys
from pathlib import Path
import json
import numpy as np

# DIRECTORY SETTING
nao_dir = Path(__file__).resolve().parents[1]

if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from config import (
    VOCABULARY_PATH, 
    CHECKPOINT_PATH,
    EMBEDDING_DIM,
    FF_DIM,
    LEARNING_RATE,
    NUM_EPOCHS
)


# IMPORT FUNCTION AND VARIABLES
from embeddings.embedding import (
    embedding_table,
    positional_embedding
)

from embeddings.data_preparation import (
    training_sequences
)

from transformer_flow.utils.layer_normalization import (
    layer_norm,
    layer_norm_backward
)

from transformer_flow.utils.self_attention import (
    self_attention,
    self_attention_backward
)

from transformer_flow.utils.ffn import ffn
from transformer_flow.utils.loss import logits, probs, loss


# =========== CONFIGURATION

with VOCABULARY_PATH.open("r", encoding="utf-8") as f:
    vocabulary = json.load(f)

vocab_size = len(vocabulary)

if CHECKPOINT_PATH.exists():

    print("Loading trained model...")

    checkpoint = np.load(CHECKPOINT_PATH)

    embedding_table = checkpoint["embedding_table"].copy()
    positional_embedding = checkpoint["positional_embedding"].copy()

    W_Q = checkpoint["W_Q"].copy()
    W_K = checkpoint["W_K"].copy()
    W_V = checkpoint["W_V"].copy()
    W_O = checkpoint["W_O"].copy()

    W1 = checkpoint["W1"].copy()
    b1 = checkpoint["b1"].copy()
    W2 = checkpoint["W2"].copy()
    b2 = checkpoint["b2"].copy()

    W_vocab = checkpoint["W_vocab"].copy()

    gamma = checkpoint["gamma"].copy()
    beta = checkpoint["beta"].copy()

    print("Model loaded successfully!")

else:

    print("No checkpoint found. Initializing a new model...")

    # LayerNorm parameters (temporarily fixed)
    gamma = np.ones(EMBEDDING_DIM)
    beta = np.zeros(EMBEDDING_DIM)

    # FFN
    W1 = np.random.randn(EMBEDDING_DIM, FF_DIM) * 0.02
    b1 = np.zeros(FF_DIM)

    W2 = np.random.randn(FF_DIM, EMBEDDING_DIM) * 0.02
    b2 = np.zeros(EMBEDDING_DIM)

    # Self-Attention
    W_Q = np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * 0.02
    W_K = np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * 0.02
    W_V = np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * 0.02
    W_O = np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * 0.02

    # Vocabulary Head
    W_vocab = np.random.randn(EMBEDDING_DIM, vocab_size) * 0.02

print("Embedding shape:", embedding_table.shape)
print("W_Q shape:", W_Q.shape)
print("W1 shape:", W1.shape)
print("W_vocab shape:", W_vocab.shape)

print("All finite:", all([
    np.all(np.isfinite(embedding_table)),
    np.all(np.isfinite(positional_embedding)),
    np.all(np.isfinite(W_Q)),
    np.all(np.isfinite(W_K)),
    np.all(np.isfinite(W_V)),
    np.all(np.isfinite(W_O)),
    np.all(np.isfinite(W1)),
    np.all(np.isfinite(W2)),
    np.all(np.isfinite(W_vocab))
]))

# =========== TRAINING

loss_history = []

for epoch in range(NUM_EPOCHS):

    epoch_loss = []

    for sequence_index, (X_ids, target_ids) in enumerate(training_sequences):

        # 1. --------- EMBEDDING

        position_ids = np.arange(len(X_ids))

        token_embeddings = embedding_table[X_ids]
        position_embeddings = positional_embedding[position_ids]

        COMBINED = token_embeddings + position_embeddings

        X = COMBINED

        T = X.shape[0]

        # 2. --------- FORWARD PASS

        # LayerNorm 1 + Self-Attention
        norm1, X_norm, variance1 = layer_norm(X, gamma, beta)

        attention_output, attention_cache = self_attention(
            X_norm, W_Q, W_K, W_V, W_O
        )

        # Residual 1
        Z = X + attention_output

        # LayerNorm 2 + FFN
        norm2, Z_norm, variance2 = layer_norm(Z, gamma, beta)

        ffn_result = ffn(Z_norm, W1, W2, b1, b2)

        H = ffn_result[0]
        A = ffn_result[1]
        Y_FFN = ffn_result[2]

        # Residual 2
        output = Z + Y_FFN

        # Vocabulary Head
        logits_output = logits(output, W_vocab)
        probs_matrix = probs(logits_output)

        # 3. --------- LOSS

        loss_output = loss(probs_matrix, target_ids)

        epoch_loss.append(loss_output)
        loss_history.append(loss_output)

        # 4. --------- BACKPROPAGATION

        # Cross-entropy + Softmax gradient
        dLogits = probs_matrix.copy()

        dLogits[np.arange(T), target_ids] -= 1

        # Normalize ONCE
        dLogits /= T

        # Vocabulary Head backward
        dW_vocab = output.T @ dLogits

        dOutput = dLogits @ W_vocab.T

        # Residual 2 backward
        dZ_from_residual = dOutput
        dFFN_output = dOutput

        # FFN backward
        dW2 = A.T @ dFFN_output
        db2 = np.sum(dFFN_output, axis=0)

        dA = dFFN_output @ W2.T

        # ReLU backward
        dH = dA * (H > 0)

        # First linear layer backward
        dW1 = Z_norm.T @ dH
        db1 = np.sum(dH, axis=0)

        dZ_norm = dH @ W1.T

        # LayerNorm 2 backward
        dZ_from_layer_norm2 = layer_norm_backward(
            dZ_norm,
            norm2,
            variance2,
            gamma
        )

        # Combine residual gradients
        dZ_total = dZ_from_residual + dZ_from_layer_norm2

        # Self-Attention backward
        dW_Q, dW_K, dW_V, dW_O, dX_norm = self_attention_backward(
            dZ_total,
            attention_cache,
            W_Q, W_K, W_V, W_O
        )

        # LayerNorm 1 backward
        dX_from_layer_norm1 = layer_norm_backward(
            dX_norm,
            norm1,
            variance1,
            gamma
        )

        # Combine residual gradients
        dX_total = dZ_total + dX_from_layer_norm1

        # 5. --------- EMBEDDING GRADIENTS

        dEmbedding_table = np.zeros_like(embedding_table)

        np.add.at(
            dEmbedding_table,
            X_ids,
            dX_total
        )

        dPositional_embedding = np.zeros_like(positional_embedding)

        np.add.at(
            dPositional_embedding,
            position_ids,
            dX_total
        )

        # 6. --------- UPDATE PARAMETERS

        # FFN
        W1 -= LEARNING_RATE * dW1
        b1 -= LEARNING_RATE * db1

        W2 -= LEARNING_RATE * dW2
        b2 -= LEARNING_RATE * db2

        # Self-Attention
        W_Q -= LEARNING_RATE * dW_Q
        W_K -= LEARNING_RATE * dW_K
        W_V -= LEARNING_RATE * dW_V
        W_O -= LEARNING_RATE * dW_O

        # Vocabulary Head
        W_vocab -= LEARNING_RATE * dW_vocab

        # Embeddings
        embedding_table -= LEARNING_RATE * dEmbedding_table

        positional_embedding -= LEARNING_RATE * dPositional_embedding

    # END OF ONE EPOCH

    average_loss = np.mean(epoch_loss)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} "
        f"| Average Loss: {average_loss}"
    )

print("Training completed!")

# # SAVE MODEL CHECKPOINT

# CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

# np.savez_compressed(
#     CHECKPOINT_PATH,

#     # Embeddings
#     embedding_table=embedding_table,
#     positional_embedding=positional_embedding,

#     # Self-attention
#     W_Q=W_Q,
#     W_K=W_K,
#     W_V=W_V,
#     W_O=W_O,

#     # FFN
#     W1=W1,
#     b1=b1,
#     W2=W2,
#     b2=b2,

#     # Vocabulary head
#     W_vocab=W_vocab,

#     # Layer normalization
#     gamma=gamma,
#     beta=beta,

#     # Model configuration
#     vocab_size=vocab_size,
#     embedding_dim=EMBEDDING_DIM,
#     ff_dim=FF_DIM,
#     max_sequence_length=MAX_SEQUENCE_LENGTH
# )

# print(f"Model saved to: {CHECKPOINT_PATH}")
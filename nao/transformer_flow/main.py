import numpy as np
from pathlib import Path
import sys

# DIRECTORY SETTING
nao_dir = Path(__file__).resolve().parents[1]
if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from config import CHECKPOINT_PATH, TOP_K
from transformer_flow.inference.predict import predict_next_token
from bpe.utils.tokenize_utils import id_to_token, tokenize_text, decode_token_ids

# 1. LOAD MODEL
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

# 2. INFERENCE
input_text = "Aku belajar"

predicted_id, next_token_probs = predict_next_token(
    input_text,
    embedding_table,
    positional_embedding,
    W_Q, W_K, W_V, W_O,
    W1, b1, W2, b2,
    W_vocab,
    gamma, beta
)

top_indices = np.argsort(next_token_probs)[-TOP_K:][::-1]

for token_id in top_indices:
    print(
        id_to_token[token_id],
        next_token_probs[token_id]
    )


# Decode input
IDS = tokenize_text(input_text)

input_ids = [
    token_id
    for group in IDS
    for token_id in group
]

decoded_input = decode_token_ids(input_ids)

# 3. DISPLAY
generated_ids = input_ids + [predicted_id]
decoded_output = decode_token_ids(generated_ids)

print("\n--- INFERENCE RESULT ---")
print("Original input:", input_text)
print("Decoded input:", decoded_input)
print("Predicted token ID:", predicted_id)
print("Predicted token bytes:", id_to_token[predicted_id])
print("Predicted token text:", decoded_output[len(decoded_input):])
print("Generated text:", decoded_output)

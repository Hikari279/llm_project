from hikari.embeddings.utils.bpe_functions import tokenize_bytes, encode, build_vocab, merge_rules
import numpy as np

#note: these vectors are still random. self-attention will fill values later.

text = "The cat sat."

tokens = tokenize_bytes(text, merge_rules)
byte_vocab = build_vocab(merge_rules)
token_ids = encode(byte_vocab, tokens)

print("Tokens:", tokens)
print("Token IDs:", token_ids)

#task2
vocab_size = len(byte_vocab)
embedding_dim = 8

embedding_matrix = np.random.randn(vocab_size, embedding_dim)
token_embeddings = embedding_matrix[token_ids]

print("Vocabulary size:", vocab_size)
print("Number of token IDs:", len(token_ids))
#print("Embeddings:", token_embeddings)
print("Embedding shape:", token_embeddings.shape)

#task3
max_positions = 128

positional_matrix = np.random.randn(max_positions, embedding_dim)

position_ids = np.arange(len(token_ids))
position_embeddings = positional_matrix[position_ids]

combined_embeddings = token_embeddings + position_embeddings

#print("Combined embeddings:", combined_embeddings)
print("Combined shape:", combined_embeddings.shape)
print("Tokenizer + Embedding Connected")

#task4 example of how GPT learns to predict continuations
input_ids = token_ids[:-1]
target_ids = token_ids[1:]

print("Input:", input_ids)
print("Target:", target_ids)
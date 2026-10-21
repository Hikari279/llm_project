import numpy as np

vocab_size = 5
embedding_dim = 3

#using random values
embedding_matrix = np.random.randn(vocab_size, embedding_dim)

print(embedding_matrix)
print(embedding_matrix.shape)

#task2
"""token_id = 2

embedding_vector = embedding_matrix[token_id]

print(embedding_vector)
print(embedding_vector.shape)"""

#task3
token_ids = [2, 0, 3]
token_embeddings = embedding_matrix[token_ids]

print(token_embeddings.shape)
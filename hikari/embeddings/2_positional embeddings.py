import numpy as np

"""max_positions = 4
embedding_dim = 3

positional_matrix = np.random.randn(max_positions, embedding_dim)

#print(positional_matrix)
#print(positional_matrix.shape)"""

#task2
"""token_ids = [2, 0, 3]
position_ids = [0, 1, 2]

position_embeddings = positional_matrix[position_ids]

print(position_embeddings.shape)"""

#task3
vocab_size = 5
embedding_dim = 3

embedding_matrix = np.random.randn(vocab_size, embedding_dim)

max_positions = 4
embedding_dim = 3

positional_matrix = np.random.randn(max_positions, embedding_dim)

token_ids = [2, 0, 3]
position_ids = [0, 1, 2]

token_embeddings = embedding_matrix[token_ids]
position_embeddings = positional_matrix[position_ids]

combined_embeddings = token_embeddings + position_embeddings

print(combined_embeddings)
print(combined_embeddings.shape)
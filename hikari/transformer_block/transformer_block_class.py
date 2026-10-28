import numpy as np

class TransformerBlock:
    def __init__(self, dim, hidden_dim):
        self.dim = dim
        self.hidden_dim = hidden_dim

        # *0.1 is fine for this learning exercise
        # production models typically use initialization schemes
        self.W_query = np.random.randn(dim, dim) * 0.1
        self.W_key = np.random.randn(dim, dim) * 0.1
        self.W_value = np.random.randn(dim, dim) * 0.1
        self.W_output = np.random.randn(dim, dim) * 0.1

        #ffn weights
        self.W1 = np.random.randn(dim, hidden_dim) * 0.1
        self.W2 = np.random.randn(hidden_dim, dim) * 0.1

        #layerNorm parameters
        self.gamma_attn = np.ones(dim)
        self.beta_attn = np.zeros(dim)
        self.gamma_ffn = np.ones(dim)
        self.beta_ffn = np.zeros(dim)

    def forward(self, matrix):
        normalized = self.layer_norm(matrix, self.gamma_attn, self.beta_attn)

        q = normalized @ self.W_query
        k = normalized @ self.W_key
        v = normalized @ self.W_value

        scores = q @ k.T / np.sqrt(self.dim)

        num_rows = matrix.shape[0]
        mask = np.triu(
            np.ones((num_rows, num_rows), dtype=bool),
            k=1
        )

        masked_scores = np.where(mask, -np.inf, scores)
        attention_weights = self.softmax(masked_scores)

        attention_output = attention_weights @ v
        projected_output = attention_output @ self.W_output

        attention_result = matrix + projected_output

        normalized_attention = self.layer_norm(attention_result, self.gamma_ffn, self.beta_ffn)

        hidden = normalized_attention @ self.W1
        hidden = np.maximum(0, hidden)
        ffn_output = hidden @ self.W2
    
        return attention_result + ffn_output



    def layer_norm(self, matrix, gamma, beta):
        epsilon = 1e-5

        means = np.mean(matrix, axis=1, keepdims=True)
        variances = np.var(matrix, axis=1, keepdims=True)

        normalized = (matrix - means) / np.sqrt(variances + epsilon)

        return gamma * normalized + beta

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

"""
combined_embeddings = [[0.76453109, -0.31776604,  0.50575023, -0.16434935,  1.16736472,  2.32659958,
   1.01112249, -0.32902159],
 [-0.19122348, -1.71660255,  1.01838422,  0.79143633,  0.65987913,  0.75964023,
  -1.59593029, -1.53153433],
 [-0.33700207, -0.53980668, -0.21870592,  1.42616782, -0.57219806, -0.39237729,
   0.63095138,  1.48562557],
 [-0.54671977,  0.06807552,  0.71290102, -0.51269744,  0.17983165,  1.43316483,
  -1.08239581, -1.47597607],
 [ 1.72177448,  0.74860446,  1.53200553, -1.07055617, -1.25983681,  0.59983728,
  -1.72899821,  0.94476568],
 [-1.05887463, -2.71997195, -0.35670241,  1.60048264, -0.30326748, -0.39370202,
   1.31769313,  1.77654356]]
flattened = [item for sublist in combined_embeddings for item in sublist] 
combined_embeddings = np.array(flattened).reshape(6, 8)

block1 = TransformerBlock(dim=8, hidden_dim=32)

out1 = block1.forward(combined_embeddings)
out2 = block1.forward(combined_embeddings)

print("Output shape:", out1.shape)
print("Same output:", np.allclose(out1, out2))

weights_before = block1.W_query.copy()

block1.forward(combined_embeddings)

print(np.allclose(weights_before, block1.W_query))
"""
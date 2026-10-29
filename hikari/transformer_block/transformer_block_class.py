import numpy as np
from utils.functions import layer_norm_backward, softmax_backward

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
        # ----- Attention sublayer -----
        normalized = self.layer_norm(
            matrix, self.gamma_attn, self.beta_attn
        )

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

        # ----- Feed-forward sublayer -----
        normalized_attention = self.layer_norm(
            attention_result, self.gamma_ffn, self.beta_ffn
        )

        pre_activation = normalized_attention @ self.W1
        hidden = np.maximum(0, pre_activation)
        ffn_output = hidden @ self.W2

        output = attention_result + ffn_output

        # ----- Save values for backward() -----
        self.cache = {
            "matrix": matrix,
            "normalized": normalized,
            "q": q,
            "k": k,
            "v": v,
            "mask": mask,
            "attention_weights": attention_weights,
            "attention_output": attention_output,
            "attention_result": attention_result,
            "normalized_attention": normalized_attention,
            "pre_activation": pre_activation,
            "hidden": hidden,
        }

        return output

    
    def backward(self, grad_output):
        # Retrieve values saved by forward()
        cache = self.cache

        matrix = cache["matrix"]
        normalized = cache["normalized"]
        q = cache["q"]
        k = cache["k"]
        v = cache["v"]
        mask = cache["mask"]
        attention_weights = cache["attention_weights"]
        attention_output = cache["attention_output"]
        attention_result = cache["attention_result"]
        normalized_attention = cache["normalized_attention"]
        pre_activation = cache["pre_activation"]
        hidden = cache["hidden"]

        # ----- Feed-forward network -----

        # output = attention_result + ffn_output
        grad_attention_result = grad_output.copy()
        grad_ffn_output = grad_output.copy()

        # ffn_output = hidden @ W2
        self.grad_W2 = hidden.T @ grad_ffn_output
        grad_hidden = grad_ffn_output @ self.W2.T

        # hidden = ReLU(pre_activation)
        grad_pre_activation = grad_hidden * (pre_activation > 0)

        # pre_activation = normalized_attention @ W1
        self.grad_W1 = normalized_attention.T @ grad_pre_activation
        grad_normalized_attention = grad_pre_activation @ self.W1.T

        
        # ----- LayerNorm backward -----
        grad_attention_result_from_ffn, self.grad_gamma_ffn, self.grad_beta_ffn = (
            layer_norm_backward(
                grad_normalized_attention,
                attention_result,
                self.gamma_ffn
            )
        )

        # Combine both paths through the residual connection
        grad_attention_result += grad_attention_result_from_ffn


        # ----- Attention output projection -----

        # projected_output = attention_output @ W_output
        self.grad_W_output = attention_output.T @ grad_attention_result

        grad_attention_output = grad_attention_result @ self.W_output.T

        # attention_output = attention_weights @ v
        grad_attention_weights = grad_attention_output @ v.T
        grad_v = attention_weights.T @ grad_attention_output

        # ----- Attention softmax -----
        grad_scores = softmax_backward(
            grad_attention_weights,
            attention_weights
        )

        # Masked future positions must receive zero gradient
        grad_scores = np.where(mask, 0.0, grad_scores)

        # scores = q @ k.T / sqrt(dim)
        scale = np.sqrt(self.dim)

        grad_q = (grad_scores @ k) / scale
        grad_k = (grad_scores.T @ q) / scale

            # ----- Query, key, and value projections -----

        # q = normalized @ W_query
        self.grad_W_query = normalized.T @ grad_q

        # k = normalized @ W_key
        self.grad_W_key = normalized.T @ grad_k

        # v = normalized @ W_value
        self.grad_W_value = normalized.T @ grad_v

        # Combine gradients flowing back into normalized
        grad_normalized = (
            grad_q @ self.W_query.T
            + grad_k @ self.W_key.T
            + grad_v @ self.W_value.T
        )

        # ----- Attention LayerNorm and residual -----
        grad_matrix_from_attention, self.grad_gamma_attn, self.grad_beta_attn = (
            layer_norm_backward(
                grad_normalized,
                matrix,
                self.gamma_attn
            )
        )

        # Attention residual: matrix + projected_output
        grad_matrix = grad_attention_result + grad_matrix_from_attention

        return grad_matrix


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
"""


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
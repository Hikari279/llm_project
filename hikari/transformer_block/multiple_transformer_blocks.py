from transformer_block_class import TransformerBlock
from utils.bpe_functions import build_vocab, merge_rules, tokenize_bytes

import numpy as np

#text = "The cat sat."
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
block2 = TransformerBlock(dim=8, hidden_dim=32)

gamma_final = np.ones(8)
beta_final = np.zeros(8)

output1 = block1.forward(combined_embeddings)
output2 = block2.forward(output1)

gpt_output = block1.layer_norm(output2, gamma_final, beta_final)

#logits / language model head
byte_map = build_vocab(merge_rules)
vocab_size = len(byte_map)
inverted_vocab = {token_id: token for token, token_id in byte_map.items()}

W_vocab = np.random.randn(8, vocab_size) * 0.1
logits = gpt_output @ W_vocab

# Calculate probabilities for every position
probs = block1.softmax(logits)

# NEW: Calculate cross-entropy loss
targets = np.array([10, 25, 31, 8, 15, 20])  # example targets
correct_probs = probs[np.arange(len(targets)), targets]
loss = -np.mean(np.log(correct_probs))

print("Loss:", loss)

# Next-token prediction (inference)

next_token_logits = logits[-1]
next_token_probs = block1.softmax(next_token_logits)

predicted_id = int(np.argmax(next_token_probs))
predicted_token = inverted_vocab[predicted_id]


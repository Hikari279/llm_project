from utils.functions import self_attention, feed_forward
import numpy as np

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

def transformer_block(matrix):
    attention_result = self_attention(matrix)
    block_output = feed_forward(attention_result)
    return block_output

block_output = transformer_block(combined_embeddings)

print("Input shape:", combined_embeddings.shape)
print("Output shape:", block_output.shape)
print(block_output)
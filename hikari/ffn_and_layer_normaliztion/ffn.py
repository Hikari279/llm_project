from utils.self_attention_full import output

import numpy as np

ffn_input = output

w1 = np.random.randn(8, 32)

hidden = ffn_input @ w1
#print(hidden.shape)
#print(hidden)

#task2
hidden = np.maximum(0, hidden)
#print(hidden)

#task3
w2 = np.random.randn(32, 8)

ffn_output = hidden @ w2
#print(ffn_output.shape)

#task4
block_output = ffn_input = ffn_output
print(block_output)
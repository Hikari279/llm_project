import numpy as np

scores = np.array([2.0, 1.0, 0.0])
scores_exp = np.exp(scores)
sum = np.sum(scores_exp)
result = scores_exp / sum

#print(result)

def softmax(matrix):
    exp_matrix = np.exp(matrix)
    totals = np.sum(exp_matrix, axis=1, keepdims=True)
    result = exp_matrix / totals
    return result

test_matrix = np.random.randn(6, 6)

result = softmax(test_matrix)
print(result)

mask = np.tril(np.ones((6, 6), dtype=bool))
print(mask)

#better practically to handle big numbers
def improved_softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
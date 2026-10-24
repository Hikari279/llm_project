import numpy as np

x = np.array([2.0, 4.0, 6.0, 8.0])

#x_mean = np.sum(x) / len(x)
x_mean = np.mean(x)

"""
x_value_minus_mean = []
for item in x:
    result = item -  x_mean
    result = result ** 2
    x_value_minus_mean.append(result)

x_variance = np.mean(x_value_minus_mean)
"""

x_variance = np.mean((x - x_mean) ** 2)
#print(x_variance)

epsilon = 1e-5
denominator = np.sqrt(x_variance + epsilon)

x_normalized = (x- x_mean) / denominator
#print(x_normalized)


#task2
x = np.array([
    [2.0, 4.0, 6.0, 8.0],
    [1.0, 3.0, 5.0, 7.0],
    [10.0, 10.0, 10.0, 10.0]
])

epsilon = 1e-5

means = np.mean(x, axis=1, keepdims=True)
variances = np.var(x, axis=1, keepdims=True)

denominator = np.sqrt(variances + epsilon)

normalized = (x - means) / denominator

#task3 (Gamma can amplify or reduce particular features.)
#(Beta can shift particular features up or down.)
gamma = np.ones(x.shape[1])
beta = np.zeros(x.shape[1])

output = gamma * normalized + beta
print(output)
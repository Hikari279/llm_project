import numpy as np

probs = np.array([
    [0.1, 0.6, 0.2, 0.1],
    [0.7, 0.1, 0.1, 0.1],
    [0.2, 0.2, 0.5, 0.1]
])

targets = [1, 0, 2]

correct_probs = probs[np.arange(len(targets)), targets]
loss = -np.mean(np.log(correct_probs))
print(loss)
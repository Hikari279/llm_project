import numpy as np

weight = 0.5
learning_rate = 0.1

gradients = np.array([-0.8, 0.6, 0.0])

new_weights = weight - learning_rate * gradients

print(new_weights)



weight = 0.8
learning_rate = 0.2

gradients = np.array([2.0, -1.0, 0.5])

for gradient in gradients:
    new_weight = weight - learning_rate * gradient

    print(
        f"Gradient: {gradient:>4.1f} | "
        f"Old weight: {weight:.2f} | "
        f"New weight: {new_weight:.2f}"
    )

"""
loss_before, grad_W_vocab = forward_pass(input_ids, target_ids)
print("Gradient shape:", grad_W_vocab.shape)

learning_rate = 0.1
W_vocab -= learning_rate * grad_W_vocab

loss_after, _ = forward_pass(input_ids, target_ids)

print("Loss before:", loss_before)
print("Loss after:", loss_after)
"""
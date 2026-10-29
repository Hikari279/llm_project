import numpy as np
import os, json
from utils.bpe_functions import tokenize_bytes, build_vocab, encode, load_merge_rules
from transformer_block_class import TransformerBlock
from utils.functions import layer_norm_backward
from utils.training_sample import training_sentences

merge_rules_v1 = [(b'a', b't'), (b'h', b'e'), (b'he', b' '), (b'at', b' '), (b'T', b'he ')]
merge_rules = load_merge_rules()
# Optional: makes initialization reproducible
np.random.seed(42)

# =========================
# 1. SETUP — RUN ONCE
# =========================

text = "The cat sat."
tokens = tokenize_bytes(text, merge_rules)

byte_vocab = build_vocab(merge_rules)
token_ids = encode(byte_vocab, tokens)
inverted_vocab = {token_id: token for token, token_id in byte_vocab.items()}

vocab_size = len(byte_vocab)
embedding_dim = 16 #from 8 for v3
hidden_dim = 64 #from 32 for v3
max_positions = 128

input_ids = token_ids[:-1]
target_ids = np.array(token_ids[1:])

# Embedding parameters
embedding_matrix = np.random.randn(vocab_size, embedding_dim)
positional_matrix = np.random.randn(max_positions, embedding_dim)

# Transformer blocks — initialize only once
block1 = TransformerBlock(dim=embedding_dim, hidden_dim=hidden_dim)
block2 = TransformerBlock(dim=embedding_dim, hidden_dim=hidden_dim)

# Final normalization parameters
gamma_final = np.ones(embedding_dim)
beta_final = np.zeros(embedding_dim)

# Language-model head — initialize only once
W_vocab = np.random.randn(embedding_dim, vocab_size) * 0.1

# =========================
# 2. FORWARD PASS
# =========================
def forward_pass(input_ids, target_ids):
    token_embeddings = embedding_matrix[input_ids]
    position_embeddings = positional_matrix[:len(input_ids)]
    combined_embeddings = token_embeddings + position_embeddings

    output1 = block1.forward(combined_embeddings)
    output2 = block2.forward(output1)

    gpt_output = block1.layer_norm(
        output2, gamma_final, beta_final
    )

    logits = gpt_output @ W_vocab
    probs = block1.softmax(logits)

    # =========================
    # 3. LOSS
    # =========================

    N = len(target_ids)

    correct_probs = probs[np.arange(N), target_ids]
    loss = -np.mean(np.log(np.clip(correct_probs, 1e-12, 1.0)))

    # =========================
    # 4. VOCABULARY HEAD GRADIENTS
    # =========================

    grad_logits = probs.copy()
    grad_logits[np.arange(N), target_ids] -= 1
    grad_logits /= N

    grad_W_vocab = gpt_output.T @ grad_logits
    grad_gpt_output = grad_logits @ W_vocab.T

    # =========================
    # 5. FINAL LAYERNORM BACKWARD
    # =========================

    grad_output2, grad_gamma_final, grad_beta_final = (
        layer_norm_backward(
            grad_gpt_output,
            output2,
            gamma_final
        )
    )

    # =========================
    # 6. TRANSFORMER BACKWARD
    # =========================

    # block2 returns the gradient for its input (output1)
    grad_output1 = block2.backward(grad_output2)

    # block1 returns the gradient for combined_embeddings
    grad_embeddings = block1.backward(grad_output1)

    # =========================
    # 7. EMBEDDING GRADIENTS
    # =========================

    grad_token_embeddings = np.zeros_like(embedding_matrix)
    grad_positional_embeddings = np.zeros_like(positional_matrix)

    # Several positions can share the same token ID.
    np.add.at(
        grad_token_embeddings,
        input_ids,
        grad_embeddings
    )

    np.add.at(
        grad_positional_embeddings,
        np.arange(len(input_ids)),
        grad_embeddings
    )

    # =========================
    # 8. UPDATE PARAMETERS
    # =========================

    learning_rate = 0.01

    W_vocab[:] -= learning_rate * grad_W_vocab

    gamma_final[:] -= learning_rate * grad_gamma_final
    beta_final[:] -= learning_rate * grad_beta_final

    embedding_matrix[:] -= learning_rate * grad_token_embeddings
    positional_matrix[:] -= learning_rate * grad_positional_embeddings

    for block in (block1, block2):
        block.W_query[:] -= learning_rate * block.grad_W_query
        block.W_key[:] -= learning_rate * block.grad_W_key
        block.W_value[:] -= learning_rate * block.grad_W_value
        block.W_output[:] -= learning_rate * block.grad_W_output

        block.W1[:] -= learning_rate * block.grad_W1
        block.W2[:] -= learning_rate * block.grad_W2

        block.gamma_attn[:] -= learning_rate * block.grad_gamma_attn
        block.beta_attn[:] -= learning_rate * block.grad_beta_attn

        block.gamma_ffn[:] -= learning_rate * block.grad_gamma_ffn
        block.beta_ffn[:] -= learning_rate * block.grad_beta_ffn

    return loss, probs

def predict_next_token(input_ids):
    token_embeddings = embedding_matrix[input_ids]
    position_embeddings = positional_matrix[:len(input_ids)]
    combined_embeddings = token_embeddings + position_embeddings

    output1 = block1.forward(combined_embeddings)
    output2 = block2.forward(output1)

    gpt_output = block1.layer_norm(
        output2, gamma_final, beta_final
    )

    logits = gpt_output @ W_vocab
    next_token_logits = logits[-1]

    probabilities = block1.softmax(next_token_logits)

    return probabilities

def generate_text(text, num_new_tokens=10, debug=False):
    tokens = tokenize_bytes(text, merge_rules)
    prompt_ids = encode(byte_vocab, tokens)

    if not prompt_ids:
        raise ValueError("Prompt must not be empty.")

    generated_ids = list(prompt_ids)

    for step in range(num_new_tokens):
        context_ids = generated_ids[-max_positions:]

        probabilities = predict_next_token(
            np.array(context_ids, dtype=int)
        )

        next_token_id = int(np.argmax(probabilities))
        generated_ids.append(next_token_id)

        generated_bytes = b"".join(
            inverted_vocab[int(token_id)]
            for token_id in generated_ids
        )

        # Stop after a generated sentence-ending mark
        if generated_bytes.endswith((b".", b"!", b"?")):
            break

    generated_tokens = [
        inverted_vocab[int(token_id)]
        for token_id in generated_ids
    ]

    if debug:
        print("Generated tokens:", generated_tokens)
        print("Generated IDs:", generated_ids)

    return b"".join(generated_tokens).decode(
        "utf-8", errors="replace"
    )

def save_checkpoint(filename="hikari/transformer_block/utils/tiny_gpt_v3.npz"):
    np.savez(
        filename,

        # Embeddings
        embedding_matrix=embedding_matrix,
        positional_matrix=positional_matrix,

        # Final LayerNorm
        gamma_final=gamma_final,
        beta_final=beta_final,

        # Vocabulary head
        W_vocab=W_vocab,

        # Transformer block 1
        block1_W_query=block1.W_query,
        block1_W_key=block1.W_key,
        block1_W_value=block1.W_value,
        block1_W_output=block1.W_output,
        block1_W1=block1.W1,
        block1_W2=block1.W2,
        block1_gamma_attn=block1.gamma_attn,
        block1_beta_attn=block1.beta_attn,
        block1_gamma_ffn=block1.gamma_ffn,
        block1_beta_ffn=block1.beta_ffn,

        # Transformer block 2
        block2_W_query=block2.W_query,
        block2_W_key=block2.W_key,
        block2_W_value=block2.W_value,
        block2_W_output=block2.W_output,
        block2_W1=block2.W1,
        block2_W2=block2.W2,
        block2_gamma_attn=block2.gamma_attn,
        block2_beta_attn=block2.beta_attn,
        block2_gamma_ffn=block2.gamma_ffn,
        block2_beta_ffn=block2.beta_ffn,
    )

    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename="hikari/transformer_block/utils/tiny_gpt_v3.npz"):
    with np.load(filename) as checkpoint:
        embedding_matrix[:] = checkpoint["embedding_matrix"]
        positional_matrix[:] = checkpoint["positional_matrix"]

        gamma_final[:] = checkpoint["gamma_final"]
        beta_final[:] = checkpoint["beta_final"]
        W_vocab[:] = checkpoint["W_vocab"]

        for block, prefix in (
            (block1, "block1"),
            (block2, "block2"),
        ):
            for name in (
                "W_query",
                "W_key",
                "W_value",
                "W_output",
                "W1",
                "W2",
                "gamma_attn",
                "beta_attn",
                "gamma_ffn",
                "beta_ffn",
            ):
                getattr(block, name)[:] = checkpoint[f"{prefix}_{name}"]

    print(f"Checkpoint loaded from {filename}")

def evaluate(input_ids, target_ids):
    token_embeddings = embedding_matrix[input_ids]
    position_embeddings = positional_matrix[:len(input_ids)]
    combined_embeddings = token_embeddings + position_embeddings

    output1 = block1.forward(combined_embeddings)
    output2 = block2.forward(output1)

    gpt_output = block1.layer_norm(
        output2, gamma_final, beta_final
    )

    logits = gpt_output @ W_vocab
    probs = block1.softmax(logits)

    for i, target_id in enumerate(target_ids):
        predicted_id = int(np.argmax(probs[i]))
        correct_prob = probs[i, target_id]

        print(
            f"Position {i}: "
            f"target={target_id}, "
            f"predicted={predicted_id}, "
            f"correct_probability={correct_prob:.4f}"
        )

def evaluate_sentence(sentence):
    tokens = tokenize_bytes(sentence, merge_rules)
    ids = encode(byte_vocab, tokens)

    correct = 0
    total = len(ids) - 1

    for i in range(total):
        context = np.array(ids[:i + 1], dtype=int)
        probabilities = predict_next_token(context)

        target_id = ids[i + 1]
        predicted_id = int(np.argmax(probabilities))

        if predicted_id == target_id:
            correct += 1

        print(
            f"Expected: {inverted_vocab[target_id]!r} | "
            f"Predicted: {inverted_vocab[predicted_id]!r} | "
            f"Target probability: {probabilities[target_id]:.3f}"
        )

    print(f"Next-token accuracy: {correct}/{total}")

def evaluate_corpus(sentences):
    total_correct = 0
    total_tokens = 0
    total_loss = 0.0

    for sentence in sentences:
        tokens = tokenize_bytes(sentence, merge_rules)
        ids = encode(byte_vocab, tokens)

        if len(ids) < 2:
            continue

        input_ids = np.array(ids[:-1], dtype=int)
        targets = np.array(ids[1:], dtype=int)

        # Forward pass only: no training updates
        embeddings = (
            embedding_matrix[input_ids]
            + positional_matrix[:len(input_ids)]
        )

        output1 = block1.forward(embeddings)
        output2 = block2.forward(output1)

        output = block1.layer_norm(
            output2, gamma_final, beta_final
        )

        probabilities = block1.softmax(output @ W_vocab)

        target_probs = probabilities[
            np.arange(len(targets)), targets
        ]

        predictions = np.argmax(probabilities, axis=1)

        total_correct += np.sum(predictions == targets)
        total_tokens += len(targets)

        total_loss -= np.sum(
            np.log(np.clip(target_probs, 1e-12, 1.0))
        )

    print("Token accuracy:", f"{total_correct / total_tokens:.2%}")
    print("Average token loss:", f"{total_loss / total_tokens:.4f}")

checkpoint_file = "hikari/transformer_block/utils/tiny_gpt_v3.npz"
if os.path.exists(checkpoint_file):
    load_checkpoint(checkpoint_file)
else:
    print("No checkpoint found. Starting with a fresh model.")


"""
epochs = 50

for epoch in range(epochs):
    total_loss = 0.0
    sentence_count = 0

    for sentence in training_sentences:
        # Convert the sentence into byte-level tokens
        sentence_tokens = tokenize_bytes(
            sentence, merge_rules
        )

        # Convert tokens into vocabulary IDs
        sentence_ids = encode(byte_vocab, sentence_tokens)

        # Need at least two tokens for next-token training
        if len(sentence_ids) < 2:
            continue

        # Ensure the input fits the positional embedding limit
        sentence_ids = sentence_ids[:max_positions + 1]

        # Shift by one position to create input/target pairs
        sample_input_ids = np.array(
            sentence_ids[:-1], dtype=int
        )

        sample_target_ids = np.array(
            sentence_ids[1:], dtype=int
        )

        # One training update for this sentence
        loss, probs = forward_pass(
            sample_input_ids,
            sample_target_ids
        )

        total_loss += loss
        sentence_count += 1

    average_loss = total_loss / sentence_count

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"average loss: {average_loss:.4f}"
    )

save_checkpoint()
"""

"""
epochs = 100

for epoch in range(epochs):
    total_loss = 0.0
    total_tokens = 0

    # Shuffle sentence order for this epoch
    shuffled_indices = np.random.permutation(
        len(training_sentences)
    )

    for index in shuffled_indices:
        sentence = training_sentences[index]

        sentence_tokens = tokenize_bytes(
            sentence, merge_rules
        )

        sentence_ids = encode(byte_vocab, sentence_tokens)

        if len(sentence_ids) < 2:
            continue

        sentence_ids = sentence_ids[:max_positions + 1]

        sample_input_ids = np.array(
            sentence_ids[:-1], dtype=int
        )

        sample_target_ids = np.array(
            sentence_ids[1:], dtype=int
        )

        loss, _ = forward_pass(
            sample_input_ids, sample_target_ids
        )

        # Weight each sentence's mean loss by its token count
        token_count = len(sample_target_ids)
        total_loss += loss * token_count
        total_tokens += token_count

    average_loss = total_loss / total_tokens

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"average token loss: {average_loss:.4f}"
    )

save_checkpoint()
"""

test_sentences = [
    "The rabbit sleeps under a tree.",
    "The farmer grows fresh vegetables.",
    "The bird flies over the river.",
    "The cat drinks cold water.",
    "The sun shines above the hills."
]

test_prompts = [
    "The rabbit",
    "The farmer",
    "The bird",
    "The cat",
    "The sun"
]

for prompt in test_prompts:
    result = generate_text(prompt, num_new_tokens=10)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")
    print("-" * 40)

print("Training")
evaluate_corpus(training_sentences)
print("Test")
evaluate_corpus(test_sentences)

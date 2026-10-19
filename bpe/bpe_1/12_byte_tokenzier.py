def merge(sequence, selected_pair):
    new_list = []

    i = 0
    while i < len(sequence):

        if i + 1 < len(sequence):
            pair = sequence[i], sequence[i + 1]
            if pair == selected_pair:
                new_list.append(sequence[i] + sequence[i+1])
                i += 2
                continue
            
        new_list.append(sequence[i])
        i += 1

    return new_list

def tokenize_bytes(text, merge_history):
    tokens = [bytes([b]) for b in text.encode("utf-8")]

    for pair in merge_history:
        tokens = merge(tokens, pair)
        
    return tokens



merge_history = []

text = "café 😊"
tokens = tokenize_bytes(text, [])
print(tokens)
print(b"".join(tokens).decode("utf-8") == text)
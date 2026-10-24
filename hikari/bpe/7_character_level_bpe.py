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

def tokenize(text, merge_history):
    tokens = [char for char in text]

    for pair in merge_history:
        tokens = merge(tokens, pair)
        
    return tokens

merge_history = [
    ("l", "o"),
    ("lo", "w"),
    ("e", "r"),
]

print(tokenize("low", merge_history))
print(tokenize("lower", merge_history))
print(tokenize("lowest", merge_history))
print(tokenize("hello", merge_history))
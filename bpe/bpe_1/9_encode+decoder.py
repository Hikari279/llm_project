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

def build_vocab(corpus, merge_history):
    token2id_map = {}
    token_id = 0

    ##comment this block out and run to notice difference
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    extra_chars =list(" .,!?'-")                      #NEW for special characters
    special_chars = ["\n", "\t", "\r"]
    merged_list = alphabet + extra_chars + special_chars
    for char in merged_list:
        if char in token2id_map:
            continue
        else:
            token2id_map[char] = token_id
            token_id += 1
    ##comment ends here

    for sublist in corpus:
        for char in sublist:
            if char in token2id_map:
                continue
            else:
                token2id_map[char] = token_id
                token_id += 1

    for pair in merge_history:
        token = pair[0] + pair[1]
        if token in token2id_map:
            continue
        else:
            token2id_map[token] = token_id
            token_id += 1

    return token2id_map

def encode(map, tokens):
    encoded_text = []

    for token in tokens:
        encoded_text.append(map[token])

    return encoded_text

def decode(inverted_map, token_ids):
    decoded_list = []

    for id in token_ids:
        token = inverted_map[id]
        decoded_list.append(token)

    
    decoded_text = "".join(decoded_list)
    return decoded_text

merge_history = [
    ("l", "o"),
    ("lo", "w"),
    ("e", "r"),
]

corpus = [
    ["l", "o", "w"],
    ["l", "o", "w"],
    ["l", "o", "w"],
    ["l", "o", "w", "e", "r"],
]

text = "lower!"

token2id_map = build_vocab(corpus, merge_history)

tokens = tokenize(text, merge_history)
token_ids = encode(token2id_map, tokens)

inverted_map = {value: key for key, value in token2id_map.items()}

decoded_text = decode(inverted_map, token_ids)

print("Tokens:", tokens)
print("IDs:", token_ids)
print("Decoded:", decoded_text)
print("Round trip:", decoded_text == text)
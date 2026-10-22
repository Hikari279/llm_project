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

def tokenize_bytes(text, merge_rules):
    tokens = [bytes([b]) for b in text.encode("utf-8")]

    for pair in merge_rules:
        tokens = merge(tokens, pair)
        
    return tokens

def build_vocab(merge_rules):
    byte_vocab = {}
    token_id = 256

    for i in range(256):
        token = bytes([i])
        byte_vocab[token] = i

    for pair in merge_rules:
            token = pair[0] + pair[1]
            if token in byte_vocab:
                continue
            else:
                byte_vocab[token] = token_id
                token_id += 1

    return byte_vocab

def encode(map, tokens):
    encoded_text = []

    for token in tokens:
        encoded_text.append(map[token])

    return encoded_text

#decoder changed slightly
def decode(inverted_map, token_ids):
    decoded_list = []

    for id in token_ids:
        token = inverted_map[id]
        decoded_list.append(token)

    
    decoded_text = b"".join(decoded_list)
    return decoded_text

merge_rules = [
    (b'l', b'o'),
    (b'lo', b'w'),
    (b'lo', b'\xc3'),
    (b'lo\xc3', b'\xa9'),
    (b'low', b'e')
]

text = "The cat is sleeping. 😊"

byte_map = build_vocab(merge_rules)

tokens = tokenize_bytes(text, merge_rules)
token_ids = encode(byte_map, tokens)

inverted_map = {value: key for key, value in byte_map.items()}

decoded_byte = decode(inverted_map, token_ids)
decoded_text = decoded_byte.decode("utf-8")

print("Tokens:", tokens)
print("IDs:", token_ids)
print("Decoded Bytes:", decoded_byte)
print("Decoded:", decoded_text)
print("Round trip:", decoded_text == text)
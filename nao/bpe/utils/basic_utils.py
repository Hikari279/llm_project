from collections import Counter


# ================================ MAIN CODE =====================================

# BYTE LEVEL HANDLER
def text_to_byte(text):
    encoded = list(text.encode("utf-8"))
    return encoded

def token_to_byte_token(byte): # single byte
    token = bytes([byte])
    return token

def text_to_byte_token(byte):
    token = bytes(byte)
    return token


def byte_tokens_to_text(byte_tokens):
    text = b''.join(byte_tokens).decode("utf-8")
    return text



# BASIC TRAINING BPE
def get_token(word):
    token = list(word)
    return token

def get_pair(tokens):
    pairs = list(zip(tokens, tokens[1:]))
    return pairs

def get_pairs_count(pairs):
    pairs_count = Counter(pairs)
    return pairs_count

def get_best_pair(pairs_count):
    best_pair = pairs_count.most_common(1)[0]
    pair = best_pair[0]

    return pair


def merge_pair(tokens, pair):
    new_tokens = []
    i = 0

    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1


    return new_tokens


# HELPER
# Fungsi untuk mengubah struktur list/nested list menjadi tuple agar bisa di-hash
def ke_tuple(item):
    if isinstance(item, list):
        return tuple(ke_tuple(x) for x in item)
    return item

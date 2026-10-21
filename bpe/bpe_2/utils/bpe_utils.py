from collections import Counter
import json
import re



# BYTE LEVEL HANDLER
def text_to_byte(text):
    encoded = list(text.encode("utf-8"))
    return encoded

def text_to_byte_token(token):
    token = bytes([token])
    return token
    
# BASIC TRAINING BPE
def get_token(word):
    token = list(word)
    return token

def get_pair(tokens):
    pairs = list(zip(tokens, tokens[1:])) # pairs the tokens[0] with tokens[1], tokens[1] with tokens[2], ...
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


# TOKENIZE BPE
with open("data/vocabulary.json", "r", encoding="utf-8") as f:
    vocabulary = json.load(f)
with open("data/merge_rules.json", "r", encoding="utf-8") as f:
    merge_rules = json.load(f)

# search id based on token
token_to_id = {
    token: idx
    for idx, token in enumerate(vocabulary)
}

# search token based on id
id_to_token = {
    idx: token
    for idx, token in enumerate(vocabulary)
}

def tokenize_new_word(word, merge_rules):
    tokens = get_token(word)

    for pair in merge_rules:
        tokens = merge_pair(tokens, tuple(pair))

    return tokens

def encode(token):
    return token_to_id[token]

def decode(ids):
    return id_to_token[ids]

def token_to_word(decoded):
    return "".join(decoded)



# WHITESPACE, PUNCTUATION, AND LONGER TEXT HANDLER
import re

def tokenize_text(text, merge_rules):

    pattern = r"\s+|[^\w\s]+|\w+"
    parts = re.findall(pattern, text)

    IDS = []
    TOKENS = []
    

    for part in parts:
        if re.fullmatch(r"\w+", part):
            tokens = tokenize_new_word(part, merge_rules)
            for token in tokens:
                TOKENS.append(token)
                id = encode(token)
                IDS.append(id)

        else:
            TOKENS.append(part)
            token_id = encode(part)
            IDS.append(token_id)

    return TOKENS, IDS

def token_to_text(IDS):
    DECODED = [decode(idx) for idx in IDS]
    text = token_to_word(DECODED)
    return DECODED, text
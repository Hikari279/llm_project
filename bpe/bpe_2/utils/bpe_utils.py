from collections import Counter


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
def tokenize_new_word(word, merge_rules):
    tokens = get_token(word)

    for pair in merge_rules:
        tokens = merge_pair(tokens, tuple(pair))

    return tokens
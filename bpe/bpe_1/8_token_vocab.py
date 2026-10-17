def build_vocab(corpus, merge_history):
    token2id_map = {}
    token_id = 0

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


corpus = [
    ["l", "o", "w"],
    ["l", "o", "w"],
    ["l", "o", "w"],
    ["l", "o", "w", "e", "r"],
]


merge_history = [
    ("l", "o"),
    ("lo", "w"),
    ("e", "r")
]

result = build_vocab(corpus, merge_history)
print(result)
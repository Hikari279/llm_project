from collections import Counter

text = "low lower lowest low lower newer newest newer"

def get_tokens(text):
    tokens = list(text)
    return tokens

def get_pair(tokens):
    pairs = list(zip(tokens, tokens[1:])) # pairs the tokens[0] with tokens[1], tokens[1] with tokens[2], ...
    pairs_counts = Counter(pairs) # count of pairs

    if not pairs_counts:
        return None
    
    best_pair = pairs_counts.most_common(1)[0]
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

merges = 10
tokens = get_tokens(text)
merge_rules = []

for i in range(merges):
    pair = get_pair(tokens)

    if not pair:
        break

    merge_rules.append(pair)
    tokens = merge_pair(tokens, pair)

    print(f"Token {i + 1} : {tokens}")

print(merge_rules)
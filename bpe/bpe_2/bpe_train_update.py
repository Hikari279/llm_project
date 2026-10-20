import re
from collections import Counter
from utils.bpe_utils import get_token, get_best_pair, get_pair, get_pairs_count, merge_pair


text = "lower newer, banana!"

WORDS = []
num_merges = 5
TOKENS = []
ALL_TOKENS = []
ALL_BEST_PAIR = []

parts = re.findall(r"\s+|[^\w\s]+|\w+", text)

for part in parts:
    if re.fullmatch(r"\w+", part):
        WORDS.append(part)
    else:
        ALL_TOKENS.extend(part)

for word in WORDS:
    token = get_token(word)
    TOKENS.append(token)
    ALL_TOKENS.extend(token)

for i in range(num_merges):
    PAIRS_COUNT = Counter()

    
    # PAIRS
    for token in TOKENS:
        print(token)
        pair = get_pair(token)
        PAIRS_COUNT.update(pair)

    if not PAIRS_COUNT:
        break

    # BEST PAIR
    best_pair = get_best_pair(PAIRS_COUNT)
    ALL_BEST_PAIR.append(best_pair)

    # MERGES
    curr_tokens = []
    history_tokens = []
    for token in TOKENS:
        token = merge_pair(token, best_pair)
        curr_tokens.append(token)
        history_tokens.extend(token)


    TOKENS.clear()
    TOKENS.extend(curr_tokens)
    ALL_TOKENS.extend(history_tokens)

#     print("="*30)
#     print(f"Cycle : {i + 1}")
#     print(f"Current Best Pair : {best_pair}")
#     print(f"Merges : {TOKENS}")


# VOCABULARY = list(dict.fromkeys(ALL_TOKENS))

# print(" ")
# print("="*30)
# print(f"All History Tokens : {VOCABULARY}")
# print(f"All Best Pair : {ALL_BEST_PAIR}")
# print("="*30)

# import json

# with open("data/merge_rules.json", "w", encoding="utf-8") as f:
#     json.dump(ALL_BEST_PAIR, f)
# with open("data/vocabulary.json", "w", encoding="utf-8") as f:
#     json.dump(VOCABULARY, f)
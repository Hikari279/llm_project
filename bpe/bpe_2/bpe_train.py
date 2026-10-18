from collections import Counter
from utils.bpe_utils import get_token, get_best_pair, get_pair, get_pairs_count, merge_pair

text = """low lower lowest
new newer newest
wide wider widest
play player playing played
play player playing played
work worker working worked
work worker working worked
happy happier happiest
happy happier happiest
slow slower slowest
slow slower slowest
token tokens tokenization
token tokens tokenization
learn learner learning learned
learn learner learning learned
computer computing computed
computer computing computed
language languages linguistic
language languages linguistic
banana bandana banana bandana"""

words = text.split()


num_merges = 50

TOKENS = []
ALL_BEST_PAIR = []

for word in words:
    token = get_token(word)
    TOKENS.append(token)

for i in range(num_merges):
    PAIRS_COUNT = Counter()

    # PAIRS
    for token in TOKENS:
        pair = get_pair(token)
        PAIRS_COUNT.update(pair)

    if not PAIRS_COUNT:
        break

    # BEST PAIR
    best_pair = get_best_pair(PAIRS_COUNT)
    ALL_BEST_PAIR.append(best_pair)
    # print(best_pair)

    # MERGES
    curr_tokens = []
    for token in TOKENS:
        token = merge_pair(token, best_pair)
        curr_tokens.append(token)

    TOKENS.clear()
    TOKENS.extend(curr_tokens)

    print("="*30)
    print(f"Cycle : {i + 1}")
    print(f"Current Best Pair : {best_pair}")
    print(f"Merges : {TOKENS}")

# print(f"All Best Pair : {ALL_BEST_PAIR}")


import json

with open("data/merge_rules.json", "w", encoding="utf-8") as f:
    json.dump(ALL_BEST_PAIR, f)

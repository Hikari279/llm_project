import re
from collections import Counter
from utils.bpe_utils import get_token, get_best_pair, get_pair, get_pairs_count, merge_pair, text_to_byte, text_to_byte_token

# Fungsi untuk mengubah struktur list/nested list menjadi tuple agar bisa di-hash
def ke_tuple(item):
    if isinstance(item, list):
        return tuple(ke_tuple(x) for x in item)
    return item



text = "lower newer, bananá!"

parts = re.findall(r"\s+|[^\w\s]+|\w+", text)

# DICT
ALL_TOKENS = [] # all token from text --> for vocabulary
ALL_BYTES = []
ALL_BEST_PAIR = [] # --> for merge rules

WORDS = [] # only word from text -> temporary
TOKENS = [] # --> word, temporary
BYTES = []

num_merges = 3

for part in parts:
    if re.fullmatch(r"\w+", part):
        WORDS.append(part)
    else:
        for p in part:
            part = list(text_to_byte(p))
            # print(part)
            ALL_TOKENS.append(part)

for word in WORDS:
    byte_tokens = [
        text_to_byte_token(b)
        for b in text_to_byte(word)
    ]
    TOKENS.append(byte_tokens)
    ALL_TOKENS.extend(list(token) for token in byte_tokens)

for i in range(num_merges):
    PAIRS_COUNT = Counter()

    # PAIRS
    for tokens in TOKENS:

        pair = get_pair(tokens)
        PAIRS_COUNT.update(pair)

    # BEST PAIR
    best_pair = get_best_pair(PAIRS_COUNT)
    ALL_BEST_PAIR.append(best_pair)


    # MERGES
    curr_tokens = []
    history_tokens = []
    for tokens in TOKENS:
        merged = merge_pair(tokens, best_pair)
        curr_tokens.append(merged)
        history_tokens.extend(list(token) for token in merged)

            
    TOKENS.clear()
    TOKENS.extend(curr_tokens)
    ALL_TOKENS.extend(history_tokens)

    print("="*30)
    print(f"Cycle : {i + 1}")
    print(f"Current Best Pair : {best_pair}")
    print(f"Merges : {[[list(token) for token in word_tokens] for word_tokens in TOKENS]}")

dilihat = set()
VOCABULARY = []

for item in ALL_TOKENS:
    item_hashable = ke_tuple(item)
    if item_hashable not in dilihat:
        dilihat.add(item_hashable)
        VOCABULARY.append(item)

MERGE_RULES = []
for pair in ALL_BEST_PAIR:
    all_byte_list = []
    for byte in pair:
        byte_list = list(byte)
        all_byte_list.append(byte_list)
    MERGE_RULES.append(all_byte_list)


print(" ")
print("="*30)
print(f"All History Tokens : {VOCABULARY}")
print(f"All Best Pair : {MERGE_RULES}")
print("="*30)
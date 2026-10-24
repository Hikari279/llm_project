from .basic_utils import text_to_byte, token_to_byte_token, merge_pair, ke_tuple, get_pair, get_best_pair
from collections import Counter
import re


# ========================== MAIN CODE ================================
# TRAINING BPE
def train_text(text, num_merges):
    parts = re.findall(r"\s+|[^\w\s]+|\w+", text)
    # DICT
    ALL_TOKENS = [] # all token from text --> for vocabulary
    ALL_BYTES = []
    ALL_BEST_PAIR = [] # --> for merge rules

    WORDS = [] # only word from text -> temporary
    TOKENS = [] # --> word, temporary
    BYTES = []

    # LOOPING MERGE
    for part in parts:
        if re.fullmatch(r"\w+", part):
            WORDS.append(part)
        else:
            ALL_TOKENS.extend([byte] for byte in text_to_byte(part))

    for word in WORDS:
        byte_tokens = [
            token_to_byte_token(b)
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

    # SET VOCAB
    default_tokens = []
    for i in range(256):
        default_tokens.append([i])

    # DEDUPLICATE VOCABULARY
    ALL_TOKENS.extend(default_tokens)

    dilihat = set()
    VOCABULARY = []
    for item in ALL_TOKENS:
        item_hashable = ke_tuple(item)
        if item_hashable not in dilihat:
            dilihat.add(item_hashable)
            VOCABULARY.append(item)


    # MERGE_RULES
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

    print()
    return VOCABULARY, MERGE_RULES



# GET VOCABULARY AND MERGE RULES
def get_vocab(text, num_merges):
    return train_text(text, num_merges)[0]

def get_merge_rules(text, num_merges):
    return train_text(text, num_merges)[1]


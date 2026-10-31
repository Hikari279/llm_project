import json
from pathlib import Path
import re
import sys

from .basic_utils import text_to_byte, text_to_byte_token, merge_pair 

# Import configuration
nao_dir = Path(__file__).resolve().parents[2]
if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from config import VOCABULARY_PATH, MERGE_RULES_PATH


# ========================== MAIN CODE ================================

# TOKENIZE BPE
with VOCABULARY_PATH.open("r", encoding="utf-8") as f:
    vocabulary = json.load(f)
with MERGE_RULES_PATH.open("r", encoding="utf-8") as f:
    merge_rules = json.load(f)


# id
# JSON token [84, 104] -> bytes token b"Th".
vocabulary = [bytes(token) for token in vocabulary]

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



# TOKENIZE TEXT [text] -> [[ids], [ids]]
def tokenize_text(text):

    # SPLIT
    parts = re.findall(r"\s+|[^\w\s]+|\w+", text)
    # "This is text!" -> ["This", " ", "is", " ", "text", "!"].


    BYTES = []
    TOKENS = []

    # GET TOKENS [byte_tokens] FROM PARTS (TEXT)
    # "This": part_bytes -> [[84], [104], [105], [115]];
    # byte_tokens -> [b"T", b"h", b"i", b"s"].
    for part in parts:
        part_bytes = [text_to_byte(p) for p in part]
        byte_tokens = [
            text_to_byte_token([byte])
            for character_bytes in part_bytes
            for byte in character_bytes
        ]

        BYTES.append(part_bytes)
        TOKENS.append(byte_tokens)

    # BYTES[0] = [[84], [104], ...]
    # TOKENS[0] = [b"T", b"h", ...].
    # BYTES holds integer-byte groups
    # TOKENS holds the equivalent bytes objects.

    # print(BYTES)
    # print(TOKENS)

    MERGE_RULES = []

    # TRANFORM MERGE_RULES FORMAT [int, int] -> [byte_tokens, byte_tokens]
    # [[105], [115]] -> (b"i", b"s").
    for pair in merge_rules:
        pair_token = []
        for byte in pair:
            byte_tokens = text_to_byte_token(byte)
            pair_token.append(byte_tokens)

        MERGE_RULES.append(tuple(pair_token))


    # PAIR THE TOKENS [byte_tokens]
    # [b"T", b"h", b"i", b"s"] -> [b"This"];
    # ALL_TOKENS keeps one merged token list per part, including spaces and punctuation.
    ALL_TOKENS = []
    for tokens in TOKENS:
        curr_tokens = list(tokens)

        for pair in MERGE_RULES:
            curr_tokens = merge_pair(curr_tokens, pair)

        ALL_TOKENS.append(curr_tokens)


    # TRANFORM ALL_TOKENS [byte_tokens] -> [int]
    # [b"This"] -> [index_of_b"This"]; IDS stays grouped per input part.
    # With the currently loaded vocabulary, "This is text!" -> [[11], [0], [9], [0], [6, 7, 8, 6], [1]].
    IDS = []
    for tokens in ALL_TOKENS:
        ids = [token_to_id[token] for token in tokens]

        IDS.append(ids)

    return IDS


# HELPER
def decode_token_ids(token_ids):
    token_bytes = b"".join(
        id_to_token[int(token_id)]
        for token_id in token_ids
    )

    return token_bytes.decode("utf-8", errors="replace")
from collections import Counter
import json
from pathlib import Path
import re

from .basic_utils import text_to_byte, text_to_byte_token, merge_pair 



# ========================== MAIN CODE ================================

# TOKENIZE BPE
data_dir = Path(__file__).resolve().parents[1] / "data"
with (data_dir / "vocabulary.json").open("r", encoding="utf-8") as f:
    vocabulary = json.load(f)
with (data_dir / "merge_rules.json").open("r", encoding="utf-8") as f:
    merge_rules = json.load(f)



# id
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



    BYTES = []
    TOKENS = []

    # GET TOKENS [byte_tokens] FROM PARTS (TEXT)
    for part in parts:
        part_bytes = [text_to_byte(p) for p in part]
        byte_tokens = [
            text_to_byte_token([byte])
            for character_bytes in part_bytes
            for byte in character_bytes
        ]

        BYTES.append(part_bytes)
        TOKENS.append(byte_tokens)


    # print(BYTES)
    # print(TOKENS)

    MERGE_RULES = []

    # TRANFORM MERGE_RULES FORMAT [int, int] -> [byte_tokens, byte_tokens]
    for pair in merge_rules:
        pair_token = []
        for byte in pair:
            byte_tokens = text_to_byte_token(byte)
            pair_token.append(byte_tokens)

        MERGE_RULES.append(tuple(pair_token))


    # PAIR THE TOKENS [byte_tokens]
    ALL_TOKENS = []
    for tokens in TOKENS:
        curr_tokens = list(tokens)

        for pair in MERGE_RULES:
            curr_tokens = merge_pair(curr_tokens, pair)

        ALL_TOKENS.append(curr_tokens)


    # TRANFORM ALL_TOKENS [byte_tokens] -> [int]
    IDS = []
    for tokens in ALL_TOKENS:
        ids = [token_to_id[token] for token in tokens]

        IDS.append(ids)

    return IDS

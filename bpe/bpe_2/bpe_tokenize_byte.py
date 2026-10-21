import json
import re
from utils.bpe_utils import text_to_byte, text_to_byte_token, merge_pair, token_to_id, id_to_token, byte_tokens_to_text

with open("data/merge_rules.json", "r", encoding="utf-8") as f:
    merge_rules = json.load(f)
with open("data/vocabulary.json", "r", encoding="utf-8") as f:
    vocabulary = json.load(f)



# TOKENIZE TEXT
text = "lower newer, bananá!"
# SPLIT
parts = re.findall(r"\s+|[^\w\s]+|\w+", text)



BYTES = []
TOKENS = []
WORDS = []


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

for pair in merge_rules:
    pair_token = []
    for byte in pair:
        byte_tokens = text_to_byte_token(byte)
        pair_token.append(byte_tokens)

    MERGE_RULES.append(tuple(pair_token))


ALL_TOKENS = []
for tokens in TOKENS:
    curr_tokens = list(tokens)

    for pair in MERGE_RULES:
        curr_tokens = merge_pair(curr_tokens, pair)

    ALL_TOKENS.append(curr_tokens)


# ENCODE
ENCODED = []
for tokens in ALL_TOKENS:
    ids = [token_to_id[token] for token in tokens]

    ENCODED.append(ids)

# print(ENCODED)




# DECODE
DECODED = []
for ids in ENCODED:
    tokens = [id_to_token[id] for id in ids]

    DECODED.append(tokens)

# print(DECODED)

DECODED_TOKENS = []
for tokens in DECODED:
    decoded_token = [token for token in tokens]
    DECODED_TOKENS.extend(decoded_token)
print(DECODED_TOKENS)

RESULT = byte_tokens_to_text(DECODED_TOKENS)
print(RESULT)




# print(merge_rules)
# print(new_tokens)
    
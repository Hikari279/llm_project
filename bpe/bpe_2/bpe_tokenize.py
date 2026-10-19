import json
from utils.bpe_utils import tokenize_new_word


with open("data/merge_rules.json", "r", encoding="utf-8") as f:
    merge_rules = json.load(f)
with open("data/token_rules.json", "r", encoding="utf-8") as f:
    token_rules = json.load(f)


# search id based on token
token_to_id = {
    token: idx
    for idx, token in enumerate(token_rules)
}

# search token based on id
id_to_token = {
    idx: token
    for idx, token in enumerate(token_rules)
}


def encode(token):
    return token_to_id[token]
    

def decode(ids):
    return id_to_token[ids]

def token_to_word(decoded):
    return "".join(decoded)


new_token = tokenize_new_word("lower", merge_rules)

IDS = []
for token in new_token:
    # get ID
    ids = encode(token)
    IDS.append(ids)

DECODED = [decode(idx) for idx in IDS]
word = token_to_word(DECODED)

# generated_token = token_to_word(decoded, merge_rules)


print(f"New Token: {new_token}")
print(f"ID: {IDS}")
print(f"Decoded ID: {DECODED}")
print(f"Word: {word}")
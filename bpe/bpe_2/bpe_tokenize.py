import json
from utils.bpe_utils import tokenize_text, token_to_text


with open("data/merge_rules.json", "r", encoding="utf-8") as f:
    merge_rules = json.load(f)
with open("data/vocabulary.json", "r", encoding="utf-8") as f:
    vocabulary = json.load(f)




# TOKENIZE WORD
# new_token = tokenize_new_word("banana", merge_rules)

# IDS = []
# for token in new_token:
#     # get ID
#     ids = encode(token)
#     IDS.append(ids)

# DECODED = [decode(idx) for idx in IDS]
# word = token_to_word(DECODED)


# print(f"New Token: {new_token}")
# print(f"ID: {IDS}")
# print(f"Decoded ID: {DECODED}")
# print(f"Word: {word}")

# TOKENIZE TEXT
text = "lower newer, banana!"
TOKENS = tokenize_text(text, merge_rules)[0]
IDS = tokenize_text(text, merge_rules)[1]
DECODED = token_to_text(IDS)[0]
decoded_text = token_to_text(IDS)[1]

print(f"Text: {text}")
print(f"Tokens of Text: {TOKENS}")
print(f"Token IDs of Text: {IDS}")
print(f"Decoded of IDs: {DECODED}")
print(f"Decoded Text: {decoded_text}")

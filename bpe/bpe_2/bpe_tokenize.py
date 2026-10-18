import json
from utils.bpe_utils import tokenize_new_word


with open("data/merge_rules.json", "r", encoding="utf-8") as f:
    merge_rules = json.load(f)

words = ["lowering", "newest", "banana"]

for word in words:
    result = tokenize_new_word(word, merge_rules)

    print(f"Tokenize of '{word}' : {result}")
import re
from collections import Counter
from pathlib import Path
from utils.train_utils import get_vocab, get_merge_rules



# ========================== MAIN CODE ================================
num_merges = 100
data_dir = Path(__file__).resolve().parent / "data"
text_path = data_dir / "text_train.txt"
with text_path.open("r", encoding="utf-8") as file:
    text = file.read()

VOCABULARY = get_vocab(text, num_merges)
MERGE_RULES = get_merge_rules(text, num_merges)

print(VOCABULARY)
print(MERGE_RULES)

import json

with (data_dir / "merge_rules.json").open("w", encoding="utf-8") as f:
    json.dump(MERGE_RULES, f)
with (data_dir / "vocabulary.json").open("w", encoding="utf-8") as f:
    json.dump(VOCABULARY, f)
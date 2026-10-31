import re
from collections import Counter
from pathlib import Path
import sys

# Import configuration
nao_dir = Path(__file__).resolve().parents[1]
if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from config import NUM_MERGES, TRAINING_TEXT_PATH, VOCABULARY_PATH, MERGE_RULES_PATH
from utils.train_utils import get_vocab, get_merge_rules



# ========================== MAIN CODE ================================
with TRAINING_TEXT_PATH.open("r", encoding="utf-8") as file:
    text = file.read()

VOCABULARY = get_vocab(text, NUM_MERGES)
MERGE_RULES = get_merge_rules(text, NUM_MERGES)


import json

with MERGE_RULES_PATH.open("w", encoding="utf-8") as f:
    json.dump(MERGE_RULES, f)
with VOCABULARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(VOCABULARY, f)
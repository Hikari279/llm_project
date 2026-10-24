import json
from pathlib import Path
import sys

import numpy as np

# DIRECTORY SETTING
nao_dir = Path(__file__).resolve().parents[1]

if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from bpe.utils.tokenize_utils import tokenize_text

vocabulary_path = nao_dir / "bpe" / "data" / "vocabulary.json"




# ========================== MAIN CODE ================================
with vocabulary_path.open("r", encoding="utf-8") as f:
    vocabulary = json.load(f)
    
# GET IDS

text = "I'm Learn BPE NOW! 😀"

IDS = tokenize_text(text)
print(IDS)

vocab_size = len(vocabulary)
embedding_dim = 8
max_sequence_length = 128


token_ids = len(IDS)

# CREATE MATRIX
embedding_table = np.random.randn(vocab_size, embedding_dim)
positional_embedding = np.random.randn(max_sequence_length, embedding_dim)

print(IDS)


# COMBINED based n IDS

COMBINED = []
for token_ids in IDS:
    sequence_embedding = embedding_table[token_ids]
    combined_embeddings = sequence_embedding + positional_embedding[:len(sequence_embedding)]
    COMBINED.append(combined_embeddings)

print(COMBINED)
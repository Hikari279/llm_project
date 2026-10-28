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

text = "Aku belajar BPE sekarang! 😀"

IDS = tokenize_text(text)
# print(IDS) 

vocab_size = len(vocabulary)
embedding_dim = 8
max_sequence_length = 128


# CREATE MATRIX
embedding_table = np.random.randn(vocab_size, embedding_dim)
positional_embedding = np.random.randn(max_sequence_length, embedding_dim)


# COMBINE TOKEN AND POSITIONAL EMBEDDINGS
token_ids = [token_id for token_group in IDS for token_id in token_group]

if len(token_ids) > max_sequence_length:
    raise ValueError(
        f"Token sequence length ({len(token_ids)}) exceeds "
        f"max_sequence_length ({max_sequence_length})."
    )

token_embeddings = embedding_table[token_ids]
position_embeddings = positional_embedding[:len(token_ids)]

COMBINED = token_embeddings + position_embeddings
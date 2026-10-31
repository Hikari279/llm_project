import json
from pathlib import Path
import numpy as np
import sys

nao_dir = Path(__file__).resolve().parents[1]
if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from config import VOCABULARY_PATH, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH

with VOCABULARY_PATH.open("r", encoding="utf-8") as f:
    vocabulary = json.load(f)

vocab_size = len(vocabulary)

# INITIALIZE PARAMETERS
embedding_table = np.random.randn(vocab_size, EMBEDDING_DIM)
positional_embedding = np.random.randn(
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM
)
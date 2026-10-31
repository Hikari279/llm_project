import numpy as np
from pathlib import Path
import sys

nao_dir = Path(__file__).resolve().parents[1]

if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

from config import TRAINING_TEXT_PATH, MAX_SEQUENCE_LENGTH
from bpe.utils.tokenize_utils import tokenize_text


with TRAINING_TEXT_PATH.open("r", encoding="utf-8") as f:
    text = f.read()

IDS = tokenize_text(text)

token_ids = np.array([
    token_id
    for group in IDS
    for token_id in group
], dtype=np.int64)


def create_training_sequences(token_ids, sequence_length=MAX_SEQUENCE_LENGTH):

    chunk_size = sequence_length + 1
    training_sequences = []

    for start in range(0, len(token_ids) - 1, sequence_length):

        chunk = token_ids[start : start + chunk_size]

        if len(chunk) < 2:
            continue

        X_ids = chunk[:-1]
        target_ids = chunk[1:]

        training_sequences.append((X_ids, target_ids))

    return training_sequences


training_sequences = create_training_sequences(token_ids)
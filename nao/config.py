from pathlib import Path

# ==================================================
# PROJECT DIRECTORY
# ==================================================

ROOT_DIR = Path(__file__).resolve().parent

BPE_DIR = ROOT_DIR / "bpe"
BPE_DATA_DIR = BPE_DIR / "data"

EMBEDDING_DIR = ROOT_DIR / "embeddings"

TRANSFORMER_DIR = ROOT_DIR / "transformer_flow"
CHECKPOINT_DIR = TRANSFORMER_DIR / "checkpoints"

TRAINING_TEXT_PATH = BPE_DATA_DIR / "text_train.txt"
VOCABULARY_PATH = BPE_DATA_DIR / "vocabulary.json"
MERGE_RULES_PATH = BPE_DATA_DIR / "merge_rules.json"

CHECKPOINT_PATH = CHECKPOINT_DIR / "model_checkpoint.npz"


# ==================================================
# BPE CONFIGURATION
# ==================================================

NUM_MERGES = 300


# ==================================================
# EMBEDDING CONFIGURATION
# ==================================================

EMBEDDING_DIM = 32
MAX_SEQUENCE_LENGTH = 128


# ==================================================
# TRANSFORMER CONFIGURATION
# ==================================================

FF_DIM = 64
NUM_ATTENTION_HEADS = 1


# ==================================================
# TRAINING CONFIGURATION
# ==================================================

LEARNING_RATE = 0.001
NUM_EPOCHS = 20

TRAIN_SPLIT = 0.9
SHUFFLE_DATA = True


# ==================================================
# GENERATION CONFIGURATION
# ==================================================

MAX_NEW_TOKENS = 100
TEMPERATURE = 1.0
TOP_K = 5

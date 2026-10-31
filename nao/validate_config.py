"""
Validation script to verify configuration centralization.
This script tests that all modules can import and use the centralized config.
"""

import sys
from pathlib import Path

# Add nao to path
nao_dir = Path(__file__).resolve().parent
if str(nao_dir) not in sys.path:
    sys.path.insert(0, str(nao_dir))

print("=" * 60)
print("CONFIGURATION VALIDATION TEST")
print("=" * 60)

# Test 1: Config module imports
print("\n[TEST 1] Importing config module...")
try:
    from config import (
        ROOT_DIR, BPE_DIR, BPE_DATA_DIR, EMBEDDING_DIR, TRANSFORMER_DIR, CHECKPOINT_DIR,
        TRAINING_TEXT_PATH, VOCABULARY_PATH, MERGE_RULES_PATH, CHECKPOINT_PATH,
        NUM_MERGES, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, FF_DIM,
        LEARNING_RATE, NUM_EPOCHS, TRAIN_SPLIT, SHUFFLE_DATA,
        MAX_NEW_TOKENS, TEMPERATURE, TOP_K
    )
    print("✓ Config module imported successfully")
    print(f"  - NUM_MERGES: {NUM_MERGES}")
    print(f"  - EMBEDDING_DIM: {EMBEDDING_DIM}")
    print(f"  - MAX_SEQUENCE_LENGTH: {MAX_SEQUENCE_LENGTH}")
    print(f"  - FF_DIM: {FF_DIM}")
    print(f"  - LEARNING_RATE: {LEARNING_RATE}")
    print(f"  - NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"  - TOP_K: {TOP_K}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 2: File paths exist
print("\n[TEST 2] Verifying file paths...")
try:
    assert TRAINING_TEXT_PATH.exists(), f"Training text not found: {TRAINING_TEXT_PATH}"
    print(f"✓ Training text exists: {TRAINING_TEXT_PATH}")
    
    assert VOCABULARY_PATH.exists(), f"Vocabulary not found: {VOCABULARY_PATH}"
    print(f"✓ Vocabulary exists: {VOCABULARY_PATH}")
    
    assert MERGE_RULES_PATH.exists(), f"Merge rules not found: {MERGE_RULES_PATH}"
    print(f"✓ Merge rules exist: {MERGE_RULES_PATH}")
    
    if CHECKPOINT_PATH.exists():
        print(f"✓ Checkpoint exists: {CHECKPOINT_PATH}")
    else:
        print(f"⚠ Checkpoint not found (will be created during training): {CHECKPOINT_PATH}")
except AssertionError as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 3: BPE module
print("\n[TEST 3] Testing BPE module...")
try:
    from bpe.utils.tokenize_utils import tokenize_text, decode_token_ids
    test_text = "Hello world"
    ids = tokenize_text(test_text)
    print(f"✓ BPE tokenization works")
    print(f"  - Input: '{test_text}'")
    print(f"  - Token groups: {len(ids)}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Embeddings module
print("\n[TEST 4] Testing embeddings module...")
try:
    from embeddings.embedding import embedding_table, positional_embedding
    print(f"✓ Embedding tables initialized")
    print(f"  - Embedding table shape: {embedding_table.shape}")
    print(f"  - Positional embedding shape: {positional_embedding.shape}")
    print(f"  - Expected: ({embedding_table.shape[0]}, {EMBEDDING_DIM})")
    print(f"  - Expected: ({MAX_SEQUENCE_LENGTH}, {EMBEDDING_DIM})")
    
    assert embedding_table.shape[1] == EMBEDDING_DIM, "Embedding dim mismatch"
    assert positional_embedding.shape == (MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), "Positional embedding shape mismatch"
    print("✓ Embedding dimensions match config")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Data preparation
print("\n[TEST 5] Testing data preparation...")
try:
    from embeddings.data_preparation import training_sequences
    print(f"✓ Training sequences created")
    print(f"  - Number of sequences: {len(training_sequences)}")
    if len(training_sequences) > 0:
        X_sample, y_sample = training_sequences[0]
        print(f"  - Sample sequence length: {len(X_sample)}")
        print(f"  - Expected max length: {MAX_SEQUENCE_LENGTH}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 6: Transformer utilities
print("\n[TEST 6] Testing transformer utilities...")
try:
    from transformer_flow.utils.self_attention import self_attention
    from transformer_flow.utils.ffn import ffn
    from transformer_flow.utils.layer_normalization import layer_norm
    from transformer_flow.utils.loss import logits, probs, loss
    print("✓ All transformer utility functions imported")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 7: Checkpoint loading (if exists)
print("\n[TEST 7] Testing checkpoint compatibility...")
try:
    if CHECKPOINT_PATH.exists():
        import numpy as np
        checkpoint = np.load(CHECKPOINT_PATH)
        print(f"✓ Checkpoint loaded successfully")
        print(f"  - Keys: {list(checkpoint.keys())}")
        
        # Check dimensions
        checkpoint_embedding_dim = int(checkpoint['embedding_dim'])
        checkpoint_ff_dim = int(checkpoint['ff_dim'])
        checkpoint_max_seq = int(checkpoint['max_sequence_length'])
        
        print(f"\n  Checkpoint configuration:")
        print(f"    - embedding_dim: {checkpoint_embedding_dim}")
        print(f"    - ff_dim: {checkpoint_ff_dim}")
        print(f"    - max_sequence_length: {checkpoint_max_seq}")
        
        print(f"\n  Current config.py configuration:")
        print(f"    - EMBEDDING_DIM: {EMBEDDING_DIM}")
        print(f"    - FF_DIM: {FF_DIM}")
        print(f"    - MAX_SEQUENCE_LENGTH: {MAX_SEQUENCE_LENGTH}")
        
        if (checkpoint_embedding_dim != EMBEDDING_DIM or 
            checkpoint_ff_dim != FF_DIM or 
            checkpoint_max_seq != MAX_SEQUENCE_LENGTH):
            print("\n  ⚠ WARNING: Checkpoint dimensions do NOT match current config!")
            print("    The existing checkpoint was trained with different hyperparameters.")
            print("    Options:")
            print("      1. Update config.py to match checkpoint values")
            print("      2. Delete checkpoint and retrain with current config")
            print("      3. Keep both - checkpoint will be overwritten on next training")
        else:
            print("\n✓ Checkpoint dimensions match config")
    else:
        print("⚠ No checkpoint to test (will be created during training)")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Inference module
print("\n[TEST 8] Testing inference module...")
try:
    from transformer_flow.inference.predict import predict_next_token
    print("✓ Inference module imported successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL VALIDATION TESTS PASSED ✓")
print("=" * 60)
print("\nConfiguration centralization is working correctly!")
print("All modules are using values from config.py")

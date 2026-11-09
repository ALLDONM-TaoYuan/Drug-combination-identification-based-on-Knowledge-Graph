from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RANDOM_STATE = 42

FEATURE_DIM = 3840

EMBEDDINGS_DIR = BASE_DIR / 'embeddings'
SMILES_DIR = BASE_DIR / 'smiles'
TARGETS_DIR = BASE_DIR / 'targets'
FEATURE_BATCH_DIR = BASE_DIR / 'feature_batches'
OUTPUT_DIR = BASE_DIR / 'benchmark_results'

CONFIG = {
    'threshold': 0.5,
    'pooling_row_budget': 200000,
    'batch_rows': 65536,
}

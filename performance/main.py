import argparse
import gc
import os
import random
import warnings

import numpy as np

warnings.filterwarnings('ignore')

import config
from features import ComboDataLoader, DrugDatasetGenerator

FEATURE_DIM = config.FEATURE_DIM
SPLITS = ('train', 'val', 'test')


def ensure_dataset():
    required = [config.FEATURE_BATCH_DIR / f'{split}_combo_index.npz' for split in SPLITS]
    if all(path.exists() for path in required):
        return
    random.seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    generator = DrugDatasetGenerator(
        embedding_path=config.EMBEDDINGS_DIR,
        smiles_path=config.SMILES_DIR,
        target_path=config.TARGETS_DIR,
    )
    generator.generate_dataset(output_dir=config.FEATURE_BATCH_DIR,
                               train_ratio=0.7, val_ratio=0.15)


def build_pair_features(loader, combos, max_pairs=None, row_budget=200000):
    pid = combos['pair_index']
    pairs, start, counts = np.unique(pid, return_index=True, return_counts=True)
    n = len(pairs)
    if max_pairs is not None and n > max_pairs:
        rng = np.random.RandomState(config.RANDOM_STATE)
        keep_idx = np.sort(rng.choice(n, max_pairs, replace=False))
        pairs, start, counts = pairs[keep_idx], start[keep_idx], counts[keep_idx]
        n = max_pairs

    labels = combos['label'][start].astype(np.int8)
    weights = combos['sample_weight'].astype(np.float32)
    X = np.empty((n, FEATURE_DIM), dtype=np.float32)

    k0 = 0
    while k0 < n:
        k1 = k0
        rows = 0
        while k1 < n and (rows + counts[k1]) <= row_budget:
            rows += counts[k1]
            k1 += 1
        if k1 == k0:
            k1 = k0 + 1
        s, e = int(start[k0]), int(start[k1 - 1] + counts[k1 - 1])
        Xb = loader.build_batch(combos[s:e])
        local_start = (start[k0:k1] - s).astype(np.int64)
        X[k0:k1] = np.add.reduceat(Xb * weights[s:e, None], local_start, axis=0)
        k0 = k1
    return X, labels, pairs


def load_splits(cfg):
    data = {}
    for split in SPLITS:
        loader = ComboDataLoader(config.FEATURE_BATCH_DIR, set_name=split)
        max_pairs = cfg.get(f'max_{split}_pairs')
        X, y, pairs = build_pair_features(
            loader, loader.combos, max_pairs=max_pairs,
            row_budget=config.CONFIG['pooling_row_budget'])
        data[f'X_{split}'], data[f'y_{split}'] = X, y
        data[f'{split}_pairs'] = pairs
        loader.close()
        gc.collect()
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-train-pairs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        config.RANDOM_STATE = args.seed

    cfg = {'max_train_pairs': args.max_train_pairs}

    ensure_dataset()
    data = load_splits(cfg)

    from evaluation import run_benchmark
    run_benchmark(data, config.CONFIG)


if __name__ == '__main__':
    os.environ.setdefault('PYTHONHASHSEED', str(config.RANDOM_STATE))
    if os.name == 'nt':
        os.environ.setdefault('OMP_NUM_THREADS', '32')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '32')
    main()

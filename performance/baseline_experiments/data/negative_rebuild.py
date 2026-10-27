from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_rules import (FEATURE_FILE, LABEL_FILE, PAIR_FILE, SPLITS, SPLIT_SEED,
                         TARGET_BIN_EDGES, TARGET_BIN_LABELS, WARM_DIR,
                         WEIGHT_FILE, _encoded)

WARM_DIR = Path(WARM_DIR)
FEATURE_DIM: int = 3840
KG_DIM: int = 128
SMILES_DIM: int = 768
TARGET_DIM: int = 1024
KG_FEATURES_SOURCE = Path(os.environ.get("KGP_DC_KG_FEATURES",
                                         WARM_DIR / "drug_kg_features.npy"))

def _assets():
    drugs = pd.read_csv(WARM_DIR / "drug_table.csv")
    kg = np.load(KG_FEATURES_SOURCE).astype(np.float32)
    smiles = np.load(WARM_DIR / "drug_smiles_features.npy").astype(np.float32)
    targets = np.load(WARM_DIR / "target_features.npy").astype(np.float32)
    index = np.load(WARM_DIR / "drug_target_index.npy")
    n_targets = int(len(np.load(WARM_DIR / "target_ids.npy")))
    if kg.shape[0] != len(drugs) or smiles.shape[0] != len(drugs):
        raise ValueError("drug-level assets are not row-aligned with drug_table.csv")
    return drugs, kg, smiles, targets, index, n_targets

def _target_counts(index, n_targets: int, n_drugs: int) -> np.ndarray:
    counts = np.zeros(n_drugs, dtype=np.int64)
    mask = index["target_index"] != n_targets
    np.add.at(counts, index["drug_id"][mask].astype(np.int64), 1)
    return counts

def _mean_targets(targets, index, n_targets: int, n_drugs: int) -> np.ndarray:
    sums = np.zeros((n_drugs, targets.shape[1]), dtype=np.float64)
    counts = np.zeros(n_drugs, dtype=np.float64)
    mask = index["target_index"] != n_targets
    drug_ids = index["drug_id"][mask].astype(np.int64)
    target_ids = index["target_index"][mask].astype(np.int64)
    np.add.at(sums, drug_ids, targets[target_ids].astype(np.float64))
    np.add.at(counts, drug_ids, 1.0)
    block = np.zeros((n_drugs, targets.shape[1]), dtype=np.float32)
    has = counts > 0
    block[has] = (sums[has] / counts[has, None]).astype(np.float32)
    return block

def _pair_stratum(count1: int, count2: int) -> tuple[int, int]:
    b1 = int(np.digitize(int(count1), TARGET_BIN_EDGES))
    b2 = int(np.digitize(int(count2), TARGET_BIN_EDGES))
    return (b1, b2) if b1 <= b2 else (b2, b1)

def _stratum_label(key: tuple[int, int]) -> str:
    return f"{TARGET_BIN_LABELS[key[0]]}-{TARGET_BIN_LABELS[key[1]]}"

def _features_for(frame: pd.DataFrame, kg, smiles, mean_targets) -> np.ndarray:
    d1 = frame["drug1_id"].to_numpy(dtype=np.int64)
    d2 = frame["drug2_id"].to_numpy(dtype=np.int64)
    matrix = np.empty((len(frame), FEATURE_DIM), dtype=np.float32)
    matrix[:, 0:KG_DIM] = kg[d1]
    matrix[:, KG_DIM:KG_DIM + SMILES_DIM] = smiles[d1]
    matrix[:, KG_DIM + SMILES_DIM:KG_DIM + SMILES_DIM + TARGET_DIM] = mean_targets[d1]
    base = KG_DIM + SMILES_DIM + TARGET_DIM
    matrix[:, base:base + KG_DIM] = kg[d2]
    matrix[:, base + KG_DIM:base + KG_DIM + SMILES_DIM] = smiles[d2]
    matrix[:, base + KG_DIM + SMILES_DIM:] = mean_targets[d2]
    return matrix

def rebuild(seed: int = SPLIT_SEED) -> dict:
    drugs, kg, smiles, targets, index, n_targets = _assets()
    n_drugs = len(drugs)
    target_counts = _target_counts(index, n_targets, n_drugs)
    mean_targets = _mean_targets(targets, index, n_targets, n_drugs)

    old = {split: pd.read_csv(WARM_DIR / PAIR_FILE.format(split=split))
           for split in SPLITS}
    positives = {split: frame.loc[frame["label"] == 1, ["drug1_id", "drug2_id"]]
                 .reset_index(drop=True) for split, frame in old.items()}
    positive_keys = set()
    for frame in positives.values():
        positive_keys |= set(_encoded(frame["drug1_id"].to_numpy(),
                                      frame["drug2_id"].to_numpy()).tolist())

    has_smiles = drugs["has_smiles"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    has_kg = np.abs(kg).sum(axis=1) > 0
    pool = drugs.loc[has_smiles & has_kg, "internal_id"].to_numpy(dtype=np.int64)
    pool = np.sort(pool)
    n_pool = len(pool)

    iu, ju = np.triu_indices(n_pool, k=1)
    left, right = pool[iu], pool[ju]
    keys = left.astype(np.int64) * 100_000_000 + right.astype(np.int64)
    stratum_of_drug = np.asarray([
        int(np.digitize(int(count), TARGET_BIN_EDGES)) for count in target_counts])
    bin_left = stratum_of_drug[left]
    bin_right = stratum_of_drug[right]
    lo = np.minimum(bin_left, bin_right)
    hi = np.maximum(bin_left, bin_right)
    stratum_code = lo * 10 + hi
    forbidden = np.fromiter(sorted(positive_keys), dtype=np.int64, count=len(positive_keys))
    not_positive = ~np.isin(keys, forbidden)

    used = np.zeros(len(left), dtype=bool)
    negatives: dict[str, list[tuple[int, int]]] = {split: [] for split in SPLITS}
    per_split_quota: dict[str, dict] = {split: {} for split in SPLITS}

    for split in SPLITS:
        frame = positives[split]
        counts1 = target_counts[frame["drug1_id"].to_numpy(dtype=np.int64)]
        counts2 = target_counts[frame["drug2_id"].to_numpy(dtype=np.int64)]
        quota: dict[int, int] = {}
        for c1, c2 in zip(counts1, counts2):
            b1, b2 = _pair_stratum(int(c1), int(c2))
            code = min(b1, b2) * 10 + max(b1, b2)
            quota[code] = quota.get(code, 0) + 1
        for code, need in sorted(quota.items()):
            available = (~used) & (stratum_code == code) & not_positive
            index_available = np.flatnonzero(available)
            take = int(min(need, len(index_available)))
            rng = np.random.RandomState(int(seed) + 1000 * SPLITS.index(split) + 17 * code)
            if take > 0:
                chosen = rng.choice(index_available, size=take, replace=False)
                used[chosen] = True
                negatives[split].extend(
                    (int(a), int(b)) for a, b in zip(left[chosen], right[chosen]))
            per_split_quota[split][_stratum_label(divmod(code, 10))] = {
                "needed": int(need), "sampled": int(take),
                "candidates": int(len(index_available))}

    for split in SPLITS:
        need = len(positives[split]) - len(negatives[split])
        if need > 0:
            available = np.flatnonzero(~used & not_positive)
            rng = np.random.RandomState(int(seed) + 7 * SPLITS.index(split))
            chosen = rng.choice(available, size=int(min(need, len(available))),
                                replace=False)
            used[chosen] = True
            negatives[split].extend(
                (int(a), int(b)) for a, b in zip(left[chosen], right[chosen]))

    report: dict = {"seed": int(seed), "positives_kept": True,
                    "negative_policy": "split-disjoint DTI-matched (rebuilt)",
                    "pool_size": int(n_pool), "splits": {}}
    for split in SPLITS:
        positive_frame = positives[split].assign(label=1, source="positive")
        negative_frame = pd.DataFrame(sorted(set(negatives[split])),
                                      columns=["drug1_id", "drug2_id"]).assign(
            label=0, source="negative")
        frame = pd.concat([positive_frame, negative_frame], ignore_index=True)
        frame = frame.sort_values(["label", "drug1_id", "drug2_id"],
                                  ascending=[False, True, True]).reset_index(drop=True)
        frame = frame[["drug1_id", "drug2_id", "label", "source"]]
        frame.to_csv(WARM_DIR / PAIR_FILE.format(split=split), index=False)
        np.save(WARM_DIR / LABEL_FILE.format(split=split),
                frame["label"].to_numpy(dtype=np.int8))
        np.save(WARM_DIR / WEIGHT_FILE.format(split=split),
                np.ones(len(frame), dtype=np.float32))
        matrix = _features_for(frame, kg, smiles, mean_targets)
        np.save(WARM_DIR / FEATURE_FILE.format(split=split), matrix)
        report["splits"][split] = {
            "pairs": int(len(frame)),
            "positives": int(frame["label"].sum()),
            "negatives": int((frame["label"] == 0).sum()),
            "finite": bool(np.isfinite(matrix).all()),
            "quota": per_split_quota[split],
        }
        del matrix

    report["unique_negatives"] = int(sum(
        len(set(negatives[split])) for split in SPLITS))
    (WARM_DIR / "rebuild_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="rewrite pair_meta / labels / weights / features")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps({
            "dry_run": True,
            "data_dir": str(WARM_DIR),
            "pairs": {split: int(len(pd.read_csv(
                WARM_DIR / PAIR_FILE.format(split=split)))) for split in SPLITS},
            "hint": "pass --apply to rewrite the split files",
        }, indent=2))
        return
    print(json.dumps(rebuild(int(args.seed)), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smiles import load_smiles_features
from target import load_target_features

KG_ROOT = ROOT / "kg_embedding"
SCENARIO_DIR = {"single": ROOT / "single_cold_start" / "features",
                "dual": ROOT / "dual_cold_start" / "features",
                "warm": ROOT / "warm_start" / "features"}
POOL_FILES = {
    "dual": {"train": "train_drugs.csv", "val": "val_cold_drugs.csv",
             "test": "test_cold_drugs.csv"},
    "single": {"train": "train_drugs.csv", "val": "val_drugs.csv",
               "test": "test_drugs.csv"},
    "warm": {split: f"{split}_drugs.csv" for split in ("train", "val", "test")},
}
DT_INDEX_DTYPE = np.dtype([("drug_id", np.int64), ("target_index", np.int64)])
SPLITS = ("train", "val", "test")
WARM_DIM = 3840
KG_DIM = 128
TARGET_BIN_EDGES = np.asarray([1, 2, 3, 6, 11])
TARGET_BIN_LABELS = ("0", "1", "2", "3-5", "6-10", "11+")


def resolve_data_root(root: Path = ROOT) -> Path:
    for name in ("entity_split_data", "data"):
        candidate = root / name
        if (candidate / "split").is_dir() and (candidate / "features").is_dir():
            return candidate
    raise FileNotFoundError(f"neither entity_split_data/ nor data/ found under {root}")


DATA_ROOT = resolve_data_root()


def find_database_dir() -> Path:
    for directory in (ROOT.parent.parent / "Database", ROOT.parent / "database"):
        if (directory / "DTI.csv").exists():
            return directory
    raise FileNotFoundError("Raw database (DTI.csv) not found")


def ensure_drug_target_index(features_dir: Path, force: bool = False) -> Path:
    out = features_dir / "drug_target_index.npy"
    if out.exists() and not force:
        return out
    table = pd.read_csv(features_dir / "drug_table.csv", dtype=str).fillna("")
    name_to_id = dict(zip(table["drug_name"].str.strip(),
                          table["internal_id"].astype(int)))
    target_ids = np.load(features_dir / "target_ids.npy").astype(np.int64)
    n_targets = int(len(target_ids))
    zero_target = n_targets
    target_table = pd.read_csv(features_dir / "target_table.csv", dtype=str).fillna("")
    raw_to_id = dict(zip(target_table["raw_target_id"].str.strip(),
                         target_table["target_id"].astype(int)))
    dti = pd.read_csv(find_database_dir() / "DTI.csv", header=None, dtype=str)
    rows: set[tuple[int, int]] = set()
    unresolved: list[dict] = []
    for _, (name, _, target) in dti.iterrows():
        drug = name_to_id.get(str(name).strip())
        tid = raw_to_id.get(str(target).strip())
        if drug is None:
            unresolved.append({"kind": "dti_head_not_in_drug_space", "name": str(name)})
            continue
        if tid is None:
            unresolved.append({"kind": "dti_tail_not_in_target_space", "target": str(target)})
            continue
        rows.add((int(drug), int(tid)))
    seen = {d for d, _ in rows}
    for drug in name_to_id.values():
        if int(drug) not in seen:
            rows.add((int(drug), zero_target))
    index = np.asarray(sorted(rows), dtype=DT_INDEX_DTYPE)
    np.save(out, index)
    (features_dir / "target_mapping_unresolved.csv").write_text(
        pd.DataFrame(unresolved).drop_duplicates().to_csv(index=False),
        encoding="utf-8")
    return out


def load_target_map(features_dir: Path) -> tuple[dict[int, list[int]], int]:
    ensure_drug_target_index(features_dir)
    zero_target = int(len(np.load(features_dir / "target_ids.npy")))
    rows = np.load(features_dir / "drug_target_index.npy", mmap_mode="r")
    mapping: dict[int, list[int]] = {}
    for drug, target in zip(rows["drug_id"].astype(np.int64),
                            rows["target_index"].astype(np.int64)):
        target = int(target)
        if target == zero_target:
            mapping.setdefault(int(drug), [])
            continue
        mapping.setdefault(int(drug), []).append(target)
    return {drug: sorted(values) for drug, values in mapping.items()}, zero_target


def load_kg_features(kg_dir: Path, n_drugs: int) -> tuple[np.ndarray, np.ndarray]:
    feature_path = kg_dir / "drug_kg_features.npy"
    status_path = kg_dir / "kg_embedding_status.csv"
    if not feature_path.exists() or not status_path.exists():
        raise FileNotFoundError(
            f"invalid KG asset directory: {kg_dir} (needs drug_kg_features.npy "
            "+ kg_embedding_status.csv)")
    kg = np.load(feature_path).astype(np.float32)
    status = pd.read_csv(status_path)
    has_kg = status["has_kg_relation"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}).to_numpy()
    status_ids = status["internal_id"].astype(int).to_numpy()
    if kg.shape != (n_drugs, KG_DIM):
        raise ValueError(f"KG features {kg.shape} != ({n_drugs}, {KG_DIM})")
    if not np.array_equal(status_ids, np.arange(n_drugs)):
        raise ValueError("KG status rows are not aligned with the drug row space")
    if not np.isfinite(kg).all():
        raise ValueError("KG features contain NaN or Inf")
    kg = kg.copy()
    kg[~has_kg] = 0.0
    return kg, has_kg


def read_pairs(split_dir: Path, split: str) -> set[tuple[int, int]]:
    frame = pd.read_csv(split_dir / f"{split}_pairs.csv")
    pairs: set[tuple[int, int]] = set()
    for a, b in zip(frame["drug1_id"], frame["drug2_id"]):
        ia, ib = int(a), int(b)
        if ia != ib:
            pairs.add((ia, ib) if ia < ib else (ib, ia))
    return pairs


def split_pool(mode: str, split: str) -> set[int]:
    path = DATA_ROOT / "split" / mode / POOL_FILES[mode][split]
    return set(pd.read_csv(path)["internal_id"].astype(int))


def cartesian_weighted_targets(targets: np.ndarray, t1: list[int],
                               t2: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rows1 = np.asarray(t1, dtype=np.int64)
    rows2 = np.asarray(t2, dtype=np.int64)
    if len(rows1) == 0 or len(rows2) == 0:
        raise ValueError("Cartesian expansion requires n1 > 0 and n2 > 0")
    weight = 1.0 / (len(rows1) * len(rows2))
    block1 = np.zeros(targets.shape[1], dtype=np.float64)
    block2 = np.zeros(targets.shape[1], dtype=np.float64)
    for row in rows1:
        block1 += targets[row] * (len(rows2) * weight)
    for row in rows2:
        block2 += targets[row] * (len(rows1) * weight)
    return block1.astype(np.float32), block2.astype(np.float32)


def feature_rows(drug: int, targets_of: dict[int, list[int]],
                 zero_target: int) -> list[int]:
    rows = targets_of.get(int(drug))
    return list(rows) if rows else [int(zero_target)]


def pair_feature(d1: int, d2: int, targets_of: dict[int, list[int]],
                 targets: np.ndarray, kg: np.ndarray, smiles: np.ndarray,
                 zero_target: int) -> np.ndarray:
    t1_rows = feature_rows(d1, targets_of, zero_target)
    t2_rows = feature_rows(d2, targets_of, zero_target)
    mean1 = targets[t1_rows].mean(axis=0)
    mean2 = targets[t2_rows].mean(axis=0)
    return np.concatenate([
        kg[d1], smiles[d1], mean1,
        kg[d2], smiles[d2], mean2,
    ]).astype(np.float32, copy=False)


def verify_cartesian(targets_of: dict[int, list[int]], targets: np.ndarray,
                     pairs: list[tuple[int, int]], limit: int,
                     rng: np.random.RandomState, zero_target: int) -> dict:
    if not pairs or limit <= 0:
        return {"checked": 0, "max_abs_error": 0.0}
    chosen = [pairs[i] for i in rng.choice(len(pairs), size=min(limit, len(pairs)),
                                           replace=False)]
    worst = 0.0
    for d1, d2 in chosen:
        rows1 = feature_rows(d1, targets_of, zero_target)
        rows2 = feature_rows(d2, targets_of, zero_target)
        mean1 = targets[rows1].mean(axis=0)
        mean2 = targets[rows2].mean(axis=0)
        explicit1, explicit2 = cartesian_weighted_targets(targets, rows1, rows2)
        worst = max(worst, float(np.abs(explicit1 - mean1).max()),
                    float(np.abs(explicit2 - mean2).max()))
    return {"checked": len(chosen), "max_abs_error": worst}


def target_stratum(count1: int, count2: int) -> tuple[int, int]:
    b1 = int(np.digitize(count1, TARGET_BIN_EDGES))
    b2 = int(np.digitize(count2, TARGET_BIN_EDGES))
    return (b1, b2) if b1 <= b2 else (b2, b1)


def stratum_label(key: tuple[int, int]) -> str:
    return f"{TARGET_BIN_LABELS[key[0]]}-{TARGET_BIN_LABELS[key[1]]}"


def encode_pairs(pairs: np.ndarray, n_drugs: int) -> np.ndarray:
    return pairs[:, 0] * n_drugs + pairs[:, 1]


def sample_negatives_matched(left_pool: set[int], right_pool: set[int],
                             quota: Counter, target_counts: dict[int, int],
                             forbidden: set[tuple[int, int]], n_drugs: int,
                             rng: np.random.RandomState) -> tuple[set, dict]:
    left = np.asarray(sorted(left_pool), dtype=np.int64)
    right = np.asarray(sorted(right_pool), dtype=np.int64)
    if left.size == 0 or right.size == 0:
        raise RuntimeError("empty KG-usable pool for negatives")

    grid_left = np.repeat(left, right.size)
    grid_right = np.tile(right, left.size)
    keep = grid_left != grid_right
    grid_left, grid_right = grid_left[keep], grid_right[keep]
    low = np.minimum(grid_left, grid_right)
    high = np.maximum(grid_left, grid_right)
    candidates = np.unique(np.stack([low, high], axis=1), axis=0)
    if forbidden:
        blocked = np.asarray([min(a, b) * n_drugs + max(a, b) for a, b in forbidden],
                             dtype=np.int64)
        candidates = candidates[~np.isin(encode_pairs(candidates, n_drugs), blocked)]

    counts = np.asarray([target_counts.get(int(d), 0) for d in range(n_drugs)],
                        dtype=np.int64)
    bin1 = np.digitize(counts[candidates[:, 0]], TARGET_BIN_EDGES)
    bin2 = np.digitize(counts[candidates[:, 1]], TARGET_BIN_EDGES)
    keys = np.stack([np.minimum(bin1, bin2), np.maximum(bin1, bin2)], axis=1)

    negatives: set[tuple[int, int]] = set()
    used = np.zeros(len(candidates), dtype=bool)
    per_stratum: dict[str, dict] = {}
    deficit_total = 0
    for key, need in sorted(quota.items()):
        mask = (keys[:, 0] == key[0]) & (keys[:, 1] == key[1]) & (~used)
        index = np.flatnonzero(mask)
        take = int(min(need, index.size))
        if take:
            chosen = rng.choice(index, size=take, replace=False)
            used[chosen] = True
            for a, b in candidates[chosen]:
                negatives.add((int(a), int(b)))
        deficit = int(need - take)
        deficit_total += deficit
        per_stratum[stratum_label(key)] = {
            "positives": int(need), "negatives": take,
            "candidates": int(index.size), "deficit": deficit}

    surplus = 0
    if deficit_total > 0:
        rest = np.flatnonzero(~used)
        extra = int(min(deficit_total, rest.size))
        if extra:
            chosen = rng.choice(rest, size=extra, replace=False)
            for a, b in candidates[chosen]:
                negatives.add((int(a), int(b)))
            surplus = extra

    diagnostics = {
        "quota_pairs": int(sum(quota.values())),
        "matched_negatives": int(len(negatives) - surplus),
        "surplus_negatives": int(surplus),
        "stratum_deficit": int(deficit_total),
        "per_stratum": per_stratum,
    }
    return negatives, diagnostics


def build_fold(mode: str, split: str, *, seed: int = 42, force: bool = False,
               verify_cartesian_pairs: int = 0,
               match_dti: bool = True) -> dict:
    mode, split = mode.lower(), split.lower()
    if mode not in SCENARIO_DIR or split not in SPLITS:
        raise ValueError(f"invalid mode/split: {mode}/{split}")
    out_dir = SCENARIO_DIR[mode]
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / f"{split}.npy"
    if feature_path.exists() and not force:
        raise FileExistsError(f"{feature_path} exists; pass --force to rebuild")

    features_dir = DATA_ROOT / "features"
    fold_kg_dir = KG_ROOT / mode / split
    drug_ids, smiles = load_smiles_features(features_dir)
    _, targets = load_target_features(features_dir)
    kg, has_kg = load_kg_features(fold_kg_dir, len(drug_ids))
    targets_of, zero_target = load_target_map(features_dir)
    target_counts = {drug: len(rows) for drug, rows in targets_of.items()}

    rng = np.random.RandomState(seed)
    split_dir = DATA_ROOT / "split" / mode

    positives = read_pairs(split_dir, split)
    usable = {int(i) for i, ok in enumerate(has_kg) if ok}
    all_positives = set().union(*[read_pairs(split_dir, s) for s in SPLITS])
    pos_ok = [p for p in positives if p[0] in usable and p[1] in usable]
    skipped_no_kg = len(positives) - len(pos_ok)

    if mode in {"dual", "warm"}:
        left = right = split_pool(mode, split) & usable
    elif split == "train":
        left = right = split_pool(mode, "train") & usable
    else:
        left = split_pool(mode, split) & usable
        right = split_pool(mode, "train") & usable
    if not left or not right:
        raise RuntimeError(f"{mode}/{split}: empty KG-usable pool for negatives")

    if match_dti:
        quota = Counter(target_stratum(target_counts.get(a, 0),
                                       target_counts.get(b, 0))
                        for a, b in pos_ok)
    else:
        quota = Counter({(-1, -1): len(pos_ok)})
    neg_set, sampling = sample_negatives_matched(
        left, right, quota, target_counts, all_positives, len(drug_ids), rng)
    if not match_dti:
        sampling = dict(sampling, per_stratum={}, quota_pairs=len(pos_ok))

    pos_rows = sorted(pos_ok)
    neg_rows = sorted(neg_set)
    rows = [(a, b, 1) for a, b in pos_rows] + [(a, b, 0) for a, b in neg_rows]
    labels = np.asarray([label for _, _, label in rows], dtype=np.int8)
    n_pairs = len(rows)
    X = np.empty((n_pairs, WARM_DIM), dtype=np.float32)
    for index, (d1, d2, _) in enumerate(rows):
        X[index] = pair_feature(d1, d2, targets_of, targets, kg, smiles,
                                zero_target)

    verification = verify_cartesian(targets_of, targets, pos_rows,
                                    verify_cartesian_pairs, rng, zero_target)

    def annotation_bit(pairs: list[tuple[int, int]]) -> np.ndarray:
        return np.asarray([1 if (target_counts.get(a, 0) > 0 and target_counts.get(b, 0) > 0)
                           else 0 for a, b in pairs], dtype=np.int8)

    bit_pos = annotation_bit(pos_rows)
    bit_neg = annotation_bit(neg_rows)

    np.save(feature_path, X)
    np.save(out_dir / f"{split}_labels.npy", labels)
    np.save(out_dir / f"{split}_sample_weight.npy", np.ones(n_pairs, dtype=np.float32))
    pd.DataFrame(rows, columns=["drug1_id", "drug2_id", "label"]).assign(
        source=["positive"] * len(pos_rows) + ["negative"] * len(neg_rows)).to_csv(
        out_dir / f"{split}_pair_meta.csv", index=False)

    summary = {
        "mode": mode, "split": split,
        "data_root": str(DATA_ROOT),
        "kg_dir": str(fold_kg_dir),
        "output_dir": str(out_dir),
        "target_policy": "cartesian combinations with weight 1/(n1*n2); closed-form mean aggregation",
        "negative_policy": ("DtiFrequencyMatchedStrata" if match_dti else "UniformRandom"),
        "positive_pairs": len(pos_ok),
        "skipped_positive_no_fold_kg": skipped_no_kg,
        "negative_pairs": len(neg_set),
        "pairs": n_pairs,
        "shape": list(X.shape),
        "feature_dim": WARM_DIM,
        "seed": seed,
        "finite": bool(np.isfinite(X).all()),
        "kg_usable_drugs_in_fold": int(len(usable)),
        "zero_target_placeholder_row": int(zero_target),
        "annotation_bit_positive_rate": float(bit_pos.mean()) if len(bit_pos) else None,
        "annotation_bit_negative_rate": float(bit_neg.mean()) if len(bit_neg) else None,
        "annotation_bit_auc": _binary_auc(bit_pos, bit_neg),
        "sampling": sampling,
        "cartesian_verification": verification,
    }
    (out_dir / f"{split}_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _binary_auc(positive_bit: np.ndarray, negative_bit: np.ndarray) -> float | None:
    if positive_bit.size == 0 or negative_bit.size == 0:
        return None
    positives = float(positive_bit.sum())
    negatives = float(negative_bit.sum())
    n_pos, n_neg = positive_bit.size, negative_bit.size
    return float((positives * (n_neg - negatives)
                  + 0.5 * (positives * negatives + (n_pos - positives) * (n_neg - negatives)))
                 / (n_pos * n_neg))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "dual", "warm"), required=True)
    parser.add_argument("--split", choices=SPLITS, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--verify-cartesian", type=int, default=200)
    parser.add_argument("--uniform-negatives", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ensure_drug_target_index(DATA_ROOT / "features", force=args.rebuild_index)
    splits = [args.split] if args.split else list(SPLITS)
    results = [build_fold(args.mode, split, seed=args.seed, force=args.force,
                          verify_cartesian_pairs=args.verify_cartesian,
                          match_dti=not args.uniform_negatives)
               for split in splits]


if __name__ == "__main__":
    main()

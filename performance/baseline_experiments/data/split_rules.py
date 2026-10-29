from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

def _resolve_data_dir(base: Path) -> Path:
    for name in ("split_data", "warm_data"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / "split_data"

WARM_DIR: Path = _resolve_data_dir(Path(__file__).resolve().parent)

SPLITS: tuple[str, ...] = ("train", "val", "test")
PAIR_FILE = "{split}_pair_meta.csv"
LABEL_FILE = "{split}_labels.npy"
FEATURE_FILE = "{split}.npy"
WEIGHT_FILE = "{split}_sample_weight.npy"

SPLIT_SEED: int = 42
RATIOS: dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2}
FEATURE_DIM: int = 3840
KG_DIM: int = 128
SMILES_DIM: int = 768
TARGET_DIM: int = 1024

PAIR_COLUMNS: tuple[str, ...] = ("pair_id", "drug1_id", "drug2_id", "label", "source")
TARGET_BIN_EDGES = np.asarray([1, 2, 3, 6, 11])
TARGET_BIN_LABELS = ("0", "1", "2", "3-5", "6-10", "11+")

KG_FEATURES_SOURCE = Path(os.environ.get(
    "KGP_DC_KG_FEATURES", WARM_DIR / "drug_kg_features.npy"))

def _encoded(drug1: np.ndarray, drug2: np.ndarray) -> np.ndarray:
    a = np.asarray(drug1, dtype=np.int64)
    b = np.asarray(drug2, dtype=np.int64)
    return np.minimum(a, b) * 100_000_000 + np.maximum(a, b)

def load_split(split: str) -> pd.DataFrame:
    split = str(split)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = WARM_DIR / PAIR_FILE.format(split=split)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; data/split_data is incomplete")

    frame = pd.read_csv(path)
    missing = {"drug1_id", "drug2_id", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    if "source" not in frame.columns:
        frame["source"] = np.where(frame["label"].to_numpy() == 1, "positive", "negative")

    labels = np.load(WARM_DIR / LABEL_FILE.format(split=split)).astype(np.int32)
    if len(labels) != len(frame):
        raise ValueError(f"{path}: {len(frame)} rows vs {len(labels)} labels")
    if not np.array_equal(labels, frame["label"].to_numpy().astype(np.int32)):
        raise ValueError(f"{path}: labels disagree with {LABEL_FILE.format(split=split)}")

    frame = frame.copy().drop(columns=["pair_id"], errors="ignore")
    frame.insert(0, "pair_id", np.arange(len(frame), dtype=np.int64))
    return frame[list(PAIR_COLUMNS)]

def load_all() -> dict[str, pd.DataFrame]:
    return {split: load_split(split) for split in SPLITS}

def validate() -> dict:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str = "",
            optional: bool = False) -> None:
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail, "optional": bool(optional)})

    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        pair_path = WARM_DIR / PAIR_FILE.format(split=split)
        feature_path = WARM_DIR / FEATURE_FILE.format(split=split)
        add(f"{split}: pair metadata exists", pair_path.exists(), pair_path.name)
        if not pair_path.exists():
            continue
        # The 3840-dimensional pair-level feature matrices are large (about
        # 0.9 GB for the training split) and exceed the 100 MB per-file limit
        # of the Git hosting service, so they are not version controlled.  They
        # are part of the experiment workspace and are produced by
        # `data/negative_rebuild.py --apply`.  Their absence is therefore
        # reported but does not fail the split validation, which only depends
        # on the committed pair metadata.
        add(f"{split}: feature matrix present (optional)",
            feature_path.exists(),
            "present" if feature_path.exists()
            else "absent - not version controlled (see data/split_data/README)",
            optional=True)
        frame = load_split(split)
        frames[split] = frame
        if feature_path.exists():
            features = np.load(feature_path, mmap_mode="r")
            add(f"{split}: feature shape",
                features.ndim == 2 and features.shape == (len(frame), FEATURE_DIM),
                f"{features.shape} vs ({len(frame)}, {FEATURE_DIM})",
                optional=True)
        positives = int(frame["label"].sum())
        negatives = int((frame["label"] == 0).sum())
        add(f"{split}: labels 1:1", positives == negatives,
            f"positives={positives}, negatives={negatives}")
        add(f"{split}: both classes", positives > 0 and negatives > 0)
        canonical = int((frame["drug1_id"] < frame["drug2_id"]).sum())
        add(f"{split}: canonical pairs", canonical == len(frame),
            f"{canonical}/{len(frame)} satisfy drug1_id < drug2_id")

    if len(frames) == len(SPLITS):
        keys = {split: set(_encoded(frame["drug1_id"].to_numpy(),
                                    frame["drug2_id"].to_numpy()).tolist())
                for split, frame in frames.items()}
        overlap_tr_va = len(keys["train"] & keys["val"])
        overlap_tr_te = len(keys["train"] & keys["test"])
        overlap_va_te = len(keys["val"] & keys["test"])
        add("no pair overlap across splits",
            overlap_tr_va == overlap_tr_te == overlap_va_te == 0,
            f"train&val={overlap_tr_va}, train&test={overlap_tr_te}, val&test={overlap_va_te}")
        total = sum(len(frame) for frame in frames.values())
        add("split sizes match 60/20/20",
            abs(len(frames["train"]) / total - RATIOS["train"]) < 0.01
            and abs(len(frames["val"]) / total - RATIOS["val"]) < 0.01,
            f"train={len(frames['train'])}, val={len(frames['val'])}, test={len(frames['test'])}")

    required = [item for item in checks if not item.get("optional")]
    return {
        "warm_dir": str(WARM_DIR),
        "split_seed": SPLIT_SEED,
        "ratios": RATIOS,
        "passed": all(item["passed"] for item in required),
        "checks": checks,
    }

def summary() -> dict:
    frames = load_all()
    return {
        "warm_dir": str(WARM_DIR),
        "split_seed": SPLIT_SEED,
        "ratios": RATIOS,
        "splits": {split: {
            "pairs": int(len(frame)),
            "positives": int(frame["label"].sum()),
            "negatives": int((frame["label"] == 0).sum()),
            "drugs": int(len(set(frame["drug1_id"]) | set(frame["drug2_id"]))),
        } for split, frame in frames.items()},
    }

FOLD_COUNT: int = 5
FOLD_DIR: Path = WARM_DIR / "folds"
VAL_ONE_IN: int = 8

def _target_counts_table() -> np.ndarray:
    drugs = pd.read_csv(WARM_DIR / "drug_table.csv")
    n_drugs = int(drugs["internal_id"].max()) + 1
    index = np.load(WARM_DIR / "drug_target_index.npy")
    n_targets = int(len(np.load(WARM_DIR / "target_ids.npy")))
    counts = np.zeros(n_drugs, dtype=np.int64)
    mask = index["target_index"] != n_targets
    np.add.at(counts, index["drug_id"][mask].astype(np.int64), 1)
    return counts

def _stratum_codes(pairs: pd.DataFrame, counts: np.ndarray) -> np.ndarray:
    b1 = np.digitize(counts[pairs["drug1_id"].to_numpy(dtype=np.int64)], TARGET_BIN_EDGES)
    b2 = np.digitize(counts[pairs["drug2_id"].to_numpy(dtype=np.int64)], TARGET_BIN_EDGES)
    lo = np.minimum(b1, b2)
    hi = np.maximum(b1, b2)
    return (lo * 10 + hi).astype(np.int64)

def build_folds(n_folds: int = FOLD_COUNT, seed: int = SPLIT_SEED) -> dict:
    frames = load_all()
    pairs = pd.concat([frame[["drug1_id", "drug2_id", "label", "source"]]
                       for frame in frames.values()], ignore_index=True)
    before = len(pairs)
    pairs = pairs.drop_duplicates(subset=["drug1_id", "drug2_id"]).reset_index(drop=True)
    if len(pairs) != before:
        raise ValueError(f"{before - len(pairs)} duplicate pairs across the warm splits")
    pairs.insert(0, "pair_id", np.arange(len(pairs), dtype=np.int64))

    counts = _target_counts_table()
    codes = _stratum_codes(pairs, counts)
    labels = pairs["label"].to_numpy(dtype=np.int32)
    fold_of_pair = np.full(len(pairs), -1, dtype=np.int64)

    for code in sorted(set(codes.tolist())):
        rows = np.flatnonzero(codes == code)
        positive = rows[labels[rows] == 1]
        negative = rows[labels[rows] == 0]
        if len(positive) != len(negative):
            raise ValueError(f"stratum {code} is not paired: "
                             f"{len(positive)} vs {len(negative)}")
        rng = np.random.RandomState(int(seed) + 7919 * code)
        permutation = rng.permutation(len(positive))
        fold_of_pair[positive] = permutation % n_folds
        fold_of_pair[negative] = permutation % n_folds
    if (fold_of_pair < 0).any():
        raise RuntimeError("some pairs were not assigned to a fold")

    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    written: dict = {}
    for fold in range(1, n_folds + 1):
        is_test = fold_of_pair == (fold - 1)
        is_pool = ~is_test
        is_val = np.zeros(len(pairs), dtype=bool)
        for code in sorted(set(codes.tolist())):
            rows = np.flatnonzero((codes == code) & is_pool)
            positive = rows[labels[rows] == 1]
            negative = rows[labels[rows] == 0]
            rng = np.random.RandomState(int(seed) + 104729 * code + fold)
            take = rng.permutation(len(positive)) % VAL_ONE_IN == 0
            is_val[positive[take]] = True
            is_val[negative[take]] = True
        is_train = is_pool & ~is_val
        for split, mask in (("train", is_train), ("val", is_val), ("test", is_test)):
            frame = pairs.loc[mask, ["pair_id", "drug1_id", "drug2_id", "label", "source"]]
            frame = frame.sort_values("pair_id")
            path = FOLD_DIR / f"fold_{fold}_{split}.csv"
            frame.to_csv(path, index=False)
            written[path.name] = {
                "pairs": int(len(frame)),
                "positives": int(frame["label"].sum()),
                "negatives": int((frame["label"] == 0).sum()),
            }

    manifest = {"n_folds": int(n_folds), "seed": int(seed),
                "rule": "pair-level stratified fold + inner validation carve",
                "val_one_in": int(VAL_ONE_IN),
                "pairs_total": int(len(pairs)),
                "files": written}
    (FOLD_DIR / "folds_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest

def load_fold(fold: int, split: str) -> pd.DataFrame:
    path = FOLD_DIR / f"fold_{int(fold)}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `python data/split_rules.py --folds`")
    frame = pd.read_csv(path)
    return frame[["pair_id", "drug1_id", "drug2_id", "label", "source"]]

def validate_folds() -> dict:
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    test_keys = {fold: set(_encoded(load_fold(fold, "test")["drug1_id"].to_numpy(),
                                    load_fold(fold, "test")["drug2_id"].to_numpy()).tolist())
                 for fold in range(1, FOLD_COUNT + 1)}
    total_pairs = 0
    for fold in range(1, FOLD_COUNT + 1):
        parts = {split: load_fold(fold, split) for split in SPLITS}
        sizes = {split: len(frame) for split, frame in parts.items()}
        if fold == 1:
            total_pairs = sum(sizes.values())
        add(f"fold {fold}: sizes", True,
            f"train={sizes['train']}, val={sizes['val']}, test={sizes['test']}")
        keys = {split: set(_encoded(frame["drug1_id"].to_numpy(),
                                    frame["drug2_id"].to_numpy()).tolist())
                for split, frame in parts.items()}
        add(f"fold {fold}: disjoint",
            len(keys["train"] & keys["val"]) == 0
            and len(keys["train"] & keys["test"]) == 0
            and len(keys["val"] & keys["test"]) == 0,
            f"train&val={len(keys['train'] & keys['val'])}, "
            f"train&test={len(keys['train'] & keys['test'])}, "
            f"val&test={len(keys['val'] & keys['test'])}")
        add(f"fold {fold}: labels 1:1",
            all(int(frame["label"].sum()) == int((frame["label"] == 0).sum())
                for frame in parts.values()),
            ", ".join(f"{split}={int(frame['label'].sum())}/"
                      f"{int((frame['label'] == 0).sum())}"
                      for split, frame in parts.items()))
    duplicates = 0
    seen: set = set()
    for fold in range(1, FOLD_COUNT + 1):
        duplicates += len(seen & test_keys[fold])
        seen |= test_keys[fold]
    add("test folds partition all pairs",
        duplicates == 0 and len(seen) == total_pairs,
        f"duplicates={duplicates}, covered={len(seen)}/{total_pairs}")
    return {"fold_dir": str(FOLD_DIR), "n_folds": FOLD_COUNT,
            "passed": all(item["passed"] for item in checks), "checks": checks}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", action="store_true",
                        help="build the 5-fold cross-validation splits (seed 42)")
    args = parser.parse_args()
    if args.folds:
        print(json.dumps(build_folds(), indent=2, ensure_ascii=False))
        print(json.dumps(validate_folds(), indent=2, ensure_ascii=False))
    print(json.dumps(validate(), indent=2, ensure_ascii=False))
    print(json.dumps(summary(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

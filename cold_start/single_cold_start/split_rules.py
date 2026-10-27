from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT.parent / "Database"
SPLIT_DIR = ROOT / "data" / "split" / "single"
DRUG_TABLE = ROOT / "data" / "features" / "drug_table.csv"

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2


def feature_capable_pairs(drug_table: pd.DataFrame) -> set[tuple[int, int]]:
    name_to_id = dict(zip(drug_table["drug_name"], drug_table["internal_id"]))
    ddi = pd.read_csv(DB_DIR / "DDI.csv", header=None, dtype=str)
    ddi.columns = ["drug1", "relation", "drug2"]
    pairs: set[tuple[int, int]] = set()
    for a, b in zip(ddi["drug1"], ddi["drug2"]):
        ia, ib = name_to_id.get(str(a).strip()), name_to_id.get(str(b).strip())
        if ia is None or ib is None or ia == ib:
            continue
        pairs.add((int(ia), int(ib)) if ia < ib else (int(ib), int(ia)))
    return pairs


def main(seed: int = 42, overwrite: bool = False) -> dict:
    drug_table = pd.read_csv(DRUG_TABLE, dtype=str).fillna("")
    drug_table["internal_id"] = drug_table["internal_id"].astype(int)
    capable = drug_table[drug_table["smiles"].str.strip().astype(bool)].copy()

    positives = feature_capable_pairs(drug_table)

    rng = np.random.RandomState(seed)
    ids = capable["internal_id"].to_numpy()
    shuffled = rng.permutation(ids)
    n = len(shuffled)
    n_tr = int(round(n * TRAIN_RATIO))
    n_va = int(round(n * VAL_RATIO))
    pools = {
        "train": set(shuffled[:n_tr].tolist()),
        "val": set(shuffled[n_tr:n_tr + n_va].tolist()),
        "test": set(shuffled[n_tr + n_va:].tolist()),
    }

    train_pos = sorted(p for p in positives
                       if p[0] in pools["train"] and p[1] in pools["train"])

    def cross_pos(pool_name: str) -> list[tuple[int, int]]:
        out = set()
        for a, b in positives:
            if (a in pools[pool_name] and b in pools["train"]) or \
               (b in pools[pool_name] and a in pools["train"]):
                out.add((a, b))
        return sorted(out)

    val_pos, test_pos = cross_pos("val"), cross_pos("test")

    if SPLIT_DIR.exists() and any(SPLIT_DIR.iterdir()) and not overwrite:
        raise FileExistsError(f"{SPLIT_DIR} already exists; use --overwrite")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for pool_name in ("train", "val", "test"):
        pd.DataFrame({"internal_id": sorted(pools[pool_name])}).to_csv(
            SPLIT_DIR / f"{pool_name}_drugs.csv", index=False)
    for split_name, pairs in (("train", train_pos), ("val", val_pos), ("test", test_pos)):
        pd.DataFrame(pairs, columns=["drug1_id", "drug2_id"]).to_csv(
            SPLIT_DIR / f"{split_name}_pairs.csv", index=False)
    pd.DataFrame(sorted(positives), columns=["drug1_id", "drug2_id"]).to_csv(
        SPLIT_DIR / "global_positive_pairs.csv", index=False)
    capable.to_csv(SPLIT_DIR / "all_drug_smiles.csv", index=False)

    summary = {
        "seed": seed,
        "mode": "single",
        "split_method": f"random_{TRAIN_RATIO}_{VAL_RATIO}_{TEST_RATIO}_pools",
        "drugs": {k: len(v) for k, v in pools.items()},
        "positive_pairs": {"train": len(train_pos), "val": len(val_pos), "test": len(test_pos)},
        "global_positive_pairs": len(positives),
    }
    (SPLIT_DIR / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(seed=args.seed, overwrite=args.overwrite)

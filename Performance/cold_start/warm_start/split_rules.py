from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "entity_split_data"
SPLIT_DIR = DATA_ROOT / "split" / "warm"
DRUG_TABLE = DATA_ROOT / "features" / "drug_table.csv"

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2


def find_database_dir() -> Path:
    candidates = [ROOT.parent / "database" / "Triples",
                  ROOT.parent / "database",
                  ROOT.parent / "Database" / "Triples",
                  ROOT.parent / "Database"]
    for directory in candidates:
        if (directory / "DDI.csv").exists():
            return directory
    raise FileNotFoundError(
        "DDI.csv not found; tried " + " | ".join(str(c) for c in candidates))


def feature_capable_pairs(drug_table: pd.DataFrame, db_dir: Path):
    name_to_id = dict(zip(drug_table["drug_name"].str.strip(),
                          drug_table["internal_id"].astype(int)))
    ddi = pd.read_csv(db_dir / "DDI.csv", header=None, dtype=str)
    ddi.columns = ["drug1", "relation", "drug2"]
    pairs: set[tuple[int, int]] = set()
    dropped = {"self_loop": 0, "no_smiles_endpoint": 0}
    for a, b in zip(ddi["drug1"], ddi["drug2"]):
        ia = name_to_id.get(str(a).strip())
        ib = name_to_id.get(str(b).strip())
        if ia is None or ib is None or ia == ib:
            dropped["self_loop" if ia == ib else "no_smiles_endpoint"] += 1
            continue
        pairs.add((int(ia), int(ib)) if ia < ib else (int(ib), int(ia)))
    return pairs, dropped


def main(seed: int = 42, overwrite: bool = False) -> dict:
    if not DRUG_TABLE.exists():
        raise FileNotFoundError(f"{DRUG_TABLE} missing; run data_pipeline/build_ids.py")
    drug_table = pd.read_csv(DRUG_TABLE, dtype=str).fillna("")
    drug_table["internal_id"] = drug_table["internal_id"].astype(int)
    capable = drug_table[drug_table["smiles"].str.strip().astype(bool)].copy()
    if len(capable) == 0:
        raise RuntimeError("no SMILES-capable drugs found")

    db_dir = find_database_dir()
    positives, dropped = feature_capable_pairs(drug_table, db_dir)
    if not positives:
        raise RuntimeError("no feature-capable positive pairs found")

    ordered = sorted(positives)
    rng = np.random.RandomState(seed)
    shuffled = [ordered[i] for i in rng.permutation(len(ordered))]
    n = len(shuffled)
    n_tr = int(round(n * TRAIN_RATIO))
    n_va = int(round(n * VAL_RATIO))
    split_pairs = {
        "train": sorted(shuffled[:n_tr]),
        "val": sorted(shuffled[n_tr:n_tr + n_va]),
        "test": sorted(shuffled[n_tr + n_va:]),
    }
    if set(split_pairs["train"]) & set(split_pairs["val"]) or \
            set(split_pairs["train"]) & set(split_pairs["test"]) or \
            set(split_pairs["val"]) & set(split_pairs["test"]):
        raise ValueError("pair-level splits overlap")

    all_drugs = sorted(int(i) for i in capable["internal_id"])
    seen_in_pairs = {d for pair in positives for d in pair}
    drugs_never_in_pairs = sorted(set(all_drugs) - seen_in_pairs)

    if SPLIT_DIR.exists() and any(SPLIT_DIR.iterdir()) and not overwrite:
        raise FileExistsError(f"{SPLIT_DIR} already exists; use --overwrite")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        pd.DataFrame({"internal_id": all_drugs}).to_csv(
            SPLIT_DIR / f"{split_name}_drugs.csv", index=False)
        pd.DataFrame(split_pairs[split_name],
                     columns=["drug1_id", "drug2_id"]).to_csv(
            SPLIT_DIR / f"{split_name}_pairs.csv", index=False)
    pd.DataFrame(sorted(positives), columns=["drug1_id", "drug2_id"]).to_csv(
        SPLIT_DIR / "global_positive_pairs.csv", index=False)
    capable.to_csv(SPLIT_DIR / "all_drug_smiles.csv", index=False)

    summary = {
        "seed": seed,
        "mode": "warm",
        "split_method": f"pair_level_{TRAIN_RATIO}_{VAL_RATIO}_{TEST_RATIO}_shared_drugs",
        "drug_policy": "no drug-level holdout; all SMILES-capable drugs are shared",
        "drugs": {"train": len(all_drugs), "val": len(all_drugs), "test": len(all_drugs)},
        "positive_pairs": {k: len(v) for k, v in split_pairs.items()},
        "global_positive_pairs": len(positives),
        "drugs_never_in_any_positive_pair": len(drugs_never_in_pairs),
        "kg_policy": {"combination": "train_pairs.csv only",
                      "interaction": "DTI of every drug"},
        "negative_policy": "sampled from the shared all-drug pool, excluding train/val/test positives",
        "skipped_ddi": dropped,
        "database_dir": str(db_dir),
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

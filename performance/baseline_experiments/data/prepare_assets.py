"""Assemble the drug-level assets required by the baseline experiment.

Why this script exists
----------------------
The pair-level feature matrices that were used for the reported run are large
(``{split}.npy`` holds 3840 features per drug pair) and are therefore not
distributed with the repository.  Everything they are built from *is* small and
*is* distributed: the drug-level SMILES embedding matrix, the target sequence
embedding matrix, the drug-target index and the two identifier tables.  They
live in the sibling cold-start workspace, which is part of this repository.

This script copies those drug-level assets into ``data/split_data/`` and
derives the small per-split label / weight arrays from the committed pair
metadata.  After it has run, ``python main.py check`` validates the split and
``python main.py train`` can re-create the fold features locally.

The large pair-level matrices are optional.  ``--rebuild-features`` regenerates
them from the drug-level assets (this is the step that needs the multi-GB
scratch space and a TransE embedding of the full knowledge graph).

Usage
-----
    python data/prepare_assets.py --status
    python data/prepare_assets.py --apply
    python data/prepare_assets.py --apply --rebuild-features
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
SPLIT_DATA: Path = MODULE_ROOT / "data" / "split_data"

SOURCE_DIR: Path = REPO_ROOT / "cold_start" / "entity_split_data" / "features"
KG_SOURCE_DIR: Path = REPO_ROOT / "cold_start" / "kg_embedding" / "warm" / "train"

# Drug-level assets: (source name, destination name, kind)
DRUG_LEVEL_ASSETS: tuple[tuple[str, str, str], ...] = (
    ("drug_table.csv", "drug_table.csv", "table"),
    ("target_table.csv", "target_table.csv", "table"),
    ("drug_ids.npy", "drug_ids.npy", "array"),
    ("target_ids.npy", "target_ids.npy", "array"),
    ("drug_target_index.npy", "drug_target_index.npy", "array"),
    ("drug_smiles_features.npy", "drug_smiles_features.npy", "array"),
    ("target_features.npy", "target_features.npy", "array"),
)

# Optional: train-only knowledge-graph embedding, used by negative_rebuild.py
KG_ASSETS: tuple[tuple[str, str], ...] = (
    ("drug_kg_features.npy", "drug_kg_features.npy"),
)

SPLITS: tuple[str, ...] = ("train", "val", "test")
SMILES_DIM: int = 768
TARGET_DIM: int = 1024


def _source_available() -> bool:
    return SOURCE_DIR.is_dir()


def status() -> dict:
    report: dict = {"source_dir": str(SOURCE_DIR),
                    "source_available": _source_available(),
                    "kg_source_dir": str(KG_SOURCE_DIR),
                    "assets": [], "splits": [], "missing": []}
    if not _source_available():
        report["missing"].append(str(SOURCE_DIR))
        return report

    for name, dest, kind in DRUG_LEVEL_ASSETS:
        src = SOURCE_DIR / name
        dst = SPLIT_DATA / dest
        entry = {"name": dest, "kind": kind, "source_exists": src.exists(),
                 "present": dst.exists()}
        if src.exists():
            entry["bytes"] = src.stat().st_size
        if kind == "array" and src.exists():
            array = np.load(src, mmap_mode="r")
            entry["shape"] = list(array.shape)
        report["assets"].append(entry)
        if not src.exists():
            report["missing"].append(str(src))

    for name, dest in KG_ASSETS:
        src = KG_SOURCE_DIR / name
        dst = SPLIT_DATA / dest
        entry = {"name": dest, "kind": "kg", "source_exists": src.exists(),
                 "present": dst.exists()}
        if src.exists():
            entry["bytes"] = src.stat().st_size
        report["assets"].append(entry)
        if not src.exists():
            report["missing"].append(str(src))

    for split in SPLITS:
        meta = SPLIT_DATA / f"{split}_pair_meta.csv"
        labels = SPLIT_DATA / f"{split}_labels.npy"
        entry = {"split": split, "pair_meta": meta.exists(), "labels": labels.exists()}
        if meta.exists():
            frame = pd.read_csv(meta)
            entry["pairs"] = int(len(frame))
            entry["positives"] = int(frame["label"].sum())
            entry["negatives"] = int((frame["label"] == 0).sum())
        report["splits"].append(entry)
    return report


def apply() -> dict:
    if not _source_available():
        raise FileNotFoundError(
            f"drug-level asset directory not found: {SOURCE_DIR}\n"
            "It is part of this repository under cold_start/entity_split_data/features/.")
    SPLIT_DATA.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name, dest, _kind in DRUG_LEVEL_ASSETS:
        src = SOURCE_DIR / name
        if not src.exists():
            continue
        dst = SPLIT_DATA / dest
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        copied.append(dest)
    for name, dest in KG_ASSETS:
        src = KG_SOURCE_DIR / name
        if not src.exists():
            continue
        dst = SPLIT_DATA / dest
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        copied.append(dest)

    # Derive the per-split label / weight arrays from the committed pair metadata.
    derived: list[str] = []
    for split in SPLITS:
        meta = SPLIT_DATA / f"{split}_pair_meta.csv"
        if not meta.exists():
            continue
        frame = pd.read_csv(meta)
        labels = frame["label"].to_numpy(dtype=np.int8)
        np.save(SPLIT_DATA / f"{split}_labels.npy", labels)
        np.save(SPLIT_DATA / f"{split}_sample_weight.npy",
                np.ones(len(frame), dtype=np.float32))
        derived += [f"{split}_labels.npy", f"{split}_sample_weight.npy"]

    summary = refresh_split_summary()
    return {"copied": copied, "derived": derived, "summary": summary}


def refresh_split_summary() -> dict:
    """Regenerate data/split_data/warm_split_summary.json from the committed
    pair metadata.

    The summary must always describe the data that is actually used.  A stale
    copy of the cold-start summary is worse than no summary at all, because it
    reports the pair counts before the knowledge-graph availability filter was
    applied.
    """
    summary_path = SPLIT_DATA / "warm_split_summary.json"
    kg_summary_path = KG_SOURCE_DIR / "kg_summary.json"
    kg_summary = {}
    if kg_summary_path.exists():
        kg_summary = json.loads(kg_summary_path.read_text(encoding="utf-8"))

    drugs = set()
    positives, negatives, pairs = {}, {}, {}
    for split in SPLITS:
        meta = SPLIT_DATA / f"{split}_pair_meta.csv"
        if not meta.exists():
            continue
        frame = pd.read_csv(meta)
        positives[split] = int(frame["label"].sum())
        negatives[split] = int((frame["label"] == 0).sum())
        pairs[split] = int(len(frame))
        drugs |= set(frame["drug1_id"].tolist()) | set(frame["drug2_id"].tolist())

    payload = {
        "seed": 42,
        "mode": "warm",
        "split_method": "pair_level_0.6_0.2_0.2_shared_drugs",
        "drug_policy": ("no drug-level holdout: every evaluated pair has two "
                        "drugs that are seen during training; the strict "
                        "unseen-drug setting is the separate cold-start experiment"),
        "drugs": int(len(drugs)),
        "positive_pairs": positives,
        "negative_pairs": negatives,
        "pairs": pairs,
        "global_positive_pairs": int(sum(positives.values())),
        "global_negative_pairs": int(sum(negatives.values())),
        "kg_policy": {
            "combination": "training positives of the fold only",
            "interaction": "drug-target interactions of every drug",
        },
        "negative_policy": ("one negative per positive, stratified by the number "
                            "of drug-target interactions, sampled without "
                            "replacement and disjoint across the splits"),
        "kg_embedding": {
            "source": os.path.relpath(KG_SOURCE_DIR, REPO_ROOT).replace("\\", "/"),
            "combination_edges": kg_summary.get("graph_combination"),
            "interaction_edges": kg_summary.get("graph_interaction"),
            "held_out_pairs_in_graph": kg_summary.get("held_out_pairs_in_graph"),
        },
        "note": ("counts are the numbers actually present in the committed pair "
                 "metadata, i.e. after the knowledge-graph availability filter"),
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    return payload


def check_alignment() -> dict:
    drugs = pd.read_csv(SPLIT_DATA / "drug_table.csv")
    n_drugs = int(drugs["internal_id"].max()) + 1
    smiles = np.load(SPLIT_DATA / "drug_smiles_features.npy", mmap_mode="r")
    targets = np.load(SPLIT_DATA / "target_features.npy", mmap_mode="r")
    n_targets = int(len(np.load(SPLIT_DATA / "target_ids.npy")))
    index = np.load(SPLIT_DATA / "drug_target_index.npy")
    checks = {
        "smiles_shape": list(smiles.shape) == [n_drugs, SMILES_DIM],
        "target_shape": list(targets.shape) == [n_targets + 1, TARGET_DIM],
        "dti_fields": index.dtype.names == ("drug_id", "target_index"),
        "dti_drug_range": bool(index["drug_id"].max() < n_drugs),
        "dti_target_range": bool(index["target_index"].max() <= n_targets),
    }
    checks["passed"] = all(checks.values())
    return {"n_drugs": n_drugs, "n_targets": n_targets, "checks": checks}


def rebuild_features() -> dict:
    """Regenerate the large pair-level matrices (optional, multi-GB scratch)."""
    script = MODULE_ROOT / "data" / "negative_rebuild.py"
    if not script.exists():
        raise FileNotFoundError(script)
    import subprocess
    command = [sys.executable, str(script), "--apply"]
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=str(MODULE_ROOT / "data"))
    return {"returncode": completed.returncode, "command": command}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true",
                        help="report which assets are present (default action)")
    parser.add_argument("--apply", action="store_true",
                        help="copy the drug-level assets and derive labels/weights")
    parser.add_argument("--check", action="store_true",
                        help="verify row alignment of the assembled assets")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="also regenerate the large pair-level feature matrices")
    args = parser.parse_args()

    if args.apply:
        result = apply()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(check_alignment(), indent=2, ensure_ascii=False))
    if args.check:
        print(json.dumps(check_alignment(), indent=2, ensure_ascii=False))
    if args.rebuild_features:
        print(json.dumps(rebuild_features(), indent=2, ensure_ascii=False))
    if not (args.apply or args.check or args.rebuild_features):
        print(json.dumps(status(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

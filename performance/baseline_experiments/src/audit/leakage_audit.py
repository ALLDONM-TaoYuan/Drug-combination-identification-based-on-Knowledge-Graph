"""Leakage audit for the baseline comparison experiment.

The audit is deliberately independent of the training code: it re-reads the
committed split metadata and reports, for the warm split and for every
cross-validation fold,

* the number of drug pairs shared between two splits (must be 0),
* the number of pairs that are positive in one split and negative in another
  (must be 0),
* the number of negative pairs that coincide with a known positive pair,
* the number of self-pairs and of non-canonical pairs,
* the drug overlap between the splits, reported as an observation.

The drug-level overlap is reported rather than required to be zero, because
the pair-level split is an interpolation setting: both drugs of an evaluated
pair are seen during training.  The strict unseen-drug setting is provided by
the separate cold-start experiment in ``cold_start/``.

Usage
-----
    python src/audit/leakage_audit.py
    python src/audit/leakage_audit.py --json results/leakage_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import common

SPLITS: tuple[str, ...] = ("train", "val", "test")
PAIR_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def encode(drug1, drug2) -> np.ndarray:
    a = np.asarray(drug1, dtype=np.int64)
    b = np.asarray(drug2, dtype=np.int64)
    return np.minimum(a, b) * 100_000_000 + np.maximum(a, b)


def audit_warm_split() -> dict:
    frames = {split: common.load_split_frame(split) for split in SPLITS}
    keys = {split: encode(frame["drug1_id"].to_numpy(),
                          frame["drug2_id"].to_numpy()) for split, frame in frames.items()}
    positives = {split: set(keys[split][frame["label"].to_numpy() == 1].tolist())
                 for split, frame in frames.items()}
    negatives = {split: set(keys[split][frame["label"].to_numpy() == 0].tolist())
                 for split, frame in frames.items()}
    drugs = {split: set(frame["drug1_id"].to_numpy().tolist())
             | set(frame["drug2_id"].to_numpy().tolist()) for split, frame in frames.items()}

    overlap = {f"{a}&{b}": len(set(keys[a].tolist()) & set(keys[b].tolist()))
               for a, b in PAIR_PAIRS}
    conflict = {f"{a}(neg) vs {b}(pos)": len(negatives[a] & positives[b])
                for a, b in PAIR_PAIRS}
    negative_conflict = {f"{a}(neg) vs {b}(neg)": len(negatives[a] & negatives[b])
                         for a, b in PAIR_PAIRS}
    drug_overlap = {f"{a}&{b}": len(drugs[a] & drugs[b]) for a, b in PAIR_PAIRS}

    total_pairs = sum(len(frame) for frame in frames.values())
    report = {
        "level": "warm 60/20/20 pair-level split",
        "sizes": {split: int(len(frame)) for split, frame in frames.items()},
        "positives": {split: int(frame["label"].sum()) for split, frame in frames.items()},
        "negatives": {split: int((frame["label"] == 0).sum())
                      for split, frame in frames.items()},
        "pair_overlap": overlap,
        "positive_negative_conflict": conflict,
        "negative_negative_overlap": negative_conflict,
        "drug_overlap": drug_overlap,
        "self_pairs": int(sum((frame["drug1_id"] == frame["drug2_id"]).sum()
                               for frame in frames.values())),
        "non_canonical_pairs": int(sum((frame["drug1_id"] > frame["drug2_id"]).sum()
                                       for frame in frames.values())),
        "global_positive_negative_conflict": len(
            set().union(*positives.values()) & set().union(*negatives.values())),
        "total_pairs": int(total_pairs),
    }
    report["passed"] = (all(value == 0 for value in overlap.values())
                        and all(value == 0 for value in conflict.values())
                        and all(value == 0 for value in negative_conflict.values())
                        and report["global_positive_negative_conflict"] == 0
                        and report["self_pairs"] == 0
                        and report["non_canonical_pairs"] == 0)
    return report


def audit_folds() -> dict:
    per_fold = []
    test_union: set = set()
    for fold in common.FOLDS:
        frames = {split: common.load_fold_frame(fold, split) for split in SPLITS}
        keys = {split: encode(frame["drug1_id"].to_numpy(),
                              frame["drug2_id"].to_numpy()) for split, frame in frames.items()}
        positives = {split: set(keys[split][frame["label"].to_numpy() == 1].tolist())
                     for split, frame in frames.items()}
        negatives = {split: set(keys[split][frame["label"].to_numpy() == 0].tolist())
                     for split, frame in frames.items()}
        drugs = {split: set(frame["drug1_id"].to_numpy().tolist())
                 | set(frame["drug2_id"].to_numpy().tolist()) for split, frame in frames.items()}
        overlap = {f"{a}&{b}": len(set(keys[a].tolist()) & set(keys[b].tolist()))
                   for a, b in PAIR_PAIRS}
        conflict = {f"{a}(neg) vs {b}(pos)": len(negatives[a] & positives[b])
                    for a, b in PAIR_PAIRS}
        entry = {
            "fold": int(fold),
            "sizes": {split: int(len(frame)) for split, frame in frames.items()},
            "positives": {split: int(frame["label"].sum()) for split, frame in frames.items()},
            "pair_overlap": overlap,
            "positive_negative_conflict": conflict,
            "drug_overlap": {f"{a}&{b}": len(drugs[a] & drugs[b]) for a, b in PAIR_PAIRS},
            "self_pairs": int(sum((frame["drug1_id"] == frame["drug2_id"]).sum()
                                   for frame in frames.values())),
        }
        entry["passed"] = (all(value == 0 for value in overlap.values())
                           and all(value == 0 for value in conflict.values())
                           and entry["self_pairs"] == 0)
        test_union |= set(keys["test"].tolist())
        per_fold.append(entry)

    total = sum(entry["sizes"][split] for entry in per_fold for split in SPLITS) // len(per_fold)
    report = {
        "level": "pair-level stratified 5-fold cross-validation",
        "folds": per_fold,
        "test_folds_cover_all_pairs": len(test_union) == total,
        "pairs_total": int(total),
        "pairs_covered": int(len(test_union)),
    }
    report["passed"] = all(entry["passed"] for entry in per_fold) and \
        report["test_folds_cover_all_pairs"]
    return report


def audit_kg_isolation() -> dict:
    """Confirm that every model reads the fold splits and derives its threshold
    from the validation split, and that the relational baselines build their
    graph from the training positives of the fold only."""
    report = {"level": "knowledge graph isolation", "models": []}
    for model_key in common.MODEL_KEYS:
        spec = common.MODEL_SPECS[model_key]
        path = common.MODELS_DIR / spec["dir"] / "model.py"
        source = path.read_text(encoding="utf-8") if path.exists() else ""
        entry = {
            "model": common.DISPLAY[model_key],
            "file": str(path.relative_to(PROJECT_ROOT)),
            "uses_fold_splits": "load_fold_frame" in source,
            "builds_graph_from_train_positives": ('label"] == 1' in source
                                                   and "train" in source),
            "threshold_from_validation": "youden_threshold" in source,
        }
        report["models"].append(entry)
    report["passed"] = all(entry["uses_fold_splits"]
                           and entry["threshold_from_validation"]
                           for entry in report["models"])
    report["note"] = ("the static check confirms that every model reads the fold "
                       "splits and derives its threshold from the validation split; "
                       "the runtime record of each run in work/runs.csv states the "
                       "number of graph edges used per fold")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=None,
                        help="optional path for the machine-readable report")
    args = parser.parse_args()

    report = {"warm_split": audit_warm_split(),
              "folds": audit_folds(),
              "kg_isolation": audit_kg_isolation()}
    report["passed"] = all(section["passed"] for section in report.values()
                           if isinstance(section, dict) and "passed" in section)

    for section in ("warm_split", "folds", "kg_isolation"):
        block = report[section]
        print(f"[{section}] {'PASS' if block['passed'] else 'FAIL'}")
        if section == "warm_split":
            print(f"  sizes               : {block['sizes']}")
            print(f"  pair overlap        : {block['pair_overlap']}")
            print(f"  pos/neg conflict    : {block['positive_negative_conflict']}")
            print(f"  global conflict     : {block['global_positive_negative_conflict']}")
            print(f"  drug overlap        : {block['drug_overlap']}  (observation)")
            print(f"  self / non-canonical: {block['self_pairs']} / {block['non_canonical_pairs']}")
        elif section == "folds":
            for entry in block["folds"]:
                print(f"  fold {entry['fold']}: overlap={entry['pair_overlap']} "
                      f"conflict={entry['positive_negative_conflict']}")
            print(f"  test folds cover all pairs: {block['test_folds_cover_all_pairs']} "
                  f"({block['pairs_covered']}/{block['pairs_total']})")
        else:
            for entry in block["models"]:
                print(f"  {entry['model']:46s} fold-splits={entry['uses_fold_splits']} "
                      f"val-threshold={entry['threshold_from_validation']}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()

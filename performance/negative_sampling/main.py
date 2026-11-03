"""Driver script of the negative-sampling strategy experiment."""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import settings as S
import train as T

STRATEGY_MODULES = {
    "euclidean": "euclidean_distance",
    "dti_matched": "frequency_matching",
    "uniform": "uniform_distribution",
}


def upstream_files() -> list[Path]:
    files: list[Path] = [S.SPLIT_DIR / "global_positive_pairs.csv",
                         S.SPLIT_DIR / "split_summary.json"]
    for split in S.SPLITS:
        files += [S.SPLIT_DIR / f"{split}_drugs.csv", S.SPLIT_DIR / f"{split}_pairs.csv"]
    for name in ("drug_ids.npy", "drug_smiles_features.npy", "target_features.npy",
                 "target_ids.npy", "drug_target_index.npy"):
        files.append(S.FEATURE_DIR / name)
    files += [S.KG_DIR / "drug_kg_features.npy", S.KG_DIR / "kg_embedding_status.csv"]
    for split in S.SPLITS:
        for suffix in (".npy", "_labels.npy", "_sample_weight.npy", "_pair_meta.csv"):
            files.append(S.FROZEN_DIR / f"{split}{suffix}")
    return files


def check_upstream() -> None:
    missing = [path for path in upstream_files() if not path.exists()]
    if missing:
        raise SystemExit("missing upstream files:\n" + "\n".join(map(str, missing)))
    print(f"all {len(upstream_files())} upstream files are present")


def load_upstream() -> dict:
    sys.path.insert(0, str(S.UPSTREAM_DIR))
    from smiles import load_smiles_features
    from target import load_target_features
    import build_fold_features as bff

    drug_ids, smiles = load_smiles_features(S.FEATURE_DIR)
    _, targets = load_target_features(S.FEATURE_DIR)
    kg, has_kg = bff.load_kg_features(S.KG_DIR, len(drug_ids))
    targets_of, zero_target = bff.load_target_map(S.FEATURE_DIR)

    target_means = np.zeros((len(drug_ids), targets.shape[1]), dtype=np.float32)
    for drug in range(len(drug_ids)):
        rows = targets_of.get(drug) or [zero_target]
        target_means[drug] = targets[rows].mean(axis=0)
    blocks = np.concatenate([kg, smiles, target_means], axis=1)

    positives: dict[str, list[tuple[int, int]]] = {}
    for split in S.SPLITS:
        meta = pd.read_csv(S.FROZEN_DIR / f"{split}_pair_meta.csv")
        pairs = meta.loc[meta["source"] == "positive", ["drug1_id", "drug2_id"]]
        positives[split] = sorted((min(int(a), int(b)), max(int(a), int(b)))
                                  for a, b in zip(pairs["drug1_id"], pairs["drug2_id"]))

    forbidden: set[tuple[int, int]] = set()
    frame = pd.read_csv(S.SPLIT_DIR / "global_positive_pairs.csv")
    for a, b in zip(frame["drug1_id"], frame["drug2_id"]):
        forbidden.add((min(int(a), int(b)), max(int(a), int(b))))
    for pairs in positives.values():
        forbidden.update(pairs)

    frozen_negatives: set[tuple[int, int]] = set()
    for split in ("val", "test"):
        part = pd.read_csv(S.FROZEN_DIR / f"{split}_pair_meta.csv")
        sel = part.loc[part["source"] == "negative", ["drug1_id", "drug2_id"]]
        frozen_negatives.update((min(int(a), int(b)), max(int(a), int(b)))
                                for a, b in zip(sel["drug1_id"], sel["drug2_id"]))

    return {
        "bff": bff,
        "blocks": blocks,
        "positives": positives,
        "forbidden": forbidden,
        "sample_forbidden": forbidden | frozen_negatives,
        "frozen_negatives": frozen_negatives,
        "pools": {split: set(pd.read_csv(S.SPLIT_DIR / f"{split}_drugs.csv")["internal_id"].astype(int))
                  for split in S.SPLITS},
        "usable": {int(i) for i, ok in enumerate(has_kg) if ok},
        "target_counts": {drug: len(values) for drug, values in targets_of.items()},
        "n_drugs": len(drug_ids),
    }


def candidate_pairs(left_pool: set[int], right_pool: set[int],
                    forbidden: set[tuple[int, int]], n_drugs: int) -> np.ndarray:
    left = np.asarray(sorted(left_pool), dtype=np.int64)
    right = np.asarray(sorted(right_pool), dtype=np.int64)
    grid_left = np.repeat(left, right.size)
    grid_right = np.tile(right, left.size)
    keep = grid_left != grid_right
    low = np.minimum(grid_left[keep], grid_right[keep])
    high = np.maximum(grid_left[keep], grid_right[keep])
    candidates = np.unique(np.stack([low, high], axis=1), axis=0)
    if forbidden:
        blocked = np.asarray([min(a, b) * n_drugs + max(a, b) for a, b in forbidden],
                             dtype=np.int64)
        candidates = candidates[~np.isin(candidates[:, 0] * n_drugs + candidates[:, 1], blocked)]
    return candidates


def repo_relative(path: Path) -> str:
    """Return the path relative to the repository root, with forward slashes."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(S.REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(src: Path, dst: Path) -> tuple[str, str]:
    """Materialise src at dst and return the transfer mode and the digest."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        mode = "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        mode = "copy"
    return mode, sha256(dst)


def write_block(block: dict, ctx: dict, negatives: np.ndarray, info: dict) -> dict:
    out_dir = S.block_features(block["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.npy"
    if train_path.exists():
        train_path.unlink()

    positives = ctx["positives"]["train"]
    rows = np.asarray(sorted(positives) + [tuple(pair) for pair in negatives.tolist()],
                      dtype=np.int64)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int8)
    features = np.concatenate([ctx["blocks"][rows[:, 0]], ctx["blocks"][rows[:, 1]]],
                              axis=1).astype(np.float32, copy=False)
    np.save(train_path, features)
    np.save(out_dir / "train_labels.npy", labels)
    np.save(out_dir / "train_sample_weight.npy", np.ones(len(rows), dtype=np.float32))
    pd.DataFrame({"drug1_id": rows[:, 0], "drug2_id": rows[:, 1], "label": labels,
                  "source": ["positive"] * len(positives) + ["negative"] * len(negatives)}
                 ).to_csv(out_dir / "train_pair_meta.csv", index=False)

    frozen: dict = {}
    for split in ("val", "test"):
        for suffix in (".npy", "_labels.npy", "_sample_weight.npy", "_pair_meta.csv"):
            name = f"{split}{suffix}"
            source = S.FROZEN_DIR / name
            mode, digest = link_or_copy(source, out_dir / name)
            frozen[name] = {"source": repo_relative(source), "mode": mode, "sha256": digest}

    manifest = {
        "block": block["name"],
        "strategy": block["strategy"],
        "title": block["title"],
        "mode": S.MODE,
        "sampling_seed": S.SAMPLING_SEED,
        "negative_ratio": S.NEGATIVE_RATIO,
        "positive_pairs": len(positives),
        "negative_pairs": int(len(negatives)),
        "feature_dim": int(features.shape[1]),
        "train_shape": list(features.shape),
        "path_base": "repository_root",
        "frozen_val_test": frozen,
        "sampling": info,
        "train_sha256": {
            "train.npy": sha256(out_dir / "train.npy"),
            "train_labels.npy": sha256(out_dir / "train_labels.npy"),
            "train_pair_meta.csv": sha256(out_dir / "train_pair_meta.csv"),
        },
    }
    (out_dir / "sampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def check_block(block: dict, ctx: dict) -> dict:
    features_dir = S.block_features(block["name"])
    meta = pd.read_csv(features_dir / "train_pair_meta.csv")
    pairs = list(zip(meta["drug1_id"].astype(int), meta["drug2_id"].astype(int)))
    labels = meta["label"].to_numpy()
    positives = [(min(a, b), max(a, b)) for (a, b), y in zip(pairs, labels) if y == 1]
    negatives = [(min(a, b), max(a, b)) for (a, b), y in zip(pairs, labels) if y == 0]
    negative_set = set(negatives)
    train_pool = set(ctx["pools"]["train"])
    train_frame = pd.read_csv(features_dir / "train_pair_meta.csv")
    features = np.load(features_dir / "train.npy", mmap_mode="r")
    return {
        "block": block["name"],
        "mode": S.MODE,
        "positives": len(positives),
        "negatives": len(negatives),
        "ratio_1_to_1": len(positives) == len(negatives),
        "self_pairs": sum(1 for a, b in negatives if a == b),
        "duplicate_negatives": len(negatives) - len(negative_set),
        "overlap_with_known_positives": len(negative_set & ctx["forbidden"]),
        "negative_outside_train_pool": sum(
            1 for a, b in negatives if a not in train_pool or b not in train_pool),
        "duplicate_with_val_test_negatives": len(negative_set & ctx["frozen_negatives"]),
        "feature_shape": list(features.shape),
        "finite_features": bool(np.isfinite(features).all()),
        "label_positive_fraction": float(train_frame["label"].mean()),
    }


def prepare(block_names: list[str] | None = None) -> None:
    block_names = block_names or [b["name"] for b in S.BLOCKS]
    ctx = load_upstream()
    left = ctx["pools"]["train"] & ctx["usable"]
    candidates = candidate_pairs(left, left, ctx["sample_forbidden"], ctx["n_drugs"])
    needed = int(round(len(ctx["positives"]["train"]) * S.NEGATIVE_RATIO))
    print(f"candidate pairs {len(candidates)}, negatives required {needed}")

    for name in block_names:
        block = S.get_block(name)
        rng = np.random.RandomState(S.SAMPLING_SEED)
        started = time.perf_counter()
        negatives, info = importlib.import_module(
            STRATEGY_MODULES[block["strategy"]]).sample(ctx, candidates, needed, rng)
        manifest = write_block(block, ctx, negatives, info)
        checks = check_block(block, ctx)
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        manifest["checks"] = checks
        (S.block_features(name) / "sampling_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{name}] positives/negatives {checks['positives']}/{checks['negatives']}, "
              f"conflict {checks['overlap_with_known_positives']}, "
              f"duplicates {checks['duplicate_negatives']}")


def train(block_names: list[str], seeds: list[int], force: bool = False) -> None:
    for name in block_names:
        for seed in seeds:
            T.run_one(name, seed, force=force)
        T.summarize_block(name)


def summarize() -> None:
    frames = []
    for block in S.BLOCKS:
        path = S.block_results(block["name"]) / "all_seeds_metrics.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("no results to summarise")

    data = pd.concat(frames, ignore_index=True)
    rows = []
    for block in S.BLOCKS:
        part = data[(data["block"] == block["name"]) & (data["split"] == "test")]
        if part.empty:
            continue
        row = {"block": block["name"], "seeds": int(part["seed"].nunique())}
        for metric in S.METRICS:
            row[f"{metric}_mean"] = round(float(part[metric].mean()), 6)
            row[f"{metric}_std"] = round(float(part[metric].std(ddof=1)), 6)
        rows.append(row)

    table = pd.DataFrame(rows)
    S.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(S.OUTPUT_ROOT / "comparison_test_metrics_single.csv", index=False)
    print(table.round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "prepare", "train", "summarize", "all"))
    parser.add_argument("--blocks", nargs="+", default=None, choices=[b["name"] for b in S.BLOCKS])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(S.SEEDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    block_names = args.blocks or [b["name"] for b in S.BLOCKS]

    if args.command in ("status", "all"):
        check_upstream()
    if args.command in ("prepare", "all"):
        prepare(block_names)
    if args.command in ("train", "all"):
        train(block_names, args.seeds, args.force)
    if args.command in ("summarize", "all"):
        summarize()


if __name__ == "__main__":
    main()

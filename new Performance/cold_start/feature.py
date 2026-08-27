"""Prepare KGP-DC pair indexes and assemble model features.

Warm pairs use KG + SMILES + Target. Strict-cold pairs use SMILES + Target.
Targets are always expanded as a Cartesian product; mean-pooling is not used.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm

from smiles import load_smiles_features
from target import load_target_features


ROOT = Path(__file__).parent
COMBO_DTYPE = np.dtype([
    ("pair_index", np.int32),
    ("drug1_idx", np.int32),
    ("drug2_idx", np.int32),
    ("target1_idx", np.int32),
    ("target2_idx", np.int32),
    ("label", np.int8),
    ("sample_weight", np.float32),
])


def canonical_pair(drug1: int, drug2: int) -> tuple[int, int]:
    return (drug1, drug2) if drug1 < drug2 else (drug2, drug1)


def parse_boolean_series(values: pd.Series, name: str) -> np.ndarray:
    """Parse a persisted boolean column without treating 'False' as truthy."""
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"Invalid boolean values in {name}: {unknown}")
    return normalized.map(mapping).to_numpy(dtype=bool)


def resolve_kg_assets(root: Path | str = ROOT) -> tuple[Path, str]:
    """Prefer rebuilt KG files, then fall back to pretrained features."""
    root = Path(root)
    strict_dir = root / "KG"
    strict_files = [strict_dir / name for name in
                    ("drug_kg_features.npy", "kg_embedding_status.csv", "kg_summary.json")]
    if all(path.exists() for path in strict_files):
        return strict_dir, "strict"
    if any(path.exists() for path in strict_files):
        raise FileNotFoundError("KG root contains an incomplete rebuilt feature set")
    pretrained_dir = strict_dir / "pretrained"
    pretrained_files = [pretrained_dir / name for name in
                        ("drug_kg_features.npy", "kg_embedding_status.csv",
                         "kg_summary.json", "reuse_policy.json")]
    if all(path.exists() for path in pretrained_files):
        return pretrained_dir, "pretrained"
    if any(path.exists() for path in pretrained_files):
        raise FileNotFoundError("KG/pretrained contains an incomplete reusable feature set")
    raise FileNotFoundError("No complete strict or reusable pretrained KG feature set was found")


def load_usable_kg_status(root: Path | str = ROOT) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Return KG status after restricting reusable legacy vectors to train drugs."""
    root = Path(root)
    kg_dir, source = resolve_kg_assets(root)
    status = pd.read_csv(kg_dir / "kg_embedding_status.csv")
    if not {"internal_id", "has_kg_relation"}.issubset(status.columns):
        raise ValueError("KG status is missing internal_id/has_kg_relation")
    usable = parse_boolean_series(status["has_kg_relation"], "has_kg_relation")
    if source == "pretrained":
        train_ids = set(pd.read_csv(root / "data" / "split" / "train_drugs.csv")
                        ["internal_id"].astype(int))
        usable &= status["internal_id"].astype(int).isin(train_ids).to_numpy()
    return status, usable, source


def generate_negative_pairs(root: Path | str = ROOT, seed: int = 42,
                            overwrite: bool = False) -> dict:
    """Generate one random negative for every positive inside each split."""
    root = Path(root)
    split_dir = root / "data" / "split"
    output_dir = root / "data" / "labeled_pairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any((output_dir / f"{s}_labeled_pairs.csv").exists()
                             for s in ("train", "val", "test")):
        raise FileExistsError("Labeled pairs already exist; add --overwrite to replace them")

    all_positive: set[tuple[int, int]] = set()
    positives: dict[str, list[tuple[int, int]]] = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(split_dir / f"{split}_pairs.csv")
        pairs = sorted({canonical_pair(int(a), int(b))
                        for a, b in zip(frame["drug1_id"], frame["drug2_id"])
                        if int(a) != int(b)})
        positives[split] = pairs
        all_positive.update(pairs)

    drug_files = {
        "train": "train_drugs.csv",
        "val": "val_cold_drugs.csv",
        "test": "test_cold_drugs.csv",
    }
    unresolved_path = root / "data" / "features" / "target_mapping_unresolved.csv"
    unresolved = set()
    if unresolved_path.exists():
        unresolved = set(pd.read_csv(unresolved_path)["internal_id"].astype(int))
    rng = np.random.RandomState(seed)
    summary = {"seed": seed, "strategy": "random_uniform_1_to_1",
               "excluded_unresolved_target_mapping": sorted(unresolved), "splits": {}}
    used_negatives: set[tuple[int, int]] = set()
    for split, drug_file in drug_files.items():
        pool = np.sort(pd.read_csv(split_dir / drug_file)["internal_id"].astype(int).unique())
        pool = pool[~np.isin(pool, list(unresolved))]
        if len(pool) < 2:
            raise RuntimeError(f"Not enough eligible drugs in {split} pool")
        required = len(positives[split])
        negatives: set[tuple[int, int]] = set()
        attempts = 0
        max_attempts = max(100_000, required * 1000)
        while len(negatives) < required and attempts < max_attempts:
            a, b = rng.choice(pool, size=2, replace=False)
            pair = canonical_pair(int(a), int(b))
            if pair not in all_positive and pair not in used_negatives:
                negatives.add(pair)
            attempts += 1
        if len(negatives) != required:
            raise RuntimeError(f"Unable to sample enough negatives for {split}")
        used_negatives.update(negatives)

        rows = [(a, b, 1, split, "positive") for a, b in positives[split]]
        rows += [(a, b, 0, split, "random_uniform_1_to_1") for a, b in sorted(negatives)]
        labeled = pd.DataFrame(
            rows,
            columns=["drug1_id", "drug2_id", "label", "split", "sample_source"],
        )
        labeled.insert(0, "pair_id", np.arange(len(labeled), dtype=np.int32))
        labeled.to_csv(output_dir / f"{split}_labeled_pairs.csv", index=False)
        summary["splits"][split] = {"positive": required, "negative": required,
                                    "drug_pool": int(len(pool)), "attempts": attempts}

    with open(output_dir / "negative_sampling_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def canonical_text_pair(drug1: str, drug2: str) -> tuple[str, str]:
    """Canonicalize string drug IDs without changing the legacy split order."""
    return (drug1, drug2) if drug1 < drug2 else (drug2, drug1)


def read_single_cold_positive_pairs(
    root: Path | str = ROOT,
) -> set[tuple[str, str]]:
    """Collect the unique known positives used by the single-cold split."""
    source_dir = Path(root) / "data" / "split"
    frames = []
    for name in ("train_pairs.csv", "val_pairs.csv", "test_pairs.csv", "mixed_pairs.csv"):
        frame = pd.read_csv(
            source_dir / name,
            dtype={"drug1_id": str, "drug2_id": str},
        )
        frames.append(frame[["drug1_id", "drug2_id"]])

    positives: set[tuple[str, str]] = set()
    for frame in frames:
        positives.update(
            canonical_text_pair(str(drug1), str(drug2))
            for drug1, drug2 in frame.itertuples(index=False)
        )
    return positives


def sample_single_cold_negatives(
    left: set[str],
    right: set[str],
    count: int,
    positives: set[tuple[str, str]],
    rng: np.random.RandomState,
) -> list[tuple[str, str]]:
    """Sample single-cold unlabeled pairs while excluding known positives."""
    values_left = sorted(left)
    values_right = sorted(right)
    result: set[tuple[str, str]] = set()
    while len(result) < count:
        drug1 = str(rng.choice(values_left))
        drug2 = str(rng.choice(values_right))
        if drug1 != drug2:
            candidate = canonical_text_pair(drug1, drug2)
            if candidate not in positives:
                result.add(candidate)
    return sorted(result)


def prepare_single_cold_split(
    root: Path | str = ROOT,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    """Create reproducible single-cold positive/negative pair splits."""
    root = Path(root)
    output_dir = root / "data" / "split_single_cold"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains files; use --overwrite to rebuild"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*"):
        if old.is_file():
            old.unlink()

    positives = read_single_cold_positive_pairs(root)
    drugs = np.asarray(sorted({drug for pair in positives for drug in pair}))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(drugs)
    n_train = int(round(len(drugs) * 0.60))
    n_val = int(round(len(drugs) * 0.20))
    pools = {
        "train": set(shuffled[:n_train]),
        "val": set(shuffled[n_train:n_train + n_val]),
        "test": set(shuffled[n_train + n_val:]),
    }
    positive_splits = {
        "train": sorted(pair for pair in positives if set(pair) <= pools["train"]),
        "val": sorted(
            pair for pair in positives
            if ((pair[0] in pools["val"] and pair[1] in pools["train"])
                or (pair[1] in pools["val"] and pair[0] in pools["train"]))
        ),
        "test": sorted(
            pair for pair in positives
            if ((pair[0] in pools["test"] and pair[1] in pools["train"])
                or (pair[1] in pools["test"] and pair[0] in pools["train"]))
        ),
    }

    negative_rng = np.random.RandomState(seed + 1)
    split_rows = {}
    for split in ("train", "val", "test"):
        left = pools[split]
        right = pools["train"]
        negatives = sample_single_cold_negatives(
            left,
            right,
            len(positive_splits[split]),
            positives,
            negative_rng,
        )
        records = [
            (drug1, drug2, 1) for drug1, drug2 in positive_splits[split]
        ]
        records += [(drug1, drug2, 0) for drug1, drug2 in negatives]
        frame = pd.DataFrame(
            records,
            columns=["drug1_id", "drug2_id", "label"],
        )
        frame.insert(0, "pair_index", np.arange(len(frame), dtype=np.int32))
        frame.to_csv(output_dir / f"{split}_labeled_pairs.csv", index=False)
        pd.DataFrame({"internal_id": sorted(pools[split])}).to_csv(
            output_dir / f"{split}_drugs.csv",
            index=False,
        )
        split_rows[split] = {
            "drugs": len(pools[split]),
            "positive_pairs": len(positive_splits[split]),
            "negative_pairs": len(negatives),
        }

    summary = {
        "seed": seed,
        "split_policy": "single_cold_60_20_20",
        "source_unique_positive_pairs": len(positives),
        "splits": split_rows,
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def load_drug_target_indices(root: Path | str = ROOT) -> dict[int, list[int]]:
    """Load the numeric drug-to-target mapping."""
    root = Path(root)
    path = root / "data" / "features" / "drug_target_index.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build it once from the original DTI source before preparing combos."
        )
    rows = np.load(path, mmap_mode="r")
    required = {"drug_id", "target_index"}
    if rows.dtype.names is None or not required.issubset(rows.dtype.names):
        raise ValueError(f"{path} must contain fields {sorted(required)}")
    target_count = np.load(
        root / "data" / "features" / "target_features.npy",
        mmap_mode="r",
    ).shape[0]
    targets = rows["target_index"]
    if len(targets) and (targets.min() < 0 or targets.max() >= target_count):
        raise ValueError("Drug-target mapping contains an out-of-range target index")
    mapping: dict[int, list[int]] = defaultdict(list)
    for drug_id, target_index in zip(rows["drug_id"], targets):
        mapping[int(drug_id)].append(int(target_index))
    if any(not values for values in mapping.values()):
        raise ValueError("Every drug must map to a target or the zero-target row")
    return dict(mapping)


def build_combo_indexes(root: Path | str = ROOT, overwrite: bool = False) -> dict:
    """Expand labeled pairs while preserving every target combination."""
    root = Path(root)
    output_dir = root / "data" / "combo_index"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any((output_dir / f"{s}_combo_index.npy").exists()
                             for s in ("train", "val", "test")):
        raise FileExistsError("Combo indexes already exist; add --overwrite to replace them")

    target_map = load_drug_target_indices(root)
    drug_ids = np.load(root / "data" / "features" / "drug_ids.npy").astype(int)
    if set(target_map) != set(map(int, drug_ids)):
        raise ValueError("Drug-target mapping must explicitly cover every drug")
    drug_to_index = {int(drug): index for index, drug in enumerate(drug_ids)}
    zero_target = int(np.load(root / "data" / "features" / "target_features.npy").shape[0] - 1)
    unresolved_path = root / "data" / "features" / "target_mapping_unresolved.csv"
    unresolved = set()
    if unresolved_path.exists():
        unresolved = set(pd.read_csv(unresolved_path)["internal_id"].astype(int))
    summary = {"feature_dim": 3584, "zero_target_index": zero_target, "splits": {}}
    pair_offset = 0
    for split in ("train", "val", "test"):
        pairs = pd.read_csv(root / "data" / "labeled_pairs" / f"{split}_labeled_pairs.csv")
        used_ids = set(pairs["drug1_id"].astype(int)) | set(pairs["drug2_id"].astype(int))
        bad_ids = used_ids & unresolved
        if bad_ids:
            raise ValueError(
                f"Pairs contain drugs with unresolved target mapping: {sorted(bad_ids)}"
            )
        rows = []
        for record in pairs.itertuples(index=False):
            targets1 = target_map[int(record.drug1_id)]
            targets2 = target_map[int(record.drug2_id)]
            weight = np.float32(1.0 / (len(targets1) * len(targets2)))
            pair_index = pair_offset + int(record.pair_id)
            for target1 in targets1:
                for target2 in targets2:
                    rows.append((pair_index, drug_to_index[int(record.drug1_id)],
                                 drug_to_index[int(record.drug2_id)], target1, target2,
                                 int(record.label), weight))
        combos = np.asarray(rows, dtype=COMBO_DTYPE)
        validate_pair_weights(combos)
        np.save(output_dir / f"{split}_combo_index.npy", combos)
        summary["splits"][split] = {"pairs": int(len(pairs)), "combos": int(len(combos))}
        pair_offset += len(pairs)
    with open(output_dir / "combo_index_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def build_single_cold_combo_indexes(
    root: Path | str = ROOT,
    overwrite: bool = False,
) -> dict:
    """Build weighted target-combination indexes for all single-cold pairs."""
    root = Path(root)
    split_dir = root / "data" / "split_single_cold"
    feature_dir = root / "data" / "features"
    output_dir = root / "data" / "combo_index_single_cold"
    if not split_dir.exists():
        raise FileNotFoundError(
            "Run `python feature.py prepare-single-cold` first"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            "Single-cold combo index exists; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*"):
        if path.is_file():
            path.unlink()

    target_map = load_drug_target_indices(root)
    drug_ids = np.load(feature_dir / "drug_ids.npy").astype(int)
    if set(target_map) != set(map(int, drug_ids)):
        raise ValueError("Drug-target mapping must explicitly cover every drug")
    drug_to_index = {
        int(drug): index for index, drug in enumerate(drug_ids)
    }
    summary = {
        "feature_dim": 3584,
        "split_policy": "single_cold_60_20_20",
        "splits": {},
    }
    pair_offset = 0
    for split in ("train", "val", "test"):
        pairs = pd.read_csv(split_dir / f"{split}_labeled_pairs.csv")
        rows = []
        for record in pairs.itertuples(index=False):
            drug1 = int(record.drug1_id)
            drug2 = int(record.drug2_id)
            targets1 = target_map[drug1]
            targets2 = target_map[drug2]
            weight = np.float32(1.0 / (len(targets1) * len(targets2)))
            pair_index = pair_offset + int(record.pair_index)
            for target1 in targets1:
                for target2 in targets2:
                    rows.append((
                        pair_index,
                        drug_to_index[drug1],
                        drug_to_index[drug2],
                        target1,
                        target2,
                        int(record.label),
                        weight,
                    ))
        combos = np.asarray(rows, dtype=COMBO_DTYPE)
        validate_pair_weights(combos)
        np.save(output_dir / f"{split}_combo_index.npy", combos)
        summary["splits"][split] = {
            "pairs": int(len(pairs)),
            "combos": int(len(combos)),
        }
        pair_offset += len(pairs)

    (output_dir / "combo_index_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def generate_warm_pairs(root: Path | str = ROOT, seed: int = 42,
                        overwrite: bool = False) -> dict:
    """Build a balanced train-only set from drugs with trained KG embeddings."""
    root = Path(root)
    output = root / "data" / "labeled_pairs" / "warm_labeled_pairs.csv"
    if output.exists() and not overwrite:
        raise FileExistsError("Warm labeled pairs already exist; add --overwrite to replace them")

    status, status_has_kg, _ = load_usable_kg_status(root)
    warm_ids = set(status.loc[status_has_kg, "internal_id"].astype(int))
    train_ids = set(
        pd.read_csv(root / "data" / "split" / "train_drugs.csv")
        ["internal_id"].astype(int)
    )
    pool = sorted(warm_ids & train_ids)

    unresolved_path = root / "data" / "features" / "target_mapping_unresolved.csv"
    unresolved = set()
    if unresolved_path.exists():
        unresolved = set(pd.read_csv(unresolved_path)["internal_id"].astype(int))
    pool = np.asarray([drug for drug in pool if drug not in unresolved], dtype=np.int64)
    if len(pool) < 2:
        raise RuntimeError("Not enough train drugs with trained KG embeddings")

    all_positive: set[tuple[int, int]] = set()
    train_positive: set[tuple[int, int]] = set()
    for split in ("train", "val", "test"):
        pairs = pd.read_csv(root / "data" / "split" / f"{split}_pairs.csv")
        current = {canonical_pair(int(a), int(b))
                   for a, b in zip(pairs["drug1_id"], pairs["drug2_id"])
                   if int(a) != int(b)}
        all_positive.update(current)
        if split == "train":
            train_positive = current
    positives = sorted(pair for pair in train_positive
                       if pair[0] in warm_ids and pair[1] in warm_ids)
    if not positives:
        raise RuntimeError("No positive train pairs have KG for both drugs")

    rng = np.random.RandomState(seed)
    negatives: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(100_000, len(positives) * 1000)
    while len(negatives) < len(positives) and attempts < max_attempts:
        a, b = rng.choice(pool, size=2, replace=False)
        pair = canonical_pair(int(a), int(b))
        if pair not in all_positive:
            negatives.add(pair)
        attempts += 1
    if len(negatives) != len(positives):
        raise RuntimeError("Unable to sample enough warm negative pairs")

    rows = [(a, b, 1, "warm_train", "positive") for a, b in positives]
    rows += [(a, b, 0, "warm_train", "random_uniform_1_to_1")
             for a, b in sorted(negatives)]
    frame = pd.DataFrame(rows, columns=["drug1_id", "drug2_id", "label", "split", "sample_source"])
    frame.insert(0, "pair_id", np.arange(len(frame), dtype=np.int32))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "seed": seed,
        "strategy": "random_uniform_1_to_1",
        "scope": "train drugs with trained KG embeddings on both sides",
        "positive": len(positives),
        "negative": len(negatives),
        "drug_pool": int(len(pool)),
        "attempts": attempts,
    }
    summary_path = output.parent / "warm_negative_sampling_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def build_warm_combo_index(root: Path | str = ROOT, overwrite: bool = False) -> dict:
    """Expand the balanced warm training pairs without target mean-pooling."""
    root = Path(root)
    output = root / "data" / "combo_index" / "warm_combo_index.npy"
    if output.exists() and not overwrite:
        raise FileExistsError("Warm combo index already exists; add --overwrite to replace it")
    pairs = pd.read_csv(root / "data" / "labeled_pairs" / "warm_labeled_pairs.csv")
    target_map = load_drug_target_indices(root)
    drug_ids = np.load(root / "data" / "features" / "drug_ids.npy").astype(int)
    drug_to_index = {int(drug): index for index, drug in enumerate(drug_ids)}
    status, status_has_kg, _ = load_usable_kg_status(root)
    has_kg = set(status.loc[status_has_kg, "internal_id"].astype(int))

    used_ids = set(pairs["drug1_id"].astype(int)) | set(pairs["drug2_id"].astype(int))
    if not used_ids.issubset(has_kg):
        raise ValueError("Warm pairs contain a drug without a trained KG embedding")
    rows = []
    for record in pairs.itertuples(index=False):
        targets1 = target_map[int(record.drug1_id)]
        targets2 = target_map[int(record.drug2_id)]
        weight = np.float32(1.0 / (len(targets1) * len(targets2)))
        for target1 in targets1:
            for target2 in targets2:
                rows.append((int(record.pair_id), drug_to_index[int(record.drug1_id)],
                             drug_to_index[int(record.drug2_id)], target1, target2,
                             int(record.label), weight))
    combos = np.asarray(rows, dtype=COMBO_DTYPE)
    validate_pair_weights(combos)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, combos)
    summary = {"feature_dim": 3840, "pairs": int(len(pairs)), "combos": int(len(combos))}
    with (output.parent / "warm_combo_index_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


@dataclass
class FeatureStore:
    kg: np.ndarray | None
    has_kg: np.ndarray | None
    smiles: np.ndarray
    targets: np.ndarray
    drug_ids: np.ndarray
    target_ids: np.ndarray

    @classmethod
    def load(cls, root: Path | str = ROOT, require_kg: bool = True) -> "FeatureStore":
        root = Path(root)
        drug_ids, smiles = load_smiles_features(root / "data" / "features")
        target_ids, targets = load_target_features(root / "data" / "features")
        if not require_kg:
            return cls(kg=None, has_kg=None, smiles=smiles, targets=targets,
                       drug_ids=drug_ids, target_ids=target_ids)
        kg_dir, source = resolve_kg_assets(root)
        kg = np.load(kg_dir / "drug_kg_features.npy").astype(np.float32)
        status, has_kg, status_source = load_usable_kg_status(root)
        if source != status_source:
            raise ValueError("KG feature and status sources do not match")
        with (kg_dir / "kg_summary.json").open("r", encoding="utf-8") as handle:
            kg_summary = json.load(handle)
        if source == "strict" and not kg_summary.get("strict_cold_entity_exclusion", False):
            raise ValueError("Rebuilt KG must exclude every val/test cold entity")
        if source == "pretrained":
            with (kg_dir / "reuse_policy.json").open("r", encoding="utf-8") as handle:
                reuse_policy = json.load(handle)
            if reuse_policy.get("cold_kg_used") or reuse_policy.get("isolated_xavier_used"):
                raise ValueError("Pretrained KG reuse policy is unsafe")
        status_ids = status["internal_id"].astype(int).to_numpy()
        if kg.shape != (len(drug_ids), 128) or not np.array_equal(status_ids, drug_ids):
            raise ValueError("KG features/status do not match drug_ids.npy")
        if not np.isfinite(kg).all():
            raise ValueError("KG features contain NaN or Inf")
        kg = kg.copy()
        kg[~has_kg] = 0.0
        if not np.allclose(kg[~has_kg], 0.0):
            raise ValueError("Unable to mask unusable KG rows")
        return cls(kg=kg, has_kg=has_kg, smiles=smiles, targets=targets,
                   drug_ids=drug_ids, target_ids=target_ids)

    def build_batch(self, rows: np.ndarray, feature_mode: str = "cold") -> np.ndarray:
        """Build one feature row for every indexed target combination."""
        d1 = rows["drug1_idx"].astype(np.int32)
        d2 = rows["drug2_idx"].astype(np.int32)
        t1 = rows["target1_idx"].astype(np.int32)
        t2 = rows["target2_idx"].astype(np.int32)
        if feature_mode == "cold":
            values = np.concatenate([
                self.smiles[d1], self.targets[t1],
                self.smiles[d2], self.targets[t2],
            ], axis=1).astype(np.float32)
            expected_dim = 3584
        elif feature_mode == "warm":
            if self.kg is None or self.has_kg is None:
                raise ValueError("Warm branch requires strict trained KG features")
            if not self.has_kg[d1].all() or not self.has_kg[d2].all():
                raise ValueError("Warm branch received a drug without a trained KG embedding")
            values = np.concatenate([
                self.kg[d1], self.smiles[d1], self.targets[t1],
                self.kg[d2], self.smiles[d2], self.targets[t2],
            ], axis=1).astype(np.float32)
            expected_dim = 3840
        else:
            raise ValueError(f"Unknown feature mode: {feature_mode}")
        if values.shape[1] != expected_dim or not np.isfinite(values).all():
            raise ValueError("Invalid combined feature batch")
        return values


def load_combo_index(
    split: str,
    root: Path | str = ROOT,
    combo_dir: Path | str = "data/combo_index",
) -> np.ndarray:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    directory = Path(combo_dir)
    if not directory.is_absolute():
        directory = Path(root) / directory
    return np.load(directory / f"{split}_combo_index.npy", mmap_mode="r")


def validate_pair_weights(rows: np.ndarray, atol: float = 1e-5) -> None:
    totals: dict[int, float] = defaultdict(float)
    for pair_id, weight in zip(rows["pair_index"], rows["sample_weight"]):
        totals[int(pair_id)] += float(weight)
    bad = [pair_id for pair_id, total in totals.items() if abs(total - 1.0) > atol]
    if bad:
        raise ValueError(f"{len(bad)} drug pairs have invalid total weight")


class ComboDataIter(xgb.core.DataIter):
    """Stream combo rows into one continuous XGBoost training process."""

    def __init__(self, rows: np.ndarray, store: FeatureStore, batch_size: int = 2048,
                 feature_mode: str = "cold", cache_prefix: str | None = None,
                 device: str = "cpu", progress_desc: str | None = None):
        super().__init__(cache_prefix=cache_prefix, release_data=True,
                         on_host=cache_prefix is None)
        self.rows = rows
        self.store = store
        self.batch_size = batch_size
        self.feature_mode = feature_mode
        self.device = device
        self.offset = 0
        self.progress_desc = progress_desc
        self.scan_number = 0
        self.progress_bar = None

    def reset(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
        self.offset = 0
        self.scan_number += 1

    def next(self, input_data) -> int:
        if self.offset >= len(self.rows):
            if self.progress_bar is not None:
                self.progress_bar.close()
                self.progress_bar = None
            return 0
        if self.progress_desc and self.progress_bar is None:
            total_batches = (len(self.rows) + self.batch_size - 1) // self.batch_size
            self.progress_bar = tqdm(
                total=total_batches,
                desc=f"{self.progress_desc} scan {self.scan_number}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )
        end = min(self.offset + self.batch_size, len(self.rows))
        batch = self.rows[self.offset:end]
        data = self.store.build_batch(batch, self.feature_mode)
        label = batch["label"].astype(np.float32)
        weight = batch["sample_weight"].astype(np.float32)
        if self.device == "cuda":
            import cupy as cp
            data, label, weight = cp.asarray(data), cp.asarray(label), cp.asarray(weight)
        elif self.device != "cpu":
            raise ValueError(f"Unknown iterator device: {self.device}")
        input_data(
            data=data,
            label=label,
            weight=weight,
        )
        self.offset = end
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        return 1


def aggregate_combo_predictions(
    rows: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate combo probabilities to drug-pair level with sample weights."""
    if len(rows) != len(probabilities):
        raise ValueError("Prediction count does not match combo rows")
    order = np.argsort(rows["pair_index"], kind="stable")
    pair_index = rows["pair_index"][order].astype(np.int64, copy=False)
    labels = rows["label"][order].astype(np.int8, copy=False)
    weights = rows["sample_weight"][order].astype(np.float64, copy=False)
    probabilities = np.asarray(probabilities, dtype=np.float64)[order]
    pair_ids, starts = np.unique(pair_index, return_index=True)
    ends = np.r_[starts[1:], len(rows)]
    first_labels = labels[starts]
    if any(np.any(labels[start:end] != first_labels[index])
           for index, (start, end) in enumerate(zip(starts, ends))):
        raise ValueError("A pair_index contains inconsistent labels")
    total_weight = np.add.reduceat(weights, starts)
    if np.any(total_weight <= 0):
        raise ValueError("A pair_index has non-positive total weight")
    pair_probabilities = np.add.reduceat(probabilities * weights, starts) / total_weight
    return pair_ids.astype(int), first_labels.astype(int), pair_probabilities


def predict_and_aggregate(model, rows: np.ndarray, store: FeatureStore,
                          batch_size: int = 2048, feature_mode: str = "cold",
                          iteration_range: tuple[int, int] | None = None,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict combo rows and aggregate them to drug-pair probabilities."""
    combo_probabilities = np.empty(len(rows), dtype=np.float32)

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        kwargs = {"iteration_range": iteration_range} if iteration_range else {}
        combo_probabilities[start:start + len(batch)] = model.predict(
            xgb.DMatrix(store.build_batch(batch, feature_mode)), **kwargs
        )
    return aggregate_combo_predictions(rows, combo_probabilities)


def predict_and_aggregate_many(
    model,
    rows: np.ndarray,
    store: FeatureStore,
    tree_counts: list[int],
    batch_size: int = 2048,
    feature_mode: str = "cold",
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate several tree counts while constructing each feature batch once."""
    counts = sorted(set(int(value) for value in tree_counts))
    if not counts or counts[0] < 1:
        raise ValueError("tree_counts must contain positive integers")
    predictions = {
        count: np.empty(len(rows), dtype=np.float32) for count in counts
    }
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        matrix = xgb.DMatrix(store.build_batch(batch, feature_mode))
        for count in counts:
            predictions[count][start:start + len(batch)] = model.predict(
                matrix, iteration_range=(0, count)
            )
    return {
        count: aggregate_combo_predictions(rows, probabilities)
        for count, probabilities in predictions.items()
    }


def parse_args():
    """Parse feature-preparation commands."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare KGP-DC feature and pair-index artifacts."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    split_parser = commands.add_parser(
        "prepare-single-cold",
        help="Create the single-cold drug pools and balanced labeled pairs.",
    )
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--overwrite", action="store_true")

    index_parser = commands.add_parser(
        "build-single-cold-index",
        help="Expand single-cold pairs into weighted target combinations.",
    )
    index_parser.add_argument("--overwrite", action="store_true")

    all_parser = commands.add_parser(
        "prepare-single-cold-all",
        help="Run both single-cold preparation stages in order.",
    )
    all_parser.add_argument("--seed", type=int, default=42)
    all_parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected feature-preparation stage."""
    args = parse_args()
    if args.command == "prepare-single-cold":
        summary = prepare_single_cold_split(seed=args.seed, overwrite=args.overwrite)
    elif args.command == "build-single-cold-index":
        summary = build_single_cold_combo_indexes(overwrite=args.overwrite)
    else:
        split_summary = prepare_single_cold_split(
            seed=args.seed,
            overwrite=args.overwrite,
        )
        index_summary = build_single_cold_combo_indexes(overwrite=args.overwrite)
        summary = {"split": split_summary, "combo_index": index_summary}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Euclidean/RMS reliable-negative ablation for the single-cold experiment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
METHODS = ("mean_euclidean", "rms")


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


@dataclass(frozen=True)
class Paths:
    config: Path
    base: Path
    legacy: Path
    output: Path

    @classmethod
    def from_config(cls, config_path: Path, cfg: dict) -> "Paths":
        root = config_path.resolve().parent
        result = []
        for key in ("base_project", "legacy_experiment", "output_dir"):
            path = Path(cfg["project"][key])
            result.append((root / path).resolve() if not path.is_absolute() else path.resolve())
        return cls(config_path.resolve(), *result)


def feature_module(paths: Paths):
    sys.path.insert(0, str(paths.base))
    import con_feature  # type: ignore
    return con_feature


def source_dataset(paths: Paths, seed: int, split: str) -> pd.DataFrame:
    path = paths.legacy / "output" / "datasets" / "S1_uniform" / f"seed_{seed}" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def candidate_data(paths: Paths) -> dict[str, np.ndarray]:
    path = paths.legacy / "output" / "splits" / "train_unlabeled_candidates.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def positive_train(paths: Paths) -> pd.DataFrame:
    frame = source_dataset(paths, 42, "train")
    result = frame.loc[frame["label"] == 1, ["drug1_id", "drug2_id"]].copy()
    return result.drop_duplicates().sort_values(["drug1_id", "drug2_id"]).reset_index(drop=True)


def target_csr(paths: Paths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(paths.base / "data" / "features" / "drug_target_indices.npz")
    return data["drug_ids"].astype(np.int64), data["offsets"].astype(np.int64), data["target_indices"].astype(np.int32)


def position_stats(rows: np.ndarray, target_lists: list[np.ndarray], smiles: np.ndarray,
                   targets: np.ndarray) -> dict[str, np.ndarray]:
    n = len(rows)
    sm = np.asarray(smiles[rows], dtype=np.float64)
    sm_mean = sm.mean(axis=0)
    sm_var = np.maximum((sm * sm).mean(axis=0) - sm_mean * sm_mean, 1e-8)
    tm = np.zeros(targets.shape[1], dtype=np.float64)
    ts = np.zeros(targets.shape[1], dtype=np.float64)
    for row in rows:
        values = np.asarray(targets[target_lists[int(row)]], dtype=np.float64)
        tm += values.mean(axis=0)
        ts += (values * values).mean(axis=0)
    tm /= n
    ts /= n
    tv = np.maximum(ts - tm * tm, 1e-8)
    return {"smiles_mean": sm_mean.astype(np.float32), "smiles_scale": np.sqrt(sm_var).astype(np.float32),
            "target_mean": tm.astype(np.float32), "target_scale": np.sqrt(tv).astype(np.float32)}


def squared_rows(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), 256):
        block = np.asarray(values[start:start + 256], dtype=np.float32)
        z = (block - mean) / scale
        result[start:start + len(block)] = np.einsum("ij,ij->i", z, z)
    return result


def build_components(paths: Paths) -> tuple[dict, dict, dict]:
    fm = feature_module(paths)
    store = fm.FeatureStore.load(paths.base, require_kg=False)
    drug_ids, offsets, target_indices = target_csr(paths)
    if not np.array_equal(drug_ids, store.drug_ids.astype(np.int64)):
        raise ValueError("Drug IDs and feature rows are not aligned")
    target_lists = [target_indices[offsets[i]:offsets[i + 1]] for i in range(len(drug_ids))]
    drug_to_row = {int(drug): i for i, drug in enumerate(drug_ids)}
    positives = positive_train(paths)
    p1 = positives["drug1_id"].map(drug_to_row).to_numpy(dtype=np.int32)
    p2 = positives["drug2_id"].map(drug_to_row).to_numpy(dtype=np.int32)
    stats1 = position_stats(p1, target_lists, store.smiles, store.targets)
    stats2 = position_stats(p2, target_lists, store.smiles, store.targets)
    components = {
        "sm1": squared_rows(store.smiles, stats1["smiles_mean"], stats1["smiles_scale"]),
        "sm2": squared_rows(store.smiles, stats2["smiles_mean"], stats2["smiles_scale"]),
        "tg1": squared_rows(store.targets, stats1["target_mean"], stats1["target_scale"]),
        "tg2": squared_rows(store.targets, stats2["target_mean"], stats2["target_scale"]),
    }
    meta = {"target_lists": target_lists, "drug_to_row": drug_to_row, "positive_count": len(positives)}
    return meta, components, {"position1": stats1, "position2": stats2}


def calculate_distances(paths: Paths, output: Path) -> dict:
    meta, q, stats = build_components(paths)
    candidates = candidate_data(paths)
    d1 = np.fromiter((meta["drug_to_row"][int(x)] for x in candidates["drug1_id"]), np.int32,
                     count=len(candidates["drug1_id"]))
    d2 = np.fromiter((meta["drug_to_row"][int(x)] for x in candidates["drug2_id"]), np.int32,
                     count=len(candidates["drug2_id"]))
    lists = meta["target_lists"]
    mt1 = np.asarray([q["tg1"][x].mean() for x in lists])
    mt2 = np.asarray([q["tg2"][x].mean() for x in lists])
    static_a = q["sm1"][d1] + q["sm2"][d2]
    static_b = q["sm1"][d2] + q["sm2"][d1]
    rms = np.sqrt(np.maximum(0.5 * (static_a + mt1[d1] + mt2[d2] +
                                    static_b + mt1[d2] + mt2[d1]), 0.0)).astype(np.float32)
    mean_euclidean = np.empty(len(d1), dtype=np.float32)
    for start in tqdm(range(0, len(d1), 4096), desc="Mean Euclidean", unit="block"):
        end = min(start + 4096, len(d1))
        for i in range(start, end):
            t1, t2 = lists[int(d1[i])], lists[int(d2[i])]
            da = np.sqrt(np.maximum(static_a[i] + q["tg1"][t1][:, None] + q["tg2"][t2][None, :], 0)).mean()
            db = np.sqrt(np.maximum(static_b[i] + q["tg1"][t2][:, None] + q["tg2"][t1][None, :], 0)).mean()
            mean_euclidean[i] = np.float32(0.5 * (da + db))
    if not np.isfinite(mean_euclidean).all() or not np.isfinite(rms).all():
        raise ValueError("Distances contain NaN or Inf")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "train_candidate_distances.npz", drug1_id=candidates["drug1_id"],
                        drug2_id=candidates["drug2_id"], pair_code=candidates["pair_code"],
                        mean_euclidean=mean_euclidean, rms=rms)
    np.savez_compressed(output / "train_positive_standardization.npz", **{
        f"{position}_{name}": value for position, values in stats.items() for name, value in values.items()})
    return {"mean_euclidean": mean_euclidean, "rms": rms,
            "candidate_count": len(d1), "positive_count": meta["positive_count"]}


def status(cfg: dict, paths: Paths) -> None:
    required = [paths.base / "con_feature.py",
                paths.base / "data" / "features" / "drug_target_indices.npz",
                paths.legacy / "output" / "splits" / "train_unlabeled_candidates.npz"]
    missing = [str(path) for path in required if not path.exists()]
    print(f"Experiment root: {paths.config.parent}\nCold-start source: {paths.base}")
    print(f"Legacy experiment (read-only): {paths.legacy}\nNew output: {paths.output}")
    if missing:
        print("Configured source files: MISSING")
        print("The following inputs are required to re-run this experiment but are "
              "not redistributed with the repository:")
        for item in missing:
            print(f"  - {item}")
        print("The committed report under output/reports_gpu_lite/ is the record of "
              "the completed run; re-executing the experiment requires the "
              "single-cold workspace that owns the unlabelled candidate pool.")
        raise SystemExit(2)
    print("Configured source files: OK")


def audit(cfg: dict, paths: Paths, force: bool) -> None:
    audit_dir = paths.output / "audits"
    distance_file = audit_dir / "train_candidate_distances.npz"
    report_file = audit_dir / "distance_capacity_audit.json"
    if report_file.exists() and not force:
        print(report_file.read_text(encoding="utf-8"))
        return
    if distance_file.exists() and not force:
        with np.load(distance_file) as saved:
            values = {method: saved[method] for method in METHODS}
            info = {"candidate_count": len(saved["drug1_id"]), "positive_count": len(positive_train(paths))}
    else:
        calculated = calculate_distances(paths, audit_dir)
        values = {method: calculated[method] for method in METHODS}
        info = calculated
    required = int(round(info["positive_count"] * float(cfg["experiment"]["positive_negative_ratio"])))
    report = {"definition": {
        "mean_euclidean": "weighted mean of target-combination Euclidean distances",
        "rms": "root weighted mean square of target-combination Euclidean distances",
        "symmetry": "average both drug orientations",
        "feature_mode": "SMILES + independent Target combinations; no target mean-pooling"},
        "candidate_count": int(info["candidate_count"]), "train_positive_count": int(info["positive_count"]),
        "required_negative_count": required, "methods": {}}
    passed = True
    for method, distances in values.items():
        aed = float(np.mean(distances, dtype=np.float64))
        thresholds = {}
        for multiplier in cfg["experiment"]["thresholds"]:
            value = float(multiplier) * aed
            eligible = int(np.sum(distances > value))
            enough = eligible >= required
            passed &= enough
            thresholds[str(multiplier)] = {"threshold": value, "eligible_count": eligible,
                                           "required_count": required, "enough_for_1_to_1": enough}
        report["methods"][method] = {"aed": aed, "min": float(distances.min()),
                                      "max": float(distances.max()), "std": float(distances.std()),
                                      "thresholds": thresholds}
    report["audit_passed"] = bool(passed)
    write_json(report_file, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def prepare(cfg: dict, paths: Paths, force: bool) -> None:
    report_path = paths.output / "audits" / "distance_capacity_audit.json"
    if not report_path.exists():
        raise FileNotFoundError("Run audit first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("audit_passed"):
        raise RuntimeError("Distance capacity audit did not pass")
    with np.load(paths.output / "audits" / "train_candidate_distances.npz") as data:
        distances = {key: data[key] for key in data.files}
    seeds = [int(x) for x in cfg["experiment"]["seeds"]]
    required = int(report["required_negative_count"])
    positive = positive_train(paths)
    manifest = {"fixed_val_test": "S1_uniform datasets for the matching seed", "runs": {}}
    for method_index, method in enumerate(METHODS):
        aed = float(report["methods"][method]["aed"])
        for multiplier in cfg["experiment"]["thresholds"]:
            group = f"{method}_aed_{float(multiplier):.1f}"
            eligible = np.flatnonzero(distances[method] > float(multiplier) * aed)
            for seed in seeds:
                out = paths.output / "datasets" / group / f"seed_{seed}"
                if out.exists() and not force:
                    raise FileExistsError(f"Dataset exists: {out}; use --force")
                out.mkdir(parents=True, exist_ok=True)
                rng = np.random.default_rng(seed + method_index * 10000 + int(round(multiplier * 100)))
                chosen = rng.choice(eligible, size=required, replace=False)
                neg = pd.DataFrame({"drug1_id": distances["drug1_id"][chosen],
                                    "drug2_id": distances["drug2_id"][chosen], "label": 0,
                                    "split": "train", "strategy": group, "seed": seed,
                                    "sample_source": "reliable_negative"})
                pos = positive.copy()
                pos["label"], pos["split"], pos["strategy"], pos["seed"] = 1, "train", group, seed
                pos["sample_source"] = "known_positive"
                train_frame = pd.concat([pos, neg], ignore_index=True)
                train_frame.insert(0, "pair_id", np.arange(len(train_frame), dtype=np.int32))
                train_frame.to_csv(out / "train.csv", index=False)
                for split in ("val", "test"):
                    fixed = source_dataset(paths, seed, split).copy()
                    fixed["strategy"] = group
                    fixed.to_csv(out / f"{split}.csv", index=False)
                manifest["runs"][f"{group}/seed_{seed}"] = {
                    "positive": int((train_frame.label == 1).sum()),
                    "negative": int((train_frame.label == 0).sum()),
                    "duplicate_negative": int(len(chosen) - len(np.unique(chosen)))}
    write_json(paths.output / "audits" / "prepared_dataset_manifest.json", manifest)
    print(f"Prepared datasets: {paths.output / 'datasets'}")


def selected_target_pairs(
    drug1: int, drug2: int, targets1: list[int], targets2: list[int], cap: int
) -> list[tuple[int, int]]:
    """Select a stable subset of target combinations without pooling features."""
    total = len(targets1) * len(targets2)
    if total <= cap:
        positions = np.arange(total, dtype=np.int64)
    else:
        positions = np.arange(total, dtype=np.uint64)
        with np.errstate(over="ignore"):
            hashes = positions * np.uint64(0x9E3779B185EBCA87)
            hashes ^= np.uint64(drug1) * np.uint64(0xC2B2AE3D27D4EB4F)
            hashes ^= np.uint64(drug2) * np.uint64(0x165667B19E3779F9)
            hashes ^= hashes >> np.uint64(29)
        positions = np.sort(positions[np.argsort(hashes, kind="stable")[:cap]].astype(np.int64))
    width = len(targets2)
    return [(int(targets1[pos // width]), int(targets2[pos % width])) for pos in positions]


def build_combos(frame: pd.DataFrame, paths: Paths, fm, cap: int) -> np.ndarray:
    target_map = fm.load_drug_target_indices(paths.base)
    drug_ids = np.load(paths.base / "data" / "features" / "drug_ids.npy").astype(int)
    mapping = {int(drug): i for i, drug in enumerate(drug_ids)}
    total = sum(min(cap, len(target_map[int(r.drug1_id)]) * len(target_map[int(r.drug2_id)]))
                for r in frame.itertuples())
    rows = np.empty(total, dtype=fm.COMBO_DTYPE)
    offset = 0
    for pair_index, record in enumerate(frame.itertuples(index=False)):
        t1, t2 = target_map[int(record.drug1_id)], target_map[int(record.drug2_id)]
        selected = selected_target_pairs(int(record.drug1_id), int(record.drug2_id), t1, t2, cap)
        weight = np.float32(1 / len(selected))
        for a, b in selected:
            rows[offset] = (pair_index, mapping[int(record.drug1_id)], mapping[int(record.drug2_id)],
                            a, b, int(record.label), weight)
            offset += 1
    fm.validate_pair_weights(rows)
    return rows


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n_pairs": int(len(y)), "acc": float((tp + tn) / len(y)), "sen": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp)), "precision": float(tp / (tp + fp)),
            "f1": float(f1_score(y, pred)), "mcc": float(matthews_corrcoef(y, pred)),
            "roc_auc": float(roc_auc_score(y, p)), "aupr": float(average_precision_score(y, p)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def train(cfg: dict, paths: Paths, method: str, threshold: float, seed: int, force: bool) -> None:
    import xgboost as xgb
    group = f"{method}_aed_{threshold:.1f}"
    data_dir = paths.output / "datasets" / group / f"seed_{seed}"
    if not data_dir.exists():
        raise FileNotFoundError("Run prepare first")
    fm = feature_module(paths)
    store = fm.FeatureStore.load(paths.base, require_kg=False)
    cap = int(cfg["experiment"]["max_target_combinations_per_pair"])
    combos = {split: build_combos(pd.read_csv(data_dir / f"{split}.csv"), paths, fm, cap)
              for split in SPLITS}
    result_dir = str(cfg["experiment"].get("result_dir", "results_gpu_lite"))
    out = paths.output / result_dir / group / f"seed_{seed}"
    if (out / "xgboost_model.json").exists() and not force:
        raise FileExistsError("Model already exists")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    batch = int(cfg["experiment"]["training_batch_size"])
    iterator = fm.ComboDataIter(combos["train"], store, batch_size=batch, feature_mode="cold",
                                cache_prefix=str(cache / "train"), device="cuda", progress_desc="train cache")
    dtrain = xgb.ExtMemQuantileDMatrix(iterator, max_bin=int(cfg["experiment"]["max_bin"]),
                                       max_quantile_batches=int(cfg["experiment"]["max_quantile_batches"]),
                                       cache_host_ratio=0.9,
                                       nthread=int(cfg["experiment"]["nthread"]))
    params = {"objective": "binary:logistic", "eval_metric": ["logloss", "auc"], "tree_method": "hist",
              "device": "cuda", "max_bin": int(cfg["experiment"]["max_bin"]),
              "nthread": int(cfg["experiment"]["nthread"]), "seed": seed, "learning_rate": 0.05,
              "max_depth": 4, "subsample": 0.6, "colsample_bytree": 0.6,
              "reg_alpha": 5.0, "reg_lambda": 10.0, "gamma": 1.0}
    history = {}
    model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dtrain, "train")],
                      evals_result=history, verbose_eval=10)
    model.save_model(out / "xgboost_model.json")
    pd.DataFrame(history["train"]).to_csv(out / "training_history.csv", index=False)
    for split in ("val", "test"):
        ids, labels, probs = fm.predict_and_aggregate(model, combos[split], store,
                                                       batch_size=batch, feature_mode="cold")
        pd.DataFrame({"pair_index": ids, "label": labels, "probability": probs}).to_csv(
            out / f"{split}_pair_predictions.csv", index=False)
        write_json(out / f"{split}_metrics.json", classification_metrics(labels, probs))
    write_json(out / "training_config.json", {
        "group": group, "seed": seed, "params": params,
        "target_combinations_preserved": True,
        "target_combination_policy": "stable_hash_subset_without_pooling",
        "max_target_combinations_per_pair": cap,
        "training_batch_size": batch,
    })
    print(f"Completed {group}, seed={seed}")


def summarize(cfg: dict, paths: Paths) -> None:
    rows = []
    for method in METHODS:
        for threshold in cfg["experiment"]["thresholds"]:
            group = f"{method}_aed_{float(threshold):.1f}"
            for seed in cfg["experiment"]["seeds"]:
                for split in ("val", "test"):
                    result_dir = str(cfg["experiment"].get("result_dir", "results_gpu_lite"))
                    path = paths.output / result_dir / group / f"seed_{seed}" / f"{split}_metrics.json"
                    if path.exists():
                        rows.append({"group": group, "method": method, "threshold": threshold,
                                     "seed": seed, "split": split, **json.loads(path.read_text())})
    if not rows:
        raise FileNotFoundError("No completed metrics")
    frame = pd.DataFrame(rows)
    report = paths.output / "reports_gpu_lite"
    report.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report / "all_seed_metrics.csv", index=False)
    columns = ["acc", "sen", "specificity", "precision", "f1", "mcc", "roc_auc", "aupr"]
    frame.groupby(["group", "split"])[columns].agg(["mean", "std"]).to_csv(report / "mean_std_summary.csv")
    print(f"Reports: {report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    a = sub.add_parser("audit"); a.add_argument("--force", action="store_true")
    p = sub.add_parser("prepare"); p.add_argument("--force", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--method", choices=METHODS, required=True)
    t.add_argument("--threshold", type=float, choices=(0.9, 1.0, 1.1), required=True)
    t.add_argument("--seed", type=int, choices=(42, 52, 62, 72, 82), required=True)
    t.add_argument("--force", action="store_true")
    sub.add_parser("summarize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(args.config, cfg)
    if args.command == "status": status(cfg, paths)
    elif args.command == "audit": audit(cfg, paths, args.force)
    elif args.command == "prepare": prepare(cfg, paths, args.force)
    elif args.command == "train": train(cfg, paths, args.method, args.threshold, args.seed, args.force)
    elif args.command == "summarize": summarize(cfg, paths)


if __name__ == "__main__":
    main()

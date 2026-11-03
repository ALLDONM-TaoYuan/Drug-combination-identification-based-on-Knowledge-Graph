"""Unified training script for the three negative-sampling blocks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import settings as S


def pair_metrics(labels, probabilities, threshold):
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 f1_score, matthews_corrcoef, roc_auc_score)
    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    binary = len(np.unique(labels)) > 1
    return {
        "pairs": int(len(labels)), "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()), "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "ACC": float((tp + tn) / len(labels)) if len(labels) else 0.0,
        "SEN": float(tp / (tp + fn)) if tp + fn else 0.0,
        "SPE": float(tn / (tn + fp)) if tn + fp else 0.0,
        "PRE": float(tp / (tp + fp)) if tp + fp else 0.0,
        "F1": float(f1_score(labels, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, predicted)) if binary else 0.0,
        "ROC_AUC": float(roc_auc_score(labels, probabilities)) if binary else 0.5,
        "AUPR": float(average_precision_score(labels, probabilities)) if binary else 0.0,
    }


def select_threshold(labels, probabilities):
    from sklearn.metrics import roc_curve
    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    usable = np.isfinite(thresholds)
    return float(thresholds[usable][int(np.argmax(tpr[usable] - fpr[usable]))]) if usable.any() else 0.5


def drug_level_halves(meta, seed):
    drug1 = meta["drug1_id"].to_numpy()
    drug2 = meta["drug2_id"].to_numpy()
    drugs = np.unique(np.concatenate([drug1, drug2]))
    chosen = set(np.random.RandomState(seed).choice(
        drugs, size=max(1, len(drugs) // 2), replace=False).tolist())
    mask_a = np.array([(a in chosen) and (b in chosen) for a, b in zip(drug1, drug2)])
    return mask_a, ~mask_a


def load_split(features_dir, split):
    x = np.load(features_dir / f"{split}.npy")
    y = np.load(features_dir / f"{split}_labels.npy").astype(np.int32)
    w = np.load(features_dir / f"{split}_sample_weight.npy").astype(np.float32)
    return x, y, w


def run_one(block_name, seed, force=False):
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    features_dir = S.block_features(block_name)
    results_dir = S.block_results(block_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"seed_{seed}_metrics.csv"
    if out_path.exists() and not force:
        print(f"[skip] {block_name} seed={seed}")
        return

    splits = {split: load_split(features_dir, split) for split in S.SPLITS}
    meta = pd.read_csv(features_dir / "val_pair_meta.csv")
    params = dict(S.XGB_PARAMS, seed=int(seed), nthread=S.NTHREAD, device=S.DEVICE)
    matrices = {split: xgb.DMatrix(splits[split][0], label=splits[split][1],
                                   weight=splits[split][2], nthread=S.NTHREAD)
                for split in S.SPLITS}
    model = xgb.train(params, matrices["train"], num_boost_round=S.ROUNDS,
                      evals=[(matrices["train"], "train"), (matrices["val"], "val")],
                      verbose_eval=False)
    raw = {split: np.asarray(model.predict(matrices[split])) for split in S.SPLITS}

    mask_a, mask_b = drug_level_halves(meta, seed)
    if int(mask_a.sum()) < 200 or int(mask_b.sum()) < 200:
        mask_a = np.ones(len(meta), dtype=bool)
        mask_b = mask_a
    probabilities = dict(raw)
    if S.CALIBRATION == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            np.clip(raw["val"][mask_a], 1e-6, 1 - 1e-6), splits["val"][1][mask_a])
        probabilities = {split: np.clip(calibrator.predict(np.clip(raw[split], 1e-6, 1 - 1e-6)),
                                        1e-6, 1 - 1e-6) for split in S.SPLITS}
    threshold = select_threshold(splits["val"][1][mask_b], probabilities["val"][mask_b])

    rows = [{"block": block_name, "seed": int(seed), "split": split,
             **pair_metrics(splits[split][1], probabilities[split], threshold)}
            for split in S.SPLITS]
    table = pd.DataFrame(rows)[["block", "seed", "split", "threshold", "pairs", "positives",
                                "negatives", "tp", "fp", "tn", "fn", *S.METRICS]]
    table.to_csv(out_path, index=False)
    test = table[table["split"] == "test"].iloc[0]
    print(f"[done] {block_name} seed={seed} ACC={test['ACC']:.4f} "
          f"MCC={test['MCC']:.4f} AUC={test['ROC_AUC']:.4f} AUPR={test['AUPR']:.4f}")


def summarize_block(block_name):
    results_dir = S.block_results(block_name)
    frames = [pd.read_csv(path) for path in sorted(results_dir.glob("seed_*_metrics.csv"))]
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(results_dir / "all_seeds_metrics.csv", index=False)
    summary = data.groupby("split")[list(S.METRICS)].agg(["mean", "std"]).round(6)
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.to_csv(results_dir / "summary.csv")

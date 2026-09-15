
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _THIS_DIR
sys.path[:] = [path for path in sys.path
               if os.path.abspath(path or os.getcwd()) != _THIS_DIR]

import argparse
import gc
import hashlib
import json
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm.auto import tqdm

PROJECT_ROOT = Path(_PROJECT_ROOT)


warnings.filterwarnings(
    "ignore",
    message=r".*Neither `use_rmm` nor `use_cuda_async_pool` is enabled.*",
    category=UserWarning,
    module=r"xgboost\..*",
)
xgb.set_config(verbosity=0)


def project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


class ProgressCallback(xgb.callback.TrainingCallback):

    def __init__(self, path: Path | None, total: int, description: str):
        self.path = path
        self.total = total
        self.description = description
        self.bar = None

    def before_training(self, model):
        self.bar = tqdm(total=self.total, desc=self.description, unit="round", dynamic_ncols=True)
        return model

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        if self.path is not None and ((epoch + 1) % 10 == 0 or epoch == 0):
            values = []
            for dataset, metrics in evals_log.items():
                values.extend(f"{dataset}-{name}={metric[-1]:.6f}"
                              for name, metric in metrics.items())
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(f"round={epoch + 1}; " + "; ".join(values) + "\n")
        if self.bar is not None:
            self.bar.update(1)
            latest = {f"{dataset}-{name}": f"{metric[-1]:.4f}"
                      for dataset, metrics in evals_log.items()
                      for name, metric in metrics.items()}
            self.bar.set_postfix(latest)
        return False

    def after_training(self, model):
        if self.bar is not None:
            self.bar.close()
        return model


SCENARIO_DIRS = {
    "single": PROJECT_ROOT / "single_cold_start",
    "dual": PROJECT_ROOT / "dual_cold_start",
    "warm": PROJECT_ROOT / "warm_start",
}
PAIR_SPLITS = ("train", "val", "test")


def scenario_paths(mode: str = "single", features_dir: Path | str | None = None,
                   results_dir: Path | str | None = None) -> tuple[Path, Path]:
    if mode not in SCENARIO_DIRS:
        raise ValueError(f"unknown mode {mode!r}; expected {sorted(SCENARIO_DIRS)}")
    base = SCENARIO_DIRS[mode]
    features = project_path(features_dir) if features_dir else base / "features"
    results = project_path(results_dir) if results_dir else base / "results"
    return features, results


def load_pair_split(features_dir: Path, split: str
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_path = features_dir / f"{split}.npy"
    label_path = features_dir / f"{split}_labels.npy"
    weight_path = features_dir / f"{split}_sample_weight.npy"
    for path in (feature_path, label_path, weight_path):
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run build_fold_features.py first")
    features = np.load(feature_path).astype(np.float32, copy=False)
    labels = np.load(label_path).astype(np.int32)
    weights = np.load(weight_path).astype(np.float32)
    if features.ndim != 2 or len(features) != len(labels) or len(weights) != len(labels):
        raise ValueError(f"{split}: inconsistent shapes "
                         f"X={features.shape} y={labels.shape} w={weights.shape}")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{split}: labels must contain both classes")
    if not np.isfinite(features).all():
        raise ValueError(f"{split}: features contain NaN or Inf")
    return features, labels, weights


def pair_level_metrics(labels: np.ndarray, probabilities: np.ndarray,
                       threshold: float = 0.5) -> dict:
    from sklearn.metrics import (confusion_matrix, matthews_corrcoef,
                                 roc_auc_score)
    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(labels)
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) > 0 else 0.0)
    mcc = (matthews_corrcoef(labels, predicted)
           if len(np.unique(labels)) > 1 else 0.0)
    return {
        "pairs": int(len(labels)), "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()), "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "ACC": float(accuracy), "SEN": float(sensitivity),
        "SPE": float(specificity), "PRE": float(precision),
        "F1": float(f1), "MCC": float(mcc),
        "ROC_AUC": float(roc_auc_score(labels, probabilities)),
        "AUPR": float(average_precision_score(labels, probabilities)),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    usable = np.isfinite(thresholds)
    if not usable.any():
        return 0.5
    youden = tpr[usable] - fpr[usable]
    return float(thresholds[usable][int(np.argmax(youden))])


CALIBRATION_METHODS = ("none", "isotonic")


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)


class IsotonicCalibrator:

    def __init__(self, probabilities: np.ndarray, labels: np.ndarray):
        from sklearn.isotonic import IsotonicRegression
        self.model = IsotonicRegression(out_of_bounds="clip").fit(
            _clip_probabilities(probabilities), np.asarray(labels).astype(np.int32))

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return _clip_probabilities(self.model.predict(_clip_probabilities(probabilities)))


def drug_level_halves(frame: pd.DataFrame, seed: int = 0
                      ) -> tuple[np.ndarray, np.ndarray]:
    total = len(frame)
    if not {"drug1_id", "drug2_id"}.issubset(frame.columns):
        index = np.arange(total)
        np.random.RandomState(seed).shuffle(index)
        mask_a = np.zeros(total, dtype=bool)
        mask_a[index[: total // 2]] = True
        return mask_a, ~mask_a
    drug1 = frame["drug1_id"].to_numpy()
    drug2 = frame["drug2_id"].to_numpy()
    drugs = np.unique(np.concatenate([drug1, drug2]))
    chosen = set(np.random.RandomState(seed)
                 .choice(drugs, size=max(1, len(drugs) // 2), replace=False).tolist())
    mask_a = np.array([(a in chosen) and (b in chosen) for a, b in zip(drug1, drug2)])
    mask_b = np.array([(a not in chosen) and (b not in chosen) for a, b in zip(drug1, drug2)])
    return mask_a, mask_b


def checkpoint_curves(model, matrices, labels: dict[str, np.ndarray], calibrator,
                      mask_a: np.ndarray, holdout: np.ndarray, rounds: int,
                      checkpoints: int = 40) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, log_loss
    steps = np.unique(np.linspace(max(1, rounds // checkpoints), rounds,
                                 checkpoints).astype(int))
    rows: list[dict] = []
    for step in steps:
        iteration_range = (0, int(step))
        probabilities = {
            split: np.asarray(model.predict(matrices[split], iteration_range=iteration_range))
            for split in ("train", "val")}
        score = (calibrator.transform if calibrator is not None
                 else (lambda p: _clip_probabilities(p)))
        row = {
            "round": int(step),
            "interactions": int(step) * len(labels["train"]),
            "train_loss": float(log_loss(labels["train"],
                                         _clip_probabilities(probabilities["train"]))),
            "val_loss": float(log_loss(labels["val"],
                                       _clip_probabilities(probabilities["val"]))),
            "train_acc": float(accuracy_score(labels["train"],
                                              (score(probabilities["train"]) >= 0.5).astype(int))),
            "val_acc": float(accuracy_score(labels["val"],
                                            (score(probabilities["val"]) >= 0.5).astype(int))),
        }
        if calibrator is not None and int(holdout.sum()) > 0:
            calibrator_step = IsotonicCalibrator(probabilities["val"][mask_a],
                                                 labels["val"][mask_a])
            row["holdout_loss_raw"] = float(log_loss(
                labels["val"][holdout],
                _clip_probabilities(probabilities["val"][holdout])))
            row["holdout_loss_calibrated"] = float(log_loss(
                labels["val"][holdout],
                _clip_probabilities(
                    calibrator_step.transform(probabilities["val"][holdout]))))
        rows.append(row)
    return pd.DataFrame(rows)


def performance_metrics_table(metrics: dict, metrics_raw: dict,
                              results_dir: Path) -> Path:
    rows: list[dict] = []
    for split in PAIR_SPLITS:
        for scale_name, source in (("calibrated", metrics),
                                   ("uncalibrated", metrics_raw)):
            item = dict(source[split])
            row: dict = {"split": split, "scale": scale_name}
            for key in ("threshold", "pairs", "positives", "negatives", "tp", "fp",
                        "tn", "fn", "ACC", "SEN", "SPE", "PRE", "F1", "MCC",
                        "ROC_AUC", "AUPR"):
                row[key] = item.get(key)
            rows.append(row)
    table = pd.DataFrame(rows)
    path = results_dir / "performance_metrics.csv"
    table.to_csv(path, index=False)
    return path


def plot_performance_reference(results_dir: Path, curves: pd.DataFrame,
                               predictions: dict[str, pd.DataFrame],
                               labels: dict[str, np.ndarray]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)

    colors = {"train": "#1F77B4", "val": "#D95319", "test": "#E8A838"}
    names = {"train": "Training", "val": "Validation", "test": "Test"}
    figure, axes = plt.subplots(1, 2, figsize=(11.7, 4.8))

    axes[0].plot(curves["interactions"], curves["train_loss"],
                 color=colors["train"], linewidth=1.5, label="Training")
    axes[0].plot(curves["interactions"], curves["val_loss"],
                 color=colors["val"], linewidth=1.5, label="Validation")
    axes[0].set_title("A", loc="left", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Interaction")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot([0, 1], [0, 1], linestyle=":", color="#BBBBBB", linewidth=1)
    for split in ("train", "val", "test"):
        split_labels = labels[split]
        probability = predictions[split]["probability"].to_numpy()
        fpr, tpr, _ = roc_curve(split_labels, probability)
        precision, recall, _ = precision_recall_curve(split_labels, probability)
        axes[1].plot(fpr, tpr, color=colors[split], linewidth=1.5, linestyle="-",
                     label=f"{names[split]} ROC (AUC={roc_auc_score(split_labels, probability):.3f})")
        axes[1].plot(recall, precision, color=colors[split], linewidth=1.5, linestyle="-",
                     label=f"{names[split]} PRC (AUPR={average_precision_score(split_labels, probability):.3f})")
    axes[1].text(0.30, 0.86, "ROC", fontsize=11, fontweight="bold")
    axes[1].text(0.68, 0.965, "PRC", fontsize=11, fontweight="bold")
    axes[1].set_title("B", loc="left", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("False positive rate / Recall")
    axes[1].set_ylabel("True positive rate / Precision")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=6.5, frameon=False, loc="lower left")

    figure.tight_layout()
    path = results_dir / "model_performance.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path





def learning_rate_schedule(base_rate: float, horizon: int):
    horizon = max(1, int(horizon))

    def schedule(round_index: int) -> float:
        return float(base_rate / (1.0 + round_index / horizon))

    return schedule


def train_pair_level(mode: str = "single",
                     features_dir: Path | str | None = None,
                     results_dir: Path | str | None = None, *,
                     rounds: int = 500, learning_rate: float = 0.05,
                     max_depth: int = 6, subsample: float = 0.8,
                     colsample_bytree: float = 0.8, min_child_weight: float = 1.0,
                     reg_alpha: float = 0.0, reg_lambda: float = 1.0,
                     gamma: float = 0.0, early_stopping_rounds: int = 50,
                     early_stopping_metric: str = "auc",
                     nthread: int = 8, seed: int = 42, threshold: float = 0.5,
                     threshold_mode: str = "val", calibrate: str = "isotonic",
                     lr_decay: str = "none", lr_decay_horizon: int = 500,
                     curve_checkpoints: int = 40,
                     device: str = "cpu", force: bool = False) -> dict:
    features, results = scenario_paths(mode, features_dir, results_dir)
    results.mkdir(parents=True, exist_ok=True)
    model_path = results / "xgboost_model.json"
    if model_path.exists() and not force:
        raise FileExistsError(f"{model_path} exists; pass --force to retrain")


    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in PAIR_SPLITS:
        features_split, labels, weights = load_pair_split(features, split)
        splits[split] = (features_split, labels, weights)

    params = {
        "objective": "binary:logistic",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "gamma": gamma,
        "seed": seed,
        "eval_metric": ["logloss", "auc", "aucpr"],
        "tree_method": "hist",
        "max_bin": 256,
        "verbosity": 0,
        "nthread": nthread,
        "device": device,
    }
    monitor = early_stopping_metric.strip().lower()
    if monitor not in {"auc", "aucpr", "logloss"}:
        raise ValueError("early_stopping_metric must be 'auc', 'aucpr' or 'logloss'")
    if calibrate not in CALIBRATION_METHODS:
        raise ValueError(f"calibrate must be one of {CALIBRATION_METHODS}")

    started = time.time()
    matrices = {
        split: xgb.DMatrix(splits[split][0], label=splits[split][1],
                           weight=splits[split][2], nthread=nthread)
        for split in PAIR_SPLITS
    }
    history: dict = {}
    callbacks = [ProgressCallback(None, rounds, f"{mode} XGBoost")]
    if early_stopping_rounds > 0:
        callbacks.append(xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, metric_name=monitor, data_name="val",
            maximize=monitor != "logloss", save_best=False))
    if lr_decay != "none":
        if lr_decay != "linear":
            raise ValueError("lr_decay must be 'linear' or 'none'")
        callbacks.append(xgb.callback.LearningRateScheduler(
            learning_rate_schedule(learning_rate, lr_decay_horizon)))
    model = xgb.train(
        params, matrices["train"], num_boost_round=rounds,
        evals=[(matrices["train"], "train"), (matrices["val"], "val")],
        evals_result=history, verbose_eval=False, callbacks=callbacks,
    )
    trained_rounds = len(history["train"]["logloss"])
    best_iteration = int(getattr(model, "best_iteration", trained_rounds - 1))

    iteration_range = (0, best_iteration + 1)
    predictions: dict[str, pd.DataFrame] = {}
    for split in PAIR_SPLITS:
        probabilities = np.asarray(
            model.predict(matrices[split], iteration_range=iteration_range))
        frame = pd.DataFrame({
            "pair_index": np.arange(len(probabilities), dtype=np.int64),
            "label": splits[split][1],
            "probability": probabilities,
        })
        meta_path = features / f"{split}_pair_meta.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            if len(meta) == len(frame):
                for column in ("drug1_id", "drug2_id", "source"):
                    if column in meta.columns:
                        frame[column] = meta[column].to_numpy()
        predictions[split] = frame

    primary_key = "probability"
    calibration_report: dict = {"method": calibrate, "enabled": calibrate != "none"}
    holdout = np.ones(len(predictions["val"]), dtype=bool)
    mask_a = np.ones(len(predictions["val"]), dtype=bool)
    calibrator = None
    if calibrate != "none":
        mask_a, mask_b = drug_level_halves(predictions["val"], seed)
        holdout = mask_b if int(mask_b.sum()) >= 200 else ~mask_a
        calibrator = IsotonicCalibrator(
            predictions["val"]["probability"].to_numpy()[mask_a],
            predictions["val"]["label"].to_numpy()[mask_a])
        for split in PAIR_SPLITS:
            frame = predictions[split]
            frame["probability_calibrated"] = calibrator.transform(
                frame["probability"].to_numpy())
        primary_key = "probability_calibrated"
        from sklearn.metrics import brier_score_loss, log_loss
        val_labels = predictions["val"]["label"].to_numpy()
        val_raw = predictions["val"]["probability"].to_numpy()
        val_cal = predictions["val"][primary_key].to_numpy()
        test_labels = predictions["test"]["label"].to_numpy()
        test_raw = predictions["test"]["probability"].to_numpy()
        test_cal = predictions["test"][primary_key].to_numpy()
        calibration_report.update({
            "fit_rows": int(mask_a.sum()), "holdout_rows": int(holdout.sum()),
            "val_holdout": {
                "logloss_raw": float(log_loss(val_labels[holdout],
                                              _clip_probabilities(val_raw[holdout]))),
                "logloss_calibrated": float(log_loss(val_labels[holdout],
                                                     _clip_probabilities(val_cal[holdout]))),
                "brier_raw": float(brier_score_loss(val_labels[holdout],
                                                    _clip_probabilities(val_raw[holdout]))),
                "brier_calibrated": float(brier_score_loss(val_labels[holdout],
                                                           _clip_probabilities(val_cal[holdout]))),
            },
            "test": {
                "logloss_raw": float(log_loss(test_labels, _clip_probabilities(test_raw))),
                "logloss_calibrated": float(log_loss(test_labels, _clip_probabilities(test_cal))),
                "brier_raw": float(brier_score_loss(test_labels, _clip_probabilities(test_raw))),
                "brier_calibrated": float(brier_score_loss(test_labels, _clip_probabilities(test_cal))),
            },
        })

    if calibrate != "none":
        selected_threshold = float(select_threshold(
            predictions["val"]["label"].to_numpy()[holdout],
            predictions["val"][primary_key].to_numpy()[holdout]))
    elif threshold_mode == "val":
        selected_threshold = float(select_threshold(
            predictions["val"]["label"].to_numpy(),
            predictions["val"][primary_key].to_numpy()))
    elif threshold_mode == "fixed":
        selected_threshold = float(threshold)
    else:
        raise ValueError("threshold_mode must be 'val' or 'fixed'")

    metrics: dict[str, dict] = {}
    metrics_fixed: dict[str, dict] = {}
    metrics_own: dict[str, dict] = {}
    metrics_raw: dict[str, dict] = {}
    for split in PAIR_SPLITS:
        labels = predictions[split]["label"].to_numpy()
        probabilities = predictions[split][primary_key].to_numpy()
        raw_probabilities = predictions[split]["probability"].to_numpy()
        metrics[split] = pair_level_metrics(labels, probabilities, selected_threshold)
        metrics_fixed[split] = pair_level_metrics(labels, probabilities, threshold)
        own_threshold = float(select_threshold(labels, probabilities))
        metrics_own[split] = pair_level_metrics(labels, probabilities, own_threshold)
        metrics_raw[split] = pair_level_metrics(labels, raw_probabilities, threshold)
    labels_by_split = {split: splits[split][1] for split in PAIR_SPLITS}
    curves = checkpoint_curves(
        model, matrices, labels_by_split, calibrator, mask_a, holdout,
        trained_rounds, checkpoints=max(2, int(curve_checkpoints)))
    performance_metrics_table(metrics, metrics_raw, results)
    for legacy in ("loss_curve.png", "auroc_curve.png", "aupr_curve.png",
                   "model_performance_curves.png", "calibration_diagnostics.png"):
        stale = results / legacy
        if stale.exists():
            stale.unlink()
    figure_path = plot_performance_reference(results, curves, predictions,
                                             labels_by_split)
    model.save_model(model_path)
    elapsed = round(time.time() - started, 1)
    summary = {
        "mode": mode, "features_dir": str(features), "results_dir": str(results),
        "trained_rounds": trained_rounds, "best_iteration": best_iteration + 1,
        "elapsed_seconds": elapsed, "model": str(model_path),
        "figure": str(figure_path), "metrics": metrics,
        "threshold_mode": threshold_mode, "threshold": selected_threshold,
        "calibrate": calibrate, "primary_scale": primary_key,
        "calibration": calibration_report,
        "early_stopping_monitor": f"val-{monitor}",
        "train_val_auc_gap": float(metrics["train"]["ROC_AUC"]
                                   - metrics["val"]["ROC_AUC"]),
        "metrics_at_fixed_threshold": metrics_fixed,
        "metrics_uncalibrated": metrics_raw,
    }
    del matrices
    gc.collect()
    return summary


TUNE_SPACE: dict[str, list] = {
    "learning_rate": [0.005, 0.01],
    "max_depth": [1, 2],
    "subsample": [0.6, 0.8],
    "colsample_bytree": [0.1, 0.2],
    "reg_alpha": [1.0, 5.0],
    "reg_lambda": [50.0, 200.0],
    "min_child_weight": [100.0, 300.0],
}
PLATEAU_TOLERANCE = 0.02


def trial_id(config: dict) -> str:
    payload = json.dumps({key: config[key] for key in sorted(config)}, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def sample_configs(count: int, rng: np.random.RandomState, space: dict | None = None,
                   existing_ids: set | None = None) -> list[dict]:
    space = {key: list(values) for key, values in (space or TUNE_SPACE).items()}
    seen = set(existing_ids or [])
    configs: list[dict] = []
    guard = 0
    while len(configs) < count and guard < count * 500:
        guard += 1
        config = {key: values[int(rng.randint(len(values)))]
                  for key, values in space.items()}
        identifier = trial_id(config)
        if identifier in seen:
            continue
        seen.add(identifier)
        configs.append(config)
    return configs


def tune_pair_level(mode: str = "single", features_dir: Path | str | None = None,
                    tune_dir: Path | str | None = None, *, trials: int = 8,
                    seed: int = 42, rounds: int = 1200,
                    early_stopping_rounds: int = 100,
                    early_stopping_metric: str = "logloss",
                    nthread: int = 8, device: str = "cpu",
                    space: dict | None = None) -> dict:
    features, results = scenario_paths(mode, features_dir, None)
    tuning = Path(tune_dir) if tune_dir is not None else (SCENARIO_DIRS[mode] / "tuning")
    tuning.mkdir(parents=True, exist_ok=True)
    for name in ("training_history.csv", "training_config.json",
                 "val_metrics.json", "test_metrics.json"):
        source, target = results / name, tuning / f"before_{name}"
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)

    results_csv = tuning / "tuning_results.csv"
    rows: list[dict] = []
    done: set[str] = set()
    if results_csv.exists():
        previous = pd.read_csv(results_csv)
        rows = previous.to_dict("records")
        done = set(previous["trial_id"].astype(str))

    matrices = {}
    labels: dict[str, np.ndarray] = {}
    for split in PAIR_SPLITS:
        X, y, w = load_pair_split(features, split)
        matrices[split] = xgb.DMatrix(X, label=y, weight=w, nthread=nthread)
        labels[split] = y

    rng = np.random.RandomState(seed)
    configs = sample_configs(trials, rng, space, done)
    for index, config in enumerate(configs, start=1):
        params = {
            "objective": "binary:logistic", "eval_metric": ["logloss", "auc"],
            "tree_method": "hist", "max_bin": 256, "verbosity": 0,
            "nthread": nthread, "device": device, "seed": seed,
            "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
            "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1.0,
            "max_depth": 3, "learning_rate": 0.05,
        }
        params.update(config)
        history: dict = {}
        callbacks = [xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, metric_name=early_stopping_metric,
            data_name="val", maximize=early_stopping_metric != "logloss",
            save_best=False)]
        started = time.time()
        booster = xgb.train(params, matrices["train"], num_boost_round=rounds,
                            evals=[(matrices["train"], "train"), (matrices["val"], "val")],
                            evals_result=history, verbose_eval=False, callbacks=callbacks)
        val_loss = np.asarray(history["val"]["logloss"], dtype=np.float64)
        train_loss = np.asarray(history["train"]["logloss"], dtype=np.float64)
        val_auc = np.asarray(history["val"]["auc"], dtype=np.float64)
        train_auc = np.asarray(history["train"]["auc"], dtype=np.float64)
        best_round = int(np.argmin(val_loss))
        tail = np.diff(val_loss[max(1, len(val_loss) // 10):])
        descent = np.diff(val_loss[:best_round + 1]) if best_round > 0 else np.asarray([])
        plateau_end = int(np.max(np.flatnonzero(
            val_loss <= val_loss.min() + PLATEAU_TOLERANCE)))
        iteration_range = (0, best_round + 1)
        probabilities = {
            split: np.asarray(booster.predict(matrices[split],
                                              iteration_range=iteration_range))
            for split in PAIR_SPLITS}
        threshold = float(select_threshold(labels["val"], probabilities["val"]))
        scored = {split: pair_level_metrics(labels[split], probabilities[split], threshold)
                  for split in PAIR_SPLITS}
        unique_fraction = float(len(np.unique(np.round(probabilities["val"], 6)))
                                / max(1, len(probabilities["val"])))
        identifier = trial_id(config)
        pd.DataFrame({
            "round": np.arange(1, len(val_loss) + 1),
            "train_combo_logloss": train_loss, "val_combo_logloss": val_loss,
            "train_combo_auc": train_auc, "val_combo_auc": val_auc,
        }).to_csv(tuning / f"hist_{identifier}.csv", index=False)
        row = dict(config)
        row.update({
            "trial_id": identifier,
            "rounds": int(len(val_loss)),
            "best_round": best_round + 1,
            "plateau_end": plateau_end + 1,
            "best_val_logloss": float(val_loss.min()),
            "first_val_logloss": float(val_loss[0]),
            "final_val_logloss": float(val_loss[-1]),
            "rise_after_min": float(val_loss[-1] - val_loss.min()),
            "plateau_ok": bool(val_loss[-1] - val_loss.min() <= PLATEAU_TOLERANCE),
            "best_val_auc": float(val_auc.max()),
            "val_auc_at_best_loss": float(val_auc[best_round]),
            "val_aupr": float(scored["val"]["AUPR"]),
            "val_mcc": float(scored["val"]["MCC"]),
            "test_auc": float(scored["test"]["ROC_AUC"]),
            "test_aupr": float(scored["test"]["AUPR"]),
            "test_mcc": float(scored["test"]["MCC"]),
            "train_mcc": float(scored["train"]["MCC"]),
            "threshold": threshold,
            "unique_score_fraction": unique_fraction,
            "train_val_loss_gap": float(val_loss[best_round] - train_loss[best_round]),
            "monotone_fraction": float(np.mean(tail <= 0)) if tail.size else 0.0,
            "descent_fraction": float(np.mean(descent <= 0)) if descent.size else 0.0,
            "elapsed_seconds": round(time.time() - started, 1),
        })
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_csv, index=False)

    table = pd.DataFrame(rows).sort_values(
        ["plateau_ok", "val_aupr", "val_auc_at_best_loss"],
        ascending=[False, False, False])
    table.to_csv(results_csv, index=False)
    best = table.iloc[0].to_dict()
    best_config = {key: best[key] for key in TUNE_SPACE if key in best}
    patience_report = check_patience(
        best_config, matrices, rounds, early_stopping_metric, nthread, device, seed,
        candidates=(50, 100, 200)) if early_stopping_metric else None
    payload = {
        "metric": "validation log-loss (lower is better) + curve shape + AUPR",
        "selection_rule": ("plateau first (final val loss within "
                           f"{PLATEAU_TOLERANCE} of its minimum), then max val AUPR, "
                           "then max val AUC"),
        "trials_total": int(len(table)),
        "best_config": best_config,
        "best_trial": {key: best[key] for key in
                       ("trial_id", "best_round", "plateau_end", "best_val_logloss",
                        "first_val_logloss", "rise_after_min", "plateau_ok",
                        "val_auc_at_best_loss", "val_aupr", "val_mcc", "test_auc",
                        "test_aupr", "test_mcc", "train_mcc",
                        "unique_score_fraction", "threshold",
                        "train_val_loss_gap", "descent_fraction") if key in best},
        "early_stopping_rounds_check": patience_report,
    }
    (tuning / "best_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    figures = []
    return {"tune_dir": str(tuning), "config": best_config, "selected": payload,
            "figures": [str(path) for path in figures], "table": str(results_csv)}


def check_patience(config: dict, matrices: dict, rounds: int, metric: str,
                   nthread: int, device: str, seed: int,
                   candidates=(50, 100, 200)) -> dict:
    report: dict = {}
    for patience in candidates:
        params = {
            "objective": "binary:logistic", "eval_metric": ["logloss", "auc"],
            "tree_method": "hist", "max_bin": 256, "verbosity": 0,
            "nthread": nthread, "device": device, "seed": seed, "gamma": 0.0,
        }
        params.update(config)
        history: dict = {}
        xgb.train(params, matrices["train"], num_boost_round=rounds,
                  evals=[(matrices["train"], "train"), (matrices["val"], "val")],
                  evals_result=history, verbose_eval=False,
                  callbacks=[xgb.callback.EarlyStopping(
                      rounds=patience, metric_name=metric, data_name="val",
                      maximize=metric != "logloss", save_best=False)])
        val_loss = np.asarray(history["val"]["logloss"], dtype=np.float64)
        best_round = int(np.argmin(val_loss)) + 1
        report[str(patience)] = {"rounds": int(len(val_loss)),
                                 "best_round": best_round,
                                 "best_val_logloss": float(val_loss.min())}
    return report





def _pair_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XGBoost on pre-assembled drug-pair feature matrices")
    parser.add_argument("--mode", choices=sorted(SCENARIO_DIRS), default="single",
                        help="single (Single_cold_start), dual (Dual_cold_start) or "
                             "warm (warm_start)")
    parser.add_argument("--features-dir", type=Path, default=None,
                        help="override the input feature directory")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="override the output results directory")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--early-stopping-metric", choices=("auc", "aucpr", "logloss"),
                        default="aucpr",
                        help="validation metric watched for early stopping; "
                             "'aucpr'/'auc' train the ranking fully, 'logloss' stops "
                             "at the point of best probability calibration")
    parser.add_argument("--calibrate", choices=CALIBRATION_METHODS,
                        default="isotonic",
                        help="post-hoc probability calibration fitted on a "
                             "drug-disjoint half of the validation split")
    parser.add_argument("--lr-decay", choices=("none", "linear"), default="none",
                        help="decay the learning rate over boosting rounds; 'linear' "
                             "uses lr_t = lr0 / (1 + t / horizon) so late rounds "
                             "contribute less and the validation loss flattens")
    parser.add_argument("--lr-decay-horizon", type=int, default=500)
    parser.add_argument("--nthread", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="fixed decision threshold kept as a reference")
    parser.add_argument("--threshold-mode", choices=("val", "fixed"), default="val",
                        help="'val': Youden-J threshold chosen on the validation "
                             "split (default); 'fixed': use --threshold")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tune", action="store_true",
                        help="run a random hyper-parameter search instead of one fit")
    parser.add_argument("--tune-trials", type=int, default=8,
                        help="number of new trials per invocation (resumable)")
    parser.add_argument("--tune-seed", type=int, default=42)
    parser.add_argument("--tune-dir", type=Path, default=None)
    parser.add_argument("--tune-space", type=Path, default=None,
                        help="optional JSON file overriding the search space")
    parser.add_argument("--tune-rounds", type=int, default=1200,
                        help="max boosting rounds per trial")
    parser.add_argument("--curve-checkpoints", type=int, default=40,
                        help="checkpoints used for the ACC/Loss curve panels")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _pair_arg_parser().parse_args(argv)
    if args.tune:
        space = None
        if args.tune_space is not None:
            space = json.loads(Path(args.tune_space).read_text(encoding="utf-8"))
        outcome = tune_pair_level(
            args.mode, args.features_dir, args.tune_dir,
            trials=args.tune_trials, seed=args.tune_seed,
            rounds=args.tune_rounds, early_stopping_rounds=args.early_stopping,
            early_stopping_metric=args.early_stopping_metric,
            nthread=args.nthread, device=args.device, space=space)
        json.dumps({'tune_dir': outcome['tune_dir'], 'best_config': outcome['config'], 'best_trial': outcome['selected']['best_trial'], 'figures': outcome['figures']}, ensure_ascii=False, indent=2)
        return
    summary = train_pair_level(
        args.mode, args.features_dir, args.results_dir, rounds=args.rounds,
        learning_rate=args.learning_rate, max_depth=args.max_depth,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight, reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda, gamma=args.gamma,
        early_stopping_rounds=args.early_stopping,
        early_stopping_metric=args.early_stopping_metric, nthread=args.nthread,
        seed=args.seed, threshold=args.threshold,
        threshold_mode=args.threshold_mode, calibrate=args.calibrate,
        lr_decay=args.lr_decay, lr_decay_horizon=args.lr_decay_horizon,
        curve_checkpoints=args.curve_checkpoints,
        device=args.device, force=args.force)
    json.dumps({'mode': summary['mode'], 'best_iteration': summary['best_iteration'], 'elapsed_seconds': summary['elapsed_seconds']}, ensure_ascii=False)


if __name__ == "__main__":
    main()

"""Train and evaluate the KGP-DC cold-start experiment.

Safe default: ``python main.py`` only checks the existing experiment.
Use ``python main.py train --force`` to intentionally start full training.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
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

from feature import (
    ComboDataIter,
    FeatureStore,
    load_combo_index,
    predict_and_aggregate,
    predict_and_aggregate_many,
    validate_pair_weights,
    generate_negative_pairs,
    build_combo_indexes,
    generate_warm_pairs,
    build_warm_combo_index,
)


ROOT = Path(__file__).parent
RESULTS_ROOT = ROOT / "results"
COLD_RESULTS = RESULTS_ROOT / "single_cold"
WARM_RESULTS = RESULTS_ROOT / "warm_kg"
DEFAULT_COMBO_DIR = Path("data/combo_index_single_cold")
SEED = 42

warnings.filterwarnings(
    "ignore",
    message=r".*Neither `use_rmm` nor `use_cuda_async_pool` is enabled.*",
    category=UserWarning,
    module=r"xgboost\..*",
)
xgb.set_config(verbosity=0)


def pair_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Return threshold-free pair-level metrics."""
    return {
        "n_pairs": int(len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "aupr": float(average_precision_score(y_true, y_prob)),
    }


def project_path(path: Path | str) -> Path:
    """Resolve a command-line path relative to this experiment directory."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def status(combo_dir: Path | str = DEFAULT_COMBO_DIR) -> None:
    """Validate core inputs and any prepared single-cold indexes."""
    store = FeatureStore.load(ROOT, require_kg=False)
    print(f"SMILES={store.smiles.shape}; targets={store.targets.shape}; pair_dim=3584")

    combo_directory = project_path(combo_dir)
    expected = [combo_directory / f"{split}_combo_index.npy"
                for split in ("train", "val", "test")]
    missing = [path for path in expected if not path.exists()]
    if missing:
        print("Prepared indexes: missing")
        print("Run: python feature.py prepare-single-cold-all --seed 42")
        return

    for split in ("train", "val", "test"):
        rows = load_combo_index(split, ROOT, combo_dir)
        if len(rows) == 0:
            raise ValueError(f"{split} combo index is empty")
        validate_pair_weights(rows)
        if not np.isfinite(rows["sample_weight"]).all():
            raise ValueError(f"{split} sample weights contain NaN or Inf")
        pair_labels = pd.DataFrame({
            "pair": rows["pair_index"],
            "label": rows["label"],
        }).drop_duplicates()
        if pair_labels["pair"].duplicated().any():
            raise ValueError(f"{split} contains inconsistent pair labels")
        counts = pair_labels["label"].value_counts().to_dict()
        if counts.get(0) != counts.get(1):
            raise ValueError(f"{split} pairs must be balanced 1:1")
        print(f"{split}: {len(rows):,} rows; {len(pair_labels):,} pairs [OK]")

    output = [COLD_RESULTS / "xgboost_model.json", COLD_RESULTS / "val_metrics.json",
              COLD_RESULTS / "test_metrics.json", WARM_RESULTS / "xgboost_model.json"]
    present = [str(path.relative_to(ROOT)) for path in output if path.exists()]
    missing = [str(path.relative_to(ROOT)) for path in output if not path.exists()]
    print(f"Input contract: OK; existing outputs={present or 'none'}")
    if missing:
        print(f"Training outputs not yet present: {missing}")


class ProgressCallback(xgb.callback.TrainingCallback):
    """Persist progress while XGBoost is running."""

    def __init__(self, path: Path, total: int, description: str):
        self.path = path
        self.total = total
        self.description = description
        self.bar = None

    def before_training(self, model):
        self.bar = tqdm(total=self.total, desc=self.description, unit="round", dynamic_ncols=True)
        return model

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        if (epoch + 1) % 10 == 0 or epoch == 0:
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


def evaluate(cold_results: Path = COLD_RESULTS) -> None:
    """Evaluate without selecting a classification threshold."""
    val = pd.read_csv(cold_results / "val_pair_predictions.csv")
    test = pd.read_csv(cold_results / "test_pair_predictions.csv")
    val_metrics = pair_metrics(val["label"].to_numpy(), val["probability"].to_numpy())
    test_metrics = pair_metrics(test["label"].to_numpy(), test["probability"].to_numpy())
    for name, values in (("val", val_metrics), ("test", test_metrics)):
        with open(cold_results / f"{name}_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2, ensure_ascii=False)
    print(f"Test AUC={test_metrics['roc_auc']:.4f}, AUPR={test_metrics['aupr']:.4f}")


def _load_loss_history(results_dir: Path) -> pd.DataFrame:
    """Load per-round loss, with compatibility for older progress logs."""
    history_path = results_dir / "training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        loss_columns = [
            column for column in ("train_combo_logloss", "val_combo_logloss")
            if column in history.columns
        ]
        if loss_columns:
            return history[["round", *loss_columns]].copy()

    progress_path = results_dir / "training_progress.log"
    if not progress_path.exists():
        raise FileNotFoundError(f"No loss history found in {results_dir}")
    pattern = re.compile(
        r"round=(?P<round>\d+);.*?train-logloss=(?P<train>[0-9.eE+-]+)"
        r"(?:;.*?val-logloss=(?P<val>[0-9.eE+-]+))?"
    )
    records = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            record = {
                "round": int(match.group("round")),
                "train_combo_logloss": float(match.group("train")),
            }
            if match.group("val") is not None:
                record["val_combo_logloss"] = float(match.group("val"))
            records.append(record)
    if not records:
        raise ValueError(f"No logloss values found in {progress_path}")
    return pd.DataFrame(records).drop_duplicates("round").sort_values("round")


def plot_performance(run_name: str = "single_cold") -> Path:
    """Draw only Loss, ROC, and precision-recall curves for an existing run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = RESULTS_ROOT / run_name
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {results_dir}")

    loss = _load_loss_history(results_dir)
    split_colors = {"train": "#2878B5", "val": "#D95319", "test": "#E5AE00"}
    split_labels = {"train": "Training", "val": "Validation", "test": "Test"}
    predictions = {}
    for split in ("train", "val", "test"):
        path = results_dir / f"{split}_pair_predictions.csv"
        if path.exists():
            frame = pd.read_csv(path)
            required = {"label", "probability"}
            if not required.issubset(frame.columns):
                raise ValueError(f"{path} must contain {sorted(required)}")
            if frame["label"].nunique() != 2:
                raise ValueError(f"{path} must contain both classes")
            predictions[split] = frame
    if "val" not in predictions or "test" not in predictions:
        raise FileNotFoundError("Both val and test pair-level predictions are required")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for split in ("train", "val"):
        column = f"{split}_combo_logloss"
        if column in loss.columns:
            ax.plot(loss["round"], loss[column], color=split_colors[split],
                    linewidth=1.8, label=split_labels[split])
    ax.set_title("Loss")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Log loss")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[1]
    for split, frame in predictions.items():
        labels = frame["label"].to_numpy()
        probabilities = frame["probability"].to_numpy()
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        ax.plot(fpr, tpr, color=split_colors[split], linewidth=1.8,
                label=f"{split_labels[split]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1)
    ax.set_title("ROC")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for split, frame in predictions.items():
        labels = frame["label"].to_numpy()
        probabilities = frame["probability"].to_numpy()
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        aupr = average_precision_score(labels, probabilities)
        ax.plot(recall, precision, color=split_colors[split], linewidth=1.8,
                label=f"{split_labels[split]} (AUPR={aupr:.3f})")
    ax.set_title("Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)

    for label, ax in zip(("A", "B", "C"), axes):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="top")
    fig.suptitle(f"XGBoost model performance: {run_name}", fontsize=13)
    fig.tight_layout()
    output = results_dir / "model_performance_curves.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def select_pair_level_model(
    model: xgb.Booster,
    combos: dict[str, np.ndarray],
    store: FeatureStore,
    results_dir: Path,
    rounds: int,
    batch_size: int,
    coarse_step: int = 10,
) -> tuple[xgb.Booster, dict]:
    """Select tree count using pair-level validation ROC-AUC, then evaluate once."""
    if coarse_step < 1:
        raise ValueError("coarse_step must be positive")

    def validation_metrics(tree_counts: list[int]) -> list[dict]:
        predictions = predict_and_aggregate_many(
            model, combos["val"], store, tree_counts, batch_size, "cold"
        )
        return [
            {"round": tree_count,
             **pair_metrics(predictions[tree_count][1], predictions[tree_count][2])}
            for tree_count in tree_counts
        ]

    coarse_rounds = sorted(set(range(coarse_step, rounds + 1, coarse_step)) | {1, rounds})
    coarse = validation_metrics(coarse_rounds)
    coarse_best = max(coarse, key=lambda item: (item["roc_auc"], item["aupr"]))
    lower = max(1, coarse_best["round"] - coarse_step + 1)
    upper = min(rounds, coarse_best["round"] + coarse_step - 1)
    known = {item["round"] for item in coarse}
    fine_rounds = [value for value in range(lower, upper + 1) if value not in known]
    fine = validation_metrics(fine_rounds) if fine_rounds else []
    selection = sorted(coarse + fine, key=lambda item: item["round"])
    best = max(selection, key=lambda item: (item["roc_auc"], item["aupr"]))
    pd.DataFrame(selection).to_csv(results_dir / "pair_level_selection.csv", index=False)

    best_model = model[:best["round"]]
    model.save_model(results_dir / "xgboost_full_model.json")
    best_model.save_model(results_dir / "xgboost_model.json")

    split_metrics = {}
    for split in ("train", "val", "test"):
        pair_ids, labels, probabilities = predict_and_aggregate(
            best_model, combos[split], store, batch_size, "cold"
        )
        pd.DataFrame({"pair_index": pair_ids, "label": labels,
                      "probability": probabilities}).to_csv(
            results_dir / f"{split}_pair_predictions.csv", index=False
        )
        split_metrics[split] = pair_metrics(labels, probabilities)
        with (results_dir / f"{split}_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(split_metrics[split], handle, indent=2, ensure_ascii=False)

    summary = {
        "selection_split": "val",
        "selection_metric": "pair_level_roc_auc",
        "best_round": best["round"],
        "coarse_step": coarse_step,
        "test_used_for_selection": False,
        "metrics": split_metrics,
    }
    with (results_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return best_model, summary


def retrofit_pair_level_selection(
    run_name: str,
    batch_size: int,
    coarse_step: int,
    combo_dir: Path | str = DEFAULT_COMBO_DIR,
) -> None:
    """Apply unified pair-level model selection to an existing cold run."""
    results_dir = RESULTS_ROOT / run_name
    config_path = results_dir / "training_config.json"
    model_path = results_dir / "xgboost_full_model.json"
    if not model_path.exists():
        model_path = results_dir / "xgboost_model.json"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing model/config in {results_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rounds = int(config["num_boost_round"])
    store = FeatureStore.load(ROOT, require_kg=False)
    combos = {
        split: load_combo_index(split, ROOT, combo_dir)
        for split in ("train", "val", "test")
    }
    model = xgb.Booster()
    model.load_model(model_path)
    _, summary = select_pair_level_model(
        model, combos, store, results_dir, rounds, batch_size, coarse_step
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def train(
    rounds: int,
    batch_size: int,
    nthread: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    early_stopping_rounds: int,
    seed: int,
    force: bool,
    smoke_pairs: int,
    smoke_rounds: int,
    skip_smoke: bool,
    branch: str,
    device: str,
    run_name: str = "single_cold",
    selection_step: int = 10,
    combo_dir: str | None = None,
) -> None:
    """Train the independent strict-cold and warm-KG formal models."""
    cold_results = RESULTS_ROOT / run_name if branch in {"cold", "both"} else COLD_RESULTS
    warm_results = WARM_RESULTS if run_name == "single_cold" else RESULTS_ROOT / f"{run_name}_warm"
    cold_model_path = cold_results / "xgboost_model.json"
    warm_model_path = warm_results / "xgboost_model.json"
    selected_paths = []
    if branch in {"cold", "both"}:
        selected_paths.append(cold_model_path)
    if branch in {"warm", "both"}:
        selected_paths.append(warm_model_path)
    if any(path.exists() for path in selected_paths) and not force:
        raise FileExistsError(
            "A trained model already exists. Add --force to retrain intentionally."
        )

    combo_source = combo_dir or DEFAULT_COMBO_DIR
    combo_index_dir = project_path(combo_source)
    if branch in {"cold", "both"}:
        missing = [
            combo_index_dir / f"{split}_combo_index.npy"
            for split in ("train", "val", "test")
            if not (combo_index_dir / f"{split}_combo_index.npy").exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Single-cold indexes are missing. Run "
                "`python feature.py prepare-single-cold-all --seed 42` first."
            )
    warm_index_path = ROOT / "data" / "combo_index" / "warm_combo_index.npy"
    if branch in {"warm", "both"} and not warm_index_path.exists():
        raise FileNotFoundError(
            "Warm index is missing. Run `python main.py prepare --step warm-all` first."
        )
    store = FeatureStore.load(ROOT, require_kg=branch in {"warm", "both"})
    params = {
        "objective": "binary:logistic",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "gamma": gamma,
        "min_child_weight": min_child_weight,
        "seed": seed,
        "eval_metric": ["logloss", "auc"],
        "tree_method": "hist",
        "verbosity": 0,
        "nthread": nthread,
        "device": device,
    }

    def smoke_pairs_balanced(rows: np.ndarray, count: int) -> np.ndarray:
        pair_table = pd.DataFrame({
            "pair_index": rows["pair_index"],
            "label": rows["label"],
        }).drop_duplicates()
        if pair_table["pair_index"].duplicated().any():
            raise ValueError("Smoke source contains inconsistent pair labels")
        per_class = max(1, count // 2)
        selected = []
        for label in (0, 1):
            candidates = pair_table.loc[pair_table["label"].eq(label), "pair_index"].to_numpy()
            if len(candidates) < per_class:
                raise ValueError(f"Not enough label={label} pairs for smoke test")
            selected.extend(candidates[:per_class])
        return rows[np.isin(rows["pair_index"], np.asarray(selected))]

    if branch in {"cold", "both"}:
        cold_results.mkdir(parents=True, exist_ok=True)
        combos = {
            split: load_combo_index(split, ROOT, combo_source)
            for split in ("train", "val", "test")
        }
        for rows in combos.values():
            validate_pair_weights(rows)
        if not skip_smoke:
            smoke_train = smoke_pairs_balanced(combos["train"], smoke_pairs)
            smoke_val = smoke_pairs_balanced(combos["val"], smoke_pairs)
            if len(smoke_train) == 0 or len(smoke_val) == 0:
                raise ValueError("Cold smoke test selected no rows")
            smoke_train_dm = xgb.QuantileDMatrix(
                ComboDataIter(smoke_train, store, batch_size, "cold"), max_bin=256
            )
            smoke_val_dm = xgb.QuantileDMatrix(
                ComboDataIter(smoke_val, store, batch_size, "cold"),
                ref=smoke_train_dm,
                max_bin=256,
            )
            xgb.train(params, smoke_train_dm, num_boost_round=smoke_rounds,
                      evals=[(smoke_train_dm, "smoke_train"), (smoke_val_dm, "smoke_val")],
                      verbose_eval=False)
            del smoke_train_dm, smoke_val_dm
            gc.collect()

        started = time.time()
        cold_progress = cold_results / "training_progress.log"
        cold_progress.write_text("training_started\n", encoding="utf-8")
        cold_cache = cold_results / "extmem_cache"
        if cold_cache.exists():
            shutil.rmtree(cold_cache)
        cold_cache.mkdir(parents=True)
        dtrain = xgb.ExtMemQuantileDMatrix(
            ComboDataIter(combos["train"], store, batch_size, "cold",
                          str(cold_cache / "train"), device=device,
                          progress_desc="Build train QDM"),
            max_bin=256, nthread=nthread
        )
        dval = xgb.ExtMemQuantileDMatrix(
            ComboDataIter(combos["val"], store, batch_size, "cold",
                          str(cold_cache / "val"), device=device,
                          progress_desc="Build val QDM"),
            ref=dtrain, max_bin=256, nthread=nthread
        )
        history: dict = {}
        train_kwargs = {}
        if early_stopping_rounds > 0:
            train_kwargs["early_stopping_rounds"] = early_stopping_rounds
        model = xgb.train(
            params, dtrain, num_boost_round=rounds,
            evals=[(dtrain, "train"), (dval, "val")], evals_result=history,
            verbose_eval=False,
            callbacks=[ProgressCallback(cold_progress, rounds, "Cold XGBoost")],
            **train_kwargs,
        )
        trained_rounds = len(history["train"]["auc"])
        pd.DataFrame({"round": np.arange(1, trained_rounds + 1),
                      "train_combo_logloss": history["train"]["logloss"],
                      "val_combo_logloss": history["val"]["logloss"],
                      "train_combo_auc": history["train"]["auc"],
                      "val_combo_auc": history["val"]["auc"]}).to_csv(
            cold_results / "training_history.csv", index=False
        )
        cold_config = {
            "params": params, "num_boost_round": rounds,
            "trained_rounds": trained_rounds,
            "early_stopping_rounds": early_stopping_rounds,
            "early_stopping_monitor": "val_combo_auc",
            "batch_size": batch_size,
            "feature_dim": 3584,
            "feature_policy": {"drug": "SMILES(768)+Target(1024)",
                               "pair_dimension": 3584, "kg_used": False,
                               "reason": "cold drugs do not have usable KG at inference time"},
            "elapsed_seconds": round(time.time() - started, 1),
        }
        with open(cold_results / "training_config.json", "w", encoding="utf-8") as handle:
            json.dump(cold_config, handle, indent=2)
        best_model, summary = select_pair_level_model(
            model, combos, store, cold_results, trained_rounds, batch_size, selection_step
        )
        figure_path = plot_performance(cold_results.name)
        print(f"Best pair-level val round={summary['best_round']}; "
              f"Test AUC={summary['metrics']['test']['roc_auc']:.4f}, "
              f"AUPR={summary['metrics']['test']['aupr']:.4f}")
        print(f"Performance curves: {figure_path}")
        del best_model, model, dtrain, dval, history, combos
        gc.collect()
        shutil.rmtree(cold_cache, ignore_errors=True)

    if branch in {"warm", "both"}:
        warm_results.mkdir(parents=True, exist_ok=True)
        warm_combos = np.load(warm_index_path, mmap_mode="r")
        validate_pair_weights(warm_combos)
        if (not store.has_kg[warm_combos["drug1_idx"]].all()
                or not store.has_kg[warm_combos["drug2_idx"]].all()):
            raise ValueError("Warm training data contains a drug without a trained KG embedding")
        if not skip_smoke:
            smoke_warm = smoke_pairs_balanced(warm_combos, smoke_pairs)
            if len(smoke_warm) == 0:
                raise ValueError("Warm smoke test selected no rows")
            smoke_warm_dm = xgb.QuantileDMatrix(
                ComboDataIter(smoke_warm, store, batch_size, "warm"), max_bin=256
            )
            xgb.train(params, smoke_warm_dm, num_boost_round=smoke_rounds,
                      evals=[(smoke_warm_dm, "smoke_warm")], verbose_eval=False)
            del smoke_warm_dm
            gc.collect()

        warm_started = time.time()
        warm_progress = warm_results / "training_progress.log"
        warm_progress.write_text("training_started\n", encoding="utf-8")
        warm_cache = warm_results / "extmem_cache"
        if warm_cache.exists():
            shutil.rmtree(warm_cache)
        warm_cache.mkdir(parents=True)
        dwarm = xgb.ExtMemQuantileDMatrix(
            ComboDataIter(warm_combos, store, batch_size, "warm",
                          str(warm_cache / "train"), device=device,
                          progress_desc="Build warm QDM"),
            max_bin=256, nthread=nthread
        )
        warm_history: dict = {}
        warm_model = xgb.train(
            params, dwarm, num_boost_round=rounds,
            evals=[(dwarm, "train")], evals_result=warm_history,
            verbose_eval=False,
            callbacks=[ProgressCallback(warm_progress, rounds, "Warm KG XGBoost")],
        )
        warm_model.save_model(warm_model_path)
        pd.DataFrame({"round": np.arange(1, rounds + 1),
                      "train_combo_logloss": warm_history["train"]["logloss"],
                      "train_combo_auc": warm_history["train"]["auc"]}).to_csv(
                      warm_results / "training_history.csv", index=False
        )
        warm_config = {
            "params": params, "num_boost_round": rounds, "batch_size": batch_size,
            "feature_dim": 3840,
            "feature_policy": {"drug": "KG(128)+SMILES(768)+Target(1024)",
                               "pair_dimension": 3840, "kg_used": True,
                               "scope": "train pairs where both drugs have trained KG embeddings"},
            "evaluation": "none: no independent warm validation/test split exists",
            "elapsed_seconds": round(time.time() - warm_started, 1),
        }
        with open(warm_results / "training_config.json", "w", encoding="utf-8") as handle:
            json.dump(warm_config, handle, indent=2)
        del warm_model, dwarm, warm_history, warm_combos
        gc.collect()
        shutil.rmtree(warm_cache, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    status_parser = subparsers.add_parser(
        "status", help="validate core inputs and prepared indexes"
    )
    status_parser.add_argument("--combo-dir", default=str(DEFAULT_COMBO_DIR))
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="recompute threshold-free metrics"
    )
    evaluate_parser.add_argument("--run-name", default="single_cold")
    plot_parser = subparsers.add_parser(
        "plot", help="draw Loss, ROC, and precision-recall curves"
    )
    plot_parser.add_argument("--run-name", default="single_cold")
    select_parser = subparsers.add_parser(
        "select-best", help="select a boosting round using pair-level validation"
    )
    select_parser.add_argument("--run-name", required=True)
    select_parser.add_argument("--batch-size", type=int, default=2048)
    select_parser.add_argument("--selection-step", type=int, default=10)
    select_parser.add_argument("--combo-dir", default=str(DEFAULT_COMBO_DIR))
    prepare_parser = subparsers.add_parser("prepare", help="rebuild derived training data")
    prepare_parser.add_argument(
        "--step", choices=("negatives", "combos", "warm-negatives", "warm-combos",
                           "warm-all", "all"),
        default="warm-all"
    )
    prepare_parser.add_argument("--seed", type=int, default=SEED)
    prepare_parser.add_argument("--overwrite", action="store_true")
    train_parser = subparsers.add_parser("train", help="run full XGBoost training")
    train_parser.add_argument("--rounds", type=int, default=500)
    train_parser.add_argument("--batch-size", type=int, default=2048)
    train_parser.add_argument("--nthread", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=0.05)
    train_parser.add_argument("--max-depth", type=int, default=4)
    train_parser.add_argument("--subsample", type=float, default=0.6)
    train_parser.add_argument("--colsample-bytree", type=float, default=0.6)
    train_parser.add_argument("--reg-alpha", type=float, default=5.0)
    train_parser.add_argument("--reg-lambda", type=float, default=10.0)
    train_parser.add_argument("--gamma", type=float, default=1.0)
    train_parser.add_argument("--min-child-weight", type=float, default=1.0)
    train_parser.add_argument(
        "--early-stopping-rounds", type=int, default=0,
        help="stop after this many rounds without val combo-AUC improvement; 0 disables",
    )
    train_parser.add_argument("--seed", type=int, default=SEED)
    train_parser.add_argument("--force", action="store_true")
    train_parser.add_argument("--smoke-pairs", type=int, default=500)
    train_parser.add_argument("--smoke-rounds", type=int, default=5)
    train_parser.add_argument("--skip-smoke", action="store_true")
    train_parser.add_argument("--branch", choices=("cold", "warm", "both"), default="cold")
    train_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train_parser.add_argument("--run-name", default="single_cold",
                              help="name of the result directory")
    train_parser.add_argument("--selection-step", type=int, default=10,
                              help="coarse step for pair-level validation round selection")
    train_parser.add_argument("--combo-dir", default=str(DEFAULT_COMBO_DIR),
                              help="combination-index directory relative to this experiment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command in {None, "status"}:
        status(getattr(args, "combo_dir", DEFAULT_COMBO_DIR))
    elif args.command == "evaluate":
        evaluate(RESULTS_ROOT / args.run_name)
    elif args.command == "plot":
        print(f"Saved: {plot_performance(args.run_name)}")
    elif args.command == "select-best":
        retrofit_pair_level_selection(
            args.run_name,
            args.batch_size,
            args.selection_step,
            args.combo_dir,
        )
    elif args.command == "prepare":
        if args.step in {"negatives", "all"}:
            print(json.dumps(generate_negative_pairs(ROOT, args.seed, args.overwrite), indent=2))
        if args.step in {"combos", "all"}:
            print(json.dumps(build_combo_indexes(ROOT, args.overwrite), indent=2))
        if args.step in {"warm-negatives", "warm-all", "all"}:
            print(json.dumps(generate_warm_pairs(ROOT, args.seed, args.overwrite), indent=2))
        if args.step in {"warm-combos", "warm-all", "all"}:
            print(json.dumps(build_warm_combo_index(ROOT, args.overwrite), indent=2))
    elif args.command == "train":
        train(
            rounds=args.rounds,
            batch_size=args.batch_size,
            nthread=args.nthread,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            gamma=args.gamma,
            min_child_weight=args.min_child_weight,
            early_stopping_rounds=args.early_stopping_rounds,
            seed=args.seed,
            force=args.force,
            smoke_pairs=args.smoke_pairs,
            smoke_rounds=args.smoke_rounds,
            skip_smoke=args.skip_smoke,
            branch=args.branch,
            device=args.device,
            run_name=args.run_name,
            selection_step=args.selection_step,
            combo_dir=args.combo_dir,
        )

from __future__ import annotations
import importlib.util
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"

def _resolve_data_dir(base: Path) -> Path:
    for name in ("split_data", "warm_data"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / "split_data"

WARM_DIR: Path = _resolve_data_dir(DATA_DIR)
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
WORK_DIR: Path = PROJECT_ROOT / "work"
PRED_DIR: Path = WORK_DIR / "predictions"
RUNS_CSV: Path = WORK_DIR / "runs.csv"
TABLE_CSV: Path = RESULTS_DIR / "table.csv"
FIGURE_PATH: Path = RESULTS_DIR / "figures.png"
TABLE_NUMERIC_CSV: Path = WORK_DIR / "table_numeric.csv"
TUNING_DIR: Path = WORK_DIR / "tuning"

for _dir in (RESULTS_DIR, WORK_DIR, PRED_DIR, TUNING_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

SEEDS: tuple[int, ...] = (42,)
SPLITS: tuple[str, ...] = ("train", "val", "test")
FOLDS: tuple[int, ...] = (1, 2, 3, 4, 5)
FOLD_DIR: Path = WARM_DIR / "folds"
METRIC_ORDER: tuple[str, ...] = ("ACC", "SEN", "SPE", "PRE", "F1", "MCC", "AUROC", "AUPRC")
METRIC_LABELS: dict[str, str] = {
    "ACC": "Accuracy", "SEN": "Sensitivity", "SPE": "Specificity",
    "PRE": "Precision", "F1": "F1_Score", "MCC": "MCC",
    "AUROC": "AUROC", "AUPRC": "AUPRC",
}

FEATURE_DIM: int = 3840
SMILES_DIM: int = 768
TARGET_DIM: int = 1024

MODEL_SPECS: dict[str, dict] = {
    "KGNN": {"dir": "KGNN", "display": "KGNN"},
    "GraRep": {"dir": "GraRep", "display": "GraRep + MLP"},
    "DeepDDS": {"dir": "DeepDDS",
                "display": "adapted DeepDDS (without cell-line branch)"},
    "DeepTraSynergy": {"dir": "DeepTraSynergy",
                       "display": "adapted DeepTraSynergy (without cell-line branch)"},
    "KGP-DC": {"dir": "KGP-DC", "display": "KGP-DC"},
}
MODEL_KEYS: tuple[str, ...] = tuple(MODEL_SPECS)
DISPLAY: dict[str, str] = {key: spec["display"] for key, spec in MODEL_SPECS.items()}

def model_slug(model_key: str) -> str:
    return model_key.lower().replace("-", "_")

def split_rules():
    path = DATA_DIR / "split_rules.py"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    spec = importlib.util.spec_from_file_location("warm_split_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_split_frame(split: str) -> pd.DataFrame:
    return split_rules().load_split(split)

def load_fold_frame(fold: int, split: str) -> pd.DataFrame:
    path = FOLD_DIR / f"fold_{int(fold)}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `python data/split_rules.py --folds`")
    return pd.read_csv(path)[["pair_id", "drug1_id", "drug2_id", "label", "source"]]

def fold_exists(fold: int) -> bool:
    return all((FOLD_DIR / f"fold_{int(fold)}_{split}.csv").exists() for split in SPLITS)

def validate_splits() -> dict:
    return split_rules().validate()

def load_pair_matrix(split: str) -> tuple[np.ndarray, np.ndarray]:
    feature_path = WARM_DIR / f"{split}.npy"
    label_path = WARM_DIR / f"{split}_labels.npy"
    for path in (feature_path, label_path):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; data/split_data is incomplete")
    features = np.load(feature_path)
    labels = np.load(label_path).astype(np.int32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"{feature_path}: shape {features.shape} != (n, {FEATURE_DIM})")
    if len(features) != len(labels):
        raise ValueError(f"{feature_path}: {len(features)} rows vs {len(labels)} labels")
    return np.ascontiguousarray(features, dtype=np.float32), labels

def load_drug_table() -> pd.DataFrame:
    path = WARM_DIR / "drug_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_csv(path)

def load_target_table() -> pd.DataFrame:
    path = WARM_DIR / "target_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_csv(path)

def n_drugs() -> int:
    return int(load_drug_table()["internal_id"].max()) + 1

def n_targets() -> int:
    return int(len(np.load(WARM_DIR / "target_ids.npy")))

def load_smiles_features() -> np.ndarray:
    return np.load(WARM_DIR / "drug_smiles_features.npy").astype(np.float32)

def load_target_features() -> np.ndarray:
    features = np.load(WARM_DIR / "target_features.npy").astype(np.float32)
    if features.shape != (n_targets() + 1, TARGET_DIM):
        raise ValueError(f"target feature shape {features.shape} != "
                         f"({n_targets() + 1}, {TARGET_DIM})")
    return features

def load_dti() -> pd.DataFrame:
    index = np.load(WARM_DIR / "drug_target_index.npy")
    frame = pd.DataFrame({"drug_id": index["drug_id"].astype(np.int64),
                          "target_id": index["target_index"].astype(np.int64)})
    frame = frame[frame["target_id"] != n_targets()]
    return frame.drop_duplicates().reset_index(drop=True)

_TARGET_MEAN_CACHE: np.ndarray | None = None

def drug_target_mean() -> np.ndarray:
    global _TARGET_MEAN_CACHE
    if _TARGET_MEAN_CACHE is not None:
        return _TARGET_MEAN_CACHE
    targets = load_target_features()
    dti = load_dti()
    n = n_drugs()
    sums = np.zeros((n, TARGET_DIM), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    drug_ids = dti["drug_id"].to_numpy(dtype=np.int64)
    target_ids = dti["target_id"].to_numpy(dtype=np.int64)
    np.add.at(sums, drug_ids, targets[target_ids].astype(np.float64))
    np.add.at(counts, drug_ids, 1.0)
    block = np.zeros((n, TARGET_DIM), dtype=np.float32)
    has = counts > 0
    block[has] = (sums[has] / counts[has, None]).astype(np.float32)
    _TARGET_MEAN_CACHE = block
    return block

def smiles_strings() -> list[str]:
    table = load_drug_table().sort_values("internal_id")
    return table["smiles"].fillna("").astype(str).tolist()

CHARCANSMISET: dict[str, int] = {
    "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, ".": 7, "1": 8, "0": 9,
    "3": 10, "2": 11, "5": 12, "4": 13, "7": 14, "6": 15, "9": 16, "8": 17,
    "=": 18, "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24, "F": 25,
    "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, "O": 31, "N": 32, "P": 33,
    "S": 34, "R": 35, "U": 36, "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41,
    "Z": 42, "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, "d": 49,
    "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, "l": 55, "o": 56, "n": 57,
    "s": 58, "r": 59, "u": 60, "t": 61, "y": 62, "@": 63, "/": 64, "\\": 0,
}
SMILES_MAX_LEN: int = 200
_TOKEN_CACHE: np.ndarray | None = None

def smiles_token_matrix() -> np.ndarray:
    global _TOKEN_CACHE
    if _TOKEN_CACHE is not None:
        return _TOKEN_CACHE
    matrix = np.zeros((n_drugs(), SMILES_MAX_LEN), dtype=np.int64)
    for drug_id, smiles in enumerate(smiles_strings()):
        for position, char in enumerate(smiles[:SMILES_MAX_LEN]):
            matrix[drug_id, position] = CHARCANSMISET.get(char, 0)
    _TOKEN_CACHE = matrix
    return matrix

ATOM_SYMBOLS: tuple[str, ...] = (
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As",
    "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti",
    "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt",
    "Hg", "Pb", "Unknown")
ATOM_DIM: int = len(ATOM_SYMBOLS) + 11 + 11 + 11 + 1
_GRAPH_CACHE: dict[int, dict] | None = None

def _one_of_k(value, allowable):
    if value not in allowable:
        raise ValueError(f"{value} not in {allowable}")
    return [value == item for item in allowable]

def _one_of_k_unk(value, allowable):
    if value not in allowable:
        value = allowable[-1]
    return [value == item for item in allowable]

def atom_features(atom) -> np.ndarray:
    return np.asarray(
        _one_of_k_unk(atom.GetSymbol(), ATOM_SYMBOLS)
        + _one_of_k(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + _one_of_k_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + _one_of_k_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + [atom.GetIsAromatic()], dtype=np.float32)

def molecular_graphs() -> dict[int, dict]:
    global _GRAPH_CACHE
    if _GRAPH_CACHE is not None:
        return _GRAPH_CACHE
    from rdkit import Chem

    graphs: dict[int, dict] = {}
    for drug_id, smiles in enumerate(smiles_strings()):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature / max(feature.sum(), 1e-9))
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges += [[i, j], [j, i]]
        graphs[drug_id] = {
            "x": np.asarray(features, dtype=np.float32),
            "edge_index": np.asarray(edges, dtype=np.int64).T,
        }
    _GRAPH_CACHE = graphs
    return graphs

def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def binary_metrics(labels: np.ndarray, probabilities: np.ndarray,
                   threshold: float) -> dict:
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 matthews_corrcoef, roc_auc_score)

    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        raise ValueError("metrics require both classes")
    predicted = (probabilities >= float(threshold)).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / len(labels)
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) else 0.0)
    mcc = matthews_corrcoef(labels, predicted) if len(np.unique(predicted)) > 1 else 0.0
    return {
        "ACC": float(accuracy), "SEN": float(sensitivity), "SPE": float(specificity),
        "PRE": float(precision), "F1": float(f1), "MCC": float(mcc),
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "threshold": float(threshold), "pairs": int(len(labels)),
        "positives": int(labels.sum()), "negatives": int((labels == 0).sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

def youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    usable = np.isfinite(thresholds)
    if not usable.any():
        return 0.5
    youden = tpr[usable] - fpr[usable]
    return float(thresholds[usable][int(np.argmax(youden))])

def prediction_path(model_key: str, fold: int, seed: int) -> Path:
    return PRED_DIR / f"{model_slug(model_key)}_fold{int(fold)}_seed{int(seed)}.csv"

def save_predictions(model_key: str, fold: int, seed: int, frame: pd.DataFrame) -> Path:
    columns = ["model", "fold", "seed", "pair_id", "true_label",
               "predicted_label", "predicted_probability"]
    path = prediction_path(model_key, fold, seed)
    frame[columns].to_csv(path, index=False)
    return path

def load_predictions(model_key: str, fold: int, seed: int) -> pd.DataFrame:
    path = prediction_path(model_key, fold, seed)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `python main.py train` first")
    return pd.read_csv(path)

def append_run_record(record: dict) -> None:
    frame = pd.DataFrame([record])
    if RUNS_CSV.exists():
        previous = pd.read_csv(RUNS_CSV)
        previous = previous[~((previous["model_key"] == record["model_key"])
                              & (previous["fold"] == int(record["fold"]))
                              & (previous["seed"] == int(record["seed"])))]
        frame = pd.concat([previous, frame], ignore_index=True)
    order = {key: index for index, key in enumerate(MODEL_KEYS)}
    frame["_order"] = frame["model_key"].map(order).fillna(99).astype(int)
    frame = frame.sort_values(["_order", "fold", "seed"]).drop(columns="_order")
    frame.to_csv(RUNS_CSV, index=False)

def load_run_records() -> pd.DataFrame:
    if not RUNS_CSV.exists():
        raise FileNotFoundError(f"missing {RUNS_CSV}; run `python main.py train` first")
    return pd.read_csv(RUNS_CSV)

def write_summary_table(records: pd.DataFrame) -> pd.DataFrame:
    rows, numeric_rows = [], []
    for model_key in MODEL_KEYS:
        group = records[records["model_key"] == model_key]
        if group.empty:
            continue
        row = {"model": DISPLAY[model_key]}
        numeric = {"model": DISPLAY[model_key], "model_key": model_key,
                   "n_runs": int(len(group))}
        for metric in METRIC_ORDER:
            values = group[metric].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[METRIC_LABELS[metric]] = f"{mean:.4f} ± {std:.4f}"
            numeric[f"{metric}_mean"] = mean
            numeric[f"{metric}_std"] = std
        rows.append(row)
        numeric_rows.append(numeric)
    columns = ["model"] + [METRIC_LABELS[metric] for metric in METRIC_ORDER]
    table = pd.DataFrame(rows)[columns]
    table.to_csv(TABLE_CSV, index=False)
    pd.DataFrame(numeric_rows).to_csv(TABLE_NUMERIC_CSV, index=False)
    return table

def subsample_indices(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    n = int(n)
    if n <= 0 or n >= len(labels):
        return np.arange(len(labels))
    rng = np.random.RandomState(int(seed))
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    n_positive = min(len(positive), n // 2)
    n_negative = min(len(negative), n - n_positive)
    chosen = np.concatenate([
        rng.choice(positive, size=n_positive, replace=False),
        rng.choice(negative, size=n_negative, replace=False)])
    return np.sort(chosen)

def subsample_frame(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    indices = subsample_indices(frame["label"].to_numpy(), n, seed)
    return frame.iloc[indices].reset_index(drop=True)

def load_best_config(model_key: str) -> dict | None:
    path = TUNING_DIR / f"{model_slug(model_key)}_best.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("config")

def write_json(path: str | Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

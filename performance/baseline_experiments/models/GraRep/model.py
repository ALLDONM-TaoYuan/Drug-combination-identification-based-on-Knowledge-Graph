from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
# Optional location of the upstream GraRep MATLAB core. It is not part of this
# repository; when it is unavailable the identical formulas are re-implemented
# in NumPy (see _grarep_python) and the backend is recorded in the run record.
COMPARISON_DIR: Path = Path(os.environ.get(
    "KGDC_COMPARISON_DIR", REPO_ROOT / "compare_exa"))
MATLAB_CORE: Path = COMPARISON_DIR / "GraRep" / "code" / "core"
MATLAB_DRIVER: Path = Path(__file__).resolve().parent / "run_grarep.m"
MATLAB_TIMEOUT: int = 1800

DEFAULT_CONFIG: dict = {
    "kstep": 4,
    "dim": 64,
    "head": "mlp",
    "hidden": (256, 128),
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "epochs": 100,
    "patience": 15,
}

TUNING_CANDIDATES: list[dict] = [
    {"head": "mlp", "hidden": (256, 128), "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4},
    {"head": "mlp", "hidden": (512, 256), "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4},
    {"head": "mlp", "hidden": (256,), "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4},
    {"head": "mlp", "hidden": (256, 128), "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4},
    {"head": "mlp", "hidden": (256, 128), "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-4},
    {"head": "mlp", "hidden": (256, 128), "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-3},
    {"head": "mlp", "hidden": (512,), "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4},
    {"head": "mlp", "hidden": (128, 64), "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4},
    {"head": "lr", "hidden": (), "dropout": 0.0, "lr": 0.0, "weight_decay": 0.0},
    {"head": "mlp", "hidden": (512, 256), "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-3},
]

_EMBEDDING_CACHE: dict = {}

def _cfg(config: dict | None) -> dict:
    return {**DEFAULT_CONFIG, **(config or {})}

def _matlab_executable() -> str | None:
    candidate = os.environ.get("MATLAB_EXE")
    if candidate and Path(candidate).exists():
        return candidate
    return shutil.which("matlab")

def _grarep_python(adjacency: np.ndarray, kstep: int, dim: int) -> np.ndarray:
    n_nodes = adjacency.shape[0]
    weights = adjacency.astype(np.float64).copy()
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=1)
    degree[degree == 0] = 1.0
    transition = weights / degree[:, None]
    ak = transition.copy()
    blocks = []
    for _ in range(kstep):
        column = ak.sum(axis=0)
        column[column == 0] = 1.0
        prob = np.log(ak / column[None, :]) - np.log(1.0 / n_nodes)
        prob[prob < 0] = 0.0
        prob[np.isnan(prob)] = 0.0
        u, s, _ = np.linalg.svd(prob, full_matrices=False)
        block = u[:, :dim] * np.sqrt(s[:dim])[None, :]
        norm = np.sqrt((block ** 2).sum(axis=1))
        norm[norm == 0] = 1.0
        blocks.append(block / norm[:, None])
        ak = ak @ transition
    representation = np.hstack(blocks)
    representation[np.isnan(representation)] = 1.0
    return representation.astype(np.float32)

def _run_matlab(edges: np.ndarray, n_nodes: int, kstep: int, dim: int
                ) -> tuple[np.ndarray | None, str]:
    exe = _matlab_executable()
    if exe is None:
        return None, "matlab executable not found"
    if not MATLAB_CORE.exists() or not MATLAB_DRIVER.exists():
        return None, f"upstream GraRep core not found at {MATLAB_CORE}"
    with tempfile.TemporaryDirectory() as tmp:
        edges_csv = Path(tmp) / "edges.csv"
        out_csv = Path(tmp) / "embedding.csv"
        np.savetxt(edges_csv, edges + 1, delimiter=",", fmt="%d")
        command = (f"addpath('{MATLAB_CORE}'); addpath('{MATLAB_DRIVER.parent}'); "
                   f"run_grarep('{edges_csv}', {n_nodes}, {kstep}, {dim}, '{out_csv}')")
        try:
            completed = subprocess.run([exe, "-batch", command], capture_output=True,
                                       text=True, timeout=MATLAB_TIMEOUT)
        except subprocess.TimeoutExpired:
            return None, "matlab timeout"
        if completed.returncode != 0 or not out_csv.exists():
            return None, f"matlab exit={completed.returncode}"
        return pd.read_csv(out_csv, header=None).to_numpy(dtype=np.float32), \
            "matlab (upstream code/core/*.m)"

def node_embeddings(fold: int, kstep: int, dim: int) -> tuple[np.ndarray, dict]:
    key = (int(fold), int(kstep), int(dim))
    if key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[key]

    n_nodes = common.n_drugs()
    train = common.load_fold_frame(fold, "train")
    positives = train.loc[train["label"] == 1, ["drug1_id", "drug2_id"]].to_numpy()
    edges = np.unique(np.sort(positives, axis=1), axis=0)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    adjacency[edges[:, 0], edges[:, 1]] = 1.0
    adjacency[edges[:, 1], edges[:, 0]] = 1.0
    degree = adjacency.sum(axis=1)
    active = np.flatnonzero(degree > 0)
    isolated = int((degree == 0).sum())
    sub_adjacency = adjacency[np.ix_(active, active)]
    index = {int(node): position for position, node in enumerate(active)}
    sub_edges = np.asarray([[index[int(a)], index[int(b)]] for a, b in edges], dtype=np.int64)

    sub, backend = _run_matlab(sub_edges, len(active), kstep, dim)
    if sub is None:
        sub = _grarep_python(sub_adjacency, kstep, dim)
        backend = f"python port of the upstream formulas ({backend})"
    column_std = sub.std(axis=0)
    if (column_std == 0).all() or not np.isfinite(sub).all():
        raise RuntimeError("GraRep produced a degenerate representation")
    embedding = np.zeros((n_nodes, kstep * dim), dtype=np.float32)
    embedding[active] = sub
    meta = {"embedding_backend": backend, "train_edges": int(len(edges)),
            "embedded_nodes": int(len(active)), "isolated_zero": isolated}
    _EMBEDDING_CACHE[key] = (embedding, meta)
    return embedding, meta

def pair_features(embedding: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    z1 = embedding[frame["drug1_id"].to_numpy(dtype=np.int64)]
    z2 = embedding[frame["drug2_id"].to_numpy(dtype=np.int64)]
    return np.concatenate([np.abs(z1 - z2), z1 * z2], axis=1).astype(np.float32)

def _fit_mlp(x_train, y_train, x_val, y_val, x_test, cfg: dict, seed: int):
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    hidden = tuple(cfg["hidden"]) or (256,)
    dropout = float(cfg["dropout"])

    class MLP(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            layers, previous = [], input_dim
            for width in hidden:
                layers += [nn.Linear(previous, int(width)), nn.ReLU(), nn.Dropout(dropout)]
                previous = int(width)
            layers.append(nn.Linear(previous, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    torch.manual_seed(int(seed))
    model = MLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                                 weight_decay=float(cfg["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(int(seed))
    train_x = torch.from_numpy(x_train)
    train_y = torch.from_numpy(y_train.astype(np.float32))
    val_x = torch.from_numpy(x_val)
    test_x = torch.from_numpy(x_test)
    best_auc, best_state, stale = -1.0, None, 0
    for _epoch in range(int(cfg["epochs"])):
        model.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), int(cfg["batch_size"])):
            index = torch.from_numpy(order[start:start + int(cfg["batch_size"])])
            optimizer.zero_grad()
            loss = criterion(model(train_x[index]), train_y[index])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_auc = roc_auc_score(y_val, torch.sigmoid(model(val_x)).numpy())
        if val_auc > best_auc + 1e-5:
            best_auc, stale = float(val_auc), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return (torch.sigmoid(model(val_x)).numpy(),
                torch.sigmoid(model(test_x)).numpy(), {"val_auroc": float(best_auc)})

def _fit_lr(x_train, y_train, x_val, y_val, x_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    best = None
    for c in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=float(c), max_iter=5000, solver="lbfgs")
        model.fit(x_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, float(c), model)
    return (best[2].predict_proba(x_val)[:, 1],
            best[2].predict_proba(x_test)[:, 1],
            {"val_auroc": float(best[0]), "selected_C": best[1]})

def train_and_predict(fold: int, seed: int = 42, config: dict | None = None,
                      write: bool = True) -> dict:
    from sklearn.preprocessing import StandardScaler

    started = time.time()
    if config is None:
        config = common.load_best_config("GraRep")
    cfg = _cfg(config)
    subsample = cfg.pop("train_subsample", None)
    embedding, embedding_meta = node_embeddings(fold, int(cfg["kstep"]), int(cfg["dim"]))
    frames = {split: common.load_fold_frame(fold, split) for split in common.SPLITS}
    if subsample:
        frames["train"] = common.subsample_frame(frames["train"], int(subsample), seed)
    features = {split: pair_features(embedding, frame) for split, frame in frames.items()}
    scaler = StandardScaler().fit(features["train"])
    scaled = {split: scaler.transform(matrix).astype(np.float32)
              for split, matrix in features.items()}

    x_train, y_train = scaled["train"], frames["train"]["label"].to_numpy()
    x_val, y_val = scaled["val"], frames["val"]["label"].to_numpy()
    if str(cfg["head"]) == "lr":
        val_probability, test_probability, head_meta = _fit_lr(
            x_train, y_train, x_val, y_val, scaled["test"])
        display = "GraRep + LR"
    else:
        val_probability, test_probability, head_meta = _fit_mlp(
            x_train, y_train, x_val, y_val, scaled["test"], cfg, seed)
        display = "GraRep + MLP"
    threshold = float(common.youden_threshold(y_val, val_probability))
    if write:
        test_frame = frames["test"]
        frame = test_frame[["pair_id"]].copy()
        frame.insert(0, "model", display)
        frame.insert(1, "fold", int(fold))
        frame.insert(2, "seed", int(seed))
        frame["true_label"] = test_frame["label"].to_numpy()
        frame["predicted_label"] = (test_probability >= threshold).astype(int)
        frame["predicted_probability"] = test_probability
        common.save_predictions("GraRep", fold, seed, frame)
    return {"val_probability": val_probability, "test_probability": test_probability,
            "threshold": threshold,
            "meta": {**embedding_meta, **head_meta, "head": str(cfg["head"]),
                     "hidden": list(cfg["hidden"]), "elapsed_sec": round(time.time() - started, 1),
                     "notes": "train-only DDI graph; symmetric pair features"}}

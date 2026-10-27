from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common

DEFAULT_CONFIG: dict = {
    "heads": 4,
    "hidden": 256,
    "drug_dim": 128,
    "head_hidden": (128, 64),
    "dropout": 0.2,
    "batch_size": 128,
    "epochs": 15,
    "lr": 1e-4,
    "patience": 5,
    "weight_decay": 0.0,
}

TUNING_CANDIDATES: list[dict] = [
    {"heads": 4, "hidden": 256, "head_hidden": (128, 64), "dropout": 0.2, "lr": 1e-4},
    {"heads": 8, "hidden": 256, "head_hidden": (128, 64), "dropout": 0.2, "lr": 1e-4},
    {"heads": 4, "hidden": 512, "head_hidden": (256, 128), "dropout": 0.2, "lr": 1e-4},
    {"heads": 8, "hidden": 128, "head_hidden": (128, 64), "dropout": 0.2, "lr": 1e-4},
    {"heads": 4, "hidden": 256, "head_hidden": (128, 64), "dropout": 0.3, "lr": 1e-4},
    {"heads": 4, "hidden": 256, "head_hidden": (128, 64), "dropout": 0.2, "lr": 3e-4},
    {"heads": 8, "hidden": 512, "head_hidden": (256, 128), "dropout": 0.2, "lr": 1e-4},
    {"heads": 4, "hidden": 128, "head_hidden": (64, 32), "dropout": 0.2, "lr": 1e-4},
    {"heads": 8, "hidden": 256, "head_hidden": (128, 64), "dropout": 0.3, "lr": 3e-4},
    {"heads": 4, "hidden": 384, "head_hidden": (192, 96), "dropout": 0.2, "lr": 1e-4},
]

_GRAPH_TENSORS: dict | None = None

def _cfg(config: dict | None) -> dict:
    return {**DEFAULT_CONFIG, **(config or {})}

def _graph_tensors() -> dict:
    global _GRAPH_TENSORS
    if _GRAPH_TENSORS is not None:
        return _GRAPH_TENSORS
    import torch

    _GRAPH_TENSORS = {drug_id: {"x": torch.from_numpy(graph["x"]),
                                "edge_index": torch.from_numpy(graph["edge_index"])}
                      for drug_id, graph in common.molecular_graphs().items()}
    return _GRAPH_TENSORS

def _pack(drug_ids: np.ndarray, graphs: dict):
    import torch

    features, edges, batch, offset = [], [], [], 0
    for position, drug_id in enumerate(drug_ids):
        graph = graphs[int(drug_id)]
        x = graph["x"]
        features.append(x)
        edges.append(graph["edge_index"] + offset)
        batch.append(torch.full((x.shape[0],), position, dtype=torch.long))
        offset += x.shape[0]
    return torch.cat(features), torch.cat(edges, dim=1), torch.cat(batch)

def build_model(cfg: dict):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    heads = int(cfg["heads"])
    hidden = int(cfg["hidden"])
    drug_dim = int(cfg["drug_dim"])
    dropout = float(cfg["dropout"])

    class GATLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, n_heads: int, concat: bool):
            super().__init__()
            self.n_heads, self.out_dim, self.concat = n_heads, out_dim, concat
            self.projection = nn.Linear(in_dim, out_dim * n_heads, bias=False)
            self.attention_src = nn.Parameter(torch.zeros(n_heads, out_dim))
            self.attention_dst = nn.Parameter(torch.zeros(n_heads, out_dim))
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.xavier_uniform_(self.attention_src)
            nn.init.xavier_uniform_(self.attention_dst)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index):
            nodes = x.shape[0]
            h = self.projection(x).view(nodes, self.n_heads, self.out_dim)
            src, dst = edge_index[0], edge_index[1]
            logits = ((h[src] * self.attention_src).sum(-1)
                      + (h[dst] * self.attention_dst).sum(-1))
            logits = F.leaky_relu(logits, 0.2)
            maximum = torch.full((nodes, self.n_heads), -1e30)
            maximum = maximum.index_reduce(0, dst, logits, "amax", include_self=True)
            weights = torch.exp(logits - maximum[dst])
            denominator = torch.zeros(nodes, self.n_heads)
            denominator = denominator.index_add(0, dst, weights)
            weights = self.dropout(weights / (denominator[dst] + 1e-16))
            out = torch.zeros(nodes, self.n_heads, self.out_dim)
            out = out.index_add(0, dst, weights.unsqueeze(-1) * h[src])
            return out.reshape(nodes, self.n_heads * self.out_dim) if self.concat \
                else out.mean(dim=1)

    class DrugEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = GATLayer(common.ATOM_DIM, hidden, heads, concat=True)
            self.layer2 = GATLayer(hidden * heads, hidden, 1, concat=True)
            self.pool_to_drug = nn.Linear(hidden, drug_dim)

        def forward(self, x, edge_index, batch):
            h = F.elu(self.layer1(x, edge_index))
            h = F.dropout(h, p=dropout, training=self.training)
            h = F.elu(self.layer2(h, edge_index))
            pooled = torch.full((int(batch.max()) + 1, h.shape[1]), -1e30, device=h.device)
            pooled = pooled.index_reduce(0, batch, h, "amax", include_self=True)
            return F.relu(self.pool_to_drug(pooled))

    class AdaptedDeepDDS(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DrugEncoder()
            layers, previous = [], drug_dim * 2
            for width in cfg["head_hidden"]:
                layers += [nn.Linear(previous, int(width)), nn.ReLU(), nn.Dropout(dropout)]
                previous = int(width)
            layers.append(nn.Linear(previous, 2))
            self.head = nn.Sequential(*layers)

        def forward(self, drug1, drug2):
            x1, e1, b1 = drug1
            x2, e2, b2 = drug2
            h1, h2 = self.encoder(x1, e1, b1), self.encoder(x2, e2, b2)
            return self.head(torch.cat([(h1 - h2).abs(), h1 * h2], dim=1))

    return AdaptedDeepDDS()

def train_and_predict(fold: int, seed: int = 42, config: dict | None = None,
                      write: bool = True) -> dict:
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    started = time.time()
    if config is None:
        config = common.load_best_config("DeepDDS")
    cfg = _cfg(config)
    subsample = cfg.pop("train_subsample", None)
    graphs = _graph_tensors()
    frames = {split: common.load_fold_frame(fold, split) for split in common.SPLITS}
    if subsample:
        frames["train"] = common.subsample_frame(frames["train"], int(subsample), seed)
    tensors = {split: (frame["drug1_id"].to_numpy(dtype=np.int64),
                       frame["drug2_id"].to_numpy(dtype=np.int64),
                       torch.from_numpy(frame["label"].to_numpy(dtype=np.int64)))
               for split, frame in frames.items()}

    torch.manual_seed(int(seed))
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                                 weight_decay=float(cfg["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    rng = np.random.RandomState(int(seed))

    def batches(split: str, shuffle: bool):
        drug1, drug2, labels = tensors[split]
        order = rng.permutation(len(drug1)) if shuffle else np.arange(len(drug1))
        for start in range(0, len(order), int(cfg["batch_size"])):
            index = order[start:start + int(cfg["batch_size"])]
            yield (_pack(drug1[index], graphs), _pack(drug2[index], graphs),
                   labels[torch.from_numpy(index)])

    def evaluate(split: str) -> np.ndarray:
        model.eval()
        probabilities = []
        with torch.no_grad():
            for packed1, packed2, _ in batches(split, shuffle=False):
                probabilities.append(torch.softmax(model(packed1, packed2), dim=1)[:, 1].numpy())
        return np.concatenate(probabilities)

    best_auc, best_state, stale = -1.0, None, 0
    for _epoch in range(int(cfg["epochs"])):
        model.train()
        for packed1, packed2, labels in batches("train", shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(packed1, packed2), labels)
            loss.backward()
            optimizer.step()
        val_auc = roc_auc_score(frames["val"]["label"].to_numpy(), evaluate("val"))
        print(f"  [DeepDDS fold{fold}] epoch {_epoch + 1}/{cfg['epochs']} "
              f"val_auroc={val_auc:.4f}", flush=True)
        if val_auc > best_auc + 1e-5:
            best_auc, stale = float(val_auc), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    val_probability = evaluate("val")
    test_probability = evaluate("test")
    threshold = float(common.youden_threshold(frames["val"]["label"].to_numpy(),
                                              val_probability))
    if write:
        test_frame = frames["test"]
        frame = test_frame[["pair_id"]].copy()
        frame.insert(0, "model", common.DISPLAY["DeepDDS"])
        frame.insert(1, "fold", int(fold))
        frame.insert(2, "seed", int(seed))
        frame["true_label"] = test_frame["label"].to_numpy()
        frame["predicted_label"] = (test_probability >= threshold).astype(int)
        frame["predicted_probability"] = test_probability
        common.save_predictions("DeepDDS", fold, seed, frame)
    return {"val_probability": val_probability, "test_probability": test_probability,
            "threshold": threshold,
            "meta": {**{k: v for k, v in cfg.items() if k != "head_hidden"},
                     "head_hidden": list(cfg["head_hidden"]),
                     "val_auroc": float(best_auc),
                     "elapsed_sec": round(time.time() - started, 1),
                     "notes": "shared-weight dual GAT; no cell-line branch"}}

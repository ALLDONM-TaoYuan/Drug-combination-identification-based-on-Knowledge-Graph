from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common

N_RELATIONS: int = 2

DEFAULT_CONFIG: dict = {
    "embed_dim": 32,
    "n_depth": 2,
    "neighbor": 4,
    "aggregator": "sum",
    "lr": 0.02,
    "batch_size": 2048,
    "epochs": 50,
    "patience": 5,
    "clip_norm": 5.0,
}

TUNING_CANDIDATES: list[dict] = [
    {"embed_dim": 32, "n_depth": 2, "aggregator": "sum", "lr": 0.02, "batch_size": 2048},
    {"embed_dim": 32, "n_depth": 2, "aggregator": "concat", "lr": 0.02, "batch_size": 2048},
    {"embed_dim": 64, "n_depth": 2, "aggregator": "sum", "lr": 0.02, "batch_size": 2048},
    {"embed_dim": 32, "n_depth": 3, "aggregator": "sum", "lr": 0.02, "batch_size": 2048},
    {"embed_dim": 64, "n_depth": 2, "aggregator": "concat", "lr": 0.02, "batch_size": 2048},
    {"embed_dim": 32, "n_depth": 2, "aggregator": "sum", "lr": 0.05, "batch_size": 2048},
    {"embed_dim": 32, "n_depth": 2, "aggregator": "sum", "lr": 0.01, "batch_size": 2048},
    {"embed_dim": 64, "n_depth": 3, "aggregator": "sum", "lr": 0.02, "batch_size": 1024},
    {"embed_dim": 32, "n_depth": 2, "aggregator": "neigh", "lr": 0.02, "batch_size": 2048},
    {"embed_dim": 64, "n_depth": 3, "aggregator": "concat", "lr": 0.02, "batch_size": 2048},
]

def _cfg(config: dict | None) -> dict:
    return {**DEFAULT_CONFIG, **(config or {})}

def build_graph(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    n_drugs = common.n_drugs()
    n_targets = common.n_targets()
    n_entities = n_drugs + n_targets
    positives = train_frame.loc[train_frame["label"] == 1, ["drug1_id", "drug2_id"]].to_numpy()
    dti = common.load_dti()
    heads = np.concatenate([positives[:, 0], positives[:, 1], dti["drug_id"].to_numpy()])
    tails = np.concatenate([positives[:, 1], positives[:, 0],
                            dti["target_id"].to_numpy() + n_drugs])
    relations = np.concatenate([np.zeros(len(positives) * 2, dtype=np.int64),
                                np.ones(len(dti), dtype=np.int64)])
    stats = {"n_drugs": int(n_drugs), "n_targets": int(n_targets),
             "combination_edges": int(len(positives)),
             "interaction_edges": int(len(dti))}
    return (heads.astype(np.int64), tails.astype(np.int64),
            relations.astype(np.int64), n_entities, stats)

def neighbour_tables(heads, tails, relations, n_entities: int, neighbor: int,
                     seed: int) -> tuple[np.ndarray, np.ndarray]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_entities)]
    for head, tail, relation in zip(heads, tails, relations):
        adjacency[int(head)].append((int(tail), int(relation)))
    rng = np.random.RandomState(int(seed))
    adj_entity = np.zeros((n_entities, neighbor), dtype=np.int64)
    adj_relation = np.zeros((n_entities, neighbor), dtype=np.int64)
    for entity in range(n_entities):
        neighbours = adjacency[entity]
        if not neighbours:
            continue
        replace = len(neighbours) < neighbor
        chosen = rng.choice(len(neighbours), size=neighbor, replace=replace)
        for position, index in enumerate(chosen):
            adj_entity[entity, position] = neighbours[index][0]
            adj_relation[entity, position] = neighbours[index][1]
    return adj_entity, adj_relation

def build_model(n_entities: int, cfg: dict):
    import torch
    import torch.nn as nn

    embed_dim = int(cfg["embed_dim"])
    n_depth = int(cfg["n_depth"])
    neighbor = int(cfg["neighbor"])
    aggregator = str(cfg["aggregator"])

    class KGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.entity = nn.Embedding(n_entities, embed_dim)
            self.relation = nn.Embedding(N_RELATIONS, embed_dim)
            nn.init.xavier_normal_(self.entity.weight)
            nn.init.xavier_normal_(self.relation.weight)
            input_dim = embed_dim * 2 if aggregator == "concat" else embed_dim
            self.projection = nn.ModuleList(
                [nn.Linear(input_dim, embed_dim) for _ in range(n_depth)])
            self.bias = nn.ParameterList(
                [nn.Parameter(torch.zeros(embed_dim)) for _ in range(n_depth)])
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()

        def aggregate(self, depth: int, entity, neighbour):
            combined = {"sum": entity + neighbour,
                        "concat": torch.cat([entity, neighbour], dim=-1),
                        "neigh": neighbour}[aggregator]
            out = self.projection[depth](combined) + self.bias[depth]
            return self.tanh(out) if depth == n_depth - 1 else self.relu(out)

        def branch(self, drug_index, adj_entity, adj_relation):
            drug_embedding = self.entity(drug_index)
            entities = [drug_embedding.unsqueeze(1)]
            relations = []
            level = drug_index.unsqueeze(1)
            for _ in range(n_depth):
                relation_level = self.relation(adj_relation[level])
                relations.append(relation_level.reshape(
                    level.shape[0], -1, relation_level.shape[-1]))
                level = adj_entity[level].reshape(level.shape[0], -1)
                entities.append(self.entity(level))
            for depth in range(n_depth):
                aggregated = []
                for hop in range(n_depth - depth):
                    relation = relations[hop]
                    neighbour_entities = entities[hop + 1]
                    score = (drug_embedding.unsqueeze(1) * relation).sum(-1, keepdim=True)
                    weighted = (score * neighbour_entities).reshape(
                        neighbour_entities.shape[0], -1, neighbor,
                        neighbour_entities.shape[-1]).sum(dim=2)
                    aggregated.append(self.aggregate(depth, entities[hop], weighted))
                entities = aggregated
            return entities[0].squeeze(1)

        def forward(self, drug1, drug2, adj_entity, adj_relation):
            e1 = self.branch(drug1, adj_entity, adj_relation)
            e2 = self.branch(drug2, adj_entity, adj_relation)
            return (e1 * e2).sum(dim=-1)

    return KGCN()

def train_and_predict(fold: int, seed: int = 42, config: dict | None = None,
                      write: bool = True) -> dict:
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    started = time.time()
    if config is None:
        config = common.load_best_config("KGNN")
    cfg = _cfg(config)
    subsample = cfg.pop("train_subsample", None)
    frames = {split: common.load_fold_frame(fold, split) for split in common.SPLITS}
    if subsample:
        frames["train"] = common.subsample_frame(frames["train"], int(subsample), seed)

    heads, tails, relations, n_entities, graph_stats = build_graph(frames["train"])
    adj_entity, adj_relation = neighbour_tables(
        heads, tails, relations, n_entities, int(cfg["neighbor"]), int(seed) * 1000 + fold)
    torch_adj_entity = torch.from_numpy(adj_entity)
    torch_adj_relation = torch.from_numpy(adj_relation)

    tensors = {
        split: (torch.from_numpy(frame["drug1_id"].to_numpy(dtype=np.int64)),
                torch.from_numpy(frame["drug2_id"].to_numpy(dtype=np.int64)),
                torch.from_numpy(frame["label"].to_numpy(dtype=np.float32)))
        for split, frame in frames.items()}

    torch.manual_seed(int(seed))
    model = build_model(n_entities, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(int(seed))

    def evaluate(split: str) -> np.ndarray:
        model.eval()
        d1, d2, _ = tensors[split]
        chunks = []
        with torch.no_grad():
            for start in range(0, len(d1), 8192):
                logits = model(d1[start:start + 8192], d2[start:start + 8192],
                               torch_adj_entity, torch_adj_relation)
                chunks.append(torch.sigmoid(logits).numpy())
        return np.concatenate(chunks)

    train_d1, train_d2, train_y = tensors["train"]
    best_auc, best_state, stale = -1.0, None, 0
    for _epoch in range(int(cfg["epochs"])):
        model.train()
        order = rng.permutation(len(train_d1))
        for start in range(0, len(order), int(cfg["batch_size"])):
            index = torch.from_numpy(order[start:start + int(cfg["batch_size"])])
            optimizer.zero_grad()
            loss = criterion(model(train_d1[index], train_d2[index],
                                   torch_adj_entity, torch_adj_relation), train_y[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip_norm"]))
            optimizer.step()
        val_auc = roc_auc_score(frames["val"]["label"].to_numpy(), evaluate("val"))
        if val_auc > best_auc + 1e-5:
            best_auc, stale = float(val_auc), 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
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
        frame.insert(0, "model", common.DISPLAY["KGNN"])
        frame.insert(1, "fold", int(fold))
        frame.insert(2, "seed", int(seed))
        frame["true_label"] = test_frame["label"].to_numpy()
        frame["predicted_label"] = (test_probability >= threshold).astype(int)
        frame["predicted_probability"] = test_probability
        common.save_predictions("KGNN", fold, seed, frame)
    return {"val_probability": val_probability, "test_probability": test_probability,
            "threshold": threshold,
            "meta": {**graph_stats, **{k: v for k, v in cfg.items()},
                     "val_auroc": float(best_auc),
                     "elapsed_sec": round(time.time() - started, 1),
                     "notes": "PyTorch port of KGCN; train-fold DDI + all DTI"}}

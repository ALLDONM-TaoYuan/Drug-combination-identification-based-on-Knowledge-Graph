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

MAX_LEN: int = 200
PATCH_WIDTH: int = 5
N_PATCHES: int = MAX_LEN // PATCH_WIDTH
PROJECTION: int = 50

DEFAULT_CONFIG: dict = {
    "layers": 2,
    "head_hidden": (256, 256, 128),
    "dropout": 0.1,
    "batch_size": 256,
    "epochs": 20,
    "lr": 1e-3,
    "patience": 8,
    "clip_norm": 5.0,
}

TUNING_CANDIDATES: list[dict] = [
    {"layers": 2, "head_hidden": (256, 256, 128), "dropout": 0.1, "lr": 1e-3},
    {"layers": 4, "head_hidden": (256, 256, 128), "dropout": 0.1, "lr": 1e-3},
    {"layers": 2, "head_hidden": (512, 256, 128), "dropout": 0.1, "lr": 1e-3},
    {"layers": 2, "head_hidden": (256, 128), "dropout": 0.1, "lr": 1e-3},
    {"layers": 2, "head_hidden": (256, 256, 128), "dropout": 0.2, "lr": 1e-3},
    {"layers": 2, "head_hidden": (256, 256, 128), "dropout": 0.1, "lr": 5e-4},
    {"layers": 4, "head_hidden": (512, 256, 128), "dropout": 0.2, "lr": 5e-4},
    {"layers": 2, "head_hidden": (128, 128), "dropout": 0.1, "lr": 1e-3},
    {"layers": 3, "head_hidden": (256, 256, 128), "dropout": 0.2, "lr": 5e-4},
    {"layers": 4, "head_hidden": (256, 256, 128), "dropout": 0.2, "lr": 1e-3},
]

_TOKEN_TENSOR = None
_TARGET_TENSOR = None

def _cfg(config: dict | None) -> dict:
    return {**DEFAULT_CONFIG, **(config or {})}

def _tokens():
    global _TOKEN_TENSOR
    if _TOKEN_TENSOR is None:
        import torch

        _TOKEN_TENSOR = torch.from_numpy(common.smiles_token_matrix())
    return _TOKEN_TENSOR

def _target_profiles():
    global _TARGET_TENSOR
    if _TARGET_TENSOR is None:
        import torch

        _TARGET_TENSOR = torch.from_numpy(common.drug_target_mean())
    return _TARGET_TENSOR

def build_model(n_drugs: int, cfg: dict):
    import torch
    import torch.nn as nn

    layers = int(cfg["layers"])
    dropout = float(cfg["dropout"])

    class DrugTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = nn.Linear(PATCH_WIDTH, PROJECTION)
            self.positions = nn.Embedding(N_PATCHES, PROJECTION)
            block = nn.TransformerEncoderLayer(
                d_model=PROJECTION, nhead=1, dim_feedforward=PROJECTION * 2,
                dropout=dropout, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(block, num_layers=layers)
            self.normalisation = nn.LayerNorm(PROJECTION)
            self.flatten = nn.Linear(N_PATCHES * PROJECTION, 200)
            self.dropout = nn.Dropout(0.5)

        def forward(self, tokens):
            batch = tokens.shape[0]
            patches = tokens.float().view(batch, N_PATCHES, PATCH_WIDTH)
            x = self.patch(patches)
            positions = torch.arange(N_PATCHES, device=tokens.device)
            x = self.normalisation(x + self.positions(positions))
            x = self.encoder(x)
            return self.flatten(self.dropout(x.reshape(batch, -1)))

    class AdaptedDeepTraSynergy(nn.Module):
        def __init__(self):
            super().__init__()
            self.drug_encoder = DrugTransformer()
            self.target_projection = nn.Linear(common.TARGET_DIM, 200)
            self.target_attention = nn.MultiheadAttention(
                embed_dim=200, num_heads=1, dropout=dropout, batch_first=True)
            self.interaction = nn.Linear(200, 200)
            head_layers, previous = [], 200 * 6
            for width in cfg["head_hidden"]:
                head_layers += [nn.Linear(previous, int(width)), nn.ReLU(),
                                nn.Dropout(dropout)]
                previous = int(width)
            head_layers.append(nn.Linear(previous, 2))
            self.head = nn.Sequential(*head_layers)
            self.register_buffer("target_profiles", torch.zeros(n_drugs, common.TARGET_DIM))

        def forward(self, tokens1, tokens2, drug1, drug2):
            h1 = self.drug_encoder(tokens1)
            h2 = self.drug_encoder(tokens2)
            p1 = self.target_projection(self.target_profiles[drug1]).unsqueeze(1)
            p2 = self.target_projection(self.target_profiles[drug2]).unsqueeze(1)
            context = torch.cat([p1, p2], dim=1)
            attention, _ = self.target_attention(context, context, context)

            def interact(h):
                weights = torch.softmax(
                    torch.bmm(h.unsqueeze(1), attention.transpose(1, 2)).squeeze(1)
                    / (200 ** 0.5), dim=-1)
                v = torch.bmm(weights.unsqueeze(1), attention).squeeze(1)
                return torch.sigmoid(self.interaction(h * v)), v

            pred1, v1 = interact(h1)
            pred2, v2 = interact(h2)
            features = torch.cat([h1, h2, (pred1 - pred2).abs(), pred1 * pred2, v1, v2],
                                 dim=1)
            return self.head(features)

    return AdaptedDeepTraSynergy()

def train_and_predict(fold: int, seed: int = 42, config: dict | None = None,
                      write: bool = True) -> dict:
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    started = time.time()
    if config is None:
        config = common.load_best_config("DeepTraSynergy")
    cfg = _cfg(config)
    subsample = cfg.pop("train_subsample", None)
    tokens = _tokens()
    profiles = _target_profiles()
    frames = {split: common.load_fold_frame(fold, split) for split in common.SPLITS}
    if subsample:
        frames["train"] = common.subsample_frame(frames["train"], int(subsample), seed)
    tensors = {split: (torch.from_numpy(frame["drug1_id"].to_numpy(dtype=np.int64)),
                       torch.from_numpy(frame["drug2_id"].to_numpy(dtype=np.int64)),
                       torch.from_numpy(frame["label"].to_numpy(dtype=np.int64)))
               for split, frame in frames.items()}

    torch.manual_seed(int(seed))
    model = build_model(common.n_drugs(), cfg)
    with torch.no_grad():
        model.target_profiles.copy_(profiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    criterion = nn.CrossEntropyLoss()
    rng = np.random.RandomState(int(seed))

    def batches(split: str, shuffle: bool):
        drug1, drug2, labels = tensors[split]
        order = rng.permutation(len(drug1)) if shuffle else np.arange(len(drug1))
        for start in range(0, len(order), int(cfg["batch_size"])):
            index = torch.from_numpy(order[start:start + int(cfg["batch_size"])])
            yield tokens[drug1[index]], tokens[drug2[index]], drug1[index], drug2[index], \
                labels[index]

    def evaluate(split: str) -> np.ndarray:
        model.eval()
        probabilities = []
        with torch.no_grad():
            for tokens1, tokens2, drug1, drug2, _ in batches(split, shuffle=False):
                probabilities.append(torch.softmax(
                    model(tokens1, tokens2, drug1, drug2), dim=1)[:, 1].numpy())
        return np.concatenate(probabilities)

    best_auc, best_state, stale = -1.0, None, 0
    for _epoch in range(int(cfg["epochs"])):
        model.train()
        for tokens1, tokens2, drug1, drug2, labels in batches("train", shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(tokens1, tokens2, drug1, drug2), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip_norm"]))
            optimizer.step()
        val_auc = roc_auc_score(frames["val"]["label"].to_numpy(), evaluate("val"))
        print(f"  [DeepTraSynergy fold{fold}] epoch {_epoch + 1}/{cfg['epochs']} "
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
        frame.insert(0, "model", common.DISPLAY["DeepTraSynergy"])
        frame.insert(1, "fold", int(fold))
        frame.insert(2, "seed", int(seed))
        frame["true_label"] = test_frame["label"].to_numpy()
        frame["predicted_label"] = (test_probability >= threshold).astype(int)
        frame["predicted_probability"] = test_probability
        common.save_predictions("DeepTraSynergy", fold, seed, frame)
    return {"val_probability": val_probability, "test_probability": test_probability,
            "threshold": threshold,
            "meta": {"layers": int(cfg["layers"]),
                     "head_hidden": list(cfg["head_hidden"]),
                     "dropout": float(cfg["dropout"]), "lr": float(cfg["lr"]),
                     "val_auroc": float(best_auc),
                     "elapsed_sec": round(time.time() - started, 1),
                     "notes": "SMILES patch-Transformer + DTI/ProtT5; no cell line"}}

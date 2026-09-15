from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "features"
DEFAULT_MODEL = r"E:\KGP-DC\Rostlab\prot_t5_xl_uniref50"
EMBED_DIM = 1024


def _ensure_local_checkpoint(model_name: str) -> None:
    candidate = Path(model_name)
    if not candidate.exists():
        raise FileNotFoundError(
            f"Cannot find local ProtT5 checkpoint: {candidate}\n"
            "Point --model-name at the local directory that contains "
            "pytorch_model.bin and spiece.model")


def _ensure_sentencepiece() -> None:
    try:
        import sentencepiece
    except ImportError:
        raise SystemExit(
            "T5Tokenizer requires the 'sentencepiece' library.\n"
            f"Interpreter: {sys.executable}\n"
            f"Fix: {sys.executable} -m pip install sentencepiece")


def _load_tokenizer(model_name: str):
    try:
        return T5Tokenizer.from_pretrained(model_name, legacy=False)
    except TypeError:
        return T5Tokenizer.from_pretrained(model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=FEAT_DIR / "target_table.csv")
    parser.add_argument("--output-dir", type=Path, default=FEAT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_sentencepiece()
    _ensure_local_checkpoint(args.model_name)
    df = pd.read_csv(args.input, dtype=str).fillna("")
    df["has_sequence"] = df["sequence"].str.strip().astype(bool)

    model = T5EncoderModel.from_pretrained(args.model_name)
    tokenizer = _load_tokenizer(args.model_name)
    model.eval()
    device = torch.device(args.device)
    model = model.to(device)

    ids = df["target_id"].astype(int).to_numpy()
    seqs = df["sequence"].str.strip().tolist()

    n_real = len(df)
    out = np.zeros((n_real + 1, EMBED_DIM), dtype=np.float32)
    todo = [i for i, s in enumerate(seqs) if s]

    with torch.no_grad(), tqdm(total=len(todo), desc="Protein") as pbar:
        for i in todo:
            clean = seqs[i].upper().replace(" ", "")
            if len(clean) < 5:
                continue
            processed = " ".join(list(clean))
            enc = tokenizer(processed, return_tensors="pt",
                            padding=True, truncation=True, max_length=args.max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            lengths = enc["attention_mask"].sum(dim=1, keepdim=True).float().clamp(min=1e-9)
            out[i] = (summed / lengths).squeeze().cpu().numpy()
            pbar.update(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "target_features.npy", out)
    np.save(args.output_dir / "target_ids.npy", ids)


if __name__ == "__main__":
    main()

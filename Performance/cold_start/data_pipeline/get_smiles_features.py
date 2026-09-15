from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "features"
DEFAULT_MODEL = r"E:\KGP-DC\Seyonec\ChemBERT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=FEAT_DIR / "drug_table.csv")
    parser.add_argument("--output-dir", type=Path, default=FEAT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, dtype=str).fillna("")
    df["has_smiles"] = df["smiles"].str.strip().astype(bool)
    ids = df["internal_id"].astype(int).to_numpy()

    model = AutoModelForMaskedLM.from_pretrained(args.model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    smiles = df["smiles"].str.strip().tolist()
    out = np.zeros((len(df), model.config.hidden_size), dtype=np.float32)
    todo = [i for i, s in enumerate(smiles) if s]

    with torch.no_grad(), tqdm(total=len(todo), desc="SMILES") as pbar:
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            text = [smiles[i] for i in chunk]
            enc = tokenizer(text, return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            feat = last_hidden.mean(dim=1).cpu().numpy()
            for k, row_index in enumerate(chunk):
                out[row_index] = feat[k]
            pbar.update(len(chunk))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "drug_smiles_features.npy", out)
    np.save(args.output_dir / "drug_ids.npy", ids)


if __name__ == "__main__":
    main()

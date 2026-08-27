"""Build or validate the KGP-DC knowledge-graph features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KG_DIR = Path(__file__).parent
ROOT = KG_DIR.parent


def resolve_feature_dir(kg_dir: Path | str = KG_DIR) -> Path:
    """Find a complete rebuilt or pretrained KG feature set."""
    kg_dir = Path(kg_dir)
    required = ("drug_kg_features.npy", "kg_embedding_status.csv", "kg_summary.json")
    direct = [kg_dir / name for name in required]
    if all(path.exists() for path in direct):
        return kg_dir
    if any(path.exists() for path in direct):
        raise FileNotFoundError(f"Incomplete KG feature set in {kg_dir}")
    pretrained = kg_dir / "pretrained"
    fallback = [pretrained / name for name in required]
    if all(path.exists() for path in fallback):
        return pretrained
    raise FileNotFoundError("No complete KG feature set was found")


def load_drug_kg_features(kg_dir: Path | str = KG_DIR) -> tuple[np.ndarray, pd.DataFrame]:
    kg_dir = resolve_feature_dir(kg_dir)
    features = np.load(kg_dir / "drug_kg_features.npy").astype(np.float32)
    status = pd.read_csv(kg_dir / "kg_embedding_status.csv")
    if features.shape != (len(status), 128):
        raise ValueError(f"KG feature/status mismatch: {features.shape}, {len(status)} rows")
    if not np.isfinite(features).all():
        raise ValueError("KG drug features contain NaN or Inf")
    return features, status


def rebuild_kg(
    output_dir: Path | str = KG_DIR,
    embedding_dim: int = 128,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    seed: int = 42,
    device: str = "auto",
    force: bool = False,
) -> dict:
    """Retrain a strict-cold TransE model and zero-fill drugs without KG."""
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    output_dir = Path(output_dir)
    model_path = output_dir / "entity_embeddings.npy"
    if model_path.exists() and not force:
        raise FileExistsError("KG outputs already exist; add --force to replace them")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_triples = pd.read_csv(KG_DIR / "train_kg_triples.csv")
    columns = ["head", "relation", "tail"]
    if not set(columns).issubset(candidate_triples.columns):
        candidate_triples = candidate_triples.iloc[:, :3]
        candidate_triples.columns = columns
    candidate_triples[columns] = candidate_triples[columns].astype(str)

    drugs = pd.read_csv(ROOT / "data" / "split" / "all_drug_smiles.csv")
    drug_entity = dict(zip(
        drugs["internal_id"].astype(int),
        drugs["entity"].astype(str).str.strip(),
    ))
    train_ids = pd.read_csv(ROOT / "data" / "split" / "train_drugs.csv")["internal_id"].astype(int)
    val_ids = pd.read_csv(
        ROOT / "data" / "split" / "val_cold_drugs.csv"
    )["internal_id"].astype(int)
    test_ids = pd.read_csv(
        ROOT / "data" / "split" / "test_cold_drugs.csv"
    )["internal_id"].astype(int)
    train_entities = {drug_entity[int(drug)] for drug in train_ids}
    cold_entities = {drug_entity[int(drug)] for drug in pd.concat([val_ids, test_ids])}

    relation = candidate_triples["relation"]
    ddi_mask = (
        relation.eq("combination")
        & candidate_triples["head"].isin(train_entities)
        & candidate_triples["tail"].isin(train_entities)
    )
    dti_mask = relation.eq("interaction") & candidate_triples["head"].isin(train_entities)
    triples = (
        candidate_triples.loc[ddi_mask | dti_mask, columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if triples["head"].isin(cold_entities).any() or triples["tail"].isin(cold_entities).any():
        raise ValueError("Cold val/test drug entity leaked into strict training KG")
    if len(triples) == 0:
        raise ValueError("Strict training KG is empty")
    train_pairs = pd.read_csv(ROOT / "data" / "split" / "train_pairs.csv")
    expected_ddi = {
        tuple(sorted((drug_entity[int(a)], drug_entity[int(b)])))
        for a, b in zip(train_pairs["drug1_id"], train_pairs["drug2_id"])
    }
    strict_ddi = triples.loc[triples["relation"].eq("combination")]
    actual_ddi = {tuple(sorted((str(row.head), str(row.tail))))
                  for row in strict_ddi.itertuples(index=False)}
    if actual_ddi != expected_ddi:
        raise ValueError(
            f"Training KG DDI mismatch: missing={len(expected_ddi - actual_ddi)}, "
            f"unexpected={len(actual_ddi - expected_ddi)}"
        )
    factory = TriplesFactory.from_labeled_triples(
        triples[columns].astype(str).to_numpy(), create_inverse_triples=False
    )
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    result = pipeline(
        training=factory,
        model="TransE",
        model_kwargs={"embedding_dim": embedding_dim},
        training_loop="sLCWA",
        training_kwargs={"num_epochs": epochs, "batch_size": batch_size},
        optimizer="Adam",
        optimizer_kwargs={"lr": learning_rate},
        loss="SoftplusLoss",
        random_seed=seed,
        device=device,
    )
    model = result.model.cpu()
    entity_embeddings = model.entity_representations[0]().detach().cpu().numpy().astype(np.float32)
    relation_embeddings = (
        model.relation_representations[0]().detach().cpu().numpy().astype(np.float32)
    )
    entity_map = pd.DataFrame(sorted(factory.entity_to_id.items(), key=lambda item: item[1]),
                              columns=["entity", "id"])[["id", "entity"]]
    relation_map = pd.DataFrame(sorted(factory.relation_to_id.items(), key=lambda item: item[1]),
                                columns=["relation", "id"])[["id", "relation"]]

    drug_ids = np.load(ROOT / "data" / "features" / "drug_ids.npy").astype(int)
    drug_rows = drugs.set_index("internal_id").loc[drug_ids]
    entity_to_id = dict(zip(entity_map["entity"].astype(str), entity_map["id"].astype(int)))
    drug_features = np.zeros((len(drug_ids), embedding_dim), dtype=np.float32)
    status_rows = []
    for index, (drug_id, row) in enumerate(drug_rows.iterrows()):
        entity = str(row["entity"]).strip()
        if entity in entity_to_id:
            drug_features[index] = entity_embeddings[entity_to_id[entity]]
            source = "trained"
            has_relation = True
        else:
            source = "missing_zero"
            has_relation = False
        status_rows.append((entity, int(drug_id), source, has_relation, embedding_dim))

    split_ids = {
        "train": pd.read_csv(
            ROOT / "data" / "split" / "train_drugs.csv"
        )["internal_id"].astype(int),
        "val": pd.read_csv(
            ROOT / "data" / "split" / "val_cold_drugs.csv"
        )["internal_id"].astype(int),
        "test": pd.read_csv(
            ROOT / "data" / "split" / "test_cold_drugs.csv"
        )["internal_id"].astype(int),
    }
    drug_to_index = {drug: index for index, drug in enumerate(drug_ids)}
    np.save(output_dir / "entity_embeddings.npy", entity_embeddings)
    np.save(output_dir / "relation_embeddings.npy", relation_embeddings)
    np.save(output_dir / "mapped_triples.npy", factory.mapped_triples.cpu().numpy())
    np.save(output_dir / "drug_kg_features.npy", drug_features)
    triples.to_csv(output_dir / "strict_train_kg_triples.csv", index=False)
    entity_map.to_csv(output_dir / "entity_mapping.csv", index=False)
    relation_map.to_csv(output_dir / "relation_mapping.csv", index=False)
    status = pd.DataFrame(status_rows, columns=["entity", "internal_id", "embedding_source",
                                                "has_kg_relation", "embedding_dimension"])
    status.to_csv(output_dir / "kg_embedding_status.csv", index=False)
    for split, ids in split_ids.items():
        indexes = [drug_to_index[int(drug)] for drug in ids]
        np.save(output_dir / f"{split}_kg_features.npy", drug_features[indexes])

    summary = {
        "candidate_triples": int(len(candidate_triples)),
        "triples": int(len(triples)), "entities": int(len(entity_map)),
        "ddi_triples": int(triples["relation"].eq("combination").sum()),
        "dti_triples": int(triples["relation"].eq("interaction").sum()),
        "embedding_dim": embedding_dim, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "seed": seed,
        "final_loss": float(result.losses[-1]),
        "trained_drugs": int((status.embedding_source == "trained").sum()),
        "missing_zero_drugs": int((status.embedding_source == "missing_zero").sum()),
        "smiles_kg_inference": False,
        "strict_cold_entity_exclusion": True,
        "cold_entities_in_training_kg": 0,
    }
    with open(output_dir / "kg_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status")
    train = sub.add_parser("train")
    train.add_argument("--embedding-dim", type=int, default=128)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command in {None, "status"}:
        values, state = load_drug_kg_features()
        print(f"KG={values.shape}; sources={state.embedding_source.value_counts().to_dict()}")
    else:
        print(json.dumps(rebuild_kg(
            embedding_dim=args.embedding_dim, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            seed=args.seed, force=args.force,
            device=args.device,
        ), indent=2))

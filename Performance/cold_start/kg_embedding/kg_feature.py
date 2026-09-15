from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_ROOT = HERE

DDI_RELATION = "combination"
DTI_RELATION = "interaction"
SPLITS = ("train", "val", "test")
VALID_MODES = ("single", "dual")
POOL_FILES = {
    "dual": {"train": "train_drugs.csv", "val": "val_cold_drugs.csv",
             "test": "test_cold_drugs.csv"},
    "single": {"train": "train_drugs.csv", "val": "val_drugs.csv",
               "test": "test_drugs.csv"},
    "warm": {split: f"{split}_drugs.csv" for split in SPLITS},
}
WARM_MODES = ("warm",)
SCENARIO = "cold"

SHARED_FILES = ("triples.csv", "drug_kg_features.npy",
                "kg_embedding_status.csv", "entity_embeddings.npy",
                "relation_embeddings.npy", "mapped_triples.npy",
                "entity_mapping.csv", "relation_mapping.csv")


def scenario_modes() -> tuple[str, ...]:
    return WARM_MODES if SCENARIO == "warm" else VALID_MODES


def data_root() -> Path:
    for name in ("entity_split_data", "data"):
        candidate = ROOT / name
        if (candidate / "split").is_dir():
            return candidate
    raise FileNotFoundError(
        "neither entity_split_data/ nor data/ found under " + str(ROOT))


def shared_dir() -> Path:
    if SCENARIO == "warm":
        return OUT_ROOT / "warm" / "shared"
    return OUT_ROOT / "shared"


def fold_dir(mode: str, split: str) -> Path:
    return OUT_ROOT / mode / split


def find_database_dir() -> Path:
    candidates = [ROOT.parent / "database" / "Triples",
                  ROOT.parent / "database",
                  ROOT.parent.parent / "Database" / "Triples",
                  ROOT.parent.parent / "Database"]
    for directory in candidates:
        if (directory / "DDI.csv").exists() and (directory / "DTI.csv").exists():
            return directory
    raise FileNotFoundError(
        "Raw database not found; expected DDI.csv + DTI.csv under "
        + " | ".join(str(c) for c in candidates))


def load_tables(features_dir: Path):
    drug_ids = np.load(features_dir / "drug_ids.npy").astype(np.int64)
    table = pd.read_csv(features_dir / "drug_table.csv", dtype=str).fillna("")
    table["internal_id"] = table["internal_id"].astype(int)
    if len(table) != len(drug_ids):
        raise ValueError(f"drug_table.csv {len(table)} != drug_ids.npy {len(drug_ids)}")
    table = table.set_index("internal_id").loc[drug_ids]
    name_to_id: dict[str, int] = {}
    id_to_entity: dict[int, str] = {}
    for drug_id, name, entity in zip(table.index, table["drug_name"], table["entity"]):
        name, entity = str(name).strip(), str(entity).strip()
        if name:
            name_to_id[name] = int(drug_id)
        if entity and entity.lower() not in {"nan", "none", "null"}:
            id_to_entity[int(drug_id)] = entity
    return name_to_id, id_to_entity, int(len(drug_ids))


def read_pairs(split_dir: Path, split: str) -> set[tuple[int, int]]:
    frame = pd.read_csv(split_dir / f"{split}_pairs.csv")
    pairs: set[tuple[int, int]] = set()
    for a, b in zip(frame["drug1_id"], frame["drug2_id"]):
        ia, ib = int(a), int(b)
        if ia != ib:
            pairs.add((ia, ib) if ia < ib else (ib, ia))
    return pairs


def read_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    if frame.shape[1] != 3:
        raise ValueError(f"{path} must have exactly 3 columns, got {frame.shape[1]}")
    frame.columns = ["head", "relation", "tail"]
    return frame.apply(lambda c: c.str.strip() if c.dtype == object else c
                       ).dropna().reset_index(drop=True)


def mode_pools(mode: str) -> dict[str, set[int]]:
    split_dir = data_root() / "split" / mode
    return {name: set(pd.read_csv(split_dir / f)["internal_id"].astype(int))
            for name, f in POOL_FILES[mode].items()}


def graph_drug_universe() -> set[int]:
    universe: set[int] = set()
    for mode in scenario_modes():
        for pool in mode_pools(mode).values():
            universe |= pool
    return universe


def train_positives() -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for mode in scenario_modes():
        pairs |= read_pairs(data_root() / "split" / mode, "train")
    return pairs


def held_out_positives() -> dict[str, set[tuple[int, int]]]:
    return {f"{mode}/{split}": read_pairs(data_root() / "split" / mode, split)
            for mode in scenario_modes() for split in ("val", "test")}


def pair_to_triple(a: int, b: int, id_to_entity: dict[int, str]
                   ) -> tuple[str, str, str] | None:
    if a not in id_to_entity or b not in id_to_entity:
        return None
    head, tail = sorted((id_to_entity[a], id_to_entity[b]))
    return head, DDI_RELATION, tail


def build_shared_triples(*, db_dir: Path | None = None,
                         force: bool = False) -> dict:
    db_dir = Path(db_dir) if db_dir else find_database_dir()
    out_dir = shared_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "triples.csv"
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} exists; pass --force to rebuild")

    name_to_id, id_to_entity, n_drugs = load_tables(data_root() / "features")
    universe = graph_drug_universe() & set(id_to_entity)
    if not universe:
        raise RuntimeError("empty entity-resolvable universe")

    positives = train_positives()
    combo_triples: set[tuple[str, str, str]] = set()
    combo_skipped = 0
    for a, b in positives:
        triple = pair_to_triple(a, b, id_to_entity)
        if triple is None:
            combo_skipped += 1
        else:
            combo_triples.add(triple)

    dti = read_raw(db_dir / "DTI.csv")
    interaction_triples: set[tuple[str, str, str]] = set()
    dti_missing = dti_out_universe = 0
    for head, relation, tail in dti.itertuples(index=False):
        drug = name_to_id.get(str(head))
        if drug is None:
            dti_missing += 1
            continue
        if drug not in universe:
            dti_out_universe += 1
            continue
        interaction_triples.add((id_to_entity[drug], DTI_RELATION, str(tail)))

    rows = sorted(combo_triples) + sorted(interaction_triples)
    frame = pd.DataFrame(rows, columns=["head", "relation", "tail"]).drop_duplicates()
    frame.to_csv(out_path, index=False)

    held_rows: list[dict] = []
    for key, pairs in held_out_positives().items():
        mode, split = key.split("/")
        for a, b in sorted(pairs):
            triple = pair_to_triple(a, b, id_to_entity)
            if triple is None:
                continue
            held_rows.append({"mode": mode, "split": split, "drug1_id": a,
                              "drug2_id": b, "head": triple[0],
                              "relation": triple[1], "tail": triple[2]})
    held_frame = pd.DataFrame(held_rows).drop_duplicates()
    held_frame.to_csv(out_dir / "held_out_positive_triples.csv", index=False)

    combo_edges = {(h, t) for h, r, t in combo_triples}
    leaked = [(row.drug1_id, row.drug2_id) for row in held_frame.itertuples(index=False)
              if (row.head, row.tail) in combo_edges]
    if leaked:
        raise RuntimeError(f"{len(leaked)} held-out positive pairs leaked into "
                           f"the shared graph, e.g. {leaked[:5]}")

    summary = {
        "graph": f"{SCENARIO}-shared",
        "scenario": SCENARIO,
        "definition": ("warm train pairs (DDI) + DTI of every drug"
                       if SCENARIO == "warm" else
                       "train positives (DDI) + DTI of every split pool drug"),
        "out_path": str(out_path),
        "train_positive_pairs": int(len(positives)),
        "combination": int(len(combo_triples)),
        "interaction": int(len(interaction_triples)),
        "triples": int(len(frame)),
        "entities": int(len(set(frame["head"]) | set(frame["tail"]))),
        "universe_drugs": int(len(universe)),
        "held_out_positive_pairs": int(len(held_frame)),
        "held_out_pairs_in_graph": 0,
        "combo_pairs_skipped_no_entity": combo_skipped,
        "dti_skipped_missing_name": dti_missing,
        "dti_skipped_out_of_universe": dti_out_universe,
        "drugs": int(n_drugs),
    }
    (out_dir / "triples_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def train_shared_embedding(*, embedding_dim: int = 128, epochs: int = 50,
                           batch_size: int = 512, learning_rate: float = 0.001,
                           model: str = "TransE", loss: str = "SoftplusLoss",
                           seed: int = 42, device: str = "auto",
                           force: bool = False) -> dict:
    out_dir = shared_dir()
    triples_path = out_dir / "triples.csv"
    if not triples_path.exists():
        raise FileNotFoundError(f"{triples_path} missing; run 'graph' first")
    summary_path = out_dir / "kg_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(f"{summary_path} exists; pass --force to retrain")

    features_dir = data_root() / "features"
    _, id_to_entity, n_drugs = load_tables(features_dir)
    triples = pd.read_csv(triples_path, dtype=str)
    if len(triples) == 0:
        raise RuntimeError("shared graph is empty")

    try:
        import torch
        import pykeen.losses as pykeen_losses
        import pykeen.models as pykeen_models
        from pykeen.sampling import BasicNegativeSampler
        from pykeen.triples import TriplesFactory
        from pykeen.training import SLCWATrainingLoop
    except ImportError as error:
        raise SystemExit(
            "PyKEEN required.\n"
            f"{sys.executable} -m pip install pykeen") from error
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls = getattr(pykeen_models, model, None)
    if model_cls is None:
        raise ValueError(f"Unknown PyKEEN model {model!r}")
    loss_cls = getattr(pykeen_losses, loss, None) if loss else None
    loss_instance = loss_cls() if loss_cls is not None else None

    started = time.time()
    torch.manual_seed(seed)
    factory = TriplesFactory.from_labeled_triples(
        triples.to_numpy(), create_inverse_triples=False)

    filter_rows = {(h, r, t) for h, r, t in triples.itertuples(index=False)}
    held_path = out_dir / "held_out_positive_triples.csv"
    held_unmappable = 0
    if held_path.exists():
        held = pd.read_csv(held_path, dtype=str)
        for h, r, t in held[["head", "relation", "tail"]].itertuples(index=False):
            filter_rows.add((str(h), str(r), str(t)))
    mapped: list[tuple[int, int, int]] = []
    for h, r, t in filter_rows:
        hi = factory.entity_to_id.get(h)
        ti = factory.entity_to_id.get(t)
        ri = factory.relation_to_id.get(r)
        if hi is None or ti is None or ri is None:
            held_unmappable += 1
            continue
        mapped.append((hi, ri, ti))
    filter_triples = torch.as_tensor(mapped, dtype=torch.long)
    sampler = BasicNegativeSampler(
        mapped_triples=filter_triples, filtered=True,
        num_entities=factory.num_entities, num_relations=factory.num_relations)
    protected = len(mapped) - len(triples)

    kg_model = model_cls(triples_factory=factory, embedding_dim=embedding_dim,
                         random_seed=seed, loss=loss_instance)
    kg_model.to(torch.device(device))
    loop = SLCWATrainingLoop(model=kg_model, triples_factory=factory,
                             optimizer="Adam",
                             optimizer_kwargs={"lr": learning_rate},
                             negative_sampler=sampler)
    losses = loop.train(triples_factory=factory, num_epochs=epochs,
                        batch_size=batch_size, use_tqdm=True)
    kg_model = kg_model.cpu()
    entity_embeddings = (kg_model.entity_representations[0]()
                         .detach().cpu().numpy().astype(np.float32))
    relation_embeddings = (kg_model.relation_representations[0]()
                           .detach().cpu().numpy().astype(np.float32))
    entity_to_id = factory.entity_to_id

    drug_ids = np.load(features_dir / "drug_ids.npy").astype(np.int64)
    drug_features = np.zeros((n_drugs, embedding_dim), dtype=np.float32)
    status_rows: list[dict] = []
    trained = missing = 0
    for drug_id in drug_ids:
        entity = id_to_entity.get(int(drug_id), "")
        if entity in entity_to_id:
            drug_features[int(drug_id)] = entity_embeddings[entity_to_id[entity]]
            source, has_kg = "trained", True
            trained += 1
        else:
            source, has_kg = "missing_zero", False
            missing += 1
        status_rows.append({"entity": entity, "internal_id": int(drug_id),
                            "embedding_source": source,
                            "has_kg_relation": has_kg,
                            "embedding_dimension": embedding_dim})
    entity_map = pd.DataFrame(sorted(entity_to_id.items(),
                                     key=lambda item: item[1]),
                              columns=["entity", "id"])
    relation_map = pd.DataFrame(sorted(factory.relation_to_id.items(),
                                       key=lambda item: item[1]),
                                columns=["relation", "id"])

    np.save(out_dir / "drug_kg_features.npy", drug_features)
    pd.DataFrame(status_rows).to_csv(out_dir / "kg_embedding_status.csv",
                                     index=False)
    np.save(out_dir / "entity_embeddings.npy", entity_embeddings)
    np.save(out_dir / "relation_embeddings.npy", relation_embeddings)
    np.save(out_dir / "mapped_triples.npy", factory.mapped_triples.cpu().numpy())
    entity_map.to_csv(out_dir / "entity_mapping.csv", index=False)
    relation_map.to_csv(out_dir / "relation_mapping.csv", index=False)

    triples_summary = json.loads(
        (out_dir / "triples_summary.json").read_text(encoding="utf-8"))
    summary = {
        "graph": "shared",
        "embedding_dim": int(embedding_dim),
        "model": model, "epochs": int(epochs), "batch_size": int(batch_size),
        "learning_rate": float(learning_rate), "seed": int(seed),
        "device": str(device), "loss": loss,
        "triples_path": str(triples_path),
        "entities": int(len(entity_to_id)),
        "relations": int(len(factory.relation_to_id)),
        "final_loss": float(losses[-1]) if losses else None,
        "elapsed_seconds": round(time.time() - started, 1),
        "drugs": int(n_drugs), "trained_drugs": trained,
        "missing_zero_drugs": missing,
        "negative_sampler": "BasicNegativeSampler(filtered=True)",
        "negative_filter_triples": int(len(mapped)),
        "protected_held_out_triples": int(protected),
        "negative_filter_unmappable": int(held_unmappable),
        "cold_kg_policy": ("pair-level cold: no evaluated DDI pair is a graph "
                           "edge; cold drugs may still carry DTI-derived vectors"),
        "strict_cold_entity_exclusion": False,
    }
    summary.update({"graph_triples": triples_summary["triples"],
                    "graph_combination": triples_summary["combination"],
                    "graph_interaction": triples_summary["interaction"],
                    "graph_entities": triples_summary["entities"]})
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    return summary


def distribute_to_folds(*, force: bool = False) -> list[dict]:
    out_dir = shared_dir()
    missing = [name for name in SHARED_FILES if not (out_dir / name).exists()]
    if missing or not (out_dir / "kg_summary.json").exists():
        raise FileNotFoundError(f"shared assets incomplete: missing {missing}; "
                                "run 'graph' and 'train' first")
    shared_summary = json.loads(
        (out_dir / "kg_summary.json").read_text(encoding="utf-8"))
    triples_summary = json.loads(
        (out_dir / "triples_summary.json").read_text(encoding="utf-8"))
    _, id_to_entity, _ = load_tables(data_root() / "features")
    kg = np.load(out_dir / "drug_kg_features.npy")
    status = pd.read_csv(out_dir / "kg_embedding_status.csv")
    has_kg = status["has_kg_relation"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}).to_numpy()
    drug_ids = np.load(data_root() / "features" / "drug_ids.npy").astype(np.int64)
    if kg.shape != (len(drug_ids), int(shared_summary["embedding_dim"])):
        raise ValueError("shared drug_kg_features.npy shape mismatch")
    if not np.array_equal(status["internal_id"].astype(int).to_numpy(), drug_ids):
        raise ValueError("shared status rows are not aligned with drug_ids.npy")
    usable = {int(i) for i, ok in enumerate(has_kg) if ok}

    combo_edges = {(h, t) for h, r, t in
                   pd.read_csv(out_dir / "triples.csv", dtype=str)
                   .itertuples(index=False) if r == DDI_RELATION}

    results: list[dict] = []
    for mode in scenario_modes():
        for split in SPLITS:
            directory = fold_dir(mode, split)
            directory.mkdir(parents=True, exist_ok=True)
            for name in SHARED_FILES:
                shutil.copyfile(out_dir / name, directory / name)

            positives = read_pairs(data_root() / "split" / mode, split)
            pos_ok = skipped = in_graph = unresolvable = 0
            for a, b in positives:
                if a not in usable or b not in usable:
                    skipped += 1
                    continue
                pos_ok += 1
                triple = pair_to_triple(a, b, id_to_entity)
                if triple is None:
                    unresolvable += 1
                elif (triple[0], triple[2]) in combo_edges:
                    in_graph += 1
            cold = {x for p in positives for x in p if x not in usable}
            fold_summary = {
                "graph": "shared",
                "mode": mode, "split": split,
                "kg_dir": str(directory),
                "shared_source": str(out_dir),
                "positive_pairs": int(len(positives)),
                "positives_with_kg_both_sides": int(pos_ok),
                "skipped_positive_no_kg": int(skipped),
                "unresolvable_pairs": int(unresolvable),
                "positives_in_graph": int(in_graph),
                "cold_drugs_without_vector": int(len(cold)),
                "embedding_dim": int(shared_summary["embedding_dim"]),
                "model": shared_summary["model"], "epochs": shared_summary["epochs"],
                "seed": shared_summary["seed"],
                "graph_triples": triples_summary["triples"],
                "graph_combination": triples_summary["combination"],
                "graph_interaction": triples_summary["interaction"],
                "held_out_pairs_in_graph": 0,
                "strict_cold_entity_exclusion": False,
                "cold_kg_policy": shared_summary["cold_kg_policy"],
                "negative_sampler": shared_summary["negative_sampler"],
                "evaluated_pairs_are_graph_edges": bool(split == "train"),
            }
            (directory / "kg_summary.json").write_text(
                json.dumps(fold_summary, indent=2, ensure_ascii=False),
                encoding="utf-8")
            triples_doc = dict(triples_summary)
            triples_doc.update({"fold": f"{mode}/{split}",
                                "note": "shared graph, identical for all folds",
                                "evaluated_pairs_in_graph": 0 if split != "train"
                                else "train positives are graph edges by design"})
            (directory / "triples_summary.json").write_text(
                json.dumps(triples_doc, indent=2, ensure_ascii=False),
                encoding="utf-8")
            results.append(fold_summary)
    return results


def verify_folds() -> list[dict]:
    out_dir = shared_dir()
    _, id_to_entity, _ = load_tables(data_root() / "features")
    triples = pd.read_csv(out_dir / "triples.csv", dtype=str)
    combo_edges = {(h, t) for h, r, t in triples.itertuples(index=False)
                   if r == DDI_RELATION}
    status = pd.read_csv(out_dir / "kg_embedding_status.csv")
    has_kg = status["has_kg_relation"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}).to_numpy()
    usable = {int(i) for i, ok in enumerate(has_kg) if ok}

    reports: list[dict] = []
    for mode in scenario_modes():
        for split in SPLITS:
            positives = read_pairs(data_root() / "split" / mode, split)
            checked = leaked = unresolvable = 0
            for a, b in positives:
                triple = pair_to_triple(a, b, id_to_entity)
                if triple is None:
                    unresolvable += 1
                    continue
                checked += 1
                if (triple[0], triple[2]) in combo_edges:
                    leaked += 1
            if split != "train" and leaked:
                raise RuntimeError(
                    f"{mode}/{split}: {leaked} evaluated positive pairs are graph "
                    "edges -> KG leakage")
            directory = fold_dir(mode, split)
            assets_ok = all((directory / name).exists() for name in SHARED_FILES)
            reports.append({"fold": f"{mode}/{split}", "positives": len(positives),
                            "checked": checked, "unresolvable": unresolvable,
                            "in_graph": leaked,
                            "in_graph_expected": "train positives (by design)"
                            if split == "train" else 0,
                            "assets_present": assets_ok,
                            "drugs_with_vector": len(usable)})
    return reports


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--model", default="TransE")
    parser.add_argument("--loss", default="SoftplusLoss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")


def _add_scenario_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", choices=("cold", "warm"), default="cold")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    graph = sub.add_parser("graph")
    graph.add_argument("--db-dir", type=Path, default=None)
    graph.add_argument("--force", action="store_true")
    _add_scenario_arg(graph)
    train = sub.add_parser("train")
    _add_train_args(train)
    _add_scenario_arg(train)
    build = sub.add_parser("build")
    build.add_argument("--db-dir", type=Path, default=None)
    _add_train_args(build)
    _add_scenario_arg(build)
    verify = sub.add_parser("verify")
    _add_scenario_arg(verify)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global SCENARIO
    args = _parse_args(argv)
    SCENARIO = args.scenario
    if args.command == "graph":
        build_shared_triples(db_dir=args.db_dir, force=args.force)
    elif args.command == "train":
        train_shared_embedding(
            embedding_dim=args.embedding_dim, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            model=args.model, loss=args.loss, seed=args.seed,
            device=args.device, force=args.force)
        distribute_to_folds(force=args.force)
    elif args.command == "build":
        build_shared_triples(db_dir=args.db_dir, force=args.force)
        train_shared_embedding(
            embedding_dim=args.embedding_dim, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            model=args.model, loss=args.loss, seed=args.seed,
            device=args.device, force=args.force)
        distribute_to_folds(force=args.force)
        verify_folds()
    else:
        reports = verify_folds()


if __name__ == "__main__":
    main()

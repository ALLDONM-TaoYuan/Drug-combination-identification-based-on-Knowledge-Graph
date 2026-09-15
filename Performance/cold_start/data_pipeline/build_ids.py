from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT.parent / "Database"
FEAT_DIR = ROOT / "data" / "features"

DT_INDEX_DTYPE = np.dtype([("drug_id", np.int64), ("target_index", np.int64)])


def load_database():
    ddi = pd.read_csv(DB_DIR / "DDI.csv", header=None, dtype=str)
    ddi.columns = ["drug1", "relation", "drug2"]
    dti = pd.read_csv(DB_DIR / "DTI.csv", header=None, dtype=str)
    dti.columns = ["drug", "relation", "target"]
    smiles = pd.read_csv(DB_DIR / "smiles.csv", dtype=str)
    targets = pd.read_csv(DB_DIR / "targets.csv", dtype=str)
    for frame in (ddi, dti):
        frame.columns = frame.columns.str.strip()
    return ddi, dti, smiles, targets


def build_tables() -> dict:
    ddi, dti, smiles, targets = load_database()
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    ddi_names = sorted({str(x).strip()
                        for x in pd.concat([ddi["drug1"], ddi["drug2"]]).dropna()})
    dti_names = sorted({str(x).strip() for x in dti["drug"].dropna()})
    union_names = sorted(set(ddi_names) | set(dti_names))

    name_to_cid = dict(zip(smiles["drug_name"].str.strip(), smiles["drug_CID"].str.strip()))
    name_to_smiles = dict(zip(smiles["drug_name"].str.strip(), smiles["smiles"].str.strip()))

    drug_rows = []
    for internal, name in enumerate(union_names):
        cid = name_to_cid.get(name, "")
        smi = name_to_smiles.get(name, "")
        drug_rows.append({
            "internal_id": internal,
            "entity": cid,
            "drug_name": name,
            "smiles": smi,
            "has_smiles": bool(smi) and str(smi) != "nan",
        })
    drug_table = pd.DataFrame(drug_rows)

    dti_targets = sorted({str(x).strip() for x in dti["target"].dropna()})
    targets["target_name"] = targets["target_name"].str.strip()
    seq_map = dict(zip(targets["target_name"], targets["sequence"].str.strip()))
    target_rows = []
    for internal, raw in enumerate(dti_targets):
        seq = seq_map.get(raw, "")
        target_rows.append({
            "target_id": internal,
            "raw_target_id": raw,
            "sequence": seq,
            "has_sequence": bool(seq) and str(seq) != "nan",
        })
    target_table = pd.DataFrame(target_rows)
    n_target = len(target_table)
    zero_target_index = n_target

    name_to_drug_id = dict(zip(drug_table["drug_name"], drug_table["internal_id"]))
    raw_to_target_id = dict(zip(target_table["raw_target_id"], target_table["target_id"]))

    pairs: list[tuple[int, int]] = []
    unresolved: list[dict] = []
    for _, row in dti.iterrows():
        dname = str(row["drug"]).strip()
        tname = str(row["target"]).strip()
        did = name_to_drug_id.get(dname)
        tid = raw_to_target_id.get(tname)
        if did is None:
            unresolved.append({"kind": "dti_head_not_in_drug_space", "name": dname})
            continue
        if tid is None:
            unresolved.append({"kind": "dti_tail_not_in_target_space", "target": tname})
            continue
        pairs.append((int(did), int(tid)))

    seen: set[int] = {d for d, _ in pairs}
    for d in drug_table["internal_id"]:
        if int(d) not in seen:
            pairs.append((int(d), zero_target_index))
    pairs = sorted(set(pairs))
    index_array = np.asarray(pairs, dtype=DT_INDEX_DTYPE)

    np.save(FEAT_DIR / "drug_ids.npy", drug_table["internal_id"].to_numpy(dtype=np.int64))
    drug_table.to_csv(FEAT_DIR / "drug_table.csv", index=False)
    np.save(FEAT_DIR / "target_ids.npy", target_table["target_id"].to_numpy(dtype=np.int64))
    target_table.to_csv(FEAT_DIR / "target_table.csv", index=False)
    np.save(FEAT_DIR / "drug_target_index.npy", index_array)

    unresolved_frame = pd.DataFrame(unresolved).drop_duplicates()
    unresolved_frame.to_csv(FEAT_DIR / "target_mapping_unresolved.csv", index=False)

    summary = {
        "database_dir": str(DB_DIR),
        "drugs": int(len(drug_table)),
        "drugs_with_smiles": int(drug_table["has_smiles"].sum()),
        "targets": n_target,
        "targets_with_sequence": int(target_table["has_sequence"].sum()),
        "zero_target_index": zero_target_index,
        "dti_edges": int(len(dti)),
        "mapped_dti_edges": len(pairs) - sum(
            1 for d in drug_table["internal_id"] if int(d) not in seen),
        "unresolved_entries": int(len(unresolved_frame)),
        "feat_dir": str(FEAT_DIR),
    }
    (FEAT_DIR / "build_ids_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    build_tables()

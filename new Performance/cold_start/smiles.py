"""SMILES feature access for the cold-start experiment."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
DEFAULT_FEATURE_DIR = ROOT / "data" / "features"


def load_smiles_features(
    feature_dir: Path | str = DEFAULT_FEATURE_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load drug IDs and their 768-dimensional SMILES features."""
    feature_dir = Path(feature_dir)
    drug_ids = np.load(feature_dir / "drug_ids.npy").astype(int)
    features = np.load(feature_dir / "drug_smiles_features.npy").astype(np.float32)
    if features.shape != (len(drug_ids), 768):
        raise ValueError(f"Unexpected SMILES feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("SMILES features contain NaN or Inf")
    return drug_ids, features


if __name__ == "__main__":
    ids, values = load_smiles_features()
    print(f"SMILES: {len(ids)} drugs, shape={values.shape}, finite=True")

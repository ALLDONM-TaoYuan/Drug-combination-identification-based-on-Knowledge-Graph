"""Target feature access for the cold-start experiment."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
DEFAULT_FEATURE_DIR = ROOT / "data" / "features"


def load_target_features(
    feature_dir: Path | str = DEFAULT_FEATURE_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load target IDs and 1024-dimensional target features.

    The final feature row is the dedicated zero-target placeholder.
    """
    feature_dir = Path(feature_dir)
    target_ids = np.load(feature_dir / "target_ids.npy").astype(int)
    features = np.load(feature_dir / "target_features.npy").astype(np.float32)
    if features.ndim != 2 or features.shape[1] != 1024:
        raise ValueError(f"Unexpected target feature shape: {features.shape}")
    if features.shape[0] != len(target_ids) + 1:
        raise ValueError("Target features must contain exactly one final zero-target row")
    if not np.allclose(features[-1], 0.0):
        raise ValueError("The final target feature row must be the zero-target placeholder")
    if not np.isfinite(features).all():
        raise ValueError("Target features contain NaN or Inf")
    return target_ids, features


if __name__ == "__main__":
    ids, values = load_target_features()
    print(f"Target: {len(ids)} IDs, shape={values.shape}, finite=True")

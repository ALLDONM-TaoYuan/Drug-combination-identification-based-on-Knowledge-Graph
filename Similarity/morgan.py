import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import warnings
import os

warnings.filterwarnings('ignore')


def load_smiles_data(csv_path):
    """Load SMILES data from a CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        if 'id' not in df.columns or 'smiles' not in df.columns:
            print("Error: CSV must contain 'id' and 'smiles' columns")
            return None
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def calculate_morgan_fingerprint(smiles, radius=2, n_bits=1024):
    """Compute Morgan fingerprint for a single SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return None


def extract_rdkit_features(df, radius=2, n_bits=1024, batch_size=1000):
    """Extract Morgan fingerprints for all SMILES entries."""
    print(f"Extracting Morgan fingerprints (radius={radius}, n_bits={n_bits})...")

    features = []
    ids = []
    failed_count = 0
    total = len(df)

    for idx, row in df.iterrows():
        if idx % batch_size == 0 and idx > 0:
            print(f"  Processed {idx}/{total} ({idx / total * 100:.1f}%)")

        drug_id = row['id']
        smiles = row['smiles']

        fingerprint = calculate_morgan_fingerprint(smiles, radius, n_bits)

        if fingerprint is not None:
            features.append(fingerprint)
            ids.append(drug_id)
        else:
            failed_count += 1
            print(f"  Warning: failed to process SMILES for drug ID {drug_id}: {smiles[:50]}...")

    print(f"Feature extraction complete: {len(features)} succeeded, {failed_count} failed")

    if len(features) == 0:
        print("Error: no features were extracted")
        return None, None, failed_count

    features_array = np.array(features)
    ids_array = np.array(ids)

    return features_array, ids_array, failed_count


def save_features(features, ids, output_dir='rdkit_features'):
    """Save extracted features and IDs to disk."""
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(output_dir, 'rdkit_features.npy')
    np.save(features_path, features)

    ids_path = os.path.join(output_dir, 'rdkit_ids.npy')
    np.save(ids_path, ids)

    return features_path, ids_path


def main():
    """Main entry point for Morgan fingerprint extraction."""
    print("=" * 60)
    print("RDKit Morgan Fingerprint Extraction")
    print("=" * 60)

    csv_path = './id_smiles.csv'
    output_dir = 'db_morgan_features'

    df = load_smiles_data(csv_path)
    if df is None:
        return
    print(f"Loaded {len(df)} SMILES strings from {csv_path}")

    features, ids, failed_count = extract_rdkit_features(
        df,
        radius=2,
        n_bits=1024
    )

    if features is None:
        return

    features_path, ids_path = save_features(features, ids, output_dir)
    print(f"Features saved to: {features_path}  (shape: {features.shape})")
    print(f"IDs saved to: {ids_path}  (count: {len(ids)})")

    info_path = os.path.join(output_dir, 'processing_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("=== RDKit Feature Extraction Info ===\n")
        f.write(f"Input file: {csv_path}\n")
        f.write(f"Total SMILES: {len(df)}\n")
        f.write(f"Successfully extracted: {len(features)}\n")
        f.write(f"Failed: {failed_count}\n")
        f.write(f"Feature dimension: {features.shape[1]}\n")
        f.write(f"Features file: {features_path}\n")
        f.write(f"IDs file: {ids_path}\n")
        f.write(f"Morgan radius: 2\n")
        f.write(f"Fingerprint bits: 1024\n")

    print(f"Processing info saved to: {info_path}")
    print("=" * 60)
    print("Feature extraction completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

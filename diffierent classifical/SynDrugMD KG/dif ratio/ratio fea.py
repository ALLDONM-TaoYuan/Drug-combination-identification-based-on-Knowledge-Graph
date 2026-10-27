import numpy as np
import h5py
import json
import os
import random
from tqdm import tqdm
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


class BalancedDatasetSampler:
    """Sampler for balanced datasets with different positive-negative ratios"""

    def __init__(self, metadata_path, h5_file_path, output_dir='./balanced_datasets_new', seed=42):
        self.metadata_path = metadata_path
        self.h5_file_path = h5_file_path
        self.output_dir = output_dir
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_file_path}")

        print(f"Metadata: {metadata_path}")
        print(f"HDF5: {h5_file_path}")
        print(f"Size: {os.path.getsize(h5_file_path) / (1024 ** 3):.2f} GB")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        os.makedirs(output_dir, exist_ok=True)

    def _get_total_samples(self, set_name, data_type):
        return sum(item['samples'] for item in self.metadata[set_name][data_type])

    def _reservoir_sampling(self, batch_list, n_samples, rng):
        reservoir_features = []
        reservoir_labels = []
        count = 0

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            for batch_info in tqdm(batch_list, desc="Sampling"):
                group = h5_file[batch_info['group_path']]
                features = group['features'][:]
                labels = group['labels'][:]

                indices = list(range(len(features)))
                rng.shuffle(indices)

                for idx in indices:
                    if count < n_samples:
                        reservoir_features.append(features[idx])
                        reservoir_labels.append(labels[idx])
                    else:
                        j = rng.randint(0, count + 1)
                        if j < n_samples:
                            reservoir_features[j] = features[idx]
                            reservoir_labels[j] = labels[idx]
                    count += 1

        return np.array(reservoir_features, dtype=np.float32), np.array(reservoir_labels, dtype=np.int32)

    def create_balanced_dataset(self, pos_counts, ratios=[1, 2, 3, 4, 5]):
        print("\n" + "=" * 50)
        print("Creating balanced datasets")
        print("=" * 50)

        for set_name in ['train', 'val', 'test']:
            total_pos = self._get_total_samples(set_name, 'positive')
            total_neg = self._get_total_samples(set_name, 'negative')
            print(f"{set_name}: Available pos={total_pos:,}, neg={total_neg:,}")

            if pos_counts[set_name] > total_pos:
                print(f"Warning: Requesting {pos_counts[set_name]:,} pos, but only {total_pos:,} available")

        for ratio in ratios:
            print(f"\nProcessing ratio 1:{ratio}")
            output_file = os.path.join(self.output_dir, f'ratio_1_{ratio}.h5')
            print(f"Output: {output_file}")

            with h5py.File(output_file, 'w') as h5_out:
                h5_out.attrs['ratio'] = f'1:{ratio}'
                h5_out.attrs['seed'] = self.seed

                for set_name in ['train', 'val', 'test']:
                    n_pos = pos_counts[set_name]
                    n_neg = n_pos * ratio
                    print(f"\n{set_name}: pos={n_pos:,}, neg={n_neg:,}")

                    # Sample positive
                    pos_features, pos_labels = self._reservoir_sampling(
                        self.metadata[set_name]['positive'], n_pos, self.rng
                    )

                    # Sample negative
                    neg_features, neg_labels = self._reservoir_sampling(
                        self.metadata[set_name]['negative'], n_neg, self.rng
                    )

                    # Combine and shuffle
                    all_features = np.concatenate([pos_features, neg_features], axis=0)
                    all_labels = np.concatenate([pos_labels, neg_labels], axis=0)

                    indices = np.arange(len(all_features))
                    self.rng.shuffle(indices)
                    all_features = all_features[indices]
                    all_labels = all_labels[indices]

                    # Save
                    set_group = h5_out.create_group(set_name)
                    set_group.create_dataset('features', data=all_features, compression='gzip')
                    set_group.create_dataset('labels', data=all_labels, compression='gzip')

                    set_group.attrs['n_samples'] = len(all_features)
                    set_group.attrs['n_pos'] = n_pos
                    set_group.attrs['n_neg'] = n_neg

                    print(f"Saved: {len(all_features):,} samples")

            print(f"Ratio 1:{ratio} saved to {output_file}")


def main():
    # Paths
    metadata_path = '../dif_classifier/feature_batches/dataset_metadata.json'
    h5_file_path = '../dif_classifier/feature_batches/dataset.h5'
    output_dir = './balanced_datasets'

    # New sample sizes
    pos_counts = {
        'train': 300000,  # 300k positive
        'val': 100000,  # 100k positive
        'test': 100000  # 100k positive
    }

    ratios = [1, 2, 3, 4, 5]

    # Create sampler
    sampler = BalancedDatasetSampler(
        metadata_path=metadata_path,
        h5_file_path=h5_file_path,
        output_dir=output_dir,
        seed=42
    )

    print(f"Positive counts: train={pos_counts['train']:,}, val={pos_counts['val']:,}, test={pos_counts['test']:,}")
    print(f"Ratios: {ratios}")

    # Create datasets
    sampler.create_balanced_dataset(pos_counts, ratios)


if __name__ == "__main__":
    main()
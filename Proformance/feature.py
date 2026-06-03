import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
import h5py
import json
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


class DrugDatasetGenerator:
    def __init__(self, embedding_path='./embeddings', smiles_path='./smiles', target_path='./targets'):
        print("Loading data files...")

        entity_mapping = pd.read_csv(f'{embedding_path}/entity_mapping.csv')
        self.entity_emb = np.load(f'{embedding_path}/entity_embeddings.npy')
        self.mapped_triples = np.load(f'{embedding_path}/mapped_triples.npy')

        smiles_features = np.load(f'{smiles_path}/smiles_features.npy')
        smiles_ids = np.load(f'{smiles_path}/smiles_ids.npy')
        protein_features = np.load(f'{target_path}/target_features.npy')
        protein_ids = np.load(f'{target_path}/target_ids.npy')

        self.drug_id_to_feature = {int(drug_id): feature for drug_id, feature in zip(smiles_ids, smiles_features)}
        self.protein_id_to_feature = {int(protein_id): feature for protein_id, feature in
                                      zip(protein_ids, protein_features)}

        print("Precomputing drug-target mappings...")
        self.drug_targets_cache = {}
        self.all_drug_ids = list(self.drug_id_to_feature.keys())
        self._precompute_drug_targets()

        print(
            f"Drugs: {len(self.drug_id_to_feature)}, Proteins: {len(self.protein_id_to_feature)}, Triples: {len(self.mapped_triples)}")

    def _precompute_drug_targets(self):
        for drug_id in tqdm(self.all_drug_ids, desc="Computing targets"):
            self.drug_targets_cache[drug_id] = self._get_drug_targets(drug_id)

    def _get_drug_targets(self, drug_id):
        targets = []
        for triple in self.mapped_triples:
            h_id, rel_id, t_id = triple
            if int(rel_id) == 1 and int(h_id) == int(drug_id):
                targets.append(int(t_id))
        return targets

    def _create_feature_vector(self, head_id, head_target_id, tail_id, tail_target_id):
        h_emb = self.entity_emb[int(head_id)]
        t_emb = self.entity_emb[int(tail_id)]

        if len(h_emb.shape) == 2:
            h_emb = np.squeeze(h_emb)
        if len(t_emb.shape) == 2:
            t_emb = np.squeeze(t_emb)

        h_drug = self.drug_id_to_feature.get(int(head_id))
        t_drug = self.drug_id_to_feature.get(int(tail_id))

        if h_drug is None or t_drug is None:
            return None

        h_target = self.protein_id_to_feature.get(int(head_target_id))
        t_target = self.protein_id_to_feature.get(int(tail_target_id))

        if h_target is None or t_target is None:
            return None

        return np.concatenate([h_emb, h_drug, h_target, t_emb, t_drug, t_target], dtype=np.float32)

    def _get_drug_drug_pairs(self):
        pairs = set()
        for triple in self.mapped_triples:
            if int(triple[1]) == 0:
                pairs.add((int(triple[0]), int(triple[2])))
        return list(pairs)

    def _save_batch_to_h5(self, h5_file, set_name, data_type, batch_id, features, labels):
        group_path = f'/{set_name}/{data_type}/batch_{batch_id:04d}'

        if group_path in h5_file:
            del h5_file[group_path]

        group = h5_file.create_group(group_path)
        group.create_dataset('features', data=features, compression='gzip')
        group.create_dataset('labels', data=labels, compression='gzip')

        group.attrs.update({
            'batch_id': int(batch_id),
            'data_type': data_type,
            'set_name': set_name,
            'n_samples': int(len(features))
        })

        return group_path

    def _generate_positive_batches(self, pairs, batch_size, set_name, h5_file):
        batch_features = []
        batch_labels = []
        batch_id = 0
        total_samples = 0

        for head_id, tail_id in tqdm(pairs, desc=f"Generating {set_name} positive samples"):
            h_targets = self.drug_targets_cache.get(head_id, [])
            t_targets = self.drug_targets_cache.get(tail_id, [])

            if not h_targets or not t_targets:
                continue

            for h_target in h_targets:
                for t_target in t_targets:
                    feature = self._create_feature_vector(head_id, h_target, tail_id, t_target)
                    if feature is not None:
                        batch_features.append(feature)
                        batch_labels.append(1)

                        if len(batch_features) >= batch_size:
                            self._save_batch_to_h5(h5_file, set_name, 'positive', batch_id, batch_features,
                                                   batch_labels)
                            total_samples += len(batch_features)
                            print(f"Saved {set_name} positive batch {batch_id}: {len(batch_features)} samples")

                            batch_features = []
                            batch_labels = []
                            batch_id += 1

        if batch_features:
            self._save_batch_to_h5(h5_file, set_name, 'positive', batch_id, batch_features, batch_labels)
            total_samples += len(batch_features)
            print(f"Saved {set_name} positive batch {batch_id}: {len(batch_features)} samples")
            batch_id += 1

        return total_samples, batch_id

    def _generate_negative_batches(self, positive_set, target_samples, batch_size, set_name, h5_file, start_batch_id=0):
        batch_features = []
        batch_labels = []
        batch_id = start_batch_id
        generated_samples = 0

        pbar = tqdm(total=target_samples, desc=f"Generating {set_name} negative samples")

        while generated_samples < target_samples:
            h_id, t_id = random.sample(self.all_drug_ids, 2)

            if (h_id, t_id) in positive_set:
                continue

            h_targets = self.drug_targets_cache.get(h_id, [])
            t_targets = self.drug_targets_cache.get(t_id, [])

            if not h_targets or not t_targets:
                continue

            h_target = random.choice(h_targets)
            t_target = random.choice(t_targets)

            feature = self._create_feature_vector(h_id, h_target, t_id, t_target)
            if feature is not None:
                batch_features.append(feature)
                batch_labels.append(0)
                generated_samples += 1
                pbar.update(1)

                if len(batch_features) >= batch_size:
                    self._save_batch_to_h5(h5_file, set_name, 'negative', batch_id, batch_features, batch_labels)
                    print(f"Saved {set_name} negative batch {batch_id}: {len(batch_features)} samples")

                    batch_features = []
                    batch_labels = []
                    batch_id += 1

        pbar.close()

        if batch_features:
            self._save_batch_to_h5(h5_file, set_name, 'negative', batch_id, batch_features, batch_labels)
            print(f"Saved {set_name} negative batch {batch_id}: {len(batch_features)} samples")
            batch_id += 1

        return generated_samples, batch_id

    def generate_dataset(self, output_dir='./feature_batches', train_ratio=0.7, val_ratio=0.15,
                         positive_batch_size=1000, negative_batch_size=1000):
        print("Generating HDF5 dataset...")
        os.makedirs(output_dir, exist_ok=True)
        h5_path = os.path.join(output_dir, 'dataset.h5')

        print("Splitting drug pairs...")
        drug_pairs = self._get_drug_drug_pairs()
        train_pairs, temp_pairs = train_test_split(drug_pairs, test_size=1 - train_ratio, random_state=42)
        val_pairs, test_pairs = train_test_split(temp_pairs, test_size=val_ratio / (1 - train_ratio), random_state=42)

        print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)} pairs")

        with h5py.File(h5_path, 'w') as h5_file:
            h5_file.attrs.update({
                'dataset_name': 'Drug Combination Dataset',
                'creation_date': pd.Timestamp.now().isoformat(),
                'n_drugs': int(len(self.drug_id_to_feature)),
                'n_proteins': int(len(self.protein_id_to_feature))
            })

            for set_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
                print(f"\nProcessing {set_name} set...")
                positive_set = set(pairs)

                pos_samples, _ = self._generate_positive_batches(pairs, positive_batch_size, set_name, h5_file)

                self._generate_negative_batches(positive_set, pos_samples, negative_batch_size, set_name, h5_file)

        metadata = self._create_metadata(h5_path, output_dir)

        print(f"\nDataset generated: {h5_path}")
        return h5_path, metadata

    def _create_metadata(self, h5_path, output_dir):
        metadata = {
            'train': {'positive': [], 'negative': []},
            'val': {'positive': [], 'negative': []},
            'test': {'positive': [], 'negative': []}
        }

        with h5py.File(h5_path, 'r') as h5_file:
            for set_name in ['train', 'val', 'test']:
                for data_type in ['positive', 'negative']:
                    base_path = f'/{set_name}/{data_type}'
                    if base_path in h5_file:
                        batch_groups = sorted(list(h5_file[base_path].keys()))
                        for batch_name in batch_groups:
                            group_path = f'{base_path}/{batch_name}'
                            group = h5_file[group_path]

                            metadata[set_name][data_type].append({
                                'file': h5_path,
                                'group_path': group_path,
                                'samples': int(group.attrs['n_samples'])
                            })

        metadata_file = os.path.join(output_dir, 'dataset_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Metadata saved: {metadata_file}")
        return metadata


class H5DataLoader:
    def __init__(self, metadata_path, set_name='train', shuffle=True, seed=42):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.positive_batches = metadata[set_name]['positive']
        self.negative_batches = metadata[set_name]['negative']
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        if shuffle:
            self.rng.shuffle(self.positive_batches)
            self.rng.shuffle(self.negative_batches)

        self.pos_idx = 0
        self.neg_idx = 0
        self.h5_cache = {}

    def _get_h5_file(self, file_path):
        if file_path not in self.h5_cache:
            self.h5_cache[file_path] = h5py.File(file_path, 'r')
        return self.h5_cache[file_path]

    def get_next_batch(self, batch_type='balanced'):
        if batch_type == 'positive':
            if self.pos_idx >= len(self.positive_batches):
                self.pos_idx = 0
                if self.shuffle:
                    self.rng.shuffle(self.positive_batches)

            batch_info = self.positive_batches[self.pos_idx]
            h5_file = self._get_h5_file(batch_info['file'])
            group = h5_file[batch_info['group_path']]

            self.pos_idx += 1
            return group['features'][:], group['labels'][:]

        elif batch_type == 'negative':
            if self.neg_idx >= len(self.negative_batches):
                self.neg_idx = 0
                if self.shuffle:
                    self.rng.shuffle(self.negative_batches)

            batch_info = self.negative_batches[self.neg_idx]
            h5_file = self._get_h5_file(batch_info['file'])
            group = h5_file[batch_info['group_path']]

            self.neg_idx += 1
            return group['features'][:], group['labels'][:]

        else:
            if self.pos_idx >= len(self.positive_batches) or self.neg_idx >= len(self.negative_batches):
                self.pos_idx = 0
                self.neg_idx = 0
                if self.shuffle:
                    self.rng.shuffle(self.positive_batches)
                    self.rng.shuffle(self.negative_batches)

            pos_info = self.positive_batches[self.pos_idx]
            h5_file = self._get_h5_file(pos_info['file'])
            pos_group = h5_file[pos_info['group_path']]
            pos_features = pos_group['features'][:]
            pos_labels = pos_group['labels'][:]

            neg_info = self.negative_batches[self.neg_idx]
            neg_group = h5_file[neg_info['group_path']]
            neg_features = neg_group['features'][:]
            neg_labels = neg_group['labels'][:]

            features = np.concatenate([pos_features, neg_features], axis=0)
            labels = np.concatenate([pos_labels, neg_labels], axis=0)

            indices = self.rng.permutation(len(features))

            self.pos_idx += 1
            self.neg_idx += 1

            return features[indices], labels[indices]

    def get_total_samples(self):
        pos_total = sum(item['samples'] for item in self.positive_batches)
        neg_total = sum(item['samples'] for item in self.negative_batches)
        return pos_total, neg_total, pos_total + neg_total

    def close(self):
        for h5_file in self.h5_cache.values():
            h5_file.close()
        self.h5_cache.clear()


def main():
    random.seed(42)
    np.random.seed(42)

    generator = DrugDatasetGenerator(
        embedding_path='./embeddings',
        smiles_path='./smiles',
        target_path='./targets'
    )

    h5_path, metadata = generator.generate_dataset(
        output_dir='./feature_batches',
        train_ratio=0.7,
        val_ratio=0.15,
        positive_batch_size=1000,
        negative_batch_size=1000
    )

    print("\n" + "=" * 50)
    print("Dataset Generation Complete")
    print("=" * 50)

    total_pos = total_neg = 0
    for set_name in ['train', 'val', 'test']:
        pos = sum(item['samples'] for item in metadata[set_name]['positive'])
        neg = sum(item['samples'] for item in metadata[set_name]['negative'])
        total_pos += pos
        total_neg += neg
        print(f"{set_name}: {pos:,} positive, {neg:,} negative, {pos + neg:,} total")

    print(f"\nTotal: {total_pos:,} positive, {total_neg:,} negative, {total_pos + total_neg:,} samples")
    print(f"HDF5 file: {h5_path}")


if __name__ == "__main__":
    main()
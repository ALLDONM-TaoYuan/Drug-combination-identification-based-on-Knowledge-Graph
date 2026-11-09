import random
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

warnings.filterwarnings('ignore')

TARGET_BIN_EDGES = np.asarray([1, 2, 3, 6, 11])


class DrugDatasetGenerator:
    def __init__(self, embedding_path, smiles_path, target_path):
        self.mapped_triples = np.load(Path(embedding_path) / 'mapped_triples.npy')
        entity_mapping = pd.read_csv(Path(embedding_path) / 'entity_mapping.csv')
        self.id_to_entity = dict(zip(entity_mapping['id'].astype(int),
                                    entity_mapping['entity'].astype(str)))

        smiles_features = np.load(Path(smiles_path) / 'smiles_features.npy')
        smiles_ids = np.load(Path(smiles_path) / 'smiles_ids.npy')
        protein_features = np.load(Path(target_path) / 'target_features.npy')
        protein_ids = np.load(Path(target_path) / 'target_ids.npy')

        self.drug_id_to_feature = {int(drug_id): feature
                                   for drug_id, feature in zip(smiles_ids, smiles_features)}
        self.protein_id_to_feature = {int(protein_id): feature
                                      for protein_id, feature in zip(protein_ids, protein_features)}

        self.drug_targets_cache = {}
        self.all_drug_ids = list(self.drug_id_to_feature.keys())
        self._precompute_drug_targets()

    def _precompute_drug_targets(self):
        for drug_id in self.all_drug_ids:
            self.drug_targets_cache[drug_id] = self._get_drug_targets(drug_id)

    def _get_drug_targets(self, drug_id):
        targets = []
        for triple in self.mapped_triples:
            h_id, rel_id, t_id = triple
            if int(rel_id) == 1 and int(h_id) == int(drug_id):
                if int(t_id) in self.protein_id_to_feature:
                    targets.append(int(t_id))
        return targets

    def _get_drug_drug_pairs(self):
        pairs = set()
        for triple in self.mapped_triples:
            if int(triple[1]) == 0:
                head_id, tail_id = int(triple[0]), int(triple[2])
                pairs.add((head_id, tail_id) if head_id <= tail_id else (tail_id, head_id))
        return list(pairs)

    def _get_valid_pairs(self):
        valid_pairs = []
        for head_id, tail_id in self._get_drug_drug_pairs():
            if head_id not in self.drug_id_to_feature or tail_id not in self.drug_id_to_feature:
                continue
            if not self.drug_targets_cache.get(head_id, []) or not self.drug_targets_cache.get(tail_id, []):
                continue
            valid_pairs.append((head_id, tail_id))
        return valid_pairs

    def _build_train_kg(self, train_pairs, kg_out_dir):
        import torch
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory

        kg_out_dir = Path(kg_out_dir)
        kg_out_dir.mkdir(parents=True, exist_ok=True)

        ddi_triples = []
        for head_id, tail_id in train_pairs:
            h_entity = self.id_to_entity.get(head_id)
            t_entity = self.id_to_entity.get(tail_id)
            if h_entity is not None and t_entity is not None:
                ddi_triples.append((h_entity, 'combination', t_entity))

        dti_triples = []
        for h_id, rel_id, t_id in self.mapped_triples:
            if int(rel_id) != 1:
                continue
            h_entity = self.id_to_entity.get(int(h_id))
            t_entity = self.id_to_entity.get(int(t_id))
            if h_entity is not None and t_entity is not None:
                dti_triples.append((h_entity, 'interaction', t_entity))

        triples_df = pd.DataFrame(ddi_triples + dti_triples,
                                  columns=['head', 'relation', 'tail']).drop_duplicates()

        kg_ddi_set = set()
        for _, row in triples_df[triples_df['relation'] == 'combination'].iterrows():
            e1, e2 = str(row['head']), str(row['tail'])
            kg_ddi_set.add((e1, e2) if e1 <= e2 else (e2, e1))
        for name, pairs in [('val', self.val_pairs), ('test', self.test_pairs)]:
            leaked = 0
            for head_id, tail_id in pairs:
                e1, e2 = self.id_to_entity.get(head_id), self.id_to_entity.get(tail_id)
                if e1 is None or e2 is None:
                    continue
                pair = (e1, e2) if e1 <= e2 else (e2, e1)
                if pair in kg_ddi_set:
                    leaked += 1
            assert leaked == 0, f'FAIL: {leaked} {name} pairs leaked into KG!'

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tf = TriplesFactory.from_labeled_triples(triples=triples_df.values,
                                                 create_inverse_triples=False)
        test_tf = TriplesFactory(
            mapped_triples=tf.mapped_triples[:100],
            entity_to_id=tf.entity_to_id,
            relation_to_id=tf.relation_to_id,
        )
        results = pipeline(
            training=tf,
            testing=test_tf,
            model='TransE',
            model_kwargs={'embedding_dim': 128},
            training_loop='SLCWA',
            training_kwargs={'num_epochs': 50, 'batch_size': 256},
            loss='SoftplusLoss',
            loss_kwargs={'reduction': 'mean'},
            optimizer='Adam',
            optimizer_kwargs={'lr': 0.001},
            negative_sampler_kwargs={'num_negs_per_pos': 10},
            random_seed=42,
            evaluator='RankBasedEvaluator',
            evaluator_kwargs={'filtered': True},
            device=device,
        )

        model = results.model
        model.cpu()
        with torch.no_grad():
            entity_emb = model.entity_representations[0](indices=None).detach().cpu().numpy()

        pd.DataFrame([(i, label) for i, label in tf.entity_id_to_label.items()],
                     columns=['id', 'entity']).to_csv(kg_out_dir / 'entity_mapping.csv', index=False)
        pd.DataFrame([(i, label) for i, label in tf.relation_id_to_label.items()],
                     columns=['id', 'relation']).to_csv(kg_out_dir / 'relation_mapping.csv', index=False)
        np.save(kg_out_dir / 'entity_embeddings.npy', entity_emb)
        np.save(kg_out_dir / 'mapped_triples.npy', tf.mapped_triples.numpy())

        return tf.entity_to_id, entity_emb

    def _build_drug_features(self, kg_entity_to_id, entity_emb, kg_out_dir):
        kg_out_dir = Path(kg_out_dir)
        fan_avg = (1 + 128) / 2
        limit = np.sqrt(6 / (2 * fan_avg))
        rng = np.random.RandomState(42)

        self.drug_ids = sorted(self.drug_id_to_feature.keys())
        self.drug_to_idx = {did: i for i, did in enumerate(self.drug_ids)}

        kg_features = np.zeros((len(self.drug_ids), 128), dtype=np.float32)
        for did in self.drug_ids:
            entity = self.id_to_entity.get(did)
            kg_id = kg_entity_to_id.get(entity) if entity is not None else None
            if kg_id is not None:
                kg_features[self.drug_to_idx[did]] = entity_emb[kg_id]
            else:
                kg_features[self.drug_to_idx[did]] = rng.uniform(-limit, limit, 128).astype(np.float32)

        smiles_features = np.stack([self.drug_id_to_feature[did]
                                    for did in self.drug_ids]).astype(np.float32)

        target_ids = sorted(self.protein_id_to_feature.keys())
        self.target_to_idx = {tid: i for i, tid in enumerate(target_ids)}
        target_features = np.zeros((len(target_ids) + 1, 1024), dtype=np.float32)
        for tid in target_ids:
            target_features[self.target_to_idx[tid]] = self.protein_id_to_feature[tid]
        self.zero_target_idx = len(target_ids)

        self.drug_target_indices = {}
        for did in self.drug_ids:
            indices = [self.target_to_idx[tid] for tid in self.drug_targets_cache.get(did, [])
                       if tid in self.target_to_idx]
            self.drug_target_indices[did] = indices if indices else [self.zero_target_idx]

        np.save(kg_out_dir / 'drug_ids.npy', np.array(self.drug_ids))
        np.save(kg_out_dir / 'drug_kg_features.npy', kg_features)
        np.save(kg_out_dir / 'drug_smiles_features.npy', smiles_features)
        np.save(kg_out_dir / 'target_features.npy', target_features)

        self.drug_kg_features = kg_features
        self.drug_smiles_features = smiles_features
        self.target_features = target_features

        return kg_features, smiles_features, target_features

    def _target_stratum(self, count1, count2):
        b1 = int(np.digitize(count1, TARGET_BIN_EDGES))
        b2 = int(np.digitize(count2, TARGET_BIN_EDGES))
        return (b1, b2) if b1 <= b2 else (b2, b1)

    def _generate_negative_pairs(self, positive_pairs, all_positive_set, used_negatives, rng):
        drugs = np.asarray([d for d in self.all_drug_ids if self.drug_targets_cache.get(d, [])],
                           dtype=np.int64)
        counts = {int(d): len(self.drug_targets_cache[int(d)]) for d in drugs}
        n_drugs = int(max(self.drug_ids)) + 1

        left = np.repeat(drugs, drugs.size)
        right = np.tile(drugs, drugs.size)
        keep = left < right
        left, right = left[keep], right[keep]
        enc = left * n_drugs + right

        blocked = [min(a, b) * n_drugs + max(a, b) for (a, b) in all_positive_set]
        blocked += [min(a, b) * n_drugs + max(a, b) for (a, b) in used_negatives]
        mask = ~np.isin(enc, np.asarray(blocked, dtype=np.int64))
        left, right, enc = left[mask], right[mask], enc[mask]

        bin_l = np.digitize(np.asarray([counts[int(d)] for d in left]), TARGET_BIN_EDGES)
        bin_r = np.digitize(np.asarray([counts[int(d)] for d in right]), TARGET_BIN_EDGES)
        key_l = np.minimum(bin_l, bin_r)
        key_r = np.maximum(bin_l, bin_r)

        quota = Counter(self._target_stratum(counts.get(int(a), 0), counts.get(int(b), 0))
                        for a, b in positive_pairs)

        negatives = set()
        used = np.zeros(len(left), dtype=bool)
        deficit_total = 0
        for key, need in sorted(quota.items()):
            index = np.flatnonzero((key_l == key[0]) & (key_r == key[1]) & (~used))
            take = int(min(need, index.size))
            if take:
                chosen = rng.choice(index, size=take, replace=False)
                used[chosen] = True
                for i in chosen:
                    negatives.add((int(left[i]), int(right[i])))
            deficit_total += int(need - take)

        if deficit_total > 0:
            rest = np.flatnonzero(~used)
            extra = int(min(deficit_total, rest.size))
            if extra:
                chosen = rng.choice(rest, size=extra, replace=False)
                for i in chosen:
                    negatives.add((int(left[i]), int(right[i])))
        used_negatives.update(negatives)

        return list(negatives)

    def _generate_combo_index(self, pairs, labels, pair_offset):
        rows = []
        for pair_id, ((head_id, tail_id), label) in enumerate(zip(pairs, labels)):
            d1_idx = self.drug_to_idx[head_id]
            d2_idx = self.drug_to_idx[tail_id]
            t1_list = self.drug_target_indices[head_id]
            t2_list = self.drug_target_indices[tail_id]
            n1, n2 = len(t1_list), len(t2_list)
            weight = 1.0 / (n1 * n2)
            for t1_idx in t1_list:
                for t2_idx in t2_list:
                    rows.append((pair_id + pair_offset, d1_idx, d2_idx,
                                 t1_idx, t2_idx, label, weight))

        dtype = np.dtype([
            ('pair_index', np.int32),
            ('drug1_idx', np.int32),
            ('drug2_idx', np.int32),
            ('target1_idx', np.int32),
            ('target2_idx', np.int32),
            ('label', np.int8),
            ('sample_weight', np.float32),
        ])
        return np.array([tuple(r) for r in rows], dtype=dtype)

    def generate_dataset(self, output_dir, train_ratio=0.7, val_ratio=0.15):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        kg_out_dir = output_dir / 'kg'

        valid_pairs = self._get_valid_pairs()
        train_pairs, temp_pairs = train_test_split(valid_pairs, test_size=1 - train_ratio,
                                                   random_state=42)
        val_pairs, test_pairs = train_test_split(temp_pairs,
                                                 test_size=val_ratio / (1 - train_ratio),
                                                 random_state=42)
        self.val_pairs, self.test_pairs = val_pairs, test_pairs

        kg_emb_path = kg_out_dir / 'entity_embeddings.npy'
        if kg_emb_path.exists():
            entity_map_df = pd.read_csv(kg_out_dir / 'entity_mapping.csv')
            kg_entity_to_id = dict(zip(entity_map_df['entity'].astype(str),
                                       entity_map_df['id'].astype(int)))
            entity_emb = np.load(kg_emb_path)
        else:
            kg_entity_to_id, entity_emb = self._build_train_kg(train_pairs, kg_out_dir)

        self._build_drug_features(kg_entity_to_id, entity_emb, kg_out_dir)

        all_positive_set = set()
        for head_id, tail_id in valid_pairs:
            all_positive_set.add((head_id, tail_id) if head_id <= tail_id else (tail_id, head_id))

        rng = np.random.RandomState(42)
        used_negatives = set()
        split_data = {}
        for set_name, pos_pairs in [('train', train_pairs), ('val', val_pairs),
                                    ('test', test_pairs)]:
            neg_pairs = self._generate_negative_pairs(pos_pairs, all_positive_set,
                                                      used_negatives, rng)
            split_data[set_name] = (pos_pairs, neg_pairs)

        pair_offset = 0
        for set_name in ['train', 'val', 'test']:
            pos_pairs, neg_pairs = split_data[set_name]
            pairs = pos_pairs + neg_pairs
            labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
            combos = self._generate_combo_index(pairs, labels, pair_offset)
            pair_offset += len(pairs)
            np.savez_compressed(output_dir / f'{set_name}_combo_index.npz', combos=combos)

        return output_dir


class ComboDataLoader:
    def __init__(self, data_dir, set_name='train'):
        data_dir = Path(data_dir)
        self.set_name = set_name
        self.combos = np.load(data_dir / f'{set_name}_combo_index.npz')['combos']
        self.drug_kg = np.load(data_dir / 'kg' / 'drug_kg_features.npy')
        self.drug_smiles = np.load(data_dir / 'kg' / 'drug_smiles_features.npy')
        self.target_features = np.load(data_dir / 'kg' / 'target_features.npy')

    def build_batch(self, batch_arr):
        d1 = batch_arr['drug1_idx'].astype(np.int32)
        d2 = batch_arr['drug2_idx'].astype(np.int32)
        t1 = batch_arr['target1_idx'].astype(np.int32)
        t2 = batch_arr['target2_idx'].astype(np.int32)
        return np.concatenate([
            self.drug_kg[d1], self.drug_smiles[d1], self.target_features[t1],
            self.drug_kg[d2], self.drug_smiles[d2], self.target_features[t2],
        ], axis=1).astype(np.float32)

    def get_total_samples(self):
        pos_total = int(np.sum(self.combos['label'] == 1))
        neg_total = int(np.sum(self.combos['label'] == 0))
        return pos_total, neg_total, pos_total + neg_total

    def close(self):
        pass


def main():
    random.seed(42)
    np.random.seed(42)

    generator = DrugDatasetGenerator(
        embedding_path=config.EMBEDDINGS_DIR,
        smiles_path=config.SMILES_DIR,
        target_path=config.TARGETS_DIR,
    )
    generator.generate_dataset(output_dir=config.FEATURE_BATCH_DIR,
                               train_ratio=0.7, val_ratio=0.15)


if __name__ == '__main__':
    main()

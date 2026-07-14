import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
import warnings
import time

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


def load_syn_drug_morgan():
    drug_morgan = np.load('./syn_morgan_features/rdkit_features.npy')
    drug_ids = np.load('./syn_morgan_features/rdkit_ids.npy')
    return drug_morgan, drug_ids


def load_syn_drug_combinations_from_kg():
    entity_mapping = pd.read_csv('../database/Embeddings/entity_mapping.csv')
    mapped_triples = np.load('../database/Embeddings/mapped_triples.npy')
    drug_df = pd.read_csv('./Syn_id_smiles.csv')

    entity_id_to_mapping = {row['id']: row['entity'] for _, row in entity_mapping.iterrows()}
    drug_name_to_id = {row['drug_name']: row['id'] for _, row in drug_df.iterrows()}

    syn_drug_combinations = []
    unique_pairs = set()

    for triple in tqdm(mapped_triples, desc="Loading SynDrugMD combinations"):
        head_id, rel_id, tail_id = triple
        if rel_id == 0:
            head_entity = entity_id_to_mapping.get(head_id)
            tail_entity = entity_id_to_mapping.get(tail_id)

            if head_entity and tail_entity:
                drug_id_head = drug_name_to_id.get(head_entity)
                drug_id_tail = drug_name_to_id.get(tail_entity)

                if drug_id_head and drug_id_tail and drug_id_head != drug_id_tail:
                    pair_key = tuple(sorted((drug_id_head, drug_id_tail)))
                    if pair_key not in unique_pairs:
                        unique_pairs.add(pair_key)
                        syn_drug_combinations.append((drug_id_head, drug_id_tail))

    return syn_drug_combinations


def load_db_drug_morgan():
    drug_morgan = np.load('./db_morgan_features/rdkit_features.npy')
    drug_ids = np.load('./db_morgan_features/rdkit_ids.npy')
    return drug_morgan, drug_ids


def load_db_drug_combinations_from_kg():
    entity_mapping = pd.read_csv('../../DrugCombDB KG/Predict/aug_KG/augmented_emb/entity_mapping.csv')
    mapped_triples = np.load('../../DrugCombDB KG/Predict/aug_KG/augmented_emb/mapped_triples.npy')
    drug_df = pd.read_csv('./DB_id_smiles.csv')

    entity_id_to_mapping = {row['id']: row['entity'] for _, row in entity_mapping.iterrows()}
    drug_name_to_id = {row['drug_name']: row['id'] for _, row in drug_df.iterrows()}

    db_drug_combinations = []
    unique_pairs = set()

    for triple in tqdm(mapped_triples, desc="Loading DrugCombDB combinations"):
        head_id, rel_id, tail_id = triple
        if rel_id == 0:
            head_entity = entity_id_to_mapping.get(head_id)
            tail_entity = entity_id_to_mapping.get(tail_id)

            if head_entity and tail_entity:
                drug_id_head = drug_name_to_id.get(head_entity)
                drug_id_tail = drug_name_to_id.get(tail_entity)

                if drug_id_head and drug_id_tail and drug_id_head != drug_id_tail:
                    pair_key = tuple(sorted((drug_id_head, drug_id_tail)))
                    if pair_key not in unique_pairs:
                        unique_pairs.add(pair_key)
                        db_drug_combinations.append((drug_id_head, drug_id_tail))

    return db_drug_combinations


def compute_drug_similarity(drug_morgan, drug_ids):
    start_time = time.time()
    drug_similarity_matrix = cosine_similarity(drug_morgan)
    np.fill_diagonal(drug_similarity_matrix, 1.0)
    drug_id_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    return drug_similarity_matrix, drug_id_to_idx


def generate_drug_combinations_from_kg(drug_combinations_list, drug_ids, dataset_name=""):
    drug_id_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    combinations = []

    for head_drug_id, tail_drug_id in tqdm(drug_combinations_list, desc=f"Generating {dataset_name} combinations"):
        if head_drug_id in drug_id_to_idx and tail_drug_id in drug_id_to_idx:
            combinations.append({
                'combo_id': len(combinations),
                'head_drug_idx': drug_id_to_idx[head_drug_id],
                'tail_drug_idx': drug_id_to_idx[tail_drug_id],
                'head_drug_id': head_drug_id,
                'tail_drug_id': tail_drug_id
            })

    return pd.DataFrame(combinations)


def compute_drug_combo_features(combo_df, drug_features):
    n_combinations = len(combo_df)
    n_features = drug_features.shape[1]
    combo_features = np.zeros((n_combinations, 2 * n_features), dtype=np.float32)

    for idx, row in tqdm(combo_df.iterrows(), total=n_combinations, desc="Computing features"):
        combo_features[idx] = np.concatenate([
            drug_features[int(row['head_drug_idx'])].astype(np.float32),
            drug_features[int(row['tail_drug_idx'])].astype(np.float32)
        ])

    return combo_features


def compute_all_combo_similarity_matrix(combo_features, chunk_size=2000):
    n_combinations = combo_features.shape[0]
    similarity_matrix = np.zeros((n_combinations, n_combinations), dtype=np.float32)
    n_chunks = int(np.ceil(n_combinations / chunk_size))

    for i in tqdm(range(n_chunks), desc="Computing full similarity matrix"):
        start_i = i * chunk_size
        end_i = min((i + 1) * chunk_size, n_combinations)

        for j in range(i, n_chunks):
            start_j = j * chunk_size
            end_j = min((j + 1) * chunk_size, n_combinations)

            chunk_similarity = cosine_similarity(
                combo_features[start_i:end_i],
                combo_features[start_j:end_j]
            )

            similarity_matrix[start_i:end_i, start_j:end_j] = chunk_similarity
            if i != j:
                similarity_matrix[start_j:end_j, start_i:end_i] = chunk_similarity.T

    np.fill_diagonal(similarity_matrix, 1.0)
    return similarity_matrix


def get_similarity_statistics(similarity_matrix):
    sim_flat = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    return {
        'mean': float(sim_flat.mean()),
        'std': float(sim_flat.std()),
        'min': float(sim_flat.min()),
        'max': float(sim_flat.max())
    }


def create_complete_analysis_figure(syn_drug_sim_matrix, syn_combo_sim_matrix,
                                    db_drug_sim_matrix, db_combo_sim_matrix):
    plt.rcParams.update({'font.size': 28})
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    n_drugs = syn_drug_sim_matrix.shape[0]
    drug_ticks = [0, 800, 1600, 2400, 3200]
    drug_ticks = [t for t in drug_ticks if t < n_drugs]

    im1 = axes[0, 0].imshow(syn_drug_sim_matrix, cmap='summer', aspect='auto', vmin=0, vmax=1)
    axes[0, 0].set_xlabel('Drugs', fontsize=28)
    axes[0, 0].set_ylabel('Drugs', fontsize=28)
    axes[0, 0].set_xticks(drug_ticks)
    axes[0, 0].set_xticklabels(drug_ticks, fontsize=24)
    axes[0, 0].set_yticks(drug_ticks)
    axes[0, 0].set_yticklabels(drug_ticks, fontsize=24)
    axes[0, 0].grid(False)

    n_combos = syn_combo_sim_matrix.shape[0]
    combo_ticks = [0, 10000, 20000, 30000, 40000, 50000]
    combo_ticks = [t for t in combo_ticks if t < n_combos]

    axes[0, 1].imshow(syn_combo_sim_matrix, cmap='summer', aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_xlabel('Drug combinations', fontsize=28)
    axes[0, 1].set_ylabel('Drug combinations', fontsize=28)
    axes[0, 1].set_xticks(combo_ticks)
    axes[0, 1].set_xticklabels(combo_ticks, fontsize=24)
    axes[0, 1].set_yticks(combo_ticks)
    axes[0, 1].set_yticklabels(combo_ticks, fontsize=24)
    axes[0, 1].grid(False)

    bin_edges = np.linspace(0, 1, 11)
    bin_labels = [f'{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}' for i in range(10)]

    drug_sim_flat = syn_drug_sim_matrix[np.triu_indices_from(syn_drug_sim_matrix, k=1)]
    drug_counts, _ = np.histogram(drug_sim_flat, bins=bin_edges)
    drug_percent = drug_counts / len(drug_sim_flat)

    combo_sim_flat = syn_combo_sim_matrix[np.triu_indices_from(syn_combo_sim_matrix, k=1)]
    combo_counts, _ = np.histogram(combo_sim_flat, bins=bin_edges)
    combo_total = len(combo_sim_flat)
    combo_percent = combo_counts / combo_total

    x = np.arange(len(bin_labels))
    width = 0.4

    axes[0, 2].bar(x - width / 2, drug_percent, width, label='Drug', color='yellow', alpha=0.7, edgecolor='black')
    axes[0, 2].bar(x + width / 2, combo_percent, width, label='Drug Combination', color='blue', alpha=0.7,
                   edgecolor='black')
    axes[0, 2].set_xlabel('Similarity Range', fontsize=28)
    axes[0, 2].set_ylabel('Percentage', fontsize=28)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=22)
    axes[0, 2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0, 2].set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=22)
    axes[0, 2].grid(False)
    axes[0, 2].legend(fontsize=20)

    n_drugs = db_drug_sim_matrix.shape[0]
    drug_ticks = [0, 150, 300, 450, 600, 750]
    drug_ticks = [t for t in drug_ticks if t < n_drugs]

    axes[1, 0].imshow(db_drug_sim_matrix, cmap='summer', aspect='auto', vmin=0, vmax=1)
    axes[1, 0].set_xlabel('Drugs', fontsize=28)
    axes[1, 0].set_ylabel('Drugs', fontsize=28)
    axes[1, 0].set_xticks(drug_ticks)
    axes[1, 0].set_xticklabels(drug_ticks, fontsize=24)
    axes[1, 0].set_yticks(drug_ticks)
    axes[1, 0].set_yticklabels(drug_ticks, fontsize=24)
    axes[1, 0].grid(False)

    n_combos = db_combo_sim_matrix.shape[0]
    combo_ticks = [0, 500, 1000, 1500, 2000, 2500]
    combo_ticks = [t for t in combo_ticks if t < n_combos]

    axes[1, 1].imshow(db_combo_sim_matrix, cmap='summer', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_xlabel('Drug combinations', fontsize=28)
    axes[1, 1].set_ylabel('Drug combinations', fontsize=28)
    axes[1, 1].set_xticks(combo_ticks)
    axes[1, 1].set_xticklabels(combo_ticks, fontsize=24)
    axes[1, 1].set_yticks(combo_ticks)
    axes[1, 1].set_yticklabels(combo_ticks, fontsize=24)
    axes[1, 1].grid(False)

    drug_sim_flat = db_drug_sim_matrix[np.triu_indices_from(db_drug_sim_matrix, k=1)]
    drug_counts, _ = np.histogram(drug_sim_flat, bins=bin_edges)
    drug_percent = drug_counts / len(drug_sim_flat)

    combo_sim_flat = db_combo_sim_matrix[np.triu_indices_from(db_combo_sim_matrix, k=1)]
    combo_counts, _ = np.histogram(combo_sim_flat, bins=bin_edges)
    combo_total = len(combo_sim_flat)
    combo_percent = combo_counts / combo_total

    x = np.arange(len(bin_labels))
    width = 0.4

    axes[1, 2].bar(x - width / 2, drug_percent, width, label='Drug', color='yellow', alpha=0.7, edgecolor='black')
    axes[1, 2].bar(x + width / 2, combo_percent, width, label='Drug Combination', color='blue', alpha=0.7,
                   edgecolor='black')
    axes[1, 2].set_xlabel('Similarity Range', fontsize=28)
    axes[1, 2].set_ylabel('Percentage', fontsize=28)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=22)
    axes[1, 2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[1, 2].set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=22)
    axes[1, 2].grid(False)
    axes[1, 2].legend(fontsize=20)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    cbar.ax.tick_params(labelsize=20)

    plt.subplots_adjust(left=0.05, right=0.9, bottom=0.1, top=0.95, wspace=0.3, hspace=0.4)

    plt.savefig('complete_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results(syn_drug_sim_matrix, syn_drug_ids, syn_combo_sim_matrix, syn_combo_df,
                 db_drug_sim_matrix, db_drug_ids, db_combo_sim_matrix, db_combo_df,
                 syn_drug_stats, syn_combo_stats, db_drug_stats, db_combo_stats):
    output_dir = 'drug_combo_similarity_results'
    os.makedirs(output_dir, exist_ok=True)

    np.save(f'{output_dir}/syn_drug_similarity_matrix.npy', syn_drug_sim_matrix)
    np.save(f'{output_dir}/syn_drug_ids.npy', syn_drug_ids)
    np.save(f'{output_dir}/syn_combo_similarity_matrix.npy', syn_combo_sim_matrix)
    syn_combo_df.to_csv(f'{output_dir}/syn_drug_combinations.csv', index=False)

    np.save(f'{output_dir}/db_drug_similarity_matrix.npy', db_drug_sim_matrix)
    np.save(f'{output_dir}/db_drug_ids.npy', db_drug_ids)
    np.save(f'{output_dir}/db_combo_similarity_matrix.npy', db_combo_sim_matrix)
    db_combo_df.to_csv(f'{output_dir}/db_drug_combinations.csv', index=False)

    stats = {
        'syn_drug_similarity_mean': syn_drug_stats['mean'],
        'syn_drug_similarity_std': syn_drug_stats['std'],
        'syn_drug_similarity_min': syn_drug_stats['min'],
        'syn_drug_similarity_max': syn_drug_stats['max'],

        'syn_combo_similarity_mean': syn_combo_stats['mean'],
        'syn_combo_similarity_std': syn_combo_stats['std'],
        'syn_combo_similarity_min': syn_combo_stats['min'],
        'syn_combo_similarity_max': syn_combo_stats['max'],

        'db_drug_similarity_mean': db_drug_stats['mean'],
        'db_drug_similarity_std': db_drug_stats['std'],
        'db_drug_similarity_min': db_drug_stats['min'],
        'db_drug_similarity_max': db_drug_stats['max'],

        'db_combo_similarity_mean': db_combo_stats['mean'],
        'db_combo_similarity_std': db_combo_stats['std'],
        'db_combo_similarity_min': db_combo_stats['min'],
        'db_combo_similarity_max': db_combo_stats['max']
    }

    pd.DataFrame([stats]).to_csv(f'{output_dir}/similarity_statistics.csv', index=False)


def main():
    start_time = time.time()

    print("1. Loading SynDrugMD data...")
    syn_drug_morgan, syn_drug_ids = load_syn_drug_morgan()
    print("\n2. Calculating SynDrugMD drug similarity...")
    syn_drug_sim_matrix, syn_drug_id_to_idx = compute_drug_similarity(syn_drug_morgan, syn_drug_ids)
    syn_drug_stats = get_similarity_statistics(syn_drug_sim_matrix)

    print("\n3. Loading SynDrugMD drug combinations...")
    syn_drug_combinations_list = load_syn_drug_combinations_from_kg()
    print("\n4. Generating SynDrugMD drug combination DataFrame...")
    syn_combo_df = generate_drug_combinations_from_kg(syn_drug_combinations_list, syn_drug_ids, "SynDrugMD")

    print("\n5. Computing SynDrugMD drug combination features...")
    syn_combo_features = compute_drug_combo_features(syn_combo_df, syn_drug_morgan)

    print("\n6. Computing SynDrugMD full drug combination similarity matrix...")
    syn_combo_sim_matrix = compute_all_combo_similarity_matrix(syn_combo_features, chunk_size=2000)
    syn_combo_stats = get_similarity_statistics(syn_combo_sim_matrix)

    print("\n7. Loading DrugCombDB data...")
    db_drug_morgan, db_drug_ids = load_db_drug_morgan()
    print("\n8. Calculating DrugCombDB drug similarity...")
    db_drug_sim_matrix, db_drug_id_to_idx = compute_drug_similarity(db_drug_morgan, db_drug_ids)
    db_drug_stats = get_similarity_statistics(db_drug_sim_matrix)

    print("\n9. Loading DrugCombDB drug combinations...")
    db_drug_combinations_list = load_db_drug_combinations_from_kg()
    print("\n10. Generating DrugCombDB drug combination DataFrame...")
    db_combo_df = generate_drug_combinations_from_kg(db_drug_combinations_list, db_drug_ids, "DrugCombDB")

    print("\n11. Computing DrugCombDB drug combination features...")
    db_combo_features = compute_drug_combo_features(db_combo_df, db_drug_morgan)

    print("\n12. Computing DrugCombDB full drug combination similarity matrix...")
    db_combo_sim_matrix = compute_all_combo_similarity_matrix(db_combo_features, chunk_size=2000)
    db_combo_stats = get_similarity_statistics(db_combo_sim_matrix)

    print("\n13. Creating analysis figure...")
    create_complete_analysis_figure(syn_drug_sim_matrix, syn_combo_sim_matrix,
                                    db_drug_sim_matrix, db_combo_sim_matrix)

    print("\n14. Saving results...")
    save_results(syn_drug_sim_matrix, syn_drug_ids, syn_combo_sim_matrix, syn_combo_df,
                 db_drug_sim_matrix, db_drug_ids, db_combo_sim_matrix, db_combo_df,
                 syn_drug_stats, syn_combo_stats, db_drug_stats, db_combo_stats)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    main()

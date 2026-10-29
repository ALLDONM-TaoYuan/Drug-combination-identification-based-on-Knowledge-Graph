import pandas as pd
from tqdm import tqdm


def match_and_save_smiles(
        entity_mapping='../embeddings/entity_mapping.csv',
        drug_smiles='./CID_Syn.csv',
        output_file='id_smiles.csv'):
    print(f'Loading entity mapping file: {entity_mapping}')
    entity_df = pd.read_csv(entity_mapping)

    if 'entity' not in entity_df.columns:
        print("Error: Missing 'entity' column in entity_mapping.csv")
        return 0, 0

    entity_df['entity'] = entity_df['entity'].astype(str)
    entity_df['entity_std'] = entity_df['entity'].str.strip().str.lower()

    print(f'Loaded {len(entity_df)} entity mapping records')
    entity_to_id = dict(zip(entity_df['entity_std'], entity_df['id']))
    print(f'Created standardized entity-to-ID mapping with {len(entity_to_id)} entries')

    print(f'\nLoading SMILES file: {drug_smiles}')
    drug_df = pd.read_csv(drug_smiles)

    if 'drug_CID' not in drug_df.columns:
        print("Error: Missing 'drug_CID' column in CID_Syn.csv")
        return 0, 0

    if 'smiles' not in drug_df.columns:
        print("Error: Missing 'smiles' column in CID_Syn.csv")
        return 0, 0

    print(f'Loaded {len(drug_df)} drug SMILES records')

    print(f'drug_CID column data type: {drug_df["drug_CID"].dtype}')

    drug_df['drug_CID'] = drug_df['drug_CID'].astype(str)
    drug_df['drug_CID_std'] = drug_df['drug_CID'].str.strip().str.lower()

    original_count = len(drug_df)
    drug_df = drug_df.drop_duplicates(subset=['drug_CID_std'])
    print(f'After deduplication: {len(drug_df)} unique drug records retained ({original_count - len(drug_df)} duplicates removed)')

    matched_data = []
    missed_data = []
    matched_count = 0

    print('\nStarting to match CID with entity IDs...')

    for _, row in tqdm(drug_df.iterrows(), total=len(drug_df), desc="Matching progress"):
        drug_CID = row['drug_CID']
        drug_CID_std = row['drug_CID_std']
        smiles = row['smiles']

        if drug_CID_std in entity_to_id:
            entity_id = entity_to_id[drug_CID_std]
            matched_data.append({
                'id': entity_id,
                'smiles': smiles,
                'drug_CID': drug_CID
            })
            matched_count += 1
        else:
            missed_data.append({
                'drug_CID': drug_CID,
                'smiles': smiles
            })

    if matched_data:
        matched_df = pd.DataFrame(matched_data)
        matched_df.to_csv(output_file, index=False)
        print(f'\nSuccessfully matched {matched_count} records, saved to: {output_file}')

        print('\nMatching results example (first 5):')
        print(matched_df.head())
    else:
        print('\nNo matching records to save')

    if missed_data:
        missed_df = pd.DataFrame(missed_data)
        missed_file = 'missed_drugs.csv'
        missed_df.to_csv(missed_file, index=False)
        print(f'{len(missed_df)} records unmatched, saved to: {missed_file}')

        print('\nUnmatched records example (first 5):')
        print(missed_df.head())

    print('\n' + '=' * 50)
    print('Matching statistics:')
    print('=' * 50)
    print(f'Total drug records: {len(drug_df)}')
    print(f'Successfully matched: {matched_count} ({matched_count / len(drug_df) * 100:.2f}%)')
    print(f'Unmatched: {len(missed_data)} ({len(missed_data) / len(drug_df) * 100:.2f}%)')

    return matched_count, len(missed_data)


def check_file_formats(entity_mapping, drug_smiles):
    print("Checking file format...")

    try:
        entity_df = pd.read_csv(entity_mapping, nrows=5)
        print(f"\nEntity mapping file first 5 rows:")
        print(entity_df)
        print(f"\nEntity mapping file columns: {list(entity_df.columns)}")
    except Exception as e:
        print(f"Failed to read entity mapping file: {e}")

    try:
        drug_df = pd.read_csv(drug_smiles, nrows=5)
        print(f"\nDrug SMILES file first 5 rows:")
        print(drug_df)
        print(f"\nDrug SMILES file columns: {list(drug_df.columns)}")
        print(f"\nDrug SMILES file data types:")
        print(drug_df.dtypes)
    except Exception as e:
        print(f"Failed to read drug SMILES file: {e}")


if __name__ == '__main__':
    csv_entity = '../embeddings/entity_mapping.csv'
    csv_drug = './CID_Syn.csv'
    output = 'id_smiles.csv'

    print('=' * 50)
    print('Starting file format check')
    print('=' * 50)

    check_file_formats(csv_entity, csv_drug)

    print('\n' + '=' * 50)
    print('Starting CID and ID matching')
    print('=' * 50)

    matched, missed = match_and_save_smiles(csv_entity, csv_drug, output)

    print('\n' + '=' * 50)
    print('Matching completed')
    print('=' * 50)

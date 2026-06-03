import pandas as pd
import logging
from tqdm import tqdm
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("target_id_matching.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def match_and_save_target_id(csv_entity_mapping='../embeddings/entity_mapping.csv',
                             csv_target_sequence='Seq_Syn.csv',
                             output_file='id_target.csv'):
    """
    Match target IDs with sequences

    Args:
        csv_entity_mapping: Path to entity mapping file (contains id and entity columns)
        csv_target_sequence: Path to target sequence file (contains target_name and sequence columns)
        output_file: Output file path
    """
    try:
        logger.info(f"Loading entity mapping file: {csv_entity_mapping}")
        entity_df = pd.read_csv(csv_entity_mapping)

        if 'id' not in entity_df.columns or 'entity' not in entity_df.columns:
            raise ValueError("Entity mapping file must contain 'id' and 'entity' columns")

        entity_df['entity'] = entity_df['entity'].astype(str)

        entity_df['entity_std'] = entity_df['entity'].str.strip().str.lower()
        logger.info(f"Successfully loaded {len(entity_df)} entity mapping records")

        entity_to_id = dict(zip(entity_df['entity_std'], entity_df['id']))
        logger.info(f"Created entity-to-ID mapping with {len(entity_to_id)} entries")

    except Exception as e:
        logger.error(f"Error loading entity mapping file: {str(e)}")
        return None, None

    try:
        logger.info(f"Loading target sequence file: {csv_target_sequence}")
        target_df = pd.read_csv(csv_target_sequence)

        if 'target_name' not in target_df.columns or 'sequence' not in target_df.columns:
            raise ValueError("Target sequence file must contain 'target_name' and 'sequence' columns")

        target_df['target_name'] = target_df['target_name'].astype(str)
        target_df['sequence'] = target_df['sequence'].astype(str)

        target_df['target_name_std'] = target_df['target_name'].str.strip().str.lower()
        logger.info(f"Successfully loaded {len(target_df)} target sequence records")

        target_df = target_df.drop_duplicates(subset=['target_name_std'])
        logger.info(f"Retained {len(target_df)} unique target records after deduplication")

    except Exception as e:
        logger.error(f"Error loading target sequence file: {str(e)}")
        return None, None

    matched_data = []
    missed_targets = []
    matched_count = 0

    logger.info("Starting target name and entity ID matching...")

    for _, row in tqdm(target_df.iterrows(), total=len(target_df)):
        target_name = row['target_name']
        target_name_std = row['target_name_std']
        sequence = row['sequence']

        if target_name_std in entity_to_id:
            entity_id = entity_to_id[target_name_std]
            matched_data.append({
                'id': entity_id,
                'sequence': sequence,
                'target_name': target_name
            })
            matched_count += 1
        else:
            missed_targets.append({
                'target_name': target_name,
                'sequence': sequence
            })

    if matched_data:
        matched_df = pd.DataFrame(matched_data)
        matched_df.to_csv(output_file, index=False)
        logger.info(f"Successfully matched {matched_count} records, saved to {output_file}")
    else:
        logger.warning("No matching records found")

    if missed_targets:
        missed_df = pd.DataFrame(missed_targets)
        missed_df.to_csv('missed_targets.csv', index=False)
        logger.info(f"{len(missed_targets)} records unmatched, saved to 'missed_targets.csv'")

    match_rate = matched_count / len(target_df) * 100 if len(target_df) > 0 else 0
    logger.info(f"Matching completed! Total targets: {len(target_df)}, Matched: {matched_count}, Match rate: {match_rate:.2f}%")

    return matched_count, len(missed_targets)


if __name__ == "__main__":
    csv_entity = '../embeddings/entity_mapping.csv'
    csv_target = 'Seq_Syn.csv'
    output = 'id_target.csv'

    logger.info("Starting target ID and sequence matching program")

    try:
        matched, missed = match_and_save_target_id(
            csv_entity_mapping=csv_entity,
            csv_target_sequence=csv_target,
            output_file=output
        )

        if matched is None or missed is None:
            logger.error("Error occurred during program execution, please check the log file")
            sys.exit(1)

        logger.info(f"Program execution completed: {matched} matched, {missed} unmatched")
    except Exception as e:
        logger.error(f"Uncaught exception during program execution: {str(e)}")
        sys.exit(1)

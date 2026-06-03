import torch
import numpy as np
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def extract_protein_features_fixed(sequences, model_name="Rostlab/prot_t5_xl_uniref50"):

    print(f"Loading model from Hugging Face: {model_name}")

    try:
        model = T5EncoderModel.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    model.eval()
    print(f"Model training mode: {model.training}")

    device = torch.device("cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    features = []
    valid_sequences = []

    for sequence in tqdm(sequences, desc="Extracting features"):
        try:
            clean_seq = sequence.upper().replace(' ', '')

            if len(clean_seq) < 5:
                print(f"Warning: Skipping short sequence: {sequence[:50]}...")
                continue

            sequence_processed = " ".join(list(clean_seq))

            inputs = tokenizer(
                sequence_processed,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            last_hidden = outputs.last_hidden_state

            attention_mask = inputs['attention_mask']

            mask = attention_mask.unsqueeze(-1).float()
            masked_hidden = last_hidden * mask
            sum_embeddings = masked_hidden.sum(dim=1)
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float().clamp(min=1e-9)

            mean_embeddings = sum_embeddings / seq_lengths
            feature = mean_embeddings.squeeze().cpu().numpy()

            features.append(feature)
            valid_sequences.append(sequence)

        except Exception as e:
            print(f"Error processing sequence: {e}")
            continue

    if len(features) == 0:
        print("Error: No features extracted successfully")
        return None

    features_array = np.array(features)

    if len(features) > 1:
        similarities = cosine_similarity(features_array)

        n = similarities.shape[0]
        non_diag_count = n * (n - 1)
        avg_similarity = similarities.sum() / non_diag_count if non_diag_count > 0 else 0
        max_similarity = similarities.max()

        print(f"Feature validation - Max similarity: {max_similarity:.4f}")
        print(f"Feature validation - Average similarity: {avg_similarity:.4f}")
        print(f"Successfully processed sequences: {len(features)}/{len(sequences)}")

        if avg_similarity > 0.95:
            print("Warning: High feature similarity detected, may need to adjust feature extraction strategy")

    return features_array


def main():
    try:
        df = pd.read_csv('id_target.csv')
        sequences = df['sequence'].tolist()
        ids = df['id'].tolist()
        print(f"Loaded {len(sequences)} sequences from CSV")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print("Starting feature extraction...")
    features = extract_protein_features_fixed(sequences)

    if features is not None:
        np.save('target_features.npy', features)
        np.save('target_ids.npy', ids)
        print("Feature extraction completed!")
        print(f"Feature matrix shape: {features.shape}")
    else:
        print("Feature extraction failed!")


if __name__ == "__main__":
    main()






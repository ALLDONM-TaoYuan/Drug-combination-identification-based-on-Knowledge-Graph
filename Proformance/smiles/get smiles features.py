import numpy as np
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import torch

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

print(f"Loading model from HuggingFace: {MODEL_NAME}")

try:
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

df = pd.read_csv('id_smiles.csv')
smiles = df['smiles'].tolist()
ids = df['id'].tolist()

print(f"Processing {len(smiles)} SMILES strings...")

features = []

batch_size = 32
pbar = tqdm(total=len(smiles), desc='Processing SMILES')

for i in range(0, len(smiles), batch_size):
    batch_smiles = smiles[i:i+batch_size]
    
    inputs = tokenizer(batch_smiles, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]
    
    batch_features = last_hidden_state.mean(dim=1).cpu().numpy()
    features.extend(batch_features)
    
    pbar.update(len(batch_smiles))

pbar.close()

features_array = np.array(features)

np.save('smiles_features.npy', features_array)
np.save('smiles_ids.npy', ids)

print(f'Complete! Features shape: {features_array.shape}')
print(f'Features saved to: smiles_features.npy')
print(f'IDs saved to: smiles_ids.npy')




"""
Extract CLIP text embeddings for offline T2M training.

This script processes all text annotations from HumanML3D through CLIP
text encoder and persists the embeddings to avoid loading CLIP at training time.

Output: ./humanml3d_272/text_latents_clip/
Each sample saved as <name>.npy (shape: (num_texts, 512) for CLIP ViT-B/32)
CFG empty text: empty_cfg_text_clip.npy (shape: (512,))

CLIP output dim: 512 (ViT-B/32)
"""

import os
import torch
import clip
import numpy as np
from os.path import join as pjoin
import codecs as cs
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

##### ---- Config ---- #####
DATASET_NAME = 't2m_272'
DATA_ROOT = './humanml3d_272'
TEXT_DIR = pjoin(DATA_ROOT, 'texts')
SPLIT_FILE = pjoin(DATA_ROOT, 'split', 'train.txt')
OUTPUT_DIR = pjoin(DATA_ROOT, 'text_latents_clip')

CLIP_MODEL = 'ViT-B/32'  # Consistent with MSA-VAE
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CLIP Text Embedding Extraction")
print("=" * 70)
print(f"Dataset: {DATASET_NAME}")
print(f"CLIP Model: {CLIP_MODEL}")
print(f"Device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Batch size: {BATCH_SIZE}")
print("")

##### ---- Load CLIP model ---- #####
print("Loading CLIP model...")
clip_model, _ = clip.load(CLIP_MODEL, device=DEVICE, jit=False)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False
print(f"✓ CLIP model loaded ({CLIP_MODEL})")
print("")

##### ---- Helper functions ---- #####
def encode_text_batch(text_list, clip_model, device, batch_size=64):
    """
    Encode a list of texts through CLIP with batching for memory efficiency.
    
    Args:
        text_list: list of strings
        clip_model: CLIP model
        device: torch device
        batch_size: batch size for encoding
    
    Returns:
        embeddings: numpy array of shape (len(text_list), 512)
    """
    embeddings = []
    
    for i in tqdm(range(0, len(text_list), batch_size), 
                  desc=f"Encoding {len(text_list)} texts", 
                  leave=False):
        batch_texts = text_list[i:i+batch_size]
        tokens = clip.tokenize(batch_texts, truncate=True).to(device)
        
        with torch.no_grad():
            batch_embeddings = clip_model.encode_text(tokens).float()
        
        embeddings.append(batch_embeddings.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)  # (len(text_list), 512)
    return embeddings


##### ---- Extract per-sample text embeddings ---- #####
print(f"Loading sample IDs from {SPLIT_FILE}...")
id_list = []
with cs.open(SPLIT_FILE, 'r') as f:
    for line in f.readlines():
        id_list.append(line.strip())

print(f"✓ Found {len(id_list)} samples")
print("")

processed_count = 0
skipped_count = 0

print("Extracting text embeddings for each sample...")
for sample_id in tqdm(id_list):
    try:
        text_file = pjoin(TEXT_DIR, sample_id + '.txt')
        
        if not os.path.exists(text_file):
            skipped_count += 1
            continue
        
        # Read all text annotations for this sample
        text_lines = []
        with cs.open(text_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    # Parse: caption#tokens#from#to
                    parts = line.split('#')
                    caption = parts[0].strip()
                    if caption:
                        text_lines.append(caption)
        
        if not text_lines:
            skipped_count += 1
            continue
        
        # Encode all captions for this sample
        embeddings = encode_text_batch(
            text_lines, clip_model, DEVICE, batch_size=32
        )  # (num_captions, 512)
        
        # Save
        save_path = pjoin(OUTPUT_DIR, sample_id + '.npy')
        np.save(save_path, embeddings)
        processed_count += 1
        
    except Exception as e:
        print(f"  ⚠ Error processing {sample_id}: {str(e)}")
        skipped_count += 1
        continue

##### ---- Generate empty CFG text embedding ---- #####
print("")
print("Generating empty CFG text embedding...")
empty_text = ""
tokens = clip.tokenize([empty_text], truncate=True).to(DEVICE)

with torch.no_grad():
    empty_embedding = clip_model.encode_text(tokens).float()

empty_embedding_np = empty_embedding.cpu().numpy()  # (1, 512)
empty_embedding_np = empty_embedding_np[0]  # Extract single vector (512,)

empty_cfg_path = pjoin(OUTPUT_DIR, 'empty_cfg_text_clip.npy')
np.save(empty_cfg_path, empty_embedding_np)
print(f"✓ Empty CFG embedding saved: {empty_cfg_path}")
print(f"  Shape: {empty_embedding_np.shape}")
print("")

##### ---- Summary ---- #####
print("=" * 70)
print("✓ CLIP Text Embedding Extraction Complete")
print("=" * 70)
print(f"Processed: {processed_count} samples")
print(f"Skipped: {skipped_count} samples")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Total files: {len(os.listdir(OUTPUT_DIR))}")
print("")

# Print sample statistics
sample_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')]
if sample_files:
    sample_data = np.load(pjoin(OUTPUT_DIR, sample_files[0]))
    print(f"Sample file statistics:")
    print(f"  Shape: {sample_data.shape}")
    print(f"  dtype: {sample_data.dtype}")
    print(f"  Mean: {sample_data.mean():.6f}")
    print(f"  Std: {sample_data.std():.6f}")
    print("")

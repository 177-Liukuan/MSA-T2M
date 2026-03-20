"""
DataLoader for T2M training with cached CLIP text embeddings and TAE (Baseline) motion latents.

This DataLoader loads precomputed features from disk:
- Motion latents: from original Causal TAE (get_latent.py output)  
- Text embeddings: from get_text_latent_clip.py output (CLIP)

This is the BASELINE for ablation study: TAE latents + CLIP text embeddings.

Feature dimensions:
- Motion latent: (T'+1, latent_dim=16) where T' = T/4
- Text embedding: (num_captions, 512) for CLIP ViT-B/32
"""

import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import os
import random
import codecs as cs
from tqdm import tqdm


class Text2MotionDataset(data.Dataset):
    """
    Dataset for TAE (baseline) motion latents + CLIP text embeddings.
    
    All features are precomputed and cached on disk.
    """
    
    def __init__(self, 
                 dataset_name='t2m_272',
                 motion_latent_dir='./humanml3d_272/t2m_latents/causal_TAE_t2m_272_h100_20260203',
                 text_latent_dir='./humanml3d_272/text_latents_clip',
                 unit_length=4):
        """
        Args:
            dataset_name: 't2m_272' or similar
            motion_latent_dir: directory containing TAE motion latents (from original get_latent.py)
            text_latent_dir: directory containing CLIP text embeddings (from get_text_latent_clip.py)
            unit_length: motion downsampling ratio (already applied in get_latent.py)
        """
        self.dataset_name = dataset_name
        self.motion_latent_dir = motion_latent_dir
        self.text_latent_dir = text_latent_dir
        self.unit_length = unit_length
        
        # Setup paths
        if dataset_name == 't2m_272':
            data_root = './humanml3d_272'
            text_dir = pjoin(data_root, 'texts')
            split_file = pjoin(data_root, 'split', 'train.txt')
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Read sample IDs from split file
        id_list = []
        with codecs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        # Filter samples that have both motion latent and text embedding
        self.data = []
        self.id_list = []
        
        print(f"Loading {len(id_list)} samples...")
        for sample_id in tqdm(id_list, desc="Building dataset"):
            motion_path = pjoin(motion_latent_dir, sample_id + '.npy')
            text_path = pjoin(text_latent_dir, sample_id + '.npy')
            
            if os.path.exists(motion_path) and os.path.exists(text_path):
                self.data.append({
                    'id': sample_id,
                    'motion_path': motion_path,
                    'text_path': text_path,
                })
                self.id_list.append(sample_id)
        
        print(f"✓ Loaded {len(self.data)} samples with complete features")
        
        # Load reference motion latent for inference
        # ref_motion_path = pjoin(motion_latent_dir, f'reference_end_latent_{dataset_name}.npy')
        # if os.path.exists(ref_motion_path):
        #     self.reference_end_latent = np.load(ref_motion_path)  # (T_ref, latent_dim)
        # else:
        #     raise FileNotFoundError(f"Reference motion latent not found: {ref_motion_path}")
        ref_motion_path = './reference_end_latent_t2m_272.npy'
        
        # Load empty text embedding for CFG
        empty_text_path = pjoin(text_latent_dir, 'empty_cfg_text_clip.npy')
        if os.path.exists(empty_text_path):
            self.empty_text_embedding = np.load(empty_text_path)  # (512,)
        else:
            print(f"Warning: Empty text embedding not found at {empty_text_path}")
            self.empty_text_embedding = np.zeros(512, dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            feat_text: (512,) float32 - CLIP text embedding for one randomly selected caption
            m_tokens: (T'+1, latent_dim) float32 - motion latent with reference appended
            m_tokens_len: int - number of latent frames (T'+1)
        """
        entry = self.data[idx]
        
        # Load motion latent: (T'+K, latent_dim) where K = num reference frames
        m_tokens = np.load(entry['motion_path']).astype(np.float32)
        m_tokens_len = m_tokens.shape[0]
        
        # Load text embeddings: (num_captions, 512)
        text_embeddings = np.load(entry['text_path']).astype(np.float32)
        
        # Randomly select one caption if multiple exist
        if len(text_embeddings.shape) > 1 and text_embeddings.shape[0] > 1:
            text_idx = random.randint(0, text_embeddings.shape[0] - 1)
            feat_text = text_embeddings[text_idx]  # (512,)
        else:
            # Single caption or already 1D
            feat_text = text_embeddings[0] if len(text_embeddings.shape) > 1 else text_embeddings
        
        # Ensure feat_text is 1D
        feat_text = feat_text.reshape(-1).astype(np.float32)
        
        return feat_text, m_tokens, m_tokens_len


def DATALoader(dataset_name,
               is_test,
               batch_size,
               motion_latent_dir='./humanml3d_272/t2m_latents/causal_TAE_t2m_272_h100_20260203',
               text_latent_dir='./humanml3d_272/text_latents_clip',
               num_workers=0,
               unit_length=4):
    """
    Create DataLoader for TAE (baseline) latents + CLIP embeddings.
    
    Args:
        dataset_name: dataset identifier
        is_test: unused (kept for API compatibility)
        batch_size: batch size
        motion_latent_dir: path to TAE motion latents
        text_latent_dir: path to CLIP text embeddings
        num_workers: number of data loading workers
        unit_length: motion downsampling ratio
    
    Returns:
        DataLoader instance
    """
    dataset = Text2MotionDataset(
        dataset_name=dataset_name,
        motion_latent_dir=motion_latent_dir,
        text_latent_dir=text_latent_dir,
        unit_length=unit_length
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    return loader


def collate_fn(batch):
    """
    Custom collate function for mixed tensor/array batch.
    
    Args:
        batch: list of (feat_text, m_tokens, m_tokens_len)
    
    Returns:
        feat_text_batch: (batch_size, 512) torch float32
        m_tokens_batch: (batch_size, max_seq_len, latent_dim) torch float32
        m_tokens_len_batch: (batch_size,) torch int64
    """
    feat_texts, m_tokens_list, m_tokens_lens = zip(*batch)
    
    # Text embeddings: all same shape (512,)
    feat_text_batch = torch.from_numpy(np.stack(feat_texts)).float()  # (B, 512)
    
    # Motion latents: variable length, pad to max
    max_len = max(m_tokens_lens)
    batch_size = len(m_tokens_list)
    latent_dim = m_tokens_list[0].shape[1]
    
    m_tokens_batch = torch.zeros(batch_size, max_len, latent_dim, dtype=torch.float32)
    m_tokens_len_batch = torch.tensor(m_tokens_lens, dtype=torch.long)
    
    for i, seq in enumerate(m_tokens_list):
        m_tokens_batch[i, :len(seq)] = torch.from_numpy(seq).float()
    
    return feat_text_batch, m_tokens_batch, m_tokens_len_batch


def cycle(iterable):
    """Cycle through iterable indefinitely."""
    while True:
        for x in iterable:
            yield x


# For backwards compatibility with original imports
import codecs

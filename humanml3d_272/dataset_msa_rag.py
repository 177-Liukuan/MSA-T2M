"""RAG dataloader for MotionStreamer Stage-2 training.

This dataset only loads precomputed numpy features from disk:
- CLIP text embeddings (512d)
- MSA-VAE global motion semantics h_cls (512d)
- Motion local latents z_local / motion latents (16d)

Batch output format:
    (text_emb, top3_h_cls, top3_sim_scores, motion_latents)
"""

import os
import random
import codecs as cs
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


class Text2MotionMSARAGDataset(data.Dataset):
    """Text-to-motion dataset with retrieval-augmented h_cls prior."""

    def __init__(
        self,
        dataset_name='t2m_272',
        motion_latent_dir='./humanml3d_272/t2m_latents/causal_TAE_t2m_272_h100_20260203',
        text_latent_dir='./humanml3d_272/text_latents_clip',
        hcls_dir='./humanml3d_272/h_cls_latents_msa_vae',
        topk=3,
        exclude_self=True,
    ):
        self.dataset_name = dataset_name
        self.motion_latent_dir = motion_latent_dir
        self.text_latent_dir = text_latent_dir
        self.hcls_dir = hcls_dir
        self.topk = topk
        self.exclude_self = exclude_self

        if dataset_name == 't2m_272':
            data_root = './humanml3d_272'
            split_file = pjoin(data_root, 'split', 'train.txt')
        else:
            raise ValueError(f'Unsupported dataset: {dataset_name}')

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                sample_id = line.strip()
                if sample_id:
                    id_list.append(sample_id)

        self.data = []
        print(f'Building RAG dataset from {len(id_list)} ids...')
        for sample_id in tqdm(id_list, desc='Checking feature files'):
            motion_path = pjoin(motion_latent_dir, sample_id + '.npy')
            text_path = pjoin(text_latent_dir, sample_id + '.npy')
            hcls_path = pjoin(hcls_dir, sample_id + '.npy')

            if os.path.exists(motion_path) and os.path.exists(text_path) and os.path.exists(hcls_path):
                self.data.append(
                    {
                        'id': sample_id,
                        'motion_path': motion_path,
                        'text_path': text_path,
                        'hcls_path': hcls_path,
                    }
                )

        if len(self.data) == 0:
            raise RuntimeError('No valid samples found. Please check motion/text/h_cls directories.')

        print(f'Loaded {len(self.data)} valid samples for training.')

        # Build in-memory retrieval library once (performance-critical).
        self.library_ids = []
        self.library_hcls = []
        for entry in tqdm(self.data, desc='Loading h_cls retrieval library'):
            hcls = np.load(entry['hcls_path']).astype(np.float32)
            # Use a stable single vector per sample for retrieval indexing.
            if hcls.ndim == 2:
                hcls_vec = hcls.mean(axis=0)
            else:
                hcls_vec = hcls.reshape(-1)

            if hcls_vec.shape[0] != 512:
                raise ValueError(
                    f"h_cls dim mismatch for {entry['id']}: got {hcls_vec.shape[0]}, expected 512"
                )

            self.library_ids.append(entry['id'])
            self.library_hcls.append(hcls_vec)

        self.library_hcls = torch.from_numpy(np.stack(self.library_hcls)).float()  # [N, 512]
        self.library_hcls_norm = self._l2_normalize(self.library_hcls)
        self.id_to_library_index = {sid: i for i, sid in enumerate(self.library_ids)}

        print(
            f'Retrieval library ready: {self.library_hcls.shape[0]} vectors, '
            f'dim={self.library_hcls.shape[1]}, topk={self.topk}'
        )

    @staticmethod
    def _l2_normalize(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        motion_latents = np.load(entry['motion_path']).astype(np.float32)  # [T, 16]

        # text_emb can be [num_caps, 512] or [512]
        text_all = np.load(entry['text_path']).astype(np.float32)
        if text_all.ndim == 2 and text_all.shape[0] > 1:
            text_idx = random.randint(0, text_all.shape[0] - 1)
            text_emb = text_all[text_idx]
        elif text_all.ndim == 2:
            text_idx = 0
            text_emb = text_all[0]
        else:
            text_idx = 0
            text_emb = text_all.reshape(-1)

        if text_emb.shape[0] != 512:
            raise ValueError(
                f"text_emb dim mismatch for {entry['id']}: got {text_emb.shape[0]}, expected 512"
            )

        # Cosine retrieval against preloaded h_cls library.
        text_query = torch.from_numpy(text_emb).float().unsqueeze(0)  # [1, 512]
        text_query = self._l2_normalize(text_query)
        sim = torch.matmul(text_query, self.library_hcls_norm.t()).squeeze(0)  # [N]

        if self.exclude_self and entry['id'] in self.id_to_library_index:
            sim[self.id_to_library_index[entry['id']]] = -1e6

        k = min(self.topk, sim.shape[0])
        top_scores, top_indices = torch.topk(sim, k=k, dim=0)
        top_hcls = self.library_hcls[top_indices]  # [k, 512]

        if k < self.topk:
            pad_h = torch.zeros(self.topk - k, 512, dtype=top_hcls.dtype)
            pad_s = torch.full((self.topk - k,), -1e6, dtype=top_scores.dtype)
            top_hcls = torch.cat([top_hcls, pad_h], dim=0)
            top_scores = torch.cat([top_scores, pad_s], dim=0)

        return (
            text_emb.astype(np.float32),             # [512]
            top_hcls.numpy().astype(np.float32),     # [3, 512]
            top_scores.numpy().astype(np.float32),   # [3]
            motion_latents.astype(np.float32),       # [T, 16]
        )


def collate_fn(batch):
    text_embs, top_hcls_list, top_scores_list, motion_list = zip(*batch)

    text_emb_batch = torch.from_numpy(np.stack(text_embs)).float()  # [B, 512]
    top_hcls_batch = torch.from_numpy(np.stack(top_hcls_list)).float()  # [B, 3, 512]
    top_scores_batch = torch.from_numpy(np.stack(top_scores_list)).float()  # [B, 3]

    max_len = max(seq.shape[0] for seq in motion_list)
    latent_dim = motion_list[0].shape[1]
    motion_batch = torch.zeros(len(motion_list), max_len, latent_dim, dtype=torch.float32)
    for i, seq in enumerate(motion_list):
        motion_batch[i, : seq.shape[0]] = torch.from_numpy(seq).float()

    return text_emb_batch, top_hcls_batch, top_scores_batch, motion_batch


def DATALoader(
    dataset_name,
    is_test,
    batch_size,
    motion_latent_dir='./humanml3d_272/t2m_latents/causal_TAE_t2m_272_h100_20260203',
    text_latent_dir='./humanml3d_272/text_latents_clip',
    hcls_dir='./humanml3d_272/h_cls_latents_msa_vae',
    topk=3,
    exclude_self=True,
    num_workers=0,
):
    _ = is_test  # keep same API style as other loaders

    dataset = Text2MotionMSARAGDataset(
        dataset_name=dataset_name,
        motion_latent_dir=motion_latent_dir,
        text_latent_dir=text_latent_dir,
        hcls_dir=hcls_dir,
        topk=topk,
        exclude_self=exclude_self,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

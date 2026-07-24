"""RAG dataloader for MotionStreamer Stage-2 training with local RAG tokens.

Extends dataset_msa_rag.py by additionally loading per-frame z latents
(motion latents from t2m_latents_msa_vae) for the top-K retrieved motions.
The full z sequences are returned with padding masks; the model aggregates them
via cross-attention (LocalRAGCrossAttn).

Batch output format:
    (text_emb, topk_h_cls, topk_sim_scores, top_z_seqs, top_z_lens, motion_latents)
    top_z_seqs: [B, K, T_max, local_rag_dim]   padded z sequences
    top_z_lens: [B, K]                          valid frame counts (end token excluded)
"""

import os
import random
import codecs as cs
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


class Text2MotionMSARAGLocalDataset(data.Dataset):
    """Text-to-motion dataset with global h_cls + local z RAG priors.

    Uses motion latents (z, from t2m_latents_msa_vae) as local RAG tokens
    instead of encoder mu latents. z gives a more faithful motion reference
    since it is the actual latent used for generation/reconstruction.

    The last frame of each z sequence (end token) is stripped before use.
    """

    def __init__(
        self,
        dataset_name='t2m_272',
        motion_latent_dir='./humanml3d_272/t2m_latents_msa_vae/exp',
        text_latent_dir='./humanml3d_272/text_latents_t5',
        hcls_dir='./humanml3d_272/h_cls_latents_msa_vae/exp',
        z_latent_dir='./humanml3d_272/t2m_latents_msa_vae/exp',
        topk=3,
        L_local=4,
        local_rag_dim=16,
        exclude_self=True,
        text_embed_dim=None,
    ):
        self.dataset_name = dataset_name
        self.motion_latent_dir = motion_latent_dir
        self.text_latent_dir = text_latent_dir
        self.hcls_dir = hcls_dir
        self.z_latent_dir = z_latent_dir
        self.topk = topk
        self.L_local = int(L_local)
        self.local_rag_dim = int(local_rag_dim)
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
        print(f'Building Local-RAG dataset from {len(id_list)} ids...')
        for sample_id in tqdm(id_list, desc='Checking feature files'):
            motion_path = pjoin(motion_latent_dir, sample_id + '.npy')
            text_path = pjoin(text_latent_dir, sample_id + '.npy')
            hcls_path = pjoin(hcls_dir, sample_id + '.npy')
            z_path = pjoin(z_latent_dir, sample_id + '.npy')

            if (
                os.path.exists(motion_path)
                and os.path.exists(text_path)
                and os.path.exists(hcls_path)
                and os.path.exists(z_path)
            ):
                self.data.append({
                    'id': sample_id,
                    'motion_path': motion_path,
                    'text_path': text_path,
                    'hcls_path': hcls_path,
                    'z_path': z_path,
                })

        if len(self.data) == 0:
            raise RuntimeError('No valid samples found. Check motion/text/h_cls/z directories.')

        if text_embed_dim is None:
            first_text = np.load(self.data[0]['text_path']).astype(np.float32)
            if first_text.ndim == 2:
                text_embed_dim = int(first_text.shape[1])
            else:
                text_embed_dim = int(first_text.reshape(-1).shape[0])
        self.text_embed_dim = int(text_embed_dim)

        print(
            f'Loaded {len(self.data)} valid samples. '
            f'text_embed_dim={self.text_embed_dim}, L_local={self.L_local}'
        )

        # Build in-memory retrieval library: h_cls for similarity search.
        self.library_ids = []
        self.library_hcls = []
        for entry in tqdm(self.data, desc='Loading h_cls retrieval library'):
            hcls = np.load(entry['hcls_path']).astype(np.float32)
            hcls_vec = hcls.mean(axis=0) if hcls.ndim == 2 else hcls.reshape(-1)
            if hcls_vec.shape[0] != self.text_embed_dim:
                raise ValueError(
                    f"h_cls dim mismatch for {entry['id']}: "
                    f"got {hcls_vec.shape[0]}, expected {self.text_embed_dim}"
                )
            self.library_ids.append(entry['id'])
            self.library_hcls.append(hcls_vec)

        self.library_hcls = torch.from_numpy(np.stack(self.library_hcls)).float()  # [N, D]
        self.library_hcls_norm = self._l2_normalize(self.library_hcls)
        self.id_to_library_index = {sid: i for i, sid in enumerate(self.library_ids)}
        # Map library index -> z_path for lazy loading at __getitem__ time.
        self.library_z_paths = [entry['z_path'] for entry in self.data]

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

        # ---- Motion latents (generation target) ----
        motion_latents = np.load(entry['motion_path']).astype(np.float32)

        # ---- Text embedding (randomly pick one description) ----
        text_all = np.load(entry['text_path']).astype(np.float32)
        if text_all.ndim == 2 and text_all.shape[0] > 1:
            text_emb = text_all[random.randint(0, text_all.shape[0] - 1)]
        elif text_all.ndim == 2:
            text_emb = text_all[0]
        else:
            text_emb = text_all.reshape(-1)

        if text_emb.shape[0] != self.text_embed_dim:
            raise ValueError(
                f"text_emb dim mismatch for {entry['id']}: "
                f"got {text_emb.shape[0]}, expected {self.text_embed_dim}"
            )

        # ---- Top-K retrieval by text-h_cls cosine similarity ----
        text_query = self._l2_normalize(torch.from_numpy(text_emb).float().unsqueeze(0))
        sim = torch.matmul(text_query, self.library_hcls_norm.t()).squeeze(0)  # [N]

        if self.exclude_self and entry['id'] in self.id_to_library_index:
            sim[self.id_to_library_index[entry['id']]] = -1e6

        k = min(self.topk, sim.shape[0])
        top_scores, top_indices = torch.topk(sim, k=k, dim=0)
        top_hcls = self.library_hcls[top_indices]  # [k, D]

        # Pad if library smaller than topk.
        if k < self.topk:
            pad_h = torch.zeros(self.topk - k, self.text_embed_dim, dtype=top_hcls.dtype)
            pad_s = torch.full((self.topk - k,), -1e6, dtype=top_scores.dtype)
            top_hcls = torch.cat([top_hcls, pad_h], dim=0)
            top_scores = torch.cat([top_scores, pad_s], dim=0)

        # ---- Load z latents for top-K motions (strip end token = last frame) ----
        # z has shape (T+1, 16) where last frame is end token; we use only [:T].
        top_z_list = []    # list of K arrays, each (T_k, local_rag_dim)
        top_z_lens_list = []
        for ki in range(k):
            lib_idx = top_indices[ki].item()
            z_seq = np.load(self.library_z_paths[lib_idx]).astype(np.float32)  # (T+1, 16)
            z_seq = z_seq[:-1]  # strip end token -> (T, 16)
            top_z_list.append(z_seq)
            top_z_lens_list.append(z_seq.shape[0])

        # Pad with single-frame zeros for any missing slots (lens=0 -> fully masked).
        for _ in range(self.topk - k):
            top_z_list.append(np.zeros((1, self.local_rag_dim), dtype=np.float32))
            top_z_lens_list.append(0)

        top_z_lens_np = np.array(top_z_lens_list, dtype=np.int64)  # [K]

        return (
            text_emb.astype(np.float32),
            top_hcls.numpy().astype(np.float32),
            top_scores.numpy().astype(np.float32),
            top_z_list,          # list of K variable-length arrays
            top_z_lens_np,       # [K] int64
            motion_latents.astype(np.float32),
        )


def collate_fn(batch):
    text_embs, top_hcls_list, top_scores_list, top_z_seqs_list, top_z_lens_list, motion_list = zip(*batch)

    B = len(batch)
    K = len(top_z_seqs_list[0])  # number of retrieved motions per sample

    # Find batch-level T_max across all B*K sequences
    T_max = max(
        z_seq.shape[0]
        for z_seqs in top_z_seqs_list
        for z_seq in z_seqs
    )
    T_max = max(T_max, 1)  # guard against edge case
    local_rag_dim = top_z_seqs_list[0][0].shape[1]

    # Build padded [B, K, T_max, local_rag_dim] tensor
    top_z_batch = torch.zeros(B, K, T_max, local_rag_dim, dtype=torch.float32)
    for i, z_seqs in enumerate(top_z_seqs_list):
        for ki, z_seq in enumerate(z_seqs):
            T = z_seq.shape[0]
            if T > 0:
                top_z_batch[i, ki, :T] = torch.from_numpy(z_seq).float()

    # Build [B, K] lens tensor
    top_z_lens_batch = torch.from_numpy(np.stack(top_z_lens_list)).long()  # [B, K]

    text_emb_batch = torch.from_numpy(np.stack(text_embs)).float()          # [B, D]
    top_hcls_batch = torch.from_numpy(np.stack(top_hcls_list)).float()      # [B, K, D]
    top_scores_batch = torch.from_numpy(np.stack(top_scores_list)).float()  # [B, K]

    max_len = max(seq.shape[0] for seq in motion_list)
    latent_dim = motion_list[0].shape[1]
    motion_batch = torch.zeros(len(motion_list), max_len, latent_dim, dtype=torch.float32)
    for i, seq in enumerate(motion_list):
        motion_batch[i, : seq.shape[0]] = torch.from_numpy(seq).float()

    return (
        text_emb_batch,
        top_hcls_batch,
        top_scores_batch,
        top_z_batch,        # [B, K, T_max, local_rag_dim]
        top_z_lens_batch,   # [B, K]
        motion_batch,
    )


def DATALoader(
    dataset_name,
    is_test,
    batch_size,
    motion_latent_dir,
    text_latent_dir,
    hcls_dir,
    z_latent_dir,
    topk=3,
    L_local=4,
    local_rag_dim=16,
    exclude_self=True,
    num_workers=4,
    text_embed_dim=None,
):
    _ = is_test

    dataset = Text2MotionMSARAGLocalDataset(
        dataset_name=dataset_name,
        motion_latent_dir=motion_latent_dir,
        text_latent_dir=text_latent_dir,
        hcls_dir=hcls_dir,
        z_latent_dir=z_latent_dir,
        topk=topk,
        L_local=L_local,
        local_rag_dim=local_rag_dim,
        exclude_self=exclude_self,
        text_embed_dim=text_embed_dim,
    )

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )


def cycle(iterable):
    while True:
        for item in iterable:
            yield item

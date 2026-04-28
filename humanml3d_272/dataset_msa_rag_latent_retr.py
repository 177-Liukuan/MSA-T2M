"""Local-RAG dataloader: per-caption text-to-text retrieval of motion latents.

Library design
--------------
Each training sample has N captions (HumanML3D: N~3-5).
One library entry per (sample, caption) pair:
    Key   : single-caption T5 sentence embedding  (768,)
    Value : the sample's motion latent             (F_i, latent_dim)
            [multiple captions share the SAME latent slice in flat storage]

Offline cache (library_cache_dir)
----------------------------------
If library_cache_dir is given and all 5 cache files exist:
    load in ~2 s (5 large numpy reads) instead of opening ~42k small files.
If cache files are missing but library_cache_dir is given:
    build from scratch then save automatically.
Cache files written to library_cache_dir/:
    lib_text_embs.npy      (N_caps, 768)   float32
    lib_sample_ids.txt     N_caps lines    str
    lib_latents_flat.npy   (sum_F, D)      float32
    lib_latent_starts.npy  (N_caps,)       int64
    lib_latent_lengths.npy (N_caps,)       int64

Batch output (6 items):
    [0] text_emb          (B, 768)
    [1] top3_h_cls        (B, K, 768)
    [2] top3_sim_scores   (B, K)
    [3] motion_latents    (B, T_max, latent_dim)
    [4] retr_latents      (B, L_max, latent_dim)
    [5] retr_latent_lens  (B,)  int64
"""

import os
import random
from collections import defaultdict
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from humanml3d_272.dataset_msa_rag import Text2MotionMSARAGDataset

_CACHE_FILES = (
    'lib_text_embs.npy',
    'lib_sample_ids.txt',
    'lib_latents_flat.npy',
    'lib_latent_starts.npy',
    'lib_latent_lengths.npy',
)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def _cache_exists(cache_dir):
    return all(os.path.exists(pjoin(cache_dir, f)) for f in _CACHE_FILES)


def save_library_cache(cache_dir, lib_text_embs, lib_sample_ids,
                       lib_latents_flat, lib_latent_starts, lib_latent_lengths):
    os.makedirs(cache_dir, exist_ok=True)
    np.save(pjoin(cache_dir, 'lib_text_embs.npy'),      lib_text_embs)
    np.save(pjoin(cache_dir, 'lib_latents_flat.npy'),   lib_latents_flat)
    np.save(pjoin(cache_dir, 'lib_latent_starts.npy'),  lib_latent_starts)
    np.save(pjoin(cache_dir, 'lib_latent_lengths.npy'), lib_latent_lengths)
    with open(pjoin(cache_dir, 'lib_sample_ids.txt'), 'w') as f:
        f.write('\n'.join(lib_sample_ids))
    print(f'[LatentRetr] Library cache saved -> {cache_dir}')


def load_library_cache(cache_dir):
    print(f'[LatentRetr] Loading library cache from {cache_dir} ...')
    lib_text_embs      = np.load(pjoin(cache_dir, 'lib_text_embs.npy'))
    lib_latents_flat   = np.load(pjoin(cache_dir, 'lib_latents_flat.npy'))
    lib_latent_starts  = np.load(pjoin(cache_dir, 'lib_latent_starts.npy'))
    lib_latent_lengths = np.load(pjoin(cache_dir, 'lib_latent_lengths.npy'))
    with open(pjoin(cache_dir, 'lib_sample_ids.txt'), 'r') as f:
        lib_sample_ids = [l.strip() for l in f.readlines()]
    print(
        f'[LatentRetr] Cache loaded: {len(lib_sample_ids)} caption entries, '
        f'text_embs {lib_text_embs.shape}, latents_flat {lib_latents_flat.shape}'
    )
    return lib_text_embs, lib_sample_ids, lib_latents_flat, lib_latent_starts, lib_latent_lengths


def scan_data_entries(dataset_name, motion_latent_dir, text_latent_dir):
    """Scan split file and return data entries with only (motion, text) paths.

    Does NOT require hcls_dir — suitable for building the local-RAG library
    where only (text_embedding, motion_latent) pairs are needed.

    Returns list of dicts: {'id', 'motion_path', 'text_path'}
    """
    import codecs as cs
    if dataset_name == 't2m_272':
        split_file = './humanml3d_272/split/train.txt'
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    with cs.open(split_file, 'r') as f:
        id_list = [line.strip() for line in f if line.strip()]

    data = []
    missing_motion = 0
    missing_text   = 0
    for sid in tqdm(id_list, desc='[LatentRetr] Scanning data entries'):
        motion_path = pjoin(motion_latent_dir, sid + '.npy')
        text_path   = pjoin(text_latent_dir,   sid + '.npy')
        if not os.path.exists(motion_path):
            missing_motion += 1
            continue
        if not os.path.exists(text_path):
            missing_text += 1
            continue
        data.append({'id': sid, 'motion_path': motion_path, 'text_path': text_path})

    print(
        f'[LatentRetr] Scan done: {len(data)} valid entries '
        f'(skipped {missing_motion} missing motion, {missing_text} missing text)'
    )
    return data


def build_library_from_data(data_entries, text_embed_dim):
    """Build flat retrieval library from a list of data entry dicts."""
    lib_text_list   = []
    lib_sample_ids  = []
    lib_latent_list = []   # unique latents (one per sample)
    lib_latent_uid  = []   # caption -> unique latent index
    _latent_idx_map = {}   # sample_id -> index in lib_latent_list

    for entry in tqdm(data_entries, desc='[LatentRetr] Building library'):
        sid = entry['id']
        if sid not in _latent_idx_map:
            _latent_idx_map[sid] = len(lib_latent_list)
            lib_latent_list.append(
                np.load(entry['motion_path']).astype(np.float32)
            )
        uid = _latent_idx_map[sid]

        text_all = np.load(entry['text_path']).astype(np.float32)
        if text_all.ndim == 1:
            text_all = text_all.reshape(1, -1)
        for cap_idx in range(text_all.shape[0]):
            lib_text_list.append(text_all[cap_idx])
            lib_sample_ids.append(sid)
            lib_latent_uid.append(uid)

    lib_text_embs  = np.stack(lib_text_list).astype(np.float32)
    lib_latent_uid = np.array(lib_latent_uid, dtype=np.int64)

    # Build flat concat + offsets for unique latents
    unique_starts  = np.zeros(len(lib_latent_list), dtype=np.int64)
    unique_lengths = np.zeros(len(lib_latent_list), dtype=np.int64)
    offset = 0
    for j, lat in enumerate(lib_latent_list):
        unique_starts[j]  = offset
        unique_lengths[j] = lat.shape[0]
        offset += lat.shape[0]
    lib_latents_flat   = np.concatenate(lib_latent_list, axis=0).astype(np.float32)
    lib_latent_starts  = unique_starts[lib_latent_uid]
    lib_latent_lengths = unique_lengths[lib_latent_uid]

    print(
        f'[LatentRetr] Built: {len(lib_sample_ids)} caption entries, '
        f'{len(lib_latent_list)} unique samples, '
        f'text_embs {lib_text_embs.shape}, latents_flat {lib_latents_flat.shape}'
    )
    return lib_text_embs, lib_sample_ids, lib_latents_flat, lib_latent_starts, lib_latent_lengths


class Text2MotionLatentRetrDataset(data.Dataset):
    def __init__(
        self,
        dataset_name='t2m_272',
        motion_latent_dir='./humanml3d_272/t2m_latents_msa_vae/exp',
        text_latent_dir='./humanml3d_272/text_latents_t5',
        hcls_dir='./humanml3d_272/h_cls_latents_msa_vae/exp',
        topk=3,
        latent_retr_topk=3,
        exclude_self=True,
        text_embed_dim=None,
        library_cache_dir=None,
        precomputed_retr_dir=None,
    ):
        self._base = Text2MotionMSARAGDataset(
            dataset_name=dataset_name,
            motion_latent_dir=motion_latent_dir,
            text_latent_dir=text_latent_dir,
            hcls_dir=hcls_dir,
            topk=topk,
            exclude_self=exclude_self,
            text_embed_dim=text_embed_dim,
        )
        self.text_embed_dim   = self._base.text_embed_dim
        self.latent_retr_topk = latent_retr_topk
        self.exclude_self     = exclude_self

        if library_cache_dir is not None and _cache_exists(library_cache_dir):
            (lib_text_embs, lib_sample_ids,
             lib_latents_flat, lib_latent_starts,
             lib_latent_lengths) = load_library_cache(library_cache_dir)
        else:
            (lib_text_embs, lib_sample_ids,
             lib_latents_flat, lib_latent_starts,
             lib_latent_lengths) = build_library_from_data(
                self._base.data, self.text_embed_dim
            )
            if library_cache_dir is not None:
                save_library_cache(
                    library_cache_dir, lib_text_embs, lib_sample_ids,
                    lib_latents_flat, lib_latent_starts, lib_latent_lengths,
                )

        self.lib_text_embs      = torch.from_numpy(lib_text_embs).float()
        self.lib_text_embs_norm = self._l2_normalize(self.lib_text_embs)
        self.lib_latents_flat   = lib_latents_flat
        self.lib_latent_starts  = lib_latent_starts
        self.lib_latent_lengths = lib_latent_lengths
        self.lib_sample_ids     = lib_sample_ids

        self.sample_id_to_lib_indices = defaultdict(list)
        for j, sid in enumerate(lib_sample_ids):
            self.sample_id_to_lib_indices[sid].append(j)

        # Build exclude-index tensors once to avoid per-sample tensor creation.
        self.sample_id_to_lib_indices_t = {
            sid: torch.tensor(idxs, dtype=torch.long)
            for sid, idxs in self.sample_id_to_lib_indices.items()
        }

        # Cache per-sample arrays in RAM to avoid repeated random np.load() on NFS.
        self._motion_cache = []
        self._text_cache = []
        for entry in tqdm(self._base.data, desc='[LatentRetr] Caching sample arrays'):
            self._motion_cache.append(np.load(entry['motion_path']).astype(np.float32))
            self._text_cache.append(np.load(entry['text_path']).astype(np.float32))

        self.latent_dim = lib_latents_flat.shape[-1]

        # Precomputed retrieval lookup: precomputed_retr_dir/{sid}.npy
        # shape (num_captions, latent_retr_topk) int32
        self.precomputed_retr_dir = precomputed_retr_dir
        self._retr_lookup_cache = {}   # sid -> np.ndarray (num_caps, topk)
        if precomputed_retr_dir is not None and os.path.isdir(precomputed_retr_dir):
            print(f'[LatentRetr] Loading precomputed retrieval lookup from {precomputed_retr_dir}')
            for entry in tqdm(self._base.data, desc='[LatentRetr] Caching retr lookup'):
                sid = entry['id']
                if sid not in self._retr_lookup_cache:
                    p = os.path.join(precomputed_retr_dir, sid + '.npy')
                    if os.path.exists(p):
                        self._retr_lookup_cache[sid] = np.load(p)  # (num_caps, topk)
            print(f'[LatentRetr] Lookup cache loaded: {len(self._retr_lookup_cache)} entries')
        else:
            if precomputed_retr_dir is not None:
                print(f'[LatentRetr] WARNING: precomputed_retr_dir not found: {precomputed_retr_dir}')
                print(f'[LatentRetr] Falling back to online matmul retrieval.')

        print(
            f'[LatentRetr] Ready: {len(lib_sample_ids)} caption entries, '
            f'latent_dim={self.latent_dim}, latent_retr_topk={latent_retr_topk}, '
            f'precomputed_retr={"yes" if self._retr_lookup_cache else "no (online matmul)"}'
        )

    @staticmethod
    def _l2_normalize(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        entry = self._base.data[idx]
        sid   = entry['id']

        motion_latents = self._motion_cache[idx]

        text_all = self._text_cache[idx]
        if text_all.ndim == 2 and text_all.shape[0] > 1:
            text_emb = text_all[random.randint(0, text_all.shape[0] - 1)]
        elif text_all.ndim == 2:
            text_emb = text_all[0]
        else:
            text_emb = text_all.reshape(-1)

        text_query_t = torch.from_numpy(text_emb).float().unsqueeze(0)
        text_query_n = self._l2_normalize(text_query_t)

        # ---- Global RAG: h_cls retrieval ----
        sim_hcls = torch.matmul(
            text_query_n, self._base.library_hcls_norm.t()
        ).squeeze(0)
        if self.exclude_self and sid in self._base.id_to_library_index:
            sim_hcls[self._base.id_to_library_index[sid]] = -1e6

        k_global = min(self._base.topk, sim_hcls.shape[0])
        top_scores, top_idx = torch.topk(sim_hcls, k=k_global, dim=0)
        top_hcls = self._base.library_hcls[top_idx]

        if k_global < self._base.topk:
            pad_h = torch.zeros(self._base.topk - k_global, self.text_embed_dim)
            pad_s = torch.full((self._base.topk - k_global,), -1e6)
            top_hcls   = torch.cat([top_hcls,   pad_h], dim=0)
            top_scores = torch.cat([top_scores, pad_s], dim=0)

        # ---- Local RAG: text-to-text latent retrieval ----
        # Fast path: use precomputed top-k indices (no online matmul).
        # Fallback: full-library cosine similarity search on CPU.
        retr_parts = []
        if self._retr_lookup_cache:
            lookup = self._retr_lookup_cache.get(sid, None)
            if lookup is not None:
                # lookup shape: (num_caps, latent_retr_topk); find cap_idx by value match.
                cap_idx = 0
                if text_all.ndim == 2 and text_all.shape[0] > 1:
                    for ci in range(text_all.shape[0]):
                        if np.array_equal(text_all[ci], text_emb):
                            cap_idx = ci
                            break
                retr_lib_indices = lookup[cap_idx, :self.latent_retr_topk].tolist()
                for lib_i in retr_lib_indices:
                    start  = int(self.lib_latent_starts[lib_i])
                    length = int(self.lib_latent_lengths[lib_i])
                    retr_parts.append(self.lib_latents_flat[start:start + length])
        if not retr_parts:
            # Fallback: online cosine similarity search over full library.
            sim_text = torch.matmul(
                text_query_n, self.lib_text_embs_norm.t()
            ).squeeze(0)
            if self.exclude_self:
                exclude = self.sample_id_to_lib_indices_t.get(sid, None)
                if exclude is not None and exclude.numel() > 0:
                    sim_text[exclude] = -1e6
            k_retr = min(self.latent_retr_topk, sim_text.shape[0])
            _, retr_idx = torch.topk(sim_text, k=k_retr, dim=0)
            for i in retr_idx:
                i      = i.item()
                start  = int(self.lib_latent_starts[i])
                length = int(self.lib_latent_lengths[i])
                retr_parts.append(self.lib_latents_flat[start:start + length])

        retr_latents_cat = np.concatenate(retr_parts, axis=0).astype(np.float32)
        retr_latent_len  = int(retr_latents_cat.shape[0])

        return (
            text_emb.astype(np.float32),
            top_hcls.numpy().astype(np.float32),
            top_scores.numpy().astype(np.float32),
            motion_latents.astype(np.float32),
            retr_latents_cat,
            retr_latent_len,
        )


def collate_fn_latent_retr(batch):
    (text_embs, top_hcls_list, top_scores_list,
     motion_list, retr_latent_list, retr_len_list) = zip(*batch)

    B = len(motion_list)
    text_emb_batch   = torch.from_numpy(np.stack(text_embs)).float()
    top_hcls_batch   = torch.from_numpy(np.stack(top_hcls_list)).float()
    top_scores_batch = torch.from_numpy(np.stack(top_scores_list)).float()

    max_motion = max(s.shape[0] for s in motion_list)
    latent_dim = motion_list[0].shape[1]
    motion_batch = torch.zeros(B, max_motion, latent_dim)
    for i, seq in enumerate(motion_list):
        motion_batch[i, :seq.shape[0]] = torch.from_numpy(seq).float()

    retr_latent_lens = torch.tensor(retr_len_list, dtype=torch.long)
    max_retr         = max(max(retr_len_list), 1)
    retr_latent_dim  = retr_latent_list[0].shape[-1]
    retr_batch = torch.zeros(B, max_retr, retr_latent_dim)
    for i, retr in enumerate(retr_latent_list):
        retr_batch[i, :retr.shape[0]] = torch.from_numpy(retr).float()

    return (text_emb_batch, top_hcls_batch, top_scores_batch,
            motion_batch, retr_batch, retr_latent_lens)


def DATALoader(
    dataset_name, is_test, batch_size,
    motion_latent_dir, text_latent_dir, hcls_dir,
    topk=3, latent_retr_topk=3, exclude_self=True,
    num_workers=0, text_embed_dim=None, library_cache_dir=None,
    precomputed_retr_dir=None,
):
    _ = is_test
    dataset = Text2MotionLatentRetrDataset(
        dataset_name=dataset_name,
        motion_latent_dir=motion_latent_dir,
        text_latent_dir=text_latent_dir,
        hcls_dir=hcls_dir,
        topk=topk,
        latent_retr_topk=latent_retr_topk,
        exclude_self=exclude_self,
        text_embed_dim=text_embed_dim,
        library_cache_dir=library_cache_dir,
        precomputed_retr_dir=precomputed_retr_dir,
    )
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn_latent_retr,
        drop_last=True, persistent_workers=(num_workers > 0), pin_memory=True,
    )

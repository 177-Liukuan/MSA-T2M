"""precompute_latent_retr_lookup.py
===================================
Pre-compute top-k retrieved motion latent indices for every caption in the
training set. Eliminates the expensive per-sample CPU matmul in __getitem__.

Output: one .npy file per motion ID  →  shape (num_captions, topk)  int32
These files are loaded by dataset_msa_rag_latent_retr.py when
--precomputed_retr_dir is set.

Usage:
    python precompute_latent_retr_lookup.py \
        --motion_latent_dir   ./humanml3d_272/t2m_latents_msa_vae/... \
        --text_latent_dir     ./humanml3d_272/text_latents_t5 \
        --library_cache_dir   ./humanml3d_272/latent_retr_library_cache/... \
        --output_dir          ./humanml3d_272/latent_retr_lookup/... \
        --topk 3 \
        --batch_size 2048
"""

import os
import argparse
import codecs as cs
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin


def l2_normalize(x, eps=1e-6):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def load_library(cache_dir):
    print(f'Loading library cache from {cache_dir}')
    text_embs = np.load(pjoin(cache_dir, 'lib_text_embs.npy'))        # (N, D)
    latent_starts = np.load(pjoin(cache_dir, 'lib_latent_starts.npy'))
    latent_lengths = np.load(pjoin(cache_dir, 'lib_latent_lengths.npy'))
    with open(pjoin(cache_dir, 'lib_sample_ids.txt')) as f:
        sample_ids = [l.strip() for l in f if l.strip()]
    print(f'Library: {len(sample_ids)} captions, text_embs {text_embs.shape}')
    return text_embs, sample_ids, latent_starts, latent_lengths


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load library
    text_embs, lib_sample_ids, lib_latent_starts, lib_latent_lengths = \
        load_library(args.library_cache_dir)

    # Normalize library embeddings
    lib_norm = l2_normalize(text_embs.astype(np.float32))  # (N, D)

    # Build sample_id → lib_indices mapping for self-exclusion
    from collections import defaultdict
    sid_to_lib_idxs = defaultdict(list)
    for j, sid in enumerate(lib_sample_ids):
        sid_to_lib_idxs[sid].append(j)

    # Load training split
    if args.dataset_name == 't2m_272':
        split_file = './humanml3d_272/split/train.txt'
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset_name}')
    with cs.open(split_file) as f:
        id_list = [l.strip() for l in f if l.strip()]
    print(f'Training samples: {len(id_list)}')

    # For each training sample, precompute top-k for each caption
    ok, skip = 0, 0
    for sid in tqdm(id_list, desc='Precomputing top-k'):
        out_path = pjoin(args.output_dir, sid + '.npy')
        if os.path.exists(out_path) and not args.overwrite:
            ok += 1
            continue

        text_path = pjoin(args.text_latent_dir, sid + '.npy')
        if not os.path.exists(text_path):
            skip += 1
            continue

        text_all = np.load(text_path).astype(np.float32)  # (num_caps, D) or (D,)
        if text_all.ndim == 1:
            text_all = text_all.reshape(1, -1)
        num_caps = text_all.shape[0]

        # Normalize queries
        queries_norm = l2_normalize(text_all)  # (num_caps, D)

        # Batched matmul: (num_caps, D) × (D, N_lib) = (num_caps, N_lib)
        # Process in batches to avoid OOM
        topk_indices = np.zeros((num_caps, args.topk), dtype=np.int32)
        exclude_idxs = np.array(sid_to_lib_idxs.get(sid, []), dtype=np.int64)

        B = min(args.batch_size, num_caps)
        for start in range(0, num_caps, B):
            q_batch = queries_norm[start:start + B]            # (B, D)
            sims = q_batch @ lib_norm.T                         # (B, N_lib)
            if len(exclude_idxs) > 0:
                sims[:, exclude_idxs] = -1e6                   # exclude self
            # top-k (descending)
            topk_idx = np.argpartition(sims, -args.topk, axis=1)[:, -args.topk:]
            # sort within topk by score
            for bi in range(topk_idx.shape[0]):
                sorted_i = topk_idx[bi][np.argsort(sims[bi, topk_idx[bi]])[::-1]]
                topk_indices[start + bi] = sorted_i

        np.save(out_path, topk_indices)
        ok += 1

    print(f'\nDone. Processed: {ok}  Skipped/failed: {skip}')
    print(f'Files saved to: {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='t2m_272')
    parser.add_argument('--text_latent_dir', type=str,
                        default='./humanml3d_272/text_latents_t5')
    parser.add_argument('--library_cache_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Caption query batch size for matmul')
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()
    main(args)

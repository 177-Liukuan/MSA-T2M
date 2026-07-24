"""Standalone script to pre-build the local-RAG retrieval library.

Run once before training to avoid rebuilding the library (42k file reads) on
every training startup:

    # EXP=MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right
    python build_latent_retr_library.py \
        --motion_latent_dir  ./humanml3d_272/t2m_latents_msa_vae/$EXP/  \
        --text_latent_dir    ./humanml3d_272/text_latents_t5/            \
        --hcls_dir           ./humanml3d_272/h_cls_latents_msa_vae/$EXP/ \
        --output_cache_dir   ./humanml3d_272/latent_retr_library_cache/$EXP/

After this completes, pass --library_cache_dir to train_t2m_rag_latent_retr.py.
"""

import argparse
import os
import sys

# Make sure the repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from humanml3d_272.dataset_msa_rag_latent_retr import (
    build_library_from_data,
    save_library_cache,
    _cache_exists,
)
from humanml3d_272.dataset_msa_rag import Text2MotionMSARAGDataset


def main():
    parser = argparse.ArgumentParser(
        description='Pre-build local-RAG library cache for MotionStreamer.'
    )
    parser.add_argument('--dataset_name',      type=str, default='t2m_272')
    parser.add_argument('--motion_latent_dir', type=str, required=True,
                        help='Directory of per-sample motion latent .npy files.')
    parser.add_argument('--text_latent_dir',   type=str, required=True,
                        help='Directory of per-sample T5 text embedding .npy files.')
    parser.add_argument('--hcls_dir',          type=str, required=True,
                        help='Directory of per-sample h_cls .npy files.')
    parser.add_argument('--output_cache_dir',  type=str, required=True,
                        help='Directory to write the 5 cache files.')
    parser.add_argument('--text_embed_dim',    type=int, default=None,
                        help='Expected text embedding dim (inferred from data if None).')
    parser.add_argument('--overwrite',         action='store_true',
                        help='Re-build even if cache already exists.')
    args = parser.parse_args()

    cache_dir = args.output_cache_dir

    if not args.overwrite and _cache_exists(cache_dir):
        print(f'[build_latent_retr_library] Cache already exists at {cache_dir}.')
        print('Use --overwrite to force a rebuild.')
        return

    print('[build_latent_retr_library] Loading data entry list ...')
    base_ds = Text2MotionMSARAGDataset(
        dataset_name=args.dataset_name,
        motion_latent_dir=args.motion_latent_dir,
        text_latent_dir=args.text_latent_dir,
        hcls_dir=args.hcls_dir,
        topk=1,           # minimal; we only need the data-entry list
        exclude_self=True,
        text_embed_dim=args.text_embed_dim,
    )
    text_embed_dim = base_ds.text_embed_dim

    print(f'[build_latent_retr_library] {len(base_ds.data)} samples found.')

    (lib_text_embs, lib_sample_ids,
     lib_latents_flat, lib_latent_starts,
     lib_latent_lengths) = build_library_from_data(base_ds.data, text_embed_dim)

    save_library_cache(
        cache_dir,
        lib_text_embs, lib_sample_ids,
        lib_latents_flat, lib_latent_starts, lib_latent_lengths,
    )
    print('[build_latent_retr_library] Done.')


if __name__ == '__main__':
    main()

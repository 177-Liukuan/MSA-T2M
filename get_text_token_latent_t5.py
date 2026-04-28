"""
get_text_token_latent_t5.py
===========================
Offline pre-compute word-level T5 token embeddings for all training motions.

Output format (one .npz per motion ID)
---------------------------------------
  embs    : float32  (num_captions, max_token_len, 768)  – zero-padded
  lengths : int32    (num_captions,)                      – valid token counts
  texts   : list[str] (num_captions,)                     – source captions

These files are consumed by dataset_msa_rag_mca.py to supply the
word-level text tokens for LLaMARAGMultiTextCAWrapper.

Usage
-----
  conda activate mgpt
  python get_text_token_latent_t5.py \
      --output_dir ./humanml3d_272/text_token_latents_t5 \
      --t5_model_path sentencet5-xxl/ \
      --split train \
      --batch_size 64
"""

import os
import codecs as cs
import argparse
from os.path import join as pjoin

import numpy as np
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
#  Token-level T5 encoding
# ---------------------------------------------------------------------------

def encode_token_level(texts, st_model, batch_size=64):
    """Return list of (seq_len, D) float32 arrays, one per input text.

    Uses the underlying Transformer module of a SentenceTransformer to get
    token-level hidden states BEFORE the mean-pooling layer.

    Args
    ----
    texts     : list[str]
    st_model  : loaded SentenceTransformer instance
    batch_size: tokenization/encoding batch size
    Returns
    -------
    embs_list  : list[np.ndarray]  each (seq_len_i, D)
    """
    transformer_module = st_model[0]   # SentenceTransformers Transformer wrapper
    device = next(st_model.parameters()).device

    embs_list = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        features = transformer_module.tokenize(batch_texts)
        features = {k: v.to(device) for k, v in features.items()}
        with torch.no_grad():
            out = transformer_module.forward(features)
        token_embs = out['token_embeddings'].float()      # (B, S, D)
        attention_mask = features['attention_mask']       # (B, S)  1=valid 0=pad
        valid_lens = attention_mask.sum(dim=1).cpu().tolist()  # list[int]
        for i, length in enumerate(valid_lens):
            length = int(length)
            embs_list.append(token_embs[i, :length, :].cpu().numpy())   # (length, D)
    return embs_list


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output dir: {args.output_dir}')

    # ---- Determine data paths ----
    if args.dataset_name == 't2m_272':
        data_root = './humanml3d_272'
        text_dir = pjoin(data_root, 'texts')
        split_file = pjoin(data_root, 'split', f'{args.split}.txt')
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset_name}')

    id_list = []
    with cs.open(split_file, 'r') as f:
        for line in f.readlines():
            sid = line.strip()
            if sid:
                id_list.append(sid)
    print(f'Total motions in split ({args.split}): {len(id_list)}')

    # ---- Load T5 model ----
    print(f'Loading SentenceTransformer: {args.t5_model_path}')
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(args.t5_model_path)
    st_model.eval()
    for p in st_model.parameters():
        p.requires_grad = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st_model.to(device)
    print(f'Using device: {device}')

    # ---- Process each motion ----
    ok, skip = 0, 0
    for name in tqdm(id_list, desc='Encoding token-level embeddings'):
        out_path = pjoin(args.output_dir, name + '.npz')
        if os.path.exists(out_path) and not args.overwrite:
            ok += 1
            continue

        txt_file = pjoin(text_dir, name + '.txt')
        if not os.path.exists(txt_file):
            skip += 1
            continue

        try:
            with cs.open(txt_file) as f:
                lines = f.readlines()
            captions = []
            for line in lines:
                parts = line.strip().split('#')
                caption = parts[0].strip()
                if caption:
                    captions.append(caption)
            if not captions:
                skip += 1
                continue

            # Encode word-level token embeddings for all captions
            embs_per_cap = encode_token_level(captions, st_model, batch_size=args.batch_size)

            # Pad to common max_len within this motion's captions
            max_len = max(e.shape[0] for e in embs_per_cap)
            text_dim = embs_per_cap[0].shape[1]
            num_caps = len(captions)

            embs_padded = np.zeros((num_caps, max_len, text_dim), dtype=np.float32)
            lengths = np.zeros(num_caps, dtype=np.int32)
            for i, e in enumerate(embs_per_cap):
                L = e.shape[0]
                embs_padded[i, :L] = e
                lengths[i] = L

            np.savez_compressed(
                out_path,
                embs=embs_padded,       # (num_caps, max_len, 768)
                lengths=lengths,        # (num_caps,)
            )
            ok += 1

        except Exception as exc:
            print(f'  ERROR for {name}: {exc}')
            skip += 1
            continue

    print(f'\nDone. Processed: {ok}  Skipped/failed: {skip}')
    print(f'Files saved to: {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-compute word-level T5 token embeddings.')
    parser.add_argument('--dataset_name', type=str, default='t2m_272')
    parser.add_argument('--split', type=str, default='train',
                        help='train | val | test')
    parser.add_argument('--output_dir', type=str,
                        default='./humanml3d_272/text_token_latents_t5',
                        help='Directory to save .npz token embedding files')
    parser.add_argument('--t5_model_path', type=str, default='sentencet5-xxl/',
                        help='Path to SentenceTransformer T5 model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for T5 encoding')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Re-compute even if output file already exists')
    args = parser.parse_args()
    main(args)

# eval_msa_t2m_rag_latent_retr.py
# Evaluation script for LLaMARAGLatentRetrWrapper (local-RAG latent retrieval variant).
# Retrieves motion latents via T5 text-cosine-similarity from a pre-built 5-file
# library cache and injects them as Flamingo CA KV into the backbone.
# The global h_cls prefix RAG is retained from the base RAG wrapper.
import os
import sys
import json
import argparse
import glob
import warnings
import codecs as cs
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model import LLaMARAGWrapper
from models.llama_rag_model_latent_retr import (
    LLaMARAGLatentRetrWrapper,
    LLaMARAGLatentRetrGatedWrapper,
)
import models.msa_vae as msa_vae
import options.option_transformer as option_trans
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from humanml3d_272 import dataset_eval_t2m

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# ---------------------------------------------------------------------------
#  Offline text embedding lookup (identical to eval_msa_t2m_rag_mca.py)
# ---------------------------------------------------------------------------

class OfflineTextEmbeddingLookup:
    """Map caption string -> offline text embedding from precomputed npy files."""

    def __init__(self, data_root, text_latent_dir, split='test', text_embed_dim=768):
        self.text_embed_dim = int(text_embed_dim)
        self.caption_to_emb = {}
        self.miss_total = 0
        self.miss_warn_budget = 20

        split_file = os.path.join(data_root, 'split', f'{split}.txt')
        text_dir = os.path.join(data_root, 'texts')

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                sid = line.strip()
                if sid:
                    id_list.append(sid)

        loaded_pairs = 0

        def add_captions_with_embeddings(captions, emb):
            nonlocal loaded_pairs
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            if emb.shape[-1] != self.text_embed_dim or emb.shape[0] == 0:
                return
            max_n = min(len(captions), emb.shape[0])
            for i in range(max_n):
                key = self._normalize_caption(captions[i])
                if key not in self.caption_to_emb:
                    self.caption_to_emb[key] = emb[i]
                    loaded_pairs += 1
            if len(captions) > max_n:
                proxy = emb.mean(axis=0)
                for i in range(max_n, len(captions)):
                    key = self._normalize_caption(captions[i])
                    if key not in self.caption_to_emb:
                        self.caption_to_emb[key] = proxy
                        loaded_pairs += 1

        for sid in id_list:
            txt_path = os.path.join(text_dir, sid + '.txt')
            emb_path = os.path.join(text_latent_dir, sid + '.npy')
            if not (os.path.exists(txt_path) and os.path.exists(emb_path)):
                continue
            captions = []
            with cs.open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('#')
                    if len(parts) >= 1 and parts[0].strip():
                        captions.append(parts[0].strip())
            emb = np.load(emb_path).astype(np.float32)
            add_captions_with_embeddings(captions, emb)

        if len(self.caption_to_emb) == 0:
            txt_files = glob.glob(os.path.join(text_dir, '*.txt'))
            for txt_path in txt_files:
                sid = os.path.splitext(os.path.basename(txt_path))[0]
                emb_path = os.path.join(text_latent_dir, sid + '.npy')
                if not os.path.exists(emb_path):
                    continue
                captions = []
                with cs.open(txt_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split('#')
                        if len(parts) >= 1 and parts[0].strip():
                            captions.append(parts[0].strip())
                emb = np.load(emb_path).astype(np.float32)
                add_captions_with_embeddings(captions, emb)

        if len(self.caption_to_emb) == 0:
            raise RuntimeError('OfflineTextEmbeddingLookup built empty caption map.')

        print(
            f'Offline caption embedding map ready: {len(self.caption_to_emb)} unique captions, '
            f'loaded_pairs={loaded_pairs}, dim={self.text_embed_dim}'
        )

    @staticmethod
    def _normalize_caption(text):
        return ' '.join(text.strip().lower().split())

    def batch_lookup(self, text_list, device):
        arr = np.zeros((len(text_list), self.text_embed_dim), dtype=np.float32)
        missing = 0
        global_proxy = None
        if len(self.caption_to_emb) > 0:
            global_proxy = np.mean(np.stack(list(self.caption_to_emb.values())), axis=0)
        for i, t in enumerate(text_list):
            key = self._normalize_caption(t)
            if key in self.caption_to_emb:
                arr[i] = self.caption_to_emb[key]
            else:
                missing += 1
                if global_proxy is not None:
                    arr[i] = global_proxy
        if missing > 0:
            self.miss_total += missing
            if self.miss_warn_budget > 0:
                print(f'[warn] Offline caption miss: {missing}/{len(text_list)} -> fallback to proxy embedding. total_miss={self.miss_total}')
                self.miss_warn_budget -= 1
            elif self.miss_total % 200 == 0:
                print(f'[warn] Offline caption cumulative misses: {self.miss_total}')
        return torch.from_numpy(arr).float().to(device)


# ---------------------------------------------------------------------------
#  Global h_cls retriever (identical to eval_msa_t2m_rag_mca.py)
# ---------------------------------------------------------------------------

class RAGRetriever:
    """Build in-memory h_cls retrieval library and provide top-k retrieval."""

    def __init__(self, hcls_dir, topk=3, text_embed_dim=768, device=torch.device('cuda')):
        self.hcls_dir = hcls_dir
        self.topk = topk
        self.device = device
        self.text_embed_dim = int(text_embed_dim)

        hcls_files = sorted(glob.glob(os.path.join(hcls_dir, '*.npy')))
        if len(hcls_files) == 0:
            raise RuntimeError(f'No h_cls npy files found in: {hcls_dir}')

        vectors = []
        for path in hcls_files:
            h = np.load(path).astype(np.float32)
            if h.ndim == 2:
                h = h.mean(axis=0)
            else:
                h = h.reshape(-1)
            if h.shape[0] != self.text_embed_dim:
                continue
            vectors.append(h)

        if len(vectors) == 0:
            raise RuntimeError(f'No valid {self.text_embed_dim}-d h_cls vectors found in: {hcls_dir}')

        lib = torch.from_numpy(np.stack(vectors)).float().to(device)
        self.library = lib
        self.library_norm = self._normalize(lib)

    @staticmethod
    def _normalize(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, text_emb):
        """text_emb: [B, D] -> (top_hcls [B,K,D], top_scores [B,K])"""
        q = self._normalize(text_emb)
        sim = torch.matmul(q, self.library_norm.t())
        k = min(self.topk, sim.shape[1])
        top_scores, top_indices = torch.topk(sim, k=k, dim=1)
        top_hcls = self.library[top_indices]

        if k < self.topk:
            pad_h = torch.zeros(
                text_emb.shape[0], self.topk - k, self.text_embed_dim,
                device=text_emb.device, dtype=top_hcls.dtype
            )
            pad_s = torch.full(
                (text_emb.shape[0], self.topk - k), -1e6,
                device=text_emb.device, dtype=top_scores.dtype
            )
            top_hcls = torch.cat([top_hcls, pad_h], dim=1)
            top_scores = torch.cat([top_scores, pad_s], dim=1)

        return top_hcls, top_scores


# ---------------------------------------------------------------------------
#  Local motion latent retriever (new — loads the 5-file library cache)
# ---------------------------------------------------------------------------

class RAGLatentRetriever:
    """Load the 5-file pre-built library cache and provide top-k motion latent retrieval.

    Library cache files (under library_cache_dir/):
      lib_text_embs.npy      — (N_caps, D) float32, retrieval keys
      lib_sample_ids.txt     — N_caps lines, sample_id per entry (for self-exclusion)
      lib_latents_flat.npy   — (sum_F, latent_dim) float32, concatenated latents
      lib_latent_starts.npy  — (N_caps,) int64, start offset per entry in flat array
      lib_latent_lengths.npy — (N_caps,) int64, frame count per entry
    """

    def __init__(self, library_cache_dir, topk=3, device=torch.device('cuda')):
        self.topk = topk
        self.device = device

        def _p(name):
            return os.path.join(library_cache_dir, name)

        for fname in ('lib_text_embs.npy', 'lib_sample_ids.txt',
                      'lib_latents_flat.npy', 'lib_latent_starts.npy',
                      'lib_latent_lengths.npy'):
            if not os.path.exists(_p(fname)):
                raise FileNotFoundError(
                    f'Library cache file missing: {_p(fname)}\n'
                    f'Run build_latent_retr_library.py first.'
                )

        lib_text_embs = np.load(_p('lib_text_embs.npy')).astype(np.float32)
        self.lib_text_embs = torch.from_numpy(lib_text_embs).float().to(device)
        self.lib_text_embs_norm = self._normalize(self.lib_text_embs)

        with open(_p('lib_sample_ids.txt'), 'r') as f:
            self.lib_sample_ids = [line.strip() for line in f.readlines()]

        self.lib_latents_flat = np.load(_p('lib_latents_flat.npy')).astype(np.float32)
        self.lib_latent_starts = np.load(_p('lib_latent_starts.npy')).astype(np.int64)
        self.lib_latent_lengths = np.load(_p('lib_latent_lengths.npy')).astype(np.int64)
        self.latent_dim = self.lib_latents_flat.shape[1]

        # Build exclusion map: sample_id -> set of library indices
        self.sample_id_to_indices = {}
        for i, sid in enumerate(self.lib_sample_ids):
            if sid not in self.sample_id_to_indices:
                self.sample_id_to_indices[sid] = []
            self.sample_id_to_indices[sid].append(i)

        print(
            f'RAGLatentRetriever: library loaded from {library_cache_dir}\n'
            f'  N_caps={len(lib_text_embs)}, latent_dim={self.latent_dim}, topk={topk}'
        )

    @staticmethod
    def _normalize(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, text_emb, exclude_sample_ids=None):
        """Retrieve top-k motion latent sequences for a batch of text embeddings.

        Args:
            text_emb           : (B, D) tensor on self.device
            exclude_sample_ids : list[str|None] of length B, sample IDs to exclude
                                 (pass None or a list of Nones to skip exclusion)
        Returns:
            retr_latents     : (B, L_max, latent_dim) float32 on self.device
            retr_latent_lens : (B,) int64 on self.device
        """
        bsz = text_emb.shape[0]
        q = self._normalize(text_emb)
        sim = torch.matmul(q, self.lib_text_embs_norm.t())  # (B, N_caps)

        all_latents = []
        all_lens = []

        for b in range(bsz):
            sim_b = sim[b].clone()  # (N_caps,)

            # Self-exclusion
            if exclude_sample_ids is not None and exclude_sample_ids[b] is not None:
                excl_sid = str(exclude_sample_ids[b])
                if excl_sid in self.sample_id_to_indices:
                    for idx in self.sample_id_to_indices[excl_sid]:
                        sim_b[idx] = -1e6

            k = min(self.topk, sim_b.shape[0])
            _, top_indices = torch.topk(sim_b, k=k, dim=0)
            top_indices_np = top_indices.cpu().numpy()

            # Concatenate latent slices from flat array
            latent_parts = []
            for idx in top_indices_np:
                s = int(self.lib_latent_starts[idx])
                length = int(self.lib_latent_lengths[idx])
                latent_parts.append(self.lib_latents_flat[s:s + length])

            if latent_parts:
                concat = np.concatenate(latent_parts, axis=0)
            else:
                concat = np.zeros((1, self.latent_dim), dtype=np.float32)

            all_latents.append(concat)
            all_lens.append(len(concat))

        # Pad batch to max length
        L_max = max(all_lens)
        padded = np.zeros((bsz, L_max, self.latent_dim), dtype=np.float32)
        for b, (lat, l) in enumerate(zip(all_latents, all_lens)):
            padded[b, :l] = lat

        retr_latents = torch.from_numpy(padded).float().to(self.device)
        retr_latent_lens = torch.tensor(all_lens, dtype=torch.long, device=self.device)
        return retr_latents, retr_latent_lens


# ---------------------------------------------------------------------------
#  Eval sampler adapter for latent retrieval RAG model
# ---------------------------------------------------------------------------

class RAGEvalSampler:
    """Adapter exposing sample_for_eval_CFG interface for eval_trans."""

    def __init__(
        self,
        rag_model,
        retriever,
        latent_retriever,
        empty_text_emb,
        latent_dim=16,
        device=torch.device('cuda'),
        reference_end_latent=None,
        stop_threshold=3.0,
        enable_stopping=True,
        text_source='offline',
        text_lookup=None,
        text_encoder=None,
        text_embed_dim=768,
        disable_rag=False,
        disable_latent_retr=False,
        cfg_scale_retr=1.0,
        use_random_topk_inference=False,
    ):
        self.rag_model = rag_model
        self.retriever = retriever                # global h_cls retriever (RAGRetriever)
        self.latent_retriever = latent_retriever  # local latent retriever (RAGLatentRetriever)
        self.text_lookup = text_lookup
        self.text_encoder = text_encoder
        self.text_source = text_source
        self.text_embed_dim = int(text_embed_dim)
        self.empty_text_emb = empty_text_emb
        self.latent_dim = latent_dim
        self.device = device
        self.reference_end_latent = reference_end_latent
        self.stop_threshold = float(stop_threshold)
        self.enable_stopping = bool(enable_stopping)
        self.disable_rag = bool(disable_rag)
        self.disable_latent_retr = bool(disable_latent_retr)
        self.cfg_scale_retr = float(cfg_scale_retr)
        self.use_random_topk_inference = bool(use_random_topk_inference)

    def eval(self):
        self.rag_model.eval()
        return self

    @torch.no_grad()
    def sample_for_eval_CFG(self, text, length=196, tokenize_model=None,
                             device=torch.device('cuda'), unit_length=4, cfg=4.0):
        _ = tokenize_model
        _ = device

        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = list(text)

        # ── text embedding (sentence-level, 768-d) ──────────────────────────
        if self.text_source == 'online_t5':
            if self.text_encoder is None:
                raise RuntimeError('text_encoder is required when text_source=online_t5')
            text_np = np.asarray(self.text_encoder.encode(text_list), dtype=np.float32)
            text_emb = torch.from_numpy(text_np).float().to(self.device)
        else:
            if self.text_lookup is None:
                raise RuntimeError('text_lookup is required when text_source=offline')
            text_emb = self.text_lookup.batch_lookup(text_list, self.device)

        if text_emb.shape[-1] != self.text_embed_dim:
            raise ValueError(
                f'text embedding dim mismatch: got {text_emb.shape[-1]}, expected {self.text_embed_dim}'
            )

        # ── global h_cls retrieval ───────────────────────────────────────────
        if self.disable_rag:
            top_hcls = None
            top_scores = None
        else:
            top_hcls, top_scores = self.retriever.retrieve(text_emb)
            if self.use_random_topk_inference and top_hcls.shape[1] > 1:
                B, K, D = top_hcls.shape
                rand_ks = torch.randint(0, K, (B,), device=top_hcls.device)
                top_hcls = top_hcls[torch.arange(B, device=top_hcls.device), rand_ks].unsqueeze(1)
                top_scores = top_scores[torch.arange(B, device=top_scores.device), rand_ks].unsqueeze(1)

        # ── local motion latent retrieval ────────────────────────────────────
        retr_latents_batch = None
        retr_latent_lens_batch = None
        if self.latent_retriever is not None and not self.disable_latent_retr:
            # No sample-ID exclusion during test-set evaluation (test samples are
            # not in the training library). Pass None to skip exclusion.
            exclude_ids = [None] * len(text_list)
            retr_latents_batch, retr_latent_lens_batch = self.latent_retriever.retrieve(
                text_emb, exclude_sample_ids=exclude_ids
            )

        # ── autoregressive sampling with CFG ─────────────────────────────────
        max_token_len = max(1, int(length) // unit_length)
        xs = None
        bsz = text_emb.shape[0]
        finished = torch.zeros(bsz, dtype=torch.bool, device=self.device)

        # Proper 2-forward velocity-space CFG (joint dropout mode).
        # z_cond  = forward(real_text, real_retr)   — always in training distribution
        # z_uncond = forward(null_text, null_retr)  — always in training distribution
        # CFG scaling done in velocity space inside diff_loss.sample(cfg=cfg).
        # 3-forward pre-mixed hidden-space CFG is NOT used because
        # z_guided = s*z_both - (s-1)*z_retr is an OOD extrapolation for the
        # diffusion head, causing flying/physically-invalid motions.
        cond_mask_eval   = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        uncond_mask_eval = torch.ones(bsz,  dtype=torch.bool, device=self.device)

        for _k in range(max_token_len):
            if xs is None:
                prefix = torch.zeros((bsz, 0, self.latent_dim), device=self.device, dtype=torch.float32)
            else:
                prefix = xs

            # Forward 1: (real_text, real_retr)
            z_cond = self.rag_model.forward(
                prefix, text_emb, top_hcls, top_scores,
                cfg_drop_mask=cond_mask_eval,
                empty_text_emb=self.empty_text_emb,
                retr_latents=retr_latents_batch,
                retr_latent_lens=retr_latent_lens_batch,
                retr_cfg_drop_mask=None,   # fallback → keep retr (joint dropout)
            )[:, -1, :]

            # Forward 2: (null_text, null_retr)
            z_uncond = self.rag_model.forward(
                prefix, text_emb, top_hcls, top_scores,
                cfg_drop_mask=uncond_mask_eval,
                empty_text_emb=self.empty_text_emb,
                retr_latents=retr_latents_batch,
                retr_latent_lens=retr_latent_lens_batch,
                retr_cfg_drop_mask=None,   # fallback → null retr (joint dropout)
            )[:, -1, :]

            mix_hidden = torch.cat([z_cond, z_uncond], dim=0)  # (2*bsz, D)
            sampled = self.rag_model.base_model.diff_loss.sample(
                mix_hidden, temperature=1.0, cfg=cfg
            )
            next_token = sampled.chunk(2, dim=0)[0]  # (bsz, latent_dim)

            next_token = next_token.unsqueeze(1)
            xs = next_token if xs is None else torch.cat([xs, next_token], dim=1)

            # Continuous stopping via reference end latent distance
            if self.enable_stopping and self.reference_end_latent is not None:
                cur = next_token.squeeze(1)
                dist_l2 = torch.linalg.norm(
                    cur - self.reference_end_latent.unsqueeze(0), dim=-1
                )
                finished = finished | (dist_l2 < self.stop_threshold)
                if torch.all(finished):
                    break

        if xs is None:
            xs = torch.zeros((bsz, 1, self.latent_dim), device=self.device, dtype=torch.float32)
        return xs


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    extra_parser = argparse.ArgumentParser(add_help=False)
    # Paths
    extra_parser.add_argument('--hcls_dir', type=str,
                              default='./humanml3d_272/h_cls_latents_msa_vae/exp')
    extra_parser.add_argument('--text_latent_dir', type=str,
                              default='./humanml3d_272/text_latents_t5')
    extra_parser.add_argument('--empty_text_path', type=str,
                              default='./humanml3d_272/text_latents_t5/empty_text_embedding.npy')
    extra_parser.add_argument('--library_cache_dir', type=str, required=True,
                              help='Path to pre-built 5-file latent retrieval library cache.')
    # Retrieval
    extra_parser.add_argument('--retrieval_topk', type=int, default=3,
                              help='Top-k for global h_cls retrieval.')
    extra_parser.add_argument('--latent_retr_topk', type=int, default=3,
                              help='Top-k for local motion latent retrieval.')
    extra_parser.add_argument('--latent_dim', type=int, default=16,
                              help='Motion latent dimension (must match checkpoint).')
    # CFG / text
    extra_parser.add_argument('--cfg_scale', type=float, default=4.0)
    extra_parser.add_argument('--cfg_scale_retr', type=float, default=1.0,
                              help='Retrieval CFG scale for dual-CFG 3-forward inference. '
                                   '1.0=mild retrieval prior; 0.0=text-only CFG given retr; '
                                   '>1.0=stronger retrieval amplification.')
    extra_parser.add_argument('--text_embed_dim', type=int, default=768)
    extra_parser.add_argument('--eval_split', type=str, default='test',
                              choices=['test', 'val'])
    extra_parser.add_argument('--text_source', type=str, default='online_t5',
                              choices=['offline', 'online_t5'])
    extra_parser.add_argument('--t5_model_path', type=str, default='sentencet5-xxl/')
    # Ablation flags
    extra_parser.add_argument('--disable_rag', action='store_true', default=False,
                              help='Ablation: disable global h_cls retrieval.')
    extra_parser.add_argument('--disable_latent_retr', action='store_true', default=False,
                              help='Ablation: disable local latent CA retrieval.')
    extra_parser.add_argument('--disable_ema', action='store_true', default=False,
                              help='Do not use EMA weights even if present in checkpoint.')
    extra_parser.add_argument('--use_random_topk_inference', action='store_true', default=False,
                              help='Inference diversity: randomly pick one h_cls from top-K instead of weighted pooling.')
    # CA architecture (must match training)
    extra_parser.add_argument('--ca_n_head', type=int, default=0,
                              help='Number of CA heads (0=auto, same as backbone).')
    extra_parser.add_argument('--ca_every_n_layers', type=int, default=1,
                              help='CA block insertion interval used during training '
                                   '(e.g. 1=every layer, 2=every 2 layers).')
    extra_parser.add_argument('--ca_insertion_mode', type=str, default='before_sa',
                              choices=['before_sa', 'after_sa', 'late_after_sa'],
                              help='Must match training config. '
                                   'before_sa=A (default), after_sa=B, late_after_sa=C.')
    # MSA-VAE architecture
    extra_parser.add_argument('--trans_d_model', type=int, default=768)
    extra_parser.add_argument('--trans_nhead', type=int, default=8)
    extra_parser.add_argument('--trans_enc_layers', type=int, default=6)
    extra_parser.add_argument('--trans_dec_layers', type=int, default=6)
    extra_parser.add_argument('--trans_ff_size', type=int, default=2048)
    extra_parser.add_argument('--trans_dropout', type=float, default=0.1)
    extra_parser.add_argument('--clip_dim', type=int, default=768)
    # Continuous stopping
    extra_parser.add_argument('--reference_end_latent_path', type=str, default='')
    extra_parser.add_argument('--stop_threshold', type=float, default=3.0)
    extra_parser.add_argument('--enable_stopping', dest='enable_stopping',
                              action='store_true', default=True)
    extra_parser.add_argument('--disable_stopping', dest='enable_stopping',
                              action='store_false')

    custom_args, remaining = extra_parser.parse_known_args()

    argv_backup = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = option_trans.get_args_parser()
    finally:
        sys.argv = argv_backup

    args.hcls_dir = custom_args.hcls_dir
    args.text_latent_dir = custom_args.text_latent_dir
    args.empty_text_path = custom_args.empty_text_path
    args.library_cache_dir = custom_args.library_cache_dir
    args.retrieval_topk = custom_args.retrieval_topk
    args.latent_retr_topk = custom_args.latent_retr_topk
    args.latent_dim_retr = custom_args.latent_dim  # separate from args.latent_dim (VAE)
    args.cfg_scale = custom_args.cfg_scale
    args.cfg_scale_retr = custom_args.cfg_scale_retr
    args.text_embed_dim = custom_args.text_embed_dim
    args.eval_split = custom_args.eval_split
    args.text_source = custom_args.text_source
    args.t5_model_path = custom_args.t5_model_path
    args.disable_rag = custom_args.disable_rag
    args.disable_latent_retr = custom_args.disable_latent_retr
    args.use_ema = not custom_args.disable_ema
    args.use_random_topk_inference = custom_args.use_random_topk_inference
    args.ca_n_head = custom_args.ca_n_head
    args.ca_every_n_layers = custom_args.ca_every_n_layers
    args.ca_insertion_mode = custom_args.ca_insertion_mode
    args.trans_d_model = custom_args.trans_d_model
    args.trans_nhead = custom_args.trans_nhead
    args.trans_enc_layers = custom_args.trans_enc_layers
    args.trans_dec_layers = custom_args.trans_dec_layers
    args.trans_ff_size = custom_args.trans_ff_size
    args.trans_dropout = custom_args.trans_dropout
    args.clip_dim = custom_args.clip_dim
    args.reference_end_latent_path = custom_args.reference_end_latent_path
    args.stop_threshold = custom_args.stop_threshold
    args.enable_stopping = custom_args.enable_stopping
    return args


# ---------------------------------------------------------------------------
#  Utilities (identical to eval_msa_t2m_rag_mca.py)
# ---------------------------------------------------------------------------

def load_state_strip_module(state_dict):
    new_state = {}
    for key, value in state_dict.items():
        if key.split('.')[0] == 'module':
            new_key = '.'.join(key.split('.')[1:])
        else:
            new_key = key
        new_state[new_key] = value
    return new_state


def resolve_existing_path(path, must_be_dir=False):
    checker = os.path.isdir if must_be_dir else os.path.exists
    if checker(path):
        return path
    if not os.path.isabs(path):
        alt = os.path.join('..', path)
        if checker(alt):
            return alt
    return path


def resolve_data_root():
    candidates = ['./humanml3d_272', '../humanml3d_272']
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'texts')) and os.path.isdir(os.path.join(c, 'split')):
            return c
    return './humanml3d_272'


def resolve_reference_end_latent(args):
    candidates = []
    if args.reference_end_latent_path:
        candidates.append(args.reference_end_latent_path)
    candidates.append(os.path.join(args.latent_dir,
                                   f'reference_end_latent_msa_vae_{args.dataname}.npy'))
    candidates.append(f'reference_end_latent_msa_vae_{args.dataname}.npy')
    candidates.append(f'reference_end_latent_{args.dataname}.npy')
    for p in candidates:
        if p and os.path.exists(p):
            arr = np.load(p).astype(np.float32).reshape(-1)
            return p, arr
    return None, None


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main():
    os.chdir('Evaluator_272')
    sys.path.insert(0, os.getcwd())

    comp_device = torch.device('cuda')
    args = parse_args()

    args.resume_pth = resolve_existing_path(args.resume_pth) if args.resume_pth is not None else None
    args.resume_trans = resolve_existing_path(args.resume_trans) if args.resume_trans is not None else None
    args.latent_dir = resolve_existing_path(args.latent_dir, must_be_dir=True)
    args.text_latent_dir = resolve_existing_path(args.text_latent_dir, must_be_dir=True)
    args.hcls_dir = resolve_existing_path(args.hcls_dir, must_be_dir=True)
    args.library_cache_dir = resolve_existing_path(args.library_cache_dir, must_be_dir=True)
    args.empty_text_path = resolve_existing_path(args.empty_text_path)
    args.t5_model_path = resolve_existing_path(args.t5_model_path, must_be_dir=True)
    data_root = resolve_data_root()

    torch.manual_seed(args.seed)

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    val_loader = dataset_eval_t2m.DATALoader(args.dataname, args.eval_split == 'test', 32)

    # ── MSA-VAE decoder backbone ─────────────────────────────────────────────
    clip_range = [-30, 20]
    net = msa_vae.MSA_HumanVAE(
        hidden_size=args.hidden_size,
        down_t=args.down_t,
        stride_t=args.stride_t,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation='relu',
        latent_dim=args.latent_dim,
        clip_range=clip_range,
        trans_d_model=args.trans_d_model,
        trans_nhead=args.trans_nhead,
        trans_enc_layers=args.trans_enc_layers,
        trans_dec_layers=args.trans_dec_layers,
        trans_ff_size=args.trans_ff_size,
        trans_dropout=args.trans_dropout,
        clip_dim=args.clip_dim,
    )

    print(f'Loading MSA-VAE checkpoint from {args.resume_pth}')
    ckpt_vae = torch.load(args.resume_pth, map_location='cpu')
    vae_state = ckpt_vae['net'] if (isinstance(ckpt_vae, dict) and 'net' in ckpt_vae) else ckpt_vae
    net.load_state_dict(vae_state, strict=True)
    net.eval()
    net.to(comp_device)

    # ── RAG model (LLaMARAGLatentRetrWrapper) ────────────────────────────────
    latent_dim_retr = getattr(args, 'latent_dim_retr', args.latent_dim)
    config = LLaMAHFConfig.from_name('Normal_size')
    config.block_size = 78
    base_model = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, comp_device)
    ca_n_head = getattr(args, 'ca_n_head', 0)

    if args.resume_trans is None:
        raise ValueError('Please provide --resume-trans for RAG checkpoint.')
    print(f'Loading RAG checkpoint from {args.resume_trans}')
    ckpt = torch.load(args.resume_trans, map_location='cpu')

    trans_key = 'trans_ema' if args.use_ema and ('trans_ema' in ckpt) else 'trans'
    rag_key   = 'rag_ema'   if args.use_ema and ('rag_ema'   in ckpt) else 'rag'

    # Auto-detect ca_every_n_layers from checkpoint (count CA blocks)
    # Must happen BEFORE model construction to ensure correct number of CA blocks.
    _rag_sd = load_state_strip_module(ckpt[rag_key]) if rag_key in ckpt else {}
    _detected_ca_every = getattr(args, 'ca_every_n_layers', 1)
    _n_ca_ckpt = sum(1 for k in _rag_sd if k.startswith('ca_blocks.') and k.endswith('.norm1.weight'))
    if _n_ca_ckpt > 0:
        ca_mode = getattr(args, 'ca_insertion_mode', 'before_sa')
        if ca_mode == 'late_after_sa':
            # CA blocks are packed into the second half of layers only
            second_half = config.n_layer - (config.n_layer // 2)
            _detected_ca_every = max(1, second_half // _n_ca_ckpt)
        else:
            # Uniform distribution across all layers
            for n in range(1, config.n_layer + 1):
                if config.n_layer // n == _n_ca_ckpt:
                    _detected_ca_every = n
                    break
        if _detected_ca_every != getattr(args, 'ca_every_n_layers', 1):
            print(
                f'[INFO] Checkpoint has {_n_ca_ckpt} CA blocks → '
                f'ca_every_n_layers auto-detected as {_detected_ca_every} '
                f'(args had {args.ca_every_n_layers}). Using detected value.'
            )

    # Build model with default ff_mult=2; auto-detect and rebuild if checkpoint differs
    rag_model = LLaMARAGLatentRetrWrapper(
        base_model=base_model,
        model_dim=config.n_embd,
        disable_rag=args.disable_rag,
        latent_dim=latent_dim_retr,
        ca_every_n_layers=max(1, _detected_ca_every),
        ca_n_head=ca_n_head if ca_n_head > 0 else None,
        ff_mult=2,
        disable_latent_retr=args.disable_latent_retr,
        ca_insertion_mode=getattr(args, 'ca_insertion_mode', 'before_sa'),
    )

    # Auto-detect ff_mult from checkpoint shape to handle mismatches gracefully
    _ff_key = next((k for k in _rag_sd if 'ca_blocks' in k and 'ff_in_proj.weight' in k), None)
    if _ff_key is not None:
        _ckpt_inner_dim = _rag_sd[_ff_key].shape[0]
        _model_inner_dim = rag_model.ca_blocks[0].ff_in_proj.weight.shape[0]
        if _ckpt_inner_dim != _model_inner_dim:
            _detected_ff_mult = _ckpt_inner_dim // config.n_embd
            print(
                f'[INFO] Checkpoint ff_mult={_detected_ff_mult} (inner_dim={_ckpt_inner_dim}) '
                f'differs from model default (inner_dim={_model_inner_dim}). '
                f'Rebuilding model with ff_mult={_detected_ff_mult}.'
            )
            rag_model = LLaMARAGLatentRetrWrapper(
                base_model=base_model,
                model_dim=config.n_embd,
                disable_rag=args.disable_rag,
                latent_dim=latent_dim_retr,
                ca_every_n_layers=max(1, _detected_ca_every),
                ca_n_head=ca_n_head if ca_n_head > 0 else None,
                ff_mult=_detected_ff_mult,
                disable_latent_retr=args.disable_latent_retr,
                ca_insertion_mode=getattr(args, 'ca_insertion_mode', 'before_sa'),
            )

    if trans_key in ckpt:
        rag_model.base_model.load_state_dict(
            load_state_strip_module(ckpt[trans_key]), strict=False
        )
    else:
        raise KeyError('RAG checkpoint missing key: trans/trans_ema')

    if rag_key in ckpt:
        rag_model.load_state_dict(load_state_strip_module(ckpt[rag_key]), strict=False)
    elif not args.disable_rag and not args.disable_latent_retr:
        raise KeyError('RAG checkpoint missing key: rag/rag_ema')
    else:
        logger.info('Checkpoint has no rag key; running in ablation mode.')

    if args.use_ema:
        logger.info(f'EMA eval enabled. Loaded keys: {trans_key}, {rag_key}')

    rag_model.eval()
    rag_model.to(comp_device)

    # ── Empty text embedding for CFG ─────────────────────────────────────────
    empty_text_path = args.empty_text_path
    if not os.path.exists(empty_text_path):
        for p in [
            os.path.join(args.text_latent_dir, 'empty_text_embedding.npy'),
            os.path.join(args.text_latent_dir, 'empty_cfg_text_t5.npy'),
            os.path.join(args.text_latent_dir, 'empty_cfg_text_clip.npy'),
        ]:
            if os.path.exists(p):
                empty_text_path = p
                break

    if not os.path.exists(empty_text_path):
        raise FileNotFoundError(f'empty cfg text file not found: {args.empty_text_path}')

    empty_text_emb = (
        torch.from_numpy(np.load(empty_text_path).astype(np.float32))
        .reshape(-1).to(comp_device)
    )
    if empty_text_emb.shape[0] != args.text_embed_dim:
        raise ValueError(
            f'empty text embedding dim must be {args.text_embed_dim}, '
            f'got {empty_text_emb.shape[0]} from {empty_text_path}'
        )

    # ── Text source ──────────────────────────────────────────────────────────
    text_lookup = None
    text_encoder = None
    if args.text_source == 'online_t5':
        from sentence_transformers import SentenceTransformer
        text_encoder = SentenceTransformer(args.t5_model_path)
        text_encoder.eval()
        logger.info(f'Text source: online_t5, model={args.t5_model_path}')
    else:
        text_lookup = OfflineTextEmbeddingLookup(
            data_root=data_root,
            text_latent_dir=args.text_latent_dir,
            split=args.eval_split,
            text_embed_dim=args.text_embed_dim,
        )
        logger.info('Text source: offline precomputed latents')

    # ── Global h_cls retriever ───────────────────────────────────────────────
    retriever = None
    if not args.disable_rag:
        retriever = RAGRetriever(
            args.hcls_dir,
            topk=args.retrieval_topk,
            text_embed_dim=args.text_embed_dim,
            device=comp_device,
        )
    else:
        logger.info('Global h_cls retrieval disabled (ablation).')

    # ── Local latent retriever ───────────────────────────────────────────────
    latent_retriever = None
    if not args.disable_latent_retr:
        latent_retriever = RAGLatentRetriever(
            library_cache_dir=args.library_cache_dir,
            topk=args.latent_retr_topk,
            device=comp_device,
        )
    else:
        logger.info('Local latent retrieval disabled (ablation).')

    # ── Reference end latent for continuous stopping ─────────────────────────
    reference_path, reference_end = resolve_reference_end_latent(args)
    reference_end_latent = None
    if args.enable_stopping:
        if reference_end is None:
            raise FileNotFoundError(
                'Cannot locate reference end latent. Provide --reference_end_latent_path '
                'or ensure it exists under --latent_dir.'
            )
        reference_end_latent = torch.from_numpy(reference_end).float().to(comp_device)
        if reference_end_latent.shape[0] != args.latent_dim:
            raise ValueError(
                f'reference end latent dim mismatch: got {reference_end_latent.shape[0]}, '
                f'expected {args.latent_dim}'
            )
        logger.info(
            f'Continuous stopping: reference={reference_path}, threshold={args.stop_threshold}'
        )
    else:
        logger.info('Continuous stopping disabled.')

    # ── Build eval sampler adapter ───────────────────────────────────────────
    trans_for_eval = RAGEvalSampler(
        rag_model=rag_model,
        retriever=retriever,
        latent_retriever=latent_retriever,
        empty_text_emb=empty_text_emb,
        latent_dim=args.latent_dim,
        device=comp_device,
        reference_end_latent=reference_end_latent,
        stop_threshold=args.stop_threshold,
        enable_stopping=args.enable_stopping,
        text_source=args.text_source,
        text_lookup=text_lookup,
        text_encoder=text_encoder,
        text_embed_dim=args.text_embed_dim,
        disable_rag=args.disable_rag,
        disable_latent_retr=args.disable_latent_retr,
        cfg_scale_retr=args.cfg_scale_retr,
        use_random_topk_inference=getattr(args, "use_random_topk_inference", False),
    )

    # ── Load evaluator ───────────────────────────────────────────────────────
    from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
    from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

    modelpath = './deps/distilbert-base-uncased'
    textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
    motionencoder = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)

    ckpt_path = '../Evaluator_272/epoch=99.ckpt'
    print(f'Loading evaluator checkpoint from {ckpt_path}')
    ckpt_eval = torch.load(ckpt_path)

    textencoder_ckpt = {
        k.replace('textencoder.', ''): v
        for k, v in ckpt_eval['state_dict'].items()
        if k.split('.')[0] == 'textencoder'
    }
    textencoder.load_state_dict(textencoder_ckpt, strict=True)
    textencoder.eval()
    textencoder.to(comp_device)

    motionencoder_ckpt = {
        k.replace('motionencoder.', ''): v
        for k, v in ckpt_eval['state_dict'].items()
        if k.split('.')[0] == 'motionencoder'
    }
    motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
    motionencoder.eval()
    motionencoder.to(comp_device)

    evaluator = [textencoder, motionencoder]

    # ── Run evaluation ───────────────────────────────────────────────────────
    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger = (
        eval_trans.evaluation_transformer_272_single(
            val_loader,
            net,
            trans_for_eval,
            tokenize_model=None,
            logger=logger,
            evaluator=evaluator,
            cfg=args.cfg_scale,
        )
    )

    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)

    logger.info('final result:')
    logger.info(f'fid: {fid}')
    logger.info(f'div: {div}')
    logger.info(f'top1: {top1}')
    logger.info(f'top2: {top2}')
    logger.info(f'top3: {top3}')
    logger.info(f'MM-dist (matching score) : {matching}')


if __name__ == '__main__':
    main()

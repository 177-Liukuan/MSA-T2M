"""Evaluation script for MotionStreamer Stage-2 with global + local RAG tokens.

Extends eval_msa_t2m_rag_t5.py by:
  - Loading z_latent_dir to build a joint h_cls + z retrieval library.
  - Passing top_z_seqs [B, K, T_max, local_rag_dim] to the model at every
    inference step (sample_next_with_cfg).
  - Setting config.block_size = 78 + L_local to match training.
  - Constructing LLaMARAGWrapper with L_local / local_rag_dim parameters.

New CLI args vs eval_msa_t2m_rag_t5.py:
  --z_latent_dir    path to t2m_latents_msa_vae/<exp>/ directory
  --L_local         number of local RAG prefix tokens (default 4)
  --local_rag_dim   dim of mu latents (default 16)
"""

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
import models.msa_vae as msa_vae
import options.option_transformer as option_trans
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from humanml3d_272 import dataset_eval_t2m

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Text embedding lookup (identical to eval_msa_t2m_rag_t5.py)
# ---------------------------------------------------------------------------

class OfflineTextEmbeddingLookup:
    """Map caption string -> offline text embedding from precomputed npy files."""

    def __init__(self, data_root, text_latent_dir, split="test", text_embed_dim=768):
        self.text_embed_dim = int(text_embed_dim)
        self.caption_to_emb = {}
        self.miss_total = 0
        self.miss_warn_budget = 20

        split_file = os.path.join(data_root, "split", f"{split}.txt")
        text_dir = os.path.join(data_root, "texts")

        id_list = []
        with cs.open(split_file, "r") as f:
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
            txt_path = os.path.join(text_dir, sid + ".txt")
            emb_path = os.path.join(text_latent_dir, sid + ".npy")
            if not (os.path.exists(txt_path) and os.path.exists(emb_path)):
                continue
            captions = []
            with cs.open(txt_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split("#")
                    if len(parts) >= 1 and parts[0].strip():
                        captions.append(parts[0].strip())
            emb = np.load(emb_path).astype(np.float32)
            add_captions_with_embeddings(captions, emb)

        if len(self.caption_to_emb) == 0:
            txt_files = glob.glob(os.path.join(text_dir, "*.txt"))
            for txt_path in txt_files:
                sid = os.path.splitext(os.path.basename(txt_path))[0]
                emb_path = os.path.join(text_latent_dir, sid + ".npy")
                if not os.path.exists(emb_path):
                    continue
                captions = []
                with cs.open(txt_path, "r") as f:
                    for line in f.readlines():
                        parts = line.strip().split("#")
                        if len(parts) >= 1 and parts[0].strip():
                            captions.append(parts[0].strip())
                emb = np.load(emb_path).astype(np.float32)
                add_captions_with_embeddings(captions, emb)

        if len(self.caption_to_emb) == 0:
            raise RuntimeError("OfflineTextEmbeddingLookup built empty caption map.")

        print(
            f"Offline caption embedding map ready: {len(self.caption_to_emb)} unique captions, "
            f"loaded_pairs={loaded_pairs}, dim={self.text_embed_dim}"
        )

    @staticmethod
    def _normalize_caption(text):
        return " ".join(text.strip().lower().split())

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
                print(
                    f"[warn] Offline caption miss: {missing}/{len(text_list)} "
                    f"-> fallback to proxy. total_miss={self.miss_total}"
                )
                self.miss_warn_budget -= 1
            elif self.miss_total % 200 == 0:
                print(f"[warn] Offline caption cumulative misses: {self.miss_total}")
        return torch.from_numpy(arr).float().to(device)


# ---------------------------------------------------------------------------
# Local RAG retriever: builds joint h_cls + mu library
# ---------------------------------------------------------------------------

class RAGLocalRetriever:
    """Build in-memory retrieval library with both h_cls and mu latents.

    For each sample in hcls_dir, the corresponding mu file (same stem) is
    loaded from z_latent_dir and stored as a full sequence (variable length, end token stripped).
    All sequences are padded to the global T_max at load time for fast retrieval.

    retrieve() returns:
        top_hcls:    [B, K, D_text]
        top_scores:  [B, K]
        top_z_seqs: [B, K, T_max, local_rag_dim]  -- globally padded
        top_z_lens: [B, K]  -- valid frame counts per slot
    """

    def __init__(
        self,
        hcls_dir,
        z_latent_dir,
        topk=3,
        L_local=4,
        text_embed_dim=768,
        local_rag_dim=16,
        device=torch.device("cuda"),
    ):
        self.topk = int(topk)
        self.local_rag_dim = int(local_rag_dim)
        self.text_embed_dim = int(text_embed_dim)
        self.device = device

        hcls_files = sorted(glob.glob(os.path.join(hcls_dir, "*.npy")))
        if len(hcls_files) == 0:
            raise RuntimeError(f"No h_cls npy files found in: {hcls_dir}")

        hcls_vectors = []
        z_seqs_raw = []   # list of (T'_i, local_rag_dim) arrays
        z_lens_list = []
        skipped = 0

        for hcls_path in hcls_files:
            stem = os.path.splitext(os.path.basename(hcls_path))[0]
            z_path = os.path.join(z_latent_dir, stem + ".npy")
            if not os.path.exists(z_path):
                skipped += 1
                continue
            h = np.load(hcls_path).astype(np.float32)
            h = h.mean(axis=0) if h.ndim == 2 else h.reshape(-1)
            if h.shape[0] != self.text_embed_dim:
                skipped += 1
                continue
            z_seq = np.load(z_path).astype(np.float32)[:-1]   # strip end token -> (T, local_rag_dim)
            hcls_vectors.append(h)
            z_seqs_raw.append(z_seq)
            z_lens_list.append(z_seq.shape[0])

        if len(hcls_vectors) == 0:
            raise RuntimeError(
                f"No valid h_cls+mu pairs found. "
                f"hcls_dir={hcls_dir}, z_latent_dir={z_latent_dir}"
            )

        # Pad all sequences to global T_max for fast tensor indexing at retrieve() time
        global_T_max = max(z_lens_list)
        N = len(hcls_vectors)
        z_library_np = np.zeros((N, global_T_max, local_rag_dim), dtype=np.float32)
        for i, (z_seq, T) in enumerate(zip(z_seqs_raw, z_lens_list)):
            z_library_np[i, :T] = z_seq

        lib = torch.from_numpy(np.stack(hcls_vectors)).float().to(device)
        self.library = lib
        self.library_norm = self._normalize(lib)
        # Globally padded mu library: [N, T_max, local_rag_dim]
        self.z_library = torch.from_numpy(z_library_np).float().to(device)
        # Valid lengths per library entry: [N]
        self.z_lens_library = torch.tensor(z_lens_list, dtype=torch.long, device=device)
        self.global_T_max = global_T_max

        print(
            f"RAGLocalRetriever: {N} valid pairs "
            f"(skipped={skipped}), global_T_max={global_T_max}, topk={topk}"
        )

    @staticmethod
    def _normalize(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, text_emb):
        """Retrieve top-K entries.

        Args:
            text_emb: [B, D]
        Returns:
            top_hcls:    [B, K, D]
            top_scores:  [B, K]
            top_z_seqs: [B, K, T_max, local_rag_dim]
            top_z_lens: [B, K]
        """
        q = self._normalize(text_emb)
        sim = torch.matmul(q, self.library_norm.t())  # [B, N]
        k = min(self.topk, sim.shape[1])
        top_scores, top_indices = torch.topk(sim, k=k, dim=1)
        top_hcls    = self.library[top_indices]           # [B, k, D]
        top_z_seqs = self.z_library[top_indices]        # [B, k, T_max, dim]
        top_z_lens = self.z_lens_library[top_indices]   # [B, k]

        if k < self.topk:
            B = text_emb.shape[0]
            T_max = self.global_T_max
            pad_h   = torch.zeros(B, self.topk - k, self.text_embed_dim,
                                  device=text_emb.device, dtype=top_hcls.dtype)
            pad_s   = torch.full((B, self.topk - k), -1e6,
                                 device=text_emb.device, dtype=top_scores.dtype)
            pad_z  = torch.zeros(B, self.topk - k, T_max, self.local_rag_dim,
                                  device=text_emb.device, dtype=top_z_seqs.dtype)
            pad_len = torch.zeros(B, self.topk - k,
                                  device=text_emb.device, dtype=torch.long)
            top_hcls    = torch.cat([top_hcls, pad_h], dim=1)
            top_scores  = torch.cat([top_scores, pad_s], dim=1)
            top_z_seqs = torch.cat([top_z_seqs, pad_z], dim=1)
            top_z_lens = torch.cat([top_z_lens, pad_len], dim=1)

        return top_hcls, top_scores, top_z_seqs, top_z_lens

# ---------------------------------------------------------------------------
# Eval sampler adapter
# ---------------------------------------------------------------------------

class RAGLocalEvalSampler:
    """Adapter exposing sample_for_eval_CFG with local RAG mu token support."""

    def __init__(
        self,
        rag_model,
        retriever,
        empty_text_emb,
        latent_dim=16,
        device=torch.device("cuda"),
        reference_end_latent=None,
        stop_threshold=3.0,
        enable_stopping=True,
        text_source="offline",
        text_lookup=None,
        text_encoder=None,
        text_embed_dim=768,
        disable_rag=False,
        use_random_topk_inference=False,
    ):
        self.rag_model = rag_model
        self.retriever = retriever
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
        self.use_random_topk_inference = bool(use_random_topk_inference)
        # Cache text embeddings to avoid re-encoding identical captions.
        self._text_emb_cache: dict = {}

    def eval(self):
        self.rag_model.eval()
        return self

    @torch.no_grad()
    def sample_for_eval_CFG(
        self,
        text,
        length=196,
        tokenize_model=None,
        device=torch.device("cuda"),
        unit_length=4,
        cfg=4.0,
    ):
        _ = tokenize_model
        _ = device

        text_list = [text] if isinstance(text, str) else list(text)

        # ---- Text embedding (with cache) ----
        if self.text_source == "online_t5":
            if self.text_encoder is None:
                raise RuntimeError("text_encoder is required when text_source=online_t5")
            uncached = [t for t in text_list if t not in self._text_emb_cache]
            if uncached:
                vecs = np.asarray(self.text_encoder.encode(uncached), dtype=np.float32)
                for t, v in zip(uncached, vecs):
                    self._text_emb_cache[t] = v
            text_np = np.stack([self._text_emb_cache[t] for t in text_list])
            text_emb = torch.from_numpy(text_np).float().to(self.device)
        else:
            if self.text_lookup is None:
                raise RuntimeError("text_lookup is required when text_source=offline")
            text_emb = self.text_lookup.batch_lookup(text_list, self.device)

        if text_emb.shape[-1] != self.text_embed_dim:
            raise ValueError(
                f"text embedding dim mismatch: got {text_emb.shape[-1]}, "
                f"expected {self.text_embed_dim}"
            )

        # ---- Retrieval ----
        top_z_lens = None
        if self.disable_rag:
            top_hcls = None
            top_scores = None
            top_z_seqs = None
        else:
            top_hcls, top_scores, top_z_seqs, top_z_lens = self.retriever.retrieve(text_emb)

            if self.use_random_topk_inference and top_hcls.shape[1] > 1:
                B, K, D = top_hcls.shape
                rand_ks = torch.randint(0, K, (B,), device=top_hcls.device)
                idx = torch.arange(B, device=top_hcls.device)
                top_hcls   = top_hcls[idx, rand_ks].unsqueeze(1)
                top_scores = top_scores[idx, rand_ks].unsqueeze(1)
                top_z_seqs = top_z_seqs[idx, rand_ks].unsqueeze(1)
                top_z_lens = top_z_lens[idx, rand_ks].unsqueeze(1)

        # ---- Autoregressive generation ----
        max_token_len = max(1, int(length) // unit_length)
        xs = None
        bsz = text_emb.shape[0]
        finished = torch.zeros(bsz, dtype=torch.bool, device=self.device)

        for _k in range(max_token_len):
            prefix = (
                torch.zeros((bsz, 0, self.latent_dim), device=self.device, dtype=torch.float32)
                if xs is None else xs
            )

            next_token = self.rag_model.sample_next_with_cfg(
                motion_prefix=prefix,
                text_emb=text_emb,
                top3_h_cls=top_hcls,
                top3_sim_scores=top_scores,
                empty_text_emb=self.empty_text_emb,
                top_z_seqs=top_z_seqs,
                top_z_lens=top_z_lens,
                cfg_scale=cfg,
                temperature=1.0,
            )

            next_token = next_token.unsqueeze(1)
            xs = next_token if xs is None else torch.cat([xs, next_token], dim=1)

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
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--hcls_dir", type=str,
                              default="./humanml3d_272/h_cls_latents_msa_vae/exp")
    extra_parser.add_argument("--z_latent_dir", type=str,
                              default="./humanml3d_272/t2m_latents_msa_vae/exp")
    extra_parser.add_argument("--text_latent_dir", type=str,
                              default="./humanml3d_272/text_latents_t5")
    extra_parser.add_argument("--empty_text_path", type=str,
                              default="./humanml3d_272/text_latents_t5/empty_text_embedding.npy")
    extra_parser.add_argument("--retrieval_topk", type=int, default=3)
    extra_parser.add_argument("--L_local", type=int, default=4)
    extra_parser.add_argument("--local_rag_dim", type=int, default=16)
    extra_parser.add_argument("--cfg_scale", type=float, default=4.0)
    extra_parser.add_argument("--text_embed_dim", type=int, default=768)
    extra_parser.add_argument("--eval_split", type=str, default="test",
                              choices=["test", "val"])
    extra_parser.add_argument("--text_source", type=str, default="online_t5",
                              choices=["offline", "online_t5"])
    extra_parser.add_argument("--t5_model_path", type=str, default="sentencet5-xxl/")
    extra_parser.add_argument("--disable_rag", action="store_true", default=False)
    extra_parser.add_argument("--disable_ema", action="store_true", default=False)
    extra_parser.add_argument("--use_random_topk_inference", action="store_true", default=False)
    extra_parser.add_argument("--add_selfatten", action="store_true", default=False,
                              help="Use TransformerEncoder to encode z frames before cross-attention.")
    extra_parser.add_argument("--trans_d_model", type=int, default=768)
    extra_parser.add_argument("--trans_nhead", type=int, default=8)
    extra_parser.add_argument("--trans_enc_layers", type=int, default=6)
    extra_parser.add_argument("--trans_dec_layers", type=int, default=6)
    extra_parser.add_argument("--trans_ff_size", type=int, default=2048)
    extra_parser.add_argument("--trans_dropout", type=float, default=0.1)
    extra_parser.add_argument("--clip_dim", type=int, default=768)
    extra_parser.add_argument("--reference_end_latent_path", type=str, default="")
    extra_parser.add_argument("--stop_threshold", type=float, default=3.0)
    extra_parser.add_argument("--enable_stopping", dest="enable_stopping",
                              action="store_true", default=True)
    extra_parser.add_argument("--disable_stopping", dest="enable_stopping",
                              action="store_false")

    custom_args, remaining = extra_parser.parse_known_args()

    argv_backup = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = option_trans.get_args_parser()
    finally:
        sys.argv = argv_backup

    for attr in [
        "hcls_dir", "z_latent_dir", "text_latent_dir", "empty_text_path",
        "retrieval_topk", "L_local", "local_rag_dim",
        "cfg_scale", "text_embed_dim", "eval_split", "text_source", "t5_model_path",
        "disable_rag",
        "trans_d_model", "trans_nhead", "trans_enc_layers", "trans_dec_layers",
        "trans_ff_size", "trans_dropout", "clip_dim",
        "reference_end_latent_path", "stop_threshold", "enable_stopping",
        "use_random_topk_inference", "add_selfatten",
    ]:
        setattr(args, attr, getattr(custom_args, attr))
    args.use_ema = not custom_args.disable_ema

    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_state_strip_module(state_dict):
    return {
        (".".join(k.split(".")[1:]) if k.split(".")[0] == "module" else k): v
        for k, v in state_dict.items()
    }


def resolve_existing_path(path, must_be_dir=False):
    checker = os.path.isdir if must_be_dir else os.path.exists
    if checker(path):
        return path
    if not os.path.isabs(path):
        alt = os.path.join("..", path)
        if checker(alt):
            return alt
    return path


def resolve_data_root():
    for c in ["./humanml3d_272", "../humanml3d_272"]:
        if os.path.isdir(os.path.join(c, "texts")) and os.path.isdir(os.path.join(c, "split")):
            return c
    return "./humanml3d_272"


def resolve_reference_end_latent(args):
    candidates = []
    if args.reference_end_latent_path:
        candidates.append(args.reference_end_latent_path)
    candidates.append(
        os.path.join(args.latent_dir, f"reference_end_latent_msa_vae_{args.dataname}.npy")
    )
    candidates.append(f"reference_end_latent_msa_vae_{args.dataname}.npy")
    candidates.append(f"reference_end_latent_{args.dataname}.npy")
    for p in candidates:
        if p and os.path.exists(p):
            return p, np.load(p).astype(np.float32).reshape(-1)
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.chdir("Evaluator_272")
    sys.path.insert(0, os.getcwd())

    comp_device = torch.device("cuda")
    args = parse_args()

    # Resolve paths
    if args.resume_pth is not None:
        args.resume_pth = resolve_existing_path(args.resume_pth)
    if args.resume_trans is not None:
        args.resume_trans = resolve_existing_path(args.resume_trans)
    args.latent_dir      = resolve_existing_path(args.latent_dir, must_be_dir=True)
    args.text_latent_dir = resolve_existing_path(args.text_latent_dir, must_be_dir=True)
    args.hcls_dir        = resolve_existing_path(args.hcls_dir, must_be_dir=True)
    args.z_latent_dir   = resolve_existing_path(args.z_latent_dir, must_be_dir=True)
    args.empty_text_path = resolve_existing_path(args.empty_text_path)
    args.t5_model_path   = resolve_existing_path(args.t5_model_path, must_be_dir=True)
    data_root = resolve_data_root()

    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, f"{args.exp_name}")
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # ---- Load checkpoint first to read L_local ----
    if args.resume_trans is None:
        raise ValueError("Please provide --resume-trans for the local RAG checkpoint.")
    print(f"Loading RAG checkpoint from {args.resume_trans}")
    ckpt = torch.load(args.resume_trans, map_location="cpu")

    ckpt_L_local = ckpt.get("L_local", None) if isinstance(ckpt, dict) else None
    if ckpt_L_local is not None and int(ckpt_L_local) != args.L_local:
        logger.warning(
            f"L_local mismatch: checkpoint={ckpt_L_local}, args={args.L_local}. "
            f"Using checkpoint value."
        )
        args.L_local = int(ckpt_L_local)
    logger.info(f"L_local={args.L_local}, block_size={78 + args.L_local}")

    # Sync add_selfatten from checkpoint (command-line flag still overrides)
    ckpt_add_selfatten = bool(ckpt.get("add_selfatten", False)) if isinstance(ckpt, dict) else False
    if ckpt_add_selfatten and not args.add_selfatten:
        logger.warning("add_selfatten found in checkpoint but not set on CLI — enabling it.")
        args.add_selfatten = True
    logger.info(f"add_selfatten={args.add_selfatten}")

    val_loader = dataset_eval_t2m.DATALoader(args.dataname, args.eval_split == "test", 32)

    # ---- MSA-VAE decoder ----
    clip_range = [-30, 20]
    net = msa_vae.MSA_HumanVAE(
        hidden_size=args.hidden_size,
        down_t=args.down_t,
        stride_t=args.stride_t,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation="relu",
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
    print(f"Loading MSA-VAE checkpoint from {args.resume_pth}")
    ckpt_vae = torch.load(args.resume_pth, map_location="cpu")
    vae_state = ckpt_vae["net"] if (isinstance(ckpt_vae, dict) and "net" in ckpt_vae) else ckpt_vae
    net.load_state_dict(vae_state, strict=True)
    net.eval()
    net.to(comp_device)

    # ---- Local RAG model (block_size = 78 + L_local) ----
    config = LLaMAHFConfig.from_name("Normal_size")
    config.block_size = 78 + args.L_local

    base_model = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, comp_device)
    rag_model = LLaMARAGWrapper(
        base_model=base_model,
        model_dim=config.n_embd,
        disable_rag=args.disable_rag,
        L_local=args.L_local,
        local_rag_dim=args.local_rag_dim,
        add_selfatten=args.add_selfatten,
    )

    trans_key = "trans_ema" if args.use_ema and "trans_ema" in ckpt else "trans"
    rag_key   = "rag_ema"   if args.use_ema and "rag_ema"   in ckpt else "rag"

    if trans_key in ckpt:
        rag_model.base_model.load_state_dict(load_state_strip_module(ckpt[trans_key]), strict=False)
    else:
        raise KeyError(f"RAG checkpoint missing key: {trans_key}")

    if rag_key in ckpt:
        rag_model.load_state_dict(load_state_strip_module(ckpt[rag_key]), strict=False)
    elif not args.disable_rag:
        raise KeyError(f"RAG checkpoint missing key: {rag_key}")
    else:
        logger.info("Checkpoint has no rag key; continuing in no-RAG ablation mode.")

    if args.use_ema:
        logger.info(f"EMA eval enabled. Loaded keys: {trans_key}, {rag_key}")

    rag_model.eval()
    rag_model.to(comp_device)

    # ---- Empty text embedding for CFG ----
    empty_text_path = args.empty_text_path
    if not os.path.exists(empty_text_path):
        for p in [
            os.path.join(args.text_latent_dir, "empty_text_embedding.npy"),
            os.path.join(args.text_latent_dir, "empty_cfg_text_t5.npy"),
            os.path.join(args.text_latent_dir, "empty_cfg_text_clip.npy"),
        ]:
            if os.path.exists(p):
                empty_text_path = p
                break
    if not os.path.exists(empty_text_path):
        raise FileNotFoundError(f"empty CFG text file not found: {args.empty_text_path}")

    empty_text_emb = (
        torch.from_numpy(np.load(empty_text_path).astype(np.float32))
        .reshape(-1).to(comp_device)
    )
    if empty_text_emb.shape[0] != args.text_embed_dim:
        raise ValueError(
            f"empty text embedding dim must be {args.text_embed_dim}, "
            f"got {empty_text_emb.shape[0]}"
        )

    # ---- Text encoder / lookup ----
    text_lookup = None
    text_encoder = None
    if args.text_source == "online_t5":
        from sentence_transformers import SentenceTransformer
        text_encoder = SentenceTransformer(args.t5_model_path)
        text_encoder.eval()
        logger.info(f"Text source: online_t5, model={args.t5_model_path}")
    else:
        text_lookup = OfflineTextEmbeddingLookup(
            data_root=data_root,
            text_latent_dir=args.text_latent_dir,
            split=args.eval_split,
            text_embed_dim=args.text_embed_dim,
        )
        logger.info("Text source: offline precomputed latents")

    # ---- Local RAG retrieval library ----
    retriever = None
    if not args.disable_rag:
        retriever = RAGLocalRetriever(
            hcls_dir=args.hcls_dir,
            z_latent_dir=args.z_latent_dir,
            topk=args.retrieval_topk,
            L_local=args.L_local,
            text_embed_dim=args.text_embed_dim,
            local_rag_dim=args.local_rag_dim,
            device=comp_device,
        )
    else:
        logger.info("No-RAG ablation enabled: retrieval library bypassed.")

    # ---- Reference end latent for stopping ----
    reference_path, reference_end = resolve_reference_end_latent(args)
    reference_end_latent = None
    if args.enable_stopping:
        if reference_end is None:
            raise FileNotFoundError(
                "Cannot locate reference end latent. Provide --reference_end_latent_path "
                "or ensure it exists under --latent_dir."
            )
        reference_end_latent = torch.from_numpy(reference_end).float().to(comp_device)
        if reference_end_latent.shape[0] != args.latent_dim:
            raise ValueError(
                f"reference end latent dim mismatch: "
                f"got {reference_end_latent.shape[0]}, expected {args.latent_dim}"
            )
        logger.info(
            f"Continuous stopping enabled: path={reference_path}, threshold={args.stop_threshold}"
        )
    else:
        logger.info("Continuous stopping disabled.")

    # ---- Eval adapter ----
    trans_for_eval = RAGLocalEvalSampler(
        rag_model=rag_model,
        retriever=retriever,
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
        use_random_topk_inference=args.use_random_topk_inference,
    )

    # ---- Evaluator ----
    from mld.models.architectures.temos.textencoder.distillbert_actor import (
        DistilbertActorAgnosticEncoder,
    )
    from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

    modelpath = "./deps/distilbert-base-uncased"
    textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
    motionencoder = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)

    ckpt_path = "../Evaluator_272/epoch=99.ckpt"
    print(f"Loading evaluator checkpoint from {ckpt_path}")
    ckpt_eval = torch.load(ckpt_path)

    textencoder_ckpt = {
        k.replace("textencoder.", ""): v
        for k, v in ckpt_eval["state_dict"].items()
        if k.split(".")[0] == "textencoder"
    }
    textencoder.load_state_dict(textencoder_ckpt, strict=True)
    textencoder.eval()
    textencoder.to(comp_device)

    motionencoder_ckpt = {
        k.replace("motionencoder.", ""): v
        for k, v in ckpt_eval["state_dict"].items()
        if k.split(".")[0] == "motionencoder"
    }
    motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
    motionencoder.eval()
    motionencoder.to(comp_device)

    evaluator = [textencoder, motionencoder]

    # ---- Run evaluation ----
    fid, div, top1, top2, top3, matching = [], [], [], [], [], []

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

    logger.info("final result:")
    logger.info(f"fid: {fid}")
    logger.info(f"div: {div}")
    logger.info(f"top1: {top1}")
    logger.info(f"top2: {top2}")
    logger.info(f"top3: {top3}")
    logger.info(f"MM-dist (matching score) : {matching}")

    writer.close()


if __name__ == "__main__":
    main()

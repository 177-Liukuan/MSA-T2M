"""
msa_gen_motion_mca_op.py
------------------------
Corrected inference & visualization script for LLaMARAGLatentRetrWrapper.

KEY BUGS FIXED vs msa_gen_motion_mca.py
---------------------------------------
Bug 1 (CRITICAL): ca_every_n_layers=1 hardcoded → 12 CA blocks built, but
  both checkpoints have only 3.  strict=False loading silently placed the 3
  loaded blocks at layers [0,1,2] instead of the correct [3,7,11].
  FIX: auto-detect ca_every_n_layers from checkpoint (n_total // n_ca_blocks).

Bug 2: OLD model (6layer_top3_ddpm) was trained with JOINT CFG dropout
  (retr_cfg_drop_mask always equals cfg_drop_mask).  Using 3-forward dual CFG
  at inference forces a (null_text, real_retr) combination that is
  out-of-distribution for that model.
  FIX: set cfg_scale_retr = cfg_scale for the OLD model.  Mathematically this
  reduces the dual-CFG formula to standard 2-forward CFG:
    z_guided = z_none + s*(z_both-z_retr) + s*(z_retr-z_none)
             = z_none + s*(z_both-z_none)   ← standard CFG ✓

Usage:
  Edit the "User config" section below, then:
    python msa_gen_motion_mca_op.py
"""

import os
import re
import time
import glob
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===========================================================================
# User config  (edit here)
# ===========================================================================

# text = "A figure dances ballet elegantly"
# text = "“A person walks forward, turns around, then sits down"
text = "A man is walking forward while he is punching"

# Built-in demo checkpoint: archived joint-CFG latent-retrieval result.
resume_trans = (
    "Experiments/explorations/cross_attention/latent_retrieval/MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr"
    "_6layer_top3_ddpm/net_Iter100000.pth"
)

# The built-in checkpoint uses joint CFG; cfg_scale_retr is ignored.
cfg_scale      = 6.0   # text CFG strength
cfg_scale_retr = 2.0   # unused for the built-in joint-CFG run

# CFG mode flag  ← use True for ALL models (OLD and NEW alike)
#
#   True  => proper 2-forward velocity-space CFG:
#             z_cond  = forward(real_text, real_retr)   -- in training distribution
#             z_uncond = forward(null_text, null_retr)  -- in training distribution
#             diff_loss.sample(cat([z_cond, z_uncond]), cfg=s)
#             CFG scaling: v_guided = v_uncond + s*(v_cond - v_uncond)  [velocity space]
#             Retrieval signal is naturally baked into z_cond via the CA blocks.
#
#   False => 3-forward pre-mixed hidden-space CFG (via sample_next_with_cfg):
#             z_guided = z_none + s_text*(z_both-z_retr) + s_retr*(z_retr-z_none)
#             diff_loss.sample(z_guided, cfg=1.0)
#   WARNING: False causes FLYING MOTIONS because z_guided = 4*z_both - 3*z_retr
#   is an out-of-distribution extrapolation the diffusion head has never seen.
#   For a 12-CA-block model the difference (z_both - z_retr) is huge, so the
#   4x amplification pushes z_guided far outside the training manifold.
use_joint_cfg = True     # Matches the built-in archived checkpoint.

# Stop token threshold (L2 distance from generated token to reference_end_latent).
# Data calibration:
#   - stop tokens in training are EXACTLY reference_end_latent => dist = 0.0
#   - diffusion sampling noise floor ≈ 0.2–0.4 (consecutive token mean=0.30)
#   - 42% of library tokens within 0.5 of reference
# Original MotionStreamer reported value; works correctly when CFG is in-distribution.
# With proper use_joint_cfg for OLD model, the generated stop token should land at
# dist ≈ 0.0 from reference_end_latent, so 0.1 triggers reliably.
threshold  = 0.1       # L2 stop distance
max_length = 300       # max motion frames

# Retrieval
retrieval_topk   = 3   # global h_cls top-k
latent_retr_topk = 3   # local motion latent top-k

# VAE
resume_pth = (
    "Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right"
    "/net_best_mpjpe.pth"
)

# Data paths (common for both models)
hcls_dir = (
    "./humanml3d_272/h_cls_latents_msa_vae"
    "/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right"
)
library_cache_dir = (
    "./humanml3d_272/latent_retr_library_cache"
    "/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right"
)
empty_text_path = "./humanml3d_272/text_latents_t5/empty_text_embedding.npy"
reference_end_latent_path = (
    "humanml3d_272/t2m_latents_msa_vae"
    "/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right"
    "/reference_end_latent_msa_vae_t2m_272.npy"
)
t5_model_path = "sentencet5-xxl/"
mean_path     = "humanml3d_272/mean_std/Mean.npy"
std_path      = "humanml3d_272/mean_std/Std.npy"

output_dir = "demo_output/MSA-T2M-LatentRetr-op"

# Ablation flags
disable_rag         = False
disable_latent_retr = False
use_ema             = True

# Reproducibility
seed          = 123
deterministic = True

# Generation
fps               = 30
unit_length       = 4
latent_dim        = 16
text_embed_dim    = 768

# Diffusion head
num_diffusion_head_layers = 9
num_flow_steps    = 50
flow_solver       = "euler"
rf_time_sampling  = "uniform"
rf_loss_type      = "mse"

# MSA-VAE architecture
hidden_size          = 1024
down_t = stride_t    = 2
depth                = 3
dilation_growth_rate = 3
trans_d_model        = 768
trans_nhead          = 8
trans_enc_layers     = 6
trans_dec_layers     = 6
trans_ff_size        = 2048
trans_dropout        = 0.1
clip_dim             = 768


# ===========================================================================
# Checkpoint introspection helpers
# ===========================================================================

def inspect_rag_checkpoint(ckpt, use_ema=True):
    """Return (n_ca_blocks, ff_mult, ff_inner_dim) from checkpoint."""
    rag_key = "rag_ema" if (use_ema and "rag_ema" in ckpt) else "rag"
    rag_sd  = ckpt[rag_key]

    ca_ids = set()
    for k in rag_sd:
        if "ca_blocks." in k:
            ca_ids.add(int(k.split("ca_blocks.")[1].split(".")[0]))
    n_ca = len(ca_ids) if ca_ids else 1

    ff_key = next((k for k in rag_sd
                   if "ca_blocks.0.ff_in_proj.weight" in k), None)
    ff_inner = rag_sd[ff_key].shape[0] if ff_key is not None else 1536
    ff_mult  = ff_inner // 768          # model_dim = 768

    return n_ca, ff_mult, ff_inner


def detect_n_transformer_layers(ckpt, use_ema=True):
    """Count LLaMA transformer layers from trans checkpoint."""
    trans_key = "trans_ema" if (use_ema and "trans_ema" in ckpt) else "trans"
    trans_sd  = ckpt[trans_key]
    layer_ids = set()
    for k in trans_sd:
        if "transformer.h." in k:
            layer_ids.add(int(k.split("transformer.h.")[1].split(".")[0]))
    return max(layer_ids) + 1 if layer_ids else 12


# ===========================================================================
# Global h_cls retriever
# ===========================================================================

class RAGRetriever:
    def __init__(self, hcls_dir_path, topk=3, embed_dim=768,
                 device=torch.device("cuda")):
        self.topk     = int(topk)
        self.embed_dim = int(embed_dim)

        files = sorted(glob.glob(os.path.join(hcls_dir_path, "*.npy")))
        if not files:
            raise RuntimeError(f"No h_cls .npy files in: {hcls_dir_path}")

        vectors = []
        for path in files:
            vec = np.load(path).astype(np.float32)
            vec = vec.mean(axis=0) if vec.ndim == 2 else vec.reshape(-1)
            if vec.shape[0] == self.embed_dim:
                vectors.append(vec)

        if not vectors:
            raise RuntimeError(
                f"No valid {self.embed_dim}-d h_cls vectors in: {hcls_dir_path}"
            )

        lib = torch.from_numpy(np.stack(vectors)).float().to(device)
        self.lib      = lib
        self.lib_norm = self._norm(lib)

    @staticmethod
    def _norm(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, text_emb):
        q = self._norm(text_emb)
        sim = torch.matmul(q, self.lib_norm.t())
        k   = min(self.topk, sim.shape[1])
        top_scores, top_idx = torch.topk(sim, k=k, dim=1)
        top_hcls = self.lib[top_idx]

        if k < self.topk:
            bsz = text_emb.shape[0]
            pad_h = torch.zeros(bsz, self.topk - k, self.embed_dim,
                                device=text_emb.device, dtype=top_hcls.dtype)
            pad_s = torch.full((bsz, self.topk - k), -1e6,
                               device=text_emb.device, dtype=top_scores.dtype)
            top_hcls   = torch.cat([top_hcls, pad_h], dim=1)
            top_scores = torch.cat([top_scores, pad_s], dim=1)

        return top_hcls, top_scores


# ===========================================================================
# Local motion latent retriever (5-file library cache)
# ===========================================================================

class RAGLatentRetriever:
    def __init__(self, library_cache_dir, topk=3, device=torch.device("cuda")):
        self.topk   = topk
        self.device = device

        def _p(name):
            return os.path.join(library_cache_dir, name)

        for fname in ("lib_text_embs.npy", "lib_sample_ids.txt",
                      "lib_latents_flat.npy", "lib_latent_starts.npy",
                      "lib_latent_lengths.npy"):
            if not os.path.exists(_p(fname)):
                raise FileNotFoundError(
                    f"Library cache file missing: {_p(fname)}\n"
                    f"Run build_latent_retr_library.py first."
                )

        lib_text_embs = np.load(_p("lib_text_embs.npy")).astype(np.float32)
        self.lib_text_embs      = torch.from_numpy(lib_text_embs).float().to(device)
        self.lib_text_embs_norm = self._normalize(self.lib_text_embs)

        with open(_p("lib_sample_ids.txt"), "r") as f:
            self.lib_sample_ids = [ln.strip() for ln in f.readlines()]

        self.lib_latents_flat   = np.load(_p("lib_latents_flat.npy")).astype(np.float32)
        self.lib_latent_starts  = np.load(_p("lib_latent_starts.npy")).astype(np.int64)
        self.lib_latent_lengths = np.load(_p("lib_latent_lengths.npy")).astype(np.int64)
        self.latent_dim = self.lib_latents_flat.shape[1]

        print(f"[RAGLatentRetriever] N_caps={len(lib_text_embs)}, "
              f"latent_dim={self.latent_dim}, topk={topk}")

    @staticmethod
    def _normalize(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, text_emb, exclude_sample_ids=None):
        bsz = text_emb.shape[0]
        q   = self._normalize(text_emb)
        sim = torch.matmul(q, self.lib_text_embs_norm.t())

        all_latents, all_lens = [], []
        for b in range(bsz):
            sim_b = sim[b].clone()
            if exclude_sample_ids is not None and exclude_sample_ids[b] is not None:
                excl = str(exclude_sample_ids[b])
                for idx, sid in enumerate(self.lib_sample_ids):
                    if sid == excl:
                        sim_b[idx] = -1e6

            k = min(self.topk, sim_b.shape[0])
            _, top_idx = torch.topk(sim_b, k=k, dim=0)

            parts = []
            for idx in top_idx.cpu().numpy():
                s = int(self.lib_latent_starts[idx])
                l = int(self.lib_latent_lengths[idx])
                parts.append(self.lib_latents_flat[s:s + l])

            concat = (np.concatenate(parts, axis=0) if parts
                      else np.zeros((1, self.latent_dim), dtype=np.float32))
            all_latents.append(concat)
            all_lens.append(len(concat))

        L_max  = max(all_lens)
        padded = np.zeros((bsz, L_max, self.latent_dim), dtype=np.float32)
        for b, (lat, l) in enumerate(zip(all_latents, all_lens)):
            padded[b, :l] = lat

        retr_latents     = torch.from_numpy(padded).float().to(self.device)
        retr_latent_lens = torch.tensor(all_lens, dtype=torch.long, device=self.device)
        return retr_latents, retr_latent_lens


# ===========================================================================
# Utilities
# ===========================================================================

def set_reproducibility(seed_val, det=True):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = bool(det)
    if det:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        new_k = ".".join(k.split(".")[1:]) if k.split(".")[0] == "module" else k
        out[new_k] = v
    return out


def sanitize_for_filename(s):
    s = re.sub(r"\s+", "_", s.strip().lower())
    s = re.sub(r"[^\w\-]", "_", s)
    return (re.sub(r"_+", "_", s).strip("_") or "motion")[:80]


# ===========================================================================
# Autoregressive sampler
# ===========================================================================

@torch.no_grad()
def _sample_2forward_joint(
    rag_model, prefix, text_emb, empty_text_emb,
    top_hcls, top_scores, cfg_scale,
    retr_latents, retr_latent_lens, device,
):
    """Proper 2-forward CFG with JOINT retrieval+text dropout.

    Matches OLD model training where retr_cfg_drop_mask was not independent —
    retrieval is dropped together with text (both share cfg_drop_mask).

    Passing retr_cfg_drop_mask=None causes the model to fall back to
    effective_retr_drop = cfg_drop_mask, which:
      Forward 1 (cond_mask=False)  => keep text AND keep retrieval  ✓
      Forward 2 (uncond_mask=True) => null text  AND null retrieval ✓
    Then standard CFG is applied in the diffusion/RF velocity space.
    """
    bsz = text_emb.shape[0]
    cond_mask   = torch.zeros(bsz, dtype=torch.bool, device=device)
    uncond_mask = torch.ones( bsz, dtype=torch.bool, device=device)

    z_cond = rag_model.forward(
        prefix, text_emb, top_hcls, top_scores,
        cfg_drop_mask=cond_mask, empty_text_emb=empty_text_emb,
        retr_latents=retr_latents, retr_latent_lens=retr_latent_lens,
        retr_cfg_drop_mask=None,   # fallback → cfg_drop_mask=False → keep retr
    )[:, -1, :]   # (bsz, D)

    z_uncond = rag_model.forward(
        prefix, text_emb, top_hcls, top_scores,
        cfg_drop_mask=uncond_mask, empty_text_emb=empty_text_emb,
        retr_latents=retr_latents, retr_latent_lens=retr_latent_lens,
        retr_cfg_drop_mask=None,   # fallback → cfg_drop_mask=True → null retr
    )[:, -1, :]   # (bsz, D)

    # Standard CFG applied in diffusion velocity/noise space (not pre-mixed hidden).
    # This is the correct form for DDPM/RF: v_guided = v_uncond + s*(v_cond-v_uncond)
    mix_hidden = torch.cat([z_cond, z_uncond], dim=0)   # (2*bsz, D)
    sampled = rag_model.base_model.diff_loss.sample(
        mix_hidden, temperature=1.0, cfg=cfg_scale
    )
    # RF with cfg!=1.0 returns (2*bsz, latent_dim); take first half (conditional)
    sampled_cond = sampled.chunk(2, dim=0)[0]            # (bsz, latent_dim)
    return sampled_cond



def _sample_3forward_additive(
    rag_model, prefix, text_emb, empty_text_emb,
    top_hcls, top_scores, s_t, s_r,
    retr_latents, retr_latent_lens, device,
):
    """3-forward additive velocity-space CFG for NEW model (independent dropout).

    v_guided = v_nn + s_t*(v_tn - v_nn) + s_r*(v_tr - v_tn)

    All 3 z vectors are in-distribution (seen during training):
      z_nn => (null_text, null_retr)  ~1%  training distribution
      z_tn => (real_text, null_retr)  ~9%  training distribution
      z_tr => (real_text, real_retr)  ~81% training distribution
    """
    bsz = text_emb.shape[0]
    all_false = torch.zeros(bsz, dtype=torch.bool, device=device)
    all_true  = torch.ones( bsz, dtype=torch.bool, device=device)

    # Forward 1: null text, null retr → z_nn
    z_nn = rag_model.forward(
        prefix, text_emb, top_hcls, top_scores,
        cfg_drop_mask=all_true, empty_text_emb=empty_text_emb,
        retr_latents=retr_latents, retr_latent_lens=retr_latent_lens,
        retr_cfg_drop_mask=all_true,
    )[:, -1, :]   # (bsz, D)

    # Forward 2: real text, null retr → z_tn
    z_tn = rag_model.forward(
        prefix, text_emb, top_hcls, top_scores,
        cfg_drop_mask=all_false, empty_text_emb=empty_text_emb,
        retr_latents=retr_latents, retr_latent_lens=retr_latent_lens,
        retr_cfg_drop_mask=all_true,
    )[:, -1, :]   # (bsz, D)

    # Forward 3: real text, real retr → z_tr
    z_tr = rag_model.forward(
        prefix, text_emb, top_hcls, top_scores,
        cfg_drop_mask=all_false, empty_text_emb=empty_text_emb,
        retr_latents=retr_latents, retr_latent_lens=retr_latent_lens,
        retr_cfg_drop_mask=all_false,
    )[:, -1, :]   # (bsz, D)

    # Additive CFG in velocity/noise space — no OOD hidden extrapolation
    sampled = rag_model.base_model.diff_loss.sample_additive_cfg(
        z_nn=z_nn, z_tn=z_tn, z_tr=z_tr,
        s_t=s_t, s_r=s_r, temperature=1.0,
    )  # (bsz, latent_dim)
    return sampled


@torch.no_grad()
def sample_motion_latents(
    rag_model,
    text_emb,          # (1, D) already on device
    empty_text_emb,    # (D,)   already on device
    top_hcls,          # (1, K, D) or None
    top_scores,        # (1, K) or None
    retr_latents,      # (1, L, 16) or None
    retr_latent_lens,  # (1,) or None
    reference_end,     # (1, 16) on device
    cfg,
    cfg_retr,
    joint_cfg=False,
    token_latent_dim=16,
    stop_threshold=0.1,
    max_tokens=75,
    device=torch.device("cuda"),
):
    """Autoregressive generation with stop-token detection.

    joint_cfg=True  (OLD model): proper 2-forward CFG with joint dropout.
      Both text and retrieval drop together — matches OLD model training.
    joint_cfg=False (NEW model): 3-forward dual-CFG pre-mixed in hidden space.
      Text and retrieval guidance are decoupled.

    Stop condition: generated token L2-distance to reference_end < stop_threshold.
      Library calibration: stop tokens at dist=0.00; consecutive motion token
      spacing mean=0.30 (p95=0.81); 42% of library tokens within 0.5 of ref.
      Original MotionStreamer value 0.1 works when CFG is in-distribution.
    """
    xs = None
    for step in range(max_tokens):
        prefix = (xs if xs is not None
                  else torch.zeros(1, 0, token_latent_dim,
                                   dtype=torch.float32, device=device))

        if joint_cfg:
            # ── OLD model: proper 2-forward CFG (joint retrieval dropout) ──
            next_tok = _sample_2forward_joint(
                rag_model, prefix, text_emb, empty_text_emb,
                top_hcls, top_scores, cfg,
                retr_latents, retr_latent_lens, device,
            )
        else:
            # ── NEW model: 3-forward additive velocity-space CFG ──
            next_tok = _sample_3forward_additive(
                rag_model, prefix, text_emb, empty_text_emb,
                top_hcls, top_scores,
                s_t=cfg,      # text guidance scale
                s_r=cfg_retr, # retrieval guidance scale
                retr_latents=retr_latents,
                retr_latent_lens=retr_latent_lens,
                device=device,
            )
        # next_tok: (1, latent_dim)

        tok_norm = next_tok.norm().item()
        dist     = torch.sqrt(((next_tok - reference_end) ** 2).sum()).item()

        next_tok = next_tok.unsqueeze(1)          # (1, 1, 16)
        xs = next_tok if xs is None else torch.cat([xs, next_tok], dim=1)

        print(f"  step {step+1:3d}: norm={tok_norm:.3f}  dist_to_stop={dist:.4f}",
              end="\r")
        if dist < stop_threshold:
            print(f"\n  [Stop] step={step+1}, dist={dist:.4f} < "
                  f"threshold={stop_threshold}  (tok_norm={tok_norm:.3f})")
            break
    else:
        print(f"\n  [MaxLen] reached max_tokens={max_tokens}")

    if xs is None:
        xs = torch.zeros(1, 1, token_latent_dim, device=device)
    return xs


# ===========================================================================
# Main
# ===========================================================================

def main():
    os.makedirs(output_dir, exist_ok=True)
    set_reproducibility(seed, deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Validate paths ───────────────────────────────────────────────────
    must_exist = {
        "resume_pth"              : resume_pth,
        "resume_trans"            : resume_trans,
        "empty_text_path"         : empty_text_path,
        "reference_end_latent"    : reference_end_latent_path,
        "mean_path"               : mean_path,
        "std_path"                : std_path,
    }
    for name, p in must_exist.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file [{name}]: {p}")
    if not disable_rag and not os.path.isdir(hcls_dir):
        raise FileNotFoundError(f"h_cls dir not found: {hcls_dir}")
    if not disable_latent_retr and not os.path.isdir(library_cache_dir):
        raise FileNotFoundError(f"Library cache dir not found: {library_cache_dir}")
    if not os.path.isdir(t5_model_path):
        raise FileNotFoundError(f"T5 model dir not found: {t5_model_path}")

    # ── Load transformer checkpoint FIRST to detect architecture ────────
    print(f"Loading transformer checkpoint: {resume_trans}")
    ckpt = torch.load(resume_trans, map_location="cpu")

    resolved_head = (ckpt.get("generative_head_type", "ddpm")
                     if isinstance(ckpt, dict) else "ddpm")
    print(f"  generative_head_type: {resolved_head}")

    # --- AUTO-DETECT ca_every_n_layers from checkpoint ---
    use_ema_actual = use_ema and ("trans_ema" in ckpt)
    n_ca_blocks, ff_mult_ckpt, ff_inner = inspect_rag_checkpoint(ckpt, use_ema_actual)
    n_total_layers  = detect_n_transformer_layers(ckpt, use_ema_actual)
    ca_every_n_auto = n_total_layers // n_ca_blocks if n_ca_blocks > 0 else 4

    print(f"  [AutoDetect] n_total_layers={n_total_layers}, "
          f"n_ca_blocks={n_ca_blocks}, "
          f"→ ca_every_n_layers={ca_every_n_auto}, ff_mult={ff_mult_ckpt}")

    # Verify CA layer positions that will be active
    ca_positions = [i for i in range(n_total_layers) if (i + 1) % ca_every_n_auto == 0]
    print(f"  [AutoDetect] CA will be applied at layer indices: {ca_positions}")

    # ── Model imports ────────────────────────────────────────────────────
    from models.llama_model import LLaMAHF, LLaMAHFConfig
    from models.llama_rag_model_latent_retr import LLaMARAGLatentRetrWrapper
    import models.msa_vae as msa_vae

    # ── VAE ──────────────────────────────────────────────────────────────
    print(f"Loading VAE: {resume_pth}")
    net = msa_vae.MSA_HumanVAE(
        hidden_size          = hidden_size,
        down_t               = down_t,
        stride_t             = stride_t,
        depth                = depth,
        dilation_growth_rate = dilation_growth_rate,
        activation           = "relu",
        latent_dim           = latent_dim,
        clip_range           = [-30, 20],
        trans_d_model        = trans_d_model,
        trans_nhead          = trans_nhead,
        trans_enc_layers     = trans_enc_layers,
        trans_dec_layers     = trans_dec_layers,
        trans_ff_size        = trans_ff_size,
        trans_dropout        = trans_dropout,
        clip_dim             = clip_dim,
    ).to(device)

    ckpt_vae = torch.load(resume_pth, map_location="cpu")
    vae_state = (ckpt_vae["net"] if isinstance(ckpt_vae, dict) and "net" in ckpt_vae
                 else ckpt_vae)
    net.load_state_dict(vae_state, strict=True)
    net.eval()

    # ── LLaMA backbone ───────────────────────────────────────────────────
    config = LLaMAHFConfig.from_name("Normal_size")
    config.block_size = 78

    base_model = LLaMAHF(
        config,
        num_diffusion_head_layers,
        latent_dim,
        device,
        generative_head_type = resolved_head,
        num_flow_steps       = num_flow_steps,
        flow_solver          = flow_solver,
        rf_time_sampling     = rf_time_sampling,
        rf_loss_type         = rf_loss_type,
    )

    # ── RAG model (ca_every_n_layers AUTO-DETECTED from checkpoint) ──────
    rag_model = LLaMARAGLatentRetrWrapper(
        base_model          = base_model,
        model_dim           = config.n_embd,
        disable_rag         = disable_rag,
        latent_dim          = latent_dim,
        ca_every_n_layers   = ca_every_n_auto,    # ← KEY FIX
        ff_mult             = ff_mult_ckpt,        # ← from checkpoint
        disable_latent_retr = disable_latent_retr,
    ).to(device)
    print(f"  [Model] {rag_model.extra_repr()}")

    # ── Load weights ─────────────────────────────────────────────────────
    trans_key = "trans_ema" if (use_ema_actual and "trans_ema" in ckpt) else "trans"
    rag_key   = "rag_ema"   if (use_ema_actual and "rag_ema"   in ckpt) else "rag"

    if trans_key not in ckpt:
        raise KeyError(f"Checkpoint missing '{trans_key}'.")
    if rag_key not in ckpt:
        raise KeyError(f"Checkpoint missing '{rag_key}'.")

    missing_trans, unexpected_trans = base_model.load_state_dict(
        strip_module_prefix(ckpt[trans_key]), strict=False
    )
    missing_rag, unexpected_rag = rag_model.load_state_dict(
        strip_module_prefix(ckpt[rag_key]), strict=False
    )
    print(f"  [Load] trans: missing={len(missing_trans)}, unexpected={len(unexpected_trans)}")
    print(f"  [Load] rag:   missing={len(missing_rag)}, unexpected={len(unexpected_rag)}")
    if missing_rag:
        print(f"  [Load] missing rag keys (first 5): {missing_rag[:5]}")

    rag_model.eval()

    # ── Text encoder ─────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    text_encoder = SentenceTransformer(t5_model_path)
    text_encoder.eval()

    # ── Auxiliary tensors ─────────────────────────────────────────────────
    empty_text_emb = (
        torch.from_numpy(np.load(empty_text_path).astype(np.float32))
        .reshape(-1).to(device)
    )
    assert empty_text_emb.shape[0] == text_embed_dim, (
        f"empty_text_emb dim mismatch: {empty_text_emb.shape[0]} vs {text_embed_dim}"
    )

    ref_end = torch.from_numpy(
        np.load(reference_end_latent_path).astype(np.float32)
    ).to(device).reshape(1, latent_dim)

    # ── Retrievers ────────────────────────────────────────────────────────
    retriever       = None
    latent_retriever = None

    if not disable_rag:
        retriever = RAGRetriever(
            hcls_dir_path = hcls_dir,
            topk          = retrieval_topk,
            embed_dim     = text_embed_dim,
            device        = device,
        )
    else:
        print("[Info] disable_rag=True: global h_cls retrieval bypassed")

    if not disable_latent_retr:
        latent_retriever = RAGLatentRetriever(
            library_cache_dir = library_cache_dir,
            topk              = latent_retr_topk,
            device            = device,
        )
    else:
        print("[Info] disable_latent_retr=True: local CA retrieval bypassed")

    # ── Encode text ───────────────────────────────────────────────────────
    print(f"\nGenerating: \"{text}\"")
    print(f"  cfg_scale={cfg_scale}, cfg_scale_retr={cfg_scale_retr}, "
          f"use_joint_cfg={use_joint_cfg}")
    if use_joint_cfg:
        print("  CFG mode: JOINT 2-forward (OLD model — retrieval drops with text)")
    else:
        print("  CFG mode: DUAL 3-forward (NEW model — independent retrieval CFG)")

    text_feat = text_encoder.encode([text])
    text_emb  = torch.from_numpy(np.asarray(text_feat, dtype=np.float32)).to(device)
    # text_emb: (1, 768)

    # ── Global h_cls retrieval ────────────────────────────────────────────
    top_hcls = top_scores = None
    if retriever is not None:
        top_hcls, top_scores = retriever.retrieve(text_emb)
        print(f"  [RAG] top-{retrieval_topk} h_cls scores: "
              f"{top_scores[0].cpu().tolist()}")

    # ── Local motion latent retrieval ─────────────────────────────────────
    retr_latents = retr_latent_lens = None
    if latent_retriever is not None:
        retr_latents, retr_latent_lens = latent_retriever.retrieve(
            text_emb, exclude_sample_ids=[None]
        )
        print(f"  [LatentRetr] retrieved {retr_latent_lens[0].item()} latent frames")

    # ── Autoregressive generation ─────────────────────────────────────────
    max_tokens = max_length // unit_length
    print(f"\n  Generating (max_tokens={max_tokens})...")
    motion_latents = sample_motion_latents(
        rag_model        = rag_model,
        text_emb         = text_emb,
        empty_text_emb   = empty_text_emb,
        top_hcls         = top_hcls,
        top_scores       = top_scores,
        retr_latents     = retr_latents,
        retr_latent_lens = retr_latent_lens,
        reference_end    = ref_end,
        cfg              = cfg_scale,
        cfg_retr         = cfg_scale_retr,
        joint_cfg        = use_joint_cfg,
        token_latent_dim = latent_dim,
        stop_threshold   = threshold,
        max_tokens       = max_tokens,
        device           = device,
    )
    n_tokens = motion_latents.shape[1]
    n_frames = n_tokens * unit_length
    print(f"  Generated {n_tokens} tokens → {n_frames} frames")

    # ── Decode & denormalise ──────────────────────────────────────────────
    motion_raw = (net.forward_decoder(motion_latents)
                  .squeeze(0).detach().cpu().numpy().astype(np.float32))

    mean_arr = np.load(mean_path)
    std_arr  = np.load(std_path)
    motion   = motion_raw * std_arr + mean_arr

    # ── Recover joint positions ───────────────────────────────────────────
    from visualization.recover_visualize import recover_from_local_position
    import visualization.plot_3d_global as plot_3d

    pred_xyz = recover_from_local_position(motion, 22)  # (T, 22, 3)
    xyz = pred_xyz.reshape(1, -1, 22, 3)

    # ── Save outputs ──────────────────────────────────────────────────────
    ts   = time.strftime("%Y%m%d_%H%M%S")
    stem = f"LatentRetr_{sanitize_for_filename(text)}_{ts}"

    gif_path = os.path.join(output_dir, f"{stem}.gif")
    npy_path = os.path.join(output_dir, f"{stem}.npy")

    plot_3d.draw_to_batch(xyz, [text], [gif_path], fps=fps)
    np.save(npy_path, motion_raw)      # save raw (pre-denorm) for further use

    print(f"\n[OK] GIF  : {gif_path}")
    print(f"[OK] NPY  : {npy_path}  (raw latent-decoded motion, shape={motion_raw.shape})")
    print("     (pass npy_path to output_vis.py --input for further visualization)")


if __name__ == "__main__":
    main()

"""
msa_gen_motion_mca.py
---------------------
Inference & visualization script for the MSA T2M model with cross-attention
(LLaMARAGMultiTextCAWrapper).  Drop-in counterpart of msa_gen_motion.py.

Usage:
    python msa_gen_motion_mca.py
"""
import os
import re
import time
import glob
import random
import warnings

import numpy as np
import torch
import torch.nn as nn

from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model_mca import LLaMARAGMultiTextCAWrapper
import models.msa_vae as msa_vae
from visualization.recover_visualize import recover_from_local_position
import visualization.plot_3d_global as plot_3d


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================
# User-editable config area
# =========================
text = "A person swings a tennis racket"
cfg_scale = 4
threshold = 0.1
retrieval_topk = 3
max_length = 300
fps = 30
disable_rag = False
seed = 123
deterministic = True
num_flow_steps = 50
flow_solver = "euler"
rf_time_sampling = "uniform"
rf_loss_type = "mse"

# Use EMA weights at inference (recommended)
use_ema = True

# MCA-specific config (must match training run.log)
ca_every_n_layers = 4       # every 4 layers → CA at [3, 7, 11] for 12-layer backbone
ca_n_head = None            # None = auto (same as backbone)
text_token_dim = 1024       # T5-xxl hidden dim
ff_mult = 2                 # CA FFN expansion factor

# Fixed paths
resume_pth = "Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048/net_best_mpjpe.pth"
resume_trans = (
    "Experiments/MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_6layer_ddpm_scratch_Flamingo_gateclose_fix"
    "/net_Iter100000.pth"
)
hcls_dir = "./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048"
empty_text_path = "./humanml3d_272/text_latents_t5/empty_text_embedding.npy"
reference_end_latent = (
    "humanml3d_272/t2m_latents_msa_vae"
    "/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048"
    "/reference_end_latent_msa_vae_t2m_272.npy"
)
t5_model_path = "sentencet5-xxl/"

# Standard defaults (aligned with training scripts)
unit_length = 4
latent_dim = 16
text_embed_dim = 768
num_diffusion_head_layers = 9

hidden_size = 1024
down_t = 2
stride_t = 2
depth = 3
dilation_growth_rate = 3

trans_d_model = 768
trans_nhead = 8
trans_enc_layers = 6
trans_dec_layers = 6
trans_ff_size = 2048
trans_dropout = 0.1
clip_dim = 768

mean_path = "humanml3d_272/mean_std/Mean.npy"
std_path  = "humanml3d_272/mean_std/Std.npy"

output_dir = "demo_output/MSA-T2M-MCA"


# ---------------------------------------------------------------------------
#  RAG retriever (identical to msa_gen_motion.py)
# ---------------------------------------------------------------------------

class RAGRetriever:
    """In-memory h_cls retrieval library for inference."""

    def __init__(self, hcls_dir_path, topk=3, embed_dim=768, device=torch.device("cuda")):
        self.topk = int(topk)
        self.embed_dim = int(embed_dim)

        files = sorted(glob.glob(os.path.join(hcls_dir_path, "*.npy")))
        if len(files) == 0:
            raise RuntimeError(f"No h_cls npy files found in: {hcls_dir_path}")

        vectors = []
        for path in files:
            vec = np.load(path).astype(np.float32)
            if vec.ndim == 2:
                vec = vec.mean(axis=0)
            else:
                vec = vec.reshape(-1)
            if vec.shape[0] == self.embed_dim:
                vectors.append(vec)

        if len(vectors) == 0:
            raise RuntimeError(
                f"No valid {self.embed_dim}-d h_cls vectors found in: {hcls_dir_path}"
            )

        lib = torch.from_numpy(np.stack(vectors)).float().to(device)
        self.lib = lib
        self.lib_norm = self._norm(lib)

    @staticmethod
    def _norm(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, text_emb):
        query = self._norm(text_emb)
        sim = torch.matmul(query, self.lib_norm.t())

        k = min(self.topk, sim.shape[1])
        top_scores, top_idx = torch.topk(sim, k=k, dim=1)
        top_hcls = self.lib[top_idx]

        if k < self.topk:
            pad_h = torch.zeros(
                text_emb.shape[0], self.topk - k, self.embed_dim,
                device=text_emb.device, dtype=top_hcls.dtype,
            )
            pad_s = torch.full(
                (text_emb.shape[0], self.topk - k), -1e6,
                device=text_emb.device, dtype=top_scores.dtype,
            )
            top_hcls  = torch.cat([top_hcls, pad_h], dim=1)
            top_scores = torch.cat([top_scores, pad_s], dim=1)

        return top_hcls, top_scores


# ---------------------------------------------------------------------------
#  Token-level T5 encoding for cross-attention
# ---------------------------------------------------------------------------

def encode_token_level(text_input, st_model, device):
    """Encode one text string into per-token T5 embeddings.

    Returns
    -------
    text_tokens : torch.Tensor  (1, S, text_token_dim)  float32  on `device`
    text_token_lens : torch.Tensor  (1,)  int64  on `device`
    """
    transformer_module = st_model[0]   # SentenceTransformers Transformer wrapper
    features = transformer_module.tokenize([text_input])
    features = {k: v.to(device) for k, v in features.items()}
    with torch.no_grad():
        out = transformer_module.forward(features)
    token_embs = out['token_embeddings'].float()        # (1, S, D)
    attention_mask = features['attention_mask']         # (1, S)
    valid_len = int(attention_mask[0].sum().item())
    token_embs = token_embs[:, :valid_len, :]           # (1, valid_len, D)
    text_token_lens = torch.tensor([valid_len], dtype=torch.long, device=device)
    return token_embs, text_token_lens


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------

def set_reproducibility(seed_value, deterministic_mode=True):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic_mode)
    if deterministic_mode:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def load_state_strip_module(state_dict):
    out = {}
    for key, value in state_dict.items():
        new_key = ".".join(key.split(".")[1:]) if key.split(".")[0] == "module" else key
        out[new_key] = value
    return out


def sanitize_text_for_filename(raw_text):
    cleaned = raw_text.strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^\w\-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return (cleaned or "motion")[:120]


def check_required_files():
    required = [
        ("resume_pth",          resume_pth),
        ("resume_trans",        resume_trans),
        ("empty_text_path",     empty_text_path),
        ("reference_end_latent", reference_end_latent),
        ("mean_path",           mean_path),
        ("std_path",            std_path),
    ]
    for name, path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file missing: {name} -> {path}")
    if not disable_rag and not os.path.isdir(hcls_dir):
        raise FileNotFoundError(f"RAG retrieval directory missing: {hcls_dir}")
    if not os.path.isdir(t5_model_path):
        raise FileNotFoundError(f"T5 model path missing: {t5_model_path}")


# ---------------------------------------------------------------------------
#  Autoregressive sampling with MCA
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_motion_latents_with_stop(
    rag_model,
    text_encoder,
    retriever,
    input_text,
    empty_text_emb,
    reference_end,
    text_tokens,          # (1, S, text_token_dim) — pre-computed
    text_token_lens,      # (1,)                   — pre-computed
    disable_rag_flag=False,
    embed_dim=768,
    stop_threshold=0.1,
    length=300,
    unit_len=4,
    cfg=4.0,
    token_latent_dim=16,
    device=torch.device("cuda"),
):
    # Sentence-level text embedding (for RAG tokens)
    text_feat = text_encoder.encode([input_text])
    text_emb = torch.from_numpy(np.asarray(text_feat, dtype=np.float32)).to(device)

    if text_emb.shape[-1] != embed_dim:
        raise ValueError(
            f"text embedding dim mismatch: got {text_emb.shape[-1]}, expected {embed_dim}"
        )

    top_hcls = top_scores = None
    if not disable_rag_flag:
        top_hcls, top_scores = retriever.retrieve(text_emb)

    max_token_len = int(length) // int(unit_len)

    reference_end = reference_end.reshape(-1)
    if reference_end.numel() != token_latent_dim:
        raise ValueError(
            f"reference stop token dim mismatch: got {reference_end.numel()}, expected {token_latent_dim}"
        )
    reference_end = reference_end.view(1, token_latent_dim)

    xs = None
    for _ in range(max_token_len):
        prefix = xs if xs is not None else torch.zeros(
            (1, 0, token_latent_dim), device=device, dtype=torch.float32
        )

        next_token = rag_model.sample_next_with_cfg(
            motion_prefix=prefix,
            text_emb=text_emb,
            top3_h_cls=top_hcls,
            top3_sim_scores=top_scores,
            empty_text_emb=empty_text_emb,
            cfg_scale=cfg,
            temperature=1.0,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
        )

        distance_l2 = torch.sqrt(torch.sum((next_token - reference_end) ** 2))
        next_token = next_token.unsqueeze(1)
        xs = next_token if xs is None else torch.cat([xs, next_token], dim=1)

        if distance_l2 < stop_threshold:
            break

    if xs is None:
        xs = torch.zeros((1, 1, token_latent_dim), device=device, dtype=torch.float32)

    return xs


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    try:
        check_required_files()
        os.makedirs(output_dir, exist_ok=True)

        set_reproducibility(seed, deterministic_mode=deterministic)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Input text  : {text}")
        print(f"Device      : {device}")
        print(f"Checkpoint  : {resume_trans}")
        print(f"Use EMA     : {use_ema}")

        # ── VAE ──────────────────────────────────────────────────────────
        clip_range = [-30, 20]
        net = msa_vae.MSA_HumanVAE(
            hidden_size=hidden_size,
            down_t=down_t,
            stride_t=stride_t,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation="relu",
            latent_dim=latent_dim,
            clip_range=clip_range,
            trans_d_model=trans_d_model,
            trans_nhead=trans_nhead,
            trans_enc_layers=trans_enc_layers,
            trans_dec_layers=trans_dec_layers,
            trans_ff_size=trans_ff_size,
            trans_dropout=trans_dropout,
            clip_dim=clip_dim,
        ).to(device)

        print(f"Loading VAE checkpoint: {resume_pth}")
        ckpt_vae = torch.load(resume_pth, map_location="cpu")
        state_vae = ckpt_vae["net"] if (isinstance(ckpt_vae, dict) and "net" in ckpt_vae) else ckpt_vae
        net.load_state_dict(state_vae, strict=True)
        net.eval()

        # ── MCA transformer ───────────────────────────────────────────────
        print(f"Loading MCA checkpoint: {resume_trans}")
        ckpt = torch.load(resume_trans, map_location="cpu")

        resolved_head_type = ckpt.get("generative_head_type", "ddpm") if isinstance(ckpt, dict) else "ddpm"
        print(f"[HeadType] generative_head_type: {resolved_head_type}")

        config = LLaMAHFConfig.from_name("Normal_size")
        config.block_size = 78

        base_model = LLaMAHF(
            config,
            num_diffusion_head_layers,
            latent_dim,
            device,
            generative_head_type=resolved_head_type,
            num_flow_steps=num_flow_steps,
            flow_solver=flow_solver,
            rf_time_sampling=rf_time_sampling,
            rf_loss_type=rf_loss_type,
        )

        rag_model = LLaMARAGMultiTextCAWrapper(
            base_model=base_model,
            model_dim=config.n_embd,
            disable_rag=disable_rag,
            text_token_dim=text_token_dim,
            ca_every_n_layers=ca_every_n_layers,
            ca_n_head=ca_n_head,
            ff_mult=ff_mult,
        ).to(device)
        print(f"[MCA] {rag_model.extra_repr()}")

        # Load weights (EMA preferred)
        trans_key = "trans_ema" if (use_ema and "trans_ema" in ckpt) else "trans"
        rag_key   = "rag_ema"   if (use_ema and "rag_ema"   in ckpt) else "rag"
        if trans_key not in ckpt:
            raise KeyError(f"Checkpoint missing '{trans_key}' key.")
        if rag_key not in ckpt:
            raise KeyError(f"Checkpoint missing '{rag_key}' key.")

        base_model.load_state_dict(load_state_strip_module(ckpt[trans_key]), strict=False)
        rag_model.load_state_dict(load_state_strip_module(ckpt[rag_key]), strict=False)
        print(f"[Checkpoint] loaded '{trans_key}' + '{rag_key}'")
        rag_model.eval()

        # ── Text encoder ──────────────────────────────────────────────────
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise ImportError("Failed to import sentence_transformers.") from exc

        text_encoder = SentenceTransformer(t5_model_path)
        text_encoder.eval()

        # ── Auxiliary tensors ─────────────────────────────────────────────
        empty_text_emb = (
            torch.from_numpy(np.load(empty_text_path).astype(np.float32))
            .reshape(-1).to(device)
        )
        if empty_text_emb.shape[0] != text_embed_dim:
            raise ValueError(
                f"empty text embedding dim mismatch: got {empty_text_emb.shape[0]}, expected {text_embed_dim}"
            )

        ref_end = torch.from_numpy(np.load(reference_end_latent).astype(np.float32)).to(device)

        # ── RAG retriever ─────────────────────────────────────────────────
        retriever = None
        if not disable_rag:
            retriever = RAGRetriever(
                hcls_dir_path=hcls_dir,
                topk=retrieval_topk,
                embed_dim=text_embed_dim,
                device=device,
            )
        else:
            print("No-RAG mode: retrieval branch bypassed.")

        # ── Token-level embeddings for cross-attention ────────────────────
        text_tokens, text_token_lens = encode_token_level(text, text_encoder, device)
        print(f"[TextTokens] shape={tuple(text_tokens.shape)}, len={text_token_lens[0].item()}")

        # ── Autoregressive generation ─────────────────────────────────────
        motion_latents = sample_motion_latents_with_stop(
            rag_model=rag_model,
            text_encoder=text_encoder,
            retriever=retriever,
            input_text=text,
            empty_text_emb=empty_text_emb,
            reference_end=ref_end,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            disable_rag_flag=disable_rag,
            embed_dim=text_embed_dim,
            stop_threshold=threshold,
            length=max_length,
            unit_len=unit_length,
            cfg=cfg_scale,
            token_latent_dim=latent_dim,
            device=device,
        )
        print(f"[Generation] generated {motion_latents.shape[1]} tokens")

        # ── Decode & visualize ────────────────────────────────────────────
        motion = net.forward_decoder(motion_latents).squeeze(0).detach().cpu().numpy().astype(np.float32)

        mean = np.load(mean_path)
        std  = np.load(std_path)
        pred_xyz = recover_from_local_position(motion * std + mean, 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)

        timestamp   = time.strftime("%Y%m%d_%H%M%S")
        text_token  = sanitize_text_for_filename(text)
        stem        = f"MSA-T2M-MCA_{text_token}_{timestamp}"

        gif_path = os.path.join(output_dir, f"{stem}.gif")
        npy_path = os.path.join(output_dir, f"{stem}.npy")

        plot_3d.draw_to_batch(xyz, [text], [gif_path], fps=fps)
        np.save(npy_path, motion)

        print(f"[OK] GIF saved : {gif_path}")
        print(f"[OK] NPY saved : {npy_path}")
        print("The saved .npy is directly compatible with output_vis.py --input")

    except Exception as exc:
        print("[ERROR] Motion generation failed.")
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()

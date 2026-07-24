"""Single-query inference demo for MotionStreamer with global + local RAG tokens.

Extends msa_gen_motion.py with local RAG token conditioning:
  - Loads z latents from z_latent_dir to build a joint h_cls + z library.
  - Uses full z sequences (end token stripped) per retrieved motion.
  - Passes top_z_seqs [1, K, T_max, local_rag_dim] at every AR step.
  - Constructs LLaMARAGWrapper with L_local / local_rag_dim.

User-editable config is at the top of this file.
"""

import os
import re
import sys
import glob
import time
import random
import warnings

import numpy as np
import torch

from sentence_transformers import SentenceTransformer

from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model import LLaMARAGWrapper
import models.msa_vae as msa_vae
from visualization.recover_visualize import recover_from_local_position
import visualization.plot_3d_global as plot_3d

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# User-editable config area
# =========================
# text = "A person runs and then jumps happily because the acceptance of NeurIPS 2026"
#text = "the person rises from a laying position and walks in a clockwise circle, and then lays back down the ground"
text = "A figure dances ballet elegantly"
cfg_scale = 6.0
threshold = 0.1
retrieval_topk = 3
max_length = 300
fps = 30
disable_rag = False
seed = 123
deterministic = True
generative_head_type = "auto"  # auto|ddpm|rectified_flow
num_flow_steps = 50
flow_solver = "euler"
rf_time_sampling = "uniform"
rf_loss_type = "mse"

# Local RAG hyper-parameters — must match the trained checkpoint.
L_local = 4
local_rag_dim = 16
add_selfatten = False   # will be overridden from checkpoint if True

# Fixed paths
resume_pth = "Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/net_best_mpjpe.pth"
resume_trans = "Experiments/MotionStreamer_t2m_272_msa_rag_local_L_k3_crossattn/net_Iter100000.pth"
hcls_dir = "./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right"
z_latent_dir = "./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right"
empty_text_path = "./humanml3d_272/text_latents_t5/empty_text_embedding.npy"
reference_end_latent = "humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/reference_end_latent_msa_vae_t2m_272.npy"
t5_model_path = "sentencet5-xxl/"

# Architecture defaults (aligned with training)
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
std_path = "humanml3d_272/mean_std/Std.npy"

output_dir = "demo_output/MSA-T2M-Local"

# Inference diversity: randomly pick one from top-K retrieved entries.
# True  => each generation draws a different retrieved prior -> more diverse output
# False => weighted pooling (same as training distribution)
use_random_topk_inference = True


# ---------------------------------------------------------------------------
# Local RAG retrieval library
# ---------------------------------------------------------------------------
class RAGLocalRetriever:
    """In-memory retrieval library holding h_cls and full z sequences.

    All z sequences are padded to a global T_max at load time for efficient
    tensor indexing. retrieve() returns both padded sequences and valid lens.
    """

    def __init__(
        self,
        hcls_dir_path,
        z_latent_dir_path,
        topk=3,
        L_local=4,
        embed_dim=768,
        local_rag_dim=16,
        device=torch.device("cuda"),
    ):
        self.topk = int(topk)
        self.embed_dim = int(embed_dim)
        self.local_rag_dim = int(local_rag_dim)
        self.device = device

        hcls_files = sorted(glob.glob(os.path.join(hcls_dir_path, "*.npy")))
        if len(hcls_files) == 0:
            raise RuntimeError(f"No h_cls npy files found in: {hcls_dir_path}")

        hcls_vectors = []
        z_seqs_raw = []
        z_lens_list = []
        skipped = 0

        for hcls_path in hcls_files:
            stem = os.path.splitext(os.path.basename(hcls_path))[0]
            z_path = os.path.join(z_latent_dir_path, stem + ".npy")
            if not os.path.exists(z_path):
                skipped += 1
                continue

            h = np.load(hcls_path).astype(np.float32)
            h = h.mean(axis=0) if h.ndim == 2 else h.reshape(-1)
            if h.shape[0] != self.embed_dim:
                skipped += 1
                continue

            z_seq = np.load(z_path).astype(np.float32)[:-1]    # strip end token -> (T, local_rag_dim)
            hcls_vectors.append(h)
            z_seqs_raw.append(z_seq)
            z_lens_list.append(z_seq.shape[0])

        if len(hcls_vectors) == 0:
            raise RuntimeError(
                f"No valid h_cls+z pairs found. "
                f"hcls_dir={hcls_dir_path}, z_latent_dir={z_latent_dir_path}"
            )

        # Pad all sequences to global T_max for fast tensor indexing
        global_T_max = max(z_lens_list)
        N = len(hcls_vectors)
        z_library_np = np.zeros((N, global_T_max, local_rag_dim), dtype=np.float32)
        for i, (z_seq, T) in enumerate(zip(z_seqs_raw, z_lens_list)):
            z_library_np[i, :T] = z_seq

        lib = torch.from_numpy(np.stack(hcls_vectors)).float().to(device)
        self.library = lib
        self.library_norm = self._normalize(lib)
        self.z_library = torch.from_numpy(z_library_np).float().to(device)   # [N, T_max, dim]
        self.z_lens_library = torch.tensor(z_lens_list, dtype=torch.long, device=device)  # [N]
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
            text_emb: [B, embed_dim]
        Returns:
            top_hcls:    [B, K, embed_dim]
            top_scores:  [B, K]
            top_z_seqs: [B, K, T_max, local_rag_dim]
            top_z_lens: [B, K]
        """
        q = self._normalize(text_emb)
        sim = torch.matmul(q, self.library_norm.t())  # [B, N]

        k = min(self.topk, sim.shape[1])
        top_scores, top_idx = torch.topk(sim, k=k, dim=1)
        top_hcls    = self.library[top_idx]             # [B, k, embed_dim]
        top_z_seqs = self.z_library[top_idx]          # [B, k, T_max, local_rag_dim]
        top_z_lens = self.z_lens_library[top_idx]     # [B, k]

        # Pad if library is smaller than topk
        if k < self.topk:
            B = text_emb.shape[0]
            T_max = self.global_T_max
            pad_h   = torch.zeros(B, self.topk - k, self.embed_dim,
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
# Utilities
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


def infer_head_type_from_ckpt(ckpt, user_choice="auto"):
    valid = {"ddpm", "rectified_flow"}
    user_norm = str(user_choice).strip().lower()
    if user_norm == "auto":
        ckpt_type = ckpt.get("generative_head_type", None) if isinstance(ckpt, dict) else None
        ckpt_norm = str(ckpt_type).strip().lower() if ckpt_type is not None else None
        return ckpt_norm if ckpt_norm in valid else "ddpm"
    if user_norm in valid:
        return user_norm
    raise ValueError(
        f"Invalid generative_head_type: {user_choice}. Expected one of auto|ddpm|rectified_flow"
    )


def sanitize_text_for_filename(raw_text):
    cleaned = raw_text.strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^\w\-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return (cleaned[:120] or "motion")


def check_required_files():
    required = [
        ("resume_pth", resume_pth),
        ("resume_trans", resume_trans),
        ("empty_text_path", empty_text_path),
        ("reference_end_latent", reference_end_latent),
        ("mean_path", mean_path),
        ("std_path", std_path),
    ]
    for name, path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file missing: {name} -> {path}")
    if not disable_rag:
        if not os.path.isdir(hcls_dir):
            raise FileNotFoundError(f"RAG h_cls directory missing: {hcls_dir}")
        if not os.path.isdir(z_latent_dir):
            raise FileNotFoundError(f"RAG z_latent_dir missing: {z_latent_dir}")
    if not os.path.isdir(t5_model_path):
        raise FileNotFoundError(f"T5 model path missing: {t5_model_path}")


# ---------------------------------------------------------------------------
# Autoregressive generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_motion_latents_with_stop(
    rag_model,
    text_encoder,
    retriever,
    input_text,
    empty_text_emb,
    reference_end,
    disable_rag_flag=False,
    embed_dim=768,
    stop_threshold=0.1,
    length=300,
    unit_len=4,
    cfg=4.0,
    token_latent_dim=16,
    device=torch.device("cuda"),
    use_random_topk=False,
):
    text_feat = text_encoder.encode([input_text])
    text_emb = torch.from_numpy(np.asarray(text_feat, dtype=np.float32)).to(device)

    if text_emb.shape[-1] != embed_dim:
        raise ValueError(
            f"text embedding dim mismatch: got {text_emb.shape[-1]}, expected {embed_dim}"
        )

    top_hcls = None
    top_scores = None
    top_z_seqs = None
    top_z_lens = None

    if not disable_rag_flag:
        top_hcls, top_scores, top_z_seqs, top_z_lens = retriever.retrieve(text_emb)
        if use_random_topk and top_hcls.shape[1] > 1:
            K = top_hcls.shape[1]
            rand_k = torch.randint(0, K, (1,)).item()
            top_hcls   = top_hcls[:, rand_k : rand_k + 1, :]         # [1, 1, D]
            top_scores = top_scores[:, rand_k : rand_k + 1]           # [1, 1]
            top_z_seqs = top_z_seqs[:, rand_k : rand_k + 1, :, :]    # [1, 1, T_max, dim]
            top_z_lens = top_z_lens[:, rand_k : rand_k + 1]           # [1, 1]

    max_token_len = int(length) // int(unit_len)

    reference_end = reference_end.reshape(-1)
    if reference_end.numel() != token_latent_dim:
        raise ValueError(
            f"reference stop token dim mismatch: got {reference_end.numel()}, expected {token_latent_dim}"
        )
    reference_end = reference_end.view(1, token_latent_dim)

    xs = None
    print(f"  Generating tokens (max={max_token_len}, stop_threshold={stop_threshold}):")
    for step in range(max_token_len):
        prefix = (
            torch.zeros((1, 0, token_latent_dim), device=device, dtype=torch.float32)
            if xs is None
            else xs
        )

        next_token = rag_model.sample_next_with_cfg(
            motion_prefix=prefix,
            text_emb=text_emb,
            top3_h_cls=top_hcls,
            top3_sim_scores=top_scores,
            empty_text_emb=empty_text_emb,
            top_z_seqs=top_z_seqs,
            top_z_lens=top_z_lens,
            cfg_scale=cfg,
            temperature=1.0,
        )

        distance_l2 = torch.sqrt(torch.sum((next_token - reference_end) ** 2))
        next_token = next_token.unsqueeze(1)
        xs = next_token if xs is None else torch.cat([xs, next_token], dim=1)

        cur_frames = xs.shape[1] * unit_len
        norm_val = next_token.squeeze().norm().item()
        stop_flag = " [STOP]" if distance_l2 < stop_threshold else ""
        print(
            f"  token {step+1:3d}/{max_token_len} | "
            f"dist_to_end={distance_l2.item():.4f} | "
            f"token_norm={norm_val:.4f} | "
            f"frames_so_far={cur_frames}{stop_flag}"
        )

        if distance_l2 < stop_threshold:
            break

    print(f"  Done. Total tokens={xs.shape[1]}, frames={xs.shape[1] * unit_len}")
    if xs is None:
        xs = torch.zeros((1, 1, token_latent_dim), device=device, dtype=torch.float32)

    return xs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        check_required_files()
        os.makedirs(output_dir, exist_ok=True)

        set_reproducibility(seed, deterministic_mode=deterministic)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Input text : {text}")
        print(f"Device     : {device}")
        print(f"L_local    : {L_local}  (block_size = {78 + L_local})")

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

        print(f"Loading RAG checkpoint: {resume_trans}")
        ckpt_rag = torch.load(resume_trans, map_location="cpu")
        resolved_head_type = infer_head_type_from_ckpt(ckpt_rag, generative_head_type)
        print(f"[HeadType] resolved: {resolved_head_type}")

        # Read L_local from checkpoint if stored (allows mismatched config detection)
        ckpt_L_local = int(ckpt_rag.get("L_local", L_local)) if isinstance(ckpt_rag, dict) else L_local
        if ckpt_L_local != L_local:
            print(
                f"[WARN] L_local mismatch: script={L_local}, checkpoint={ckpt_L_local}. "
                f"Using checkpoint value {ckpt_L_local}."
            )
        effective_L_local = ckpt_L_local

        # Read add_selfatten from checkpoint
        effective_add_selfatten = bool(ckpt_rag.get("add_selfatten", add_selfatten)) if isinstance(ckpt_rag, dict) else add_selfatten
        if effective_add_selfatten:
            print("[INFO] add_selfatten=True (loaded from checkpoint)")

        config = LLaMAHFConfig.from_name("Normal_size")
        config.block_size = 78 + effective_L_local
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
        rag_model = LLaMARAGWrapper(
            base_model=base_model,
            model_dim=config.n_embd,
            disable_rag=disable_rag,
            L_local=effective_L_local,
            local_rag_dim=local_rag_dim,
            add_selfatten=effective_add_selfatten,
        ).to(device)

        if "trans" not in ckpt_rag:
            raise KeyError("Checkpoint must contain a 'trans' key.")
        rag_model.base_model.load_state_dict(
            load_state_strip_module(ckpt_rag["trans"]), strict=False
        )
        if "rag" in ckpt_rag:
            rag_model.load_state_dict(load_state_strip_module(ckpt_rag["rag"]), strict=False)
        elif not disable_rag:
            print("[WARN] Checkpoint has no 'rag' key; RAG parameters use random init.")

        rag_model.eval()

        text_encoder = SentenceTransformer(t5_model_path)
        text_encoder.eval()

        empty_text_emb = (
            torch.from_numpy(np.load(empty_text_path).astype(np.float32))
            .reshape(-1).to(device)
        )
        if empty_text_emb.shape[0] != text_embed_dim:
            raise ValueError(
                f"empty text embedding dim mismatch: got {empty_text_emb.shape[0]}, "
                f"expected {text_embed_dim}"
            )

        retriever = None
        if not disable_rag:
            retriever = RAGLocalRetriever(
                hcls_dir_path=hcls_dir,
                z_latent_dir_path=z_latent_dir,
                topk=retrieval_topk,
                L_local=effective_L_local,
                embed_dim=text_embed_dim,
                local_rag_dim=local_rag_dim,
                device=device,
            )

        ref_end = torch.from_numpy(np.load(reference_end_latent).astype(np.float32)).to(device)

        motion_latents = sample_motion_latents_with_stop(
            rag_model=rag_model,
            text_encoder=text_encoder,
            retriever=retriever,
            input_text=text,
            empty_text_emb=empty_text_emb,
            reference_end=ref_end,
            disable_rag_flag=disable_rag,
            embed_dim=text_embed_dim,
            stop_threshold=threshold,
            length=max_length,
            unit_len=unit_length,
            cfg=cfg_scale,
            token_latent_dim=latent_dim,
            device=device,
            use_random_topk=use_random_topk_inference,
        )

        motion = net.forward_decoder(motion_latents).squeeze(0).detach().cpu().numpy().astype(np.float32)

        mean = np.load(mean_path)
        std = np.load(std_path)
        pred_xyz = recover_from_local_position(motion * std + mean, 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        text_token = sanitize_text_for_filename(text)
        stem = f"MSA-T2M-Local_{text_token}_{timestamp}"

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

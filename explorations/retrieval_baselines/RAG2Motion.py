"""
RAG2Motion.py
=============
Validate that RAG tokens (h_cls) can be decoded back to motion.

Pipeline
--------
1. Text Encoder  : encode input text -> text_emb  (T5 SentenceTransformer)
2. Retrieval     : cosine search on h_cls library -> top-k h_cls tokens (768-d)
3. Decoder       : for each h_cls
                     TransformerAE decoder  -> mu  (seq_len, 16)
                     CNN decoder            -> motion  (T, 272)
4. Renderer      : motion -> joint positions -> GIF

Usage
-----
  python RAG2Motion.py
  Edit the "User Config" block below to change text / topk / paths.
"""

import os
import glob
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# User Config  (edit here)
# =============================================================================

text = "A person gracefully dancing ballet"
topk = 5                  # number of retrieved h_cls tokens to decode
fps = 30
output_dir = "demo_output/RAG2Motion"

# Original motion data directory – used to look up per-motion frame count,
# so the Transformer decoder can use the correct latent sequence length.
motion_data_dir = "humanml3d_272/motion_data"

# --- Model / data paths ---
resume_pth = (
    "Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/"
    "net_best_mpjpe.pth"
)
hcls_dir = (
    "humanml3d_272/h_cls_latents_msa_vae/"
    "MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right"
)
t5_model_path = "sentencet5-xxl/"
mean_path = "humanml3d_272/mean_std/Mean.npy"
std_path  = "humanml3d_272/mean_std/Std.npy"

# --- MSA-VAE hyper-params (must match checkpoint) ---
hidden_size          = 1024
down_t               = 2
stride_t             = 2
depth                = 3
dilation_growth_rate = 3
latent_dim           = 16
trans_d_model        = 768
trans_nhead          = 8
trans_enc_layers     = 6
trans_dec_layers     = 6
trans_ff_size        = 2048
trans_dropout        = 0.1
clip_dim             = 768

# =============================================================================
# Imports (after config so TOKENIZERS_PARALLELISM is set first)
# =============================================================================
import re
import models.msa_vae as msa_vae
from visualization.recover_visualize import recover_from_local_position
import visualization.plot_3d_global as plot_3d


# =============================================================================
# Module 1 – Text Encoder
# =============================================================================

def build_text_encoder(model_path: str):
    """Load a SentenceTransformer T5 text encoder."""
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model_path)
    encoder.eval()
    return encoder


@torch.no_grad()
def encode_text(encoder, text: str, device: torch.device) -> torch.Tensor:
    """Return (1, D) float32 tensor on *device*."""
    feat = encoder.encode([text])
    emb  = torch.from_numpy(np.asarray(feat, dtype=np.float32)).to(device)
    return emb                      # (1, D)


# =============================================================================
# Module 2 – Retriever
# =============================================================================

class HClsRetriever:
    """
    Load all h_cls .npy files, build a library, and retrieve top-k by cosine
    similarity with a query text embedding.
    """

    def __init__(self, hcls_dir_path: str, embed_dim: int = 768,
                 device: torch.device = torch.device("cpu")):
        files = sorted(glob.glob(os.path.join(hcls_dir_path, "*.npy")))
        if not files:
            raise RuntimeError(f"No h_cls .npy files found in: {hcls_dir_path}")

        vectors, names = [], []
        for path in files:
            vec = np.load(path).astype(np.float32)
            vec = vec.mean(axis=0) if vec.ndim == 2 else vec.reshape(-1)
            if vec.shape[0] == embed_dim:
                vectors.append(vec)
                names.append(os.path.basename(path))

        if not vectors:
            raise RuntimeError(
                f"No valid {embed_dim}-d h_cls vectors found in: {hcls_dir_path}"
            )

        self.lib      = torch.from_numpy(np.stack(vectors)).float().to(device)
        self.lib_norm = self._l2_norm(self.lib)
        self.names    = names
        self.embed_dim = embed_dim
        self.device    = device
        print(f"[Retriever] loaded {len(vectors)} h_cls vectors from {hcls_dir_path}")

    @staticmethod
    def _l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, text_emb: torch.Tensor, k: int):
        """
        Args:
            text_emb: (1, D) query embedding
            k:        number of results
        Returns:
            top_hcls:   (k, D) float32 tensor on self.device
            top_scores: (k,) float32 tensor
            top_names:  list[str]
        """
        q   = self._l2_norm(text_emb)              # (1, D)
        sim = torch.matmul(q, self.lib_norm.t())   # (1, N)
        k_  = min(k, sim.shape[1])
        scores, idx = torch.topk(sim, k=k_, dim=1)
        idx = idx.squeeze(0)                       # (k,)
        return (
            self.lib[idx],                         # (k, D)
            scores.squeeze(0),                     # (k,)
            [self.names[i] for i in idx.tolist()],
        )


# =============================================================================
# Module 3 – Decoder  (h_cls -> motion)
# =============================================================================

def build_vae(device: torch.device) -> msa_vae.MSA_HumanVAE:
    """Instantiate MSA_HumanVAE with the config defined in User Config."""
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

    print(f"[Decoder] loading VAE checkpoint: {resume_pth}")
    ckpt = torch.load(resume_pth, map_location="cpu")
    state = ckpt["net"] if (isinstance(ckpt, dict) and "net" in ckpt) else ckpt
    # strip DDP "module." prefix if present
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def lookup_seq_len(motion_id: str, data_dir: str, down_t: int, stride_t: int) -> int:
    """
    Look up the original frame count T for *motion_id* from the motion data
    directory, then compute the latent sequence length that the CNN encoder
    would produce:  seq_len = ceil(T / stride_t^down_t).

    This is the ground-truth length that was used during VAE training, so
    using it here gives the most faithful Transformer decoder output.
    """
    motion_path = os.path.join(data_dir, f"{motion_id}.npy")
    if not os.path.exists(motion_path):
        raise FileNotFoundError(
            f"Motion data file not found for id '{motion_id}': {motion_path}\n"
            "  Ensure motion_data_dir points to the directory containing per-motion .npy files."
        )
    T = np.load(motion_path, mmap_mode="r").shape[0]
    seq_len = T
    for _ in range(down_t):
        seq_len = (seq_len + stride_t - 1) // stride_t   # ceil division
    return seq_len


@torch.no_grad()
def hcls_to_motion(
    net: msa_vae.MSA_HumanVAE,
    h_cls: torch.Tensor,   # (D,) or (1, D)
    seq_len: int,
) -> np.ndarray:
    """
    Decode one h_cls token into a (T, 272) motion array (normalised).

    Steps
    -----
      h_cls (1, 768) -> TransformerDecoder -> mu (1, seq_len, 16)
      mu             -> CNNDecoder         -> motion (1, T, 272)

    seq_len is the number of latent tokens the CNN encoder produced for the
    original motion; it must be passed in explicitly because the Transformer
    decoder uses zero-initialised positional queries and has no mechanism to
    infer output length from h_cls alone.
    """
    if h_cls.dim() == 1:
        h_cls = h_cls.unsqueeze(0)   # (1, D)

    # Step A: Transformer AE decoder  h_cls -> mu  (reconstructed latent tokens)
    mu_recon = net.msa_vae.decode_transformer(h_cls, seq_len=seq_len)   # (1, seq_len, 16)

    # Step B: CNN decoder  mu -> motion
    motion_tensor = net.forward_decoder(mu_recon)   # (1, T, 272)

    return motion_tensor.squeeze(0).cpu().numpy()   # (T, 272)


# =============================================================================
# Module 4 – Renderer  (motion -> GIF)
# =============================================================================

def load_normalisation(mean_path: str, std_path: str):
    return np.load(mean_path), np.load(std_path)


def sanitize(text: str, max_len: int = 60) -> str:
    s = re.sub(r"\s+", "_", text.strip().lower())
    s = re.sub(r"[^\w\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_") or "motion"
    return s[:max_len]


def render_motion_to_gif(
    motion_norm: np.ndarray,   # (T, 272) – normalised
    mean: np.ndarray,
    std: np.ndarray,
    title: str,
    out_path: str,
    fps: int = 20,
) -> None:
    """Denormalise, recover joint positions, and save a GIF."""
    motion = motion_norm * std + mean       # (T, 272)
    joints = recover_from_local_position(motion, 22)   # (T, 22, 3)
    xyz    = joints.reshape(1, -1, 22, 3)  # (1, T, 22, 3)
    plot_3d.draw_to_batch(xyz, [title], [out_path], fps=fps)


def render_original_to_gif(
    motion_id: str,
    data_dir: str,
    title: str,
    out_path: str,
    fps: int = 20,
) -> None:
    """Load the raw (unnormalised) motion .npy, recover joints, and save a GIF.

    Files in motion_data/ are stored in raw space; the dataset normalises them
    on-the-fly with (x - mean) / std before training.  Here we load raw data
    directly and pass it straight to recover_from_local_position.
    """
    motion_path = os.path.join(data_dir, f"{motion_id}.npy")
    motion = np.load(motion_path).astype(np.float32)         # (T, 272) raw
    joints = recover_from_local_position(motion, 22)         # (T, 22, 3)
    xyz    = joints.reshape(1, -1, 22, 3)
    plot_3d.draw_to_batch(xyz, [title], [out_path], fps=fps)


# =============================================================================
# Main
# =============================================================================

def main():
    # ---- sanity checks ----
    for label, path in [
        ("resume_pth", resume_pth),
        ("hcls_dir",   hcls_dir),
        ("t5_model",   t5_model_path),
        ("mean",       mean_path),
        ("std",        std_path),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Config] {label} not found: {path}")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] device={device}  text='{text}'  topk={topk}")

    # ---- build components ----
    print("\n[Step 1] Building text encoder …")
    text_encoder = build_text_encoder(t5_model_path)

    print("\n[Step 2] Building h_cls retriever …")
    retriever = HClsRetriever(hcls_dir, embed_dim=trans_d_model, device=device)

    print("\n[Step 3] Building VAE decoder …")
    net = build_vae(device)

    mean, std = load_normalisation(mean_path, std_path)

    # ---- encode text ----
    print(f"\n[Step 4] Encoding text: '{text}'")
    text_emb = encode_text(text_encoder, text, device)   # (1, 768)
    print(f"         text_emb shape: {text_emb.shape}")

    # ---- retrieve top-k h_cls ----
    print(f"\n[Step 5] Retrieving top-{topk} h_cls tokens …")
    top_hcls, top_scores, top_names = retriever.retrieve(text_emb, k=topk)
    for rank, (name, score) in enumerate(zip(top_names, top_scores.tolist()), 1):
        print(f"  [{rank}] {name}  cosine={score:.4f}")

    # ---- decode each h_cls -> motion -> GIF ----
    print(f"\n[Step 6] Decoding {len(top_hcls)} h_cls token(s) to motion …")
    text_tag = sanitize(text)

    for rank, (h_cls_vec, score, name) in enumerate(
        zip(top_hcls, top_scores.tolist(), top_names), 1
    ):
        src_id = os.path.splitext(name)[0]
        out_gif = os.path.join(output_dir, f"rank{rank:02d}_{src_id}_{text_tag}.gif")

        seq_len = lookup_seq_len(src_id, motion_data_dir, down_t, stride_t)
        print(f"\n  [{rank}/{len(top_hcls)}] h_cls={name}  cosine={score:.4f}  seq_len={seq_len}")

        # --- decoded reconstruction ---
        motion_norm = hcls_to_motion(net, h_cls_vec.to(device), seq_len=seq_len)
        print(f"         decoded motion shape: {motion_norm.shape}  (T={motion_norm.shape[0]})")
        title_dec = f"[decoded rank {rank}] {text}\n(src: {src_id}, cos={score:.3f})"
        render_motion_to_gif(motion_norm, mean, std, title=title_dec, out_path=out_gif, fps=fps)
        print(f"         decoded → {out_gif}")

        # --- original motion ---
        out_gif_orig = os.path.join(output_dir, f"rank{rank:02d}_{src_id}_ORIGINAL.gif")
        title_orig = f"[original] {src_id}"
        render_original_to_gif(src_id, motion_data_dir,
                               title=title_orig, out_path=out_gif_orig, fps=fps)
        print(f"         original → {out_gif_orig}")

    print(f"\n[Done] All {len(top_hcls)} GIF(s) saved under: {output_dir}/")


if __name__ == "__main__":
    main()

"""
ReMoDiffuse: Text-to-Motion visualization script (HumanML3D-272).

Usage:
    conda activate mogen
    cd /share/home/tm878032203900000/a878044490/MotionStreamer
    python remodiffuse_gen_motion.py

On first run the retrieval database (~23K training motions) is built
automatically and cached to disk. Subsequent runs load it directly.
"""

import os
import re
import sys
import time
import random
import warnings

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REMODIFFUSE_DIR = os.path.join(SCRIPT_DIR, "ReMoDiffuse")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REMODIFFUSE_DIR)

import mmcv                                                 # noqa: E402  (mogen env)
from mogen.models import build_architecture                  # noqa: E402
from mmcv.runner import load_checkpoint                      # noqa: E402
from visualization.recover_visualize import recover_from_local_position  # noqa
import visualization.plot_3d_global as plot_3d               # noqa

warnings.filterwarnings("ignore")


# =========================================================
# User-editable configuration (edit here directly)
# =========================================================
text          = "A person gracefully dancing ballet"
motion_length = 196        # frames, must be in [16, 196]
fps           = 30
seed          = 123

# Paths
checkpoint_path   = "ReMoDiffuse/epoch_20.pth"
mean_path         = "humanml3d_272/mean_std/Mean.npy"
std_path          = "humanml3d_272/mean_std/Std.npy"
retrieval_db_path = "ReMoDiffuse/data/database/t2m_text_train_272.npz"
output_dir        = "demo_output/ReMoDiffuse"

# Training data (used only when building the retrieval database for the first time)
motion_data_dir = "humanml3d_272/motion_data"
texts_dir       = "humanml3d_272/texts"
train_split_txt = "humanml3d_272/split/train.txt"

# =========================================================
# Model architecture constants (derived from epoch_20.pth weights)
# =========================================================
INPUT_FEATS      = 272    # confirmed from model.joint_embed.weight (512, 272)
MAX_SEQ_LEN      = 196
LATENT_DIM       = 512
TIME_EMBED_DIM   = 2048
TEXT_LATENT_DIM  = 256
FF_SIZE          = 1024
NUM_HEADS        = 8
DROPOUT          = 0
NUM_LAYERS       = 4
NUM_RETRIEVAL    = 2
TOPK             = 2


# =========================================================
# Helpers
# =========================================================

def set_seed(seed_val: int = 123):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sanitize(raw_text: str, max_len: int = 100) -> str:
    s = raw_text.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "motion")[:max_len]


def motion_temporal_filter(motion, sigma: float = 1.0):
    """Smooth joint trajectories with a Gaussian filter (same as ReMoDiffuse tools)."""
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)


# =========================================================
# Build retrieval database on first run
# =========================================================

def _clip_encode_text_seq(clip_model, caption: str, device):
    """
    Return (seq_feat: np.ndarray [77, 512], pooled_feat: np.ndarray [512]).
    Uses the same CLIP (ViT-B/32) that is embedded in the model.
    """
    import clip as clip_pkg
    with torch.no_grad():
        tokens = clip_pkg.tokenize([caption], truncate=True).to(device)  # (1, 77)
        x = clip_model.token_embedding(tokens).type(clip_model.dtype)     # (1, 77, 512)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)                                            # (77, 1, 512)
        x = clip_model.transformer(x)
        x = clip_model.ln_final(x).float()                                # (77, 1, 512)
        seq_feat = x.permute(1, 0, 2).squeeze(0).cpu().numpy()            # (77, 512)
        eos_idx  = tokens.argmax(dim=-1)[0].item()
        pooled   = (x[eos_idx, 0, :] @ clip_model.text_projection.float()).cpu().numpy()  # (512,)
    return seq_feat.astype(np.float32), pooled.astype(np.float32)


def build_retrieval_database(clip_model, mean: np.ndarray, std: np.ndarray, device):
    """
    Build a retrieval database npz from the HumanML3D-272 training split.
    Saved to `retrieval_db_path` and loaded by the model on inference.
    """
    os.makedirs(os.path.dirname(os.path.abspath(retrieval_db_path)), exist_ok=True)

    with open(train_split_txt) as f:
        train_ids = [l.strip() for l in f if l.strip()]

    all_text_features = []
    all_captions      = []
    all_motions       = []
    all_m_lengths     = []
    all_clip_seq      = []

    print(f"[DB] Building retrieval database from {len(train_ids)} training samples ...")
    skipped = 0
    for idx, mid in enumerate(train_ids):
        if idx % 2000 == 0:
            print(f"    {idx}/{len(train_ids)} processed, {skipped} skipped")

        motion_path = os.path.join(motion_data_dir, mid + ".npy")
        text_path   = os.path.join(texts_dir,       mid + ".txt")
        if not os.path.exists(motion_path) or not os.path.exists(text_path):
            skipped += 1
            continue

        # --- motion ---
        motion   = np.load(motion_path).astype(np.float32)   # (T, 272)
        m_length = min(len(motion), MAX_SEQ_LEN)
        motion   = motion[:m_length]
        # normalise
        motion_norm = (motion - mean) / (std + 1e-8)
        # pad to MAX_SEQ_LEN
        if m_length < MAX_SEQ_LEN:
            pad = np.zeros((MAX_SEQ_LEN - m_length, INPUT_FEATS), dtype=np.float32)
            motion_norm = np.concatenate([motion_norm, pad], axis=0)

        # --- text (first annotation) ---
        with open(text_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        caption = lines[0].split("#")[0].strip() if lines else mid

        # --- CLIP features ---
        seq_feat, pooled = _clip_encode_text_seq(clip_model, caption, device)

        all_text_features.append(pooled)
        all_captions.append(caption)
        all_motions.append(motion_norm)
        all_m_lengths.append(m_length)
        all_clip_seq.append(seq_feat)

    np.savez(
        retrieval_db_path,
        text_features  = np.array(all_text_features,  dtype=np.float32),
        captions       = np.array(all_captions),
        motions        = np.array(all_motions,         dtype=np.float32),
        m_lengths      = np.array(all_m_lengths,       dtype=np.int32),
        clip_seq_features = np.array(all_clip_seq,     dtype=np.float32),
    )
    n = len(all_captions)
    print(f"[DB] Database built: {n} samples -> {retrieval_db_path}")
    return n


def _create_dummy_db(path: str):
    """Create a minimal 1-sample database so the model can be constructed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(
        path,
        text_features     = np.zeros((1, 512),             dtype=np.float32),
        captions          = np.array(["dummy"]),
        motions           = np.zeros((1, MAX_SEQ_LEN, INPUT_FEATS), dtype=np.float32),
        m_lengths         = np.array([16], dtype=np.int32),
        clip_seq_features = np.zeros((1, 77, 512),         dtype=np.float32),
    )


def _reload_db_into_model(model, db_path: str, device):
    """Replace in-memory retrieval data in an already-loaded model."""
    db = np.load(db_path)
    database = model.model.database
    database.text_features   = torch.Tensor(db["text_features"])
    database.captions        = db["captions"]
    database.motions         = db["motions"]
    database.m_lengths       = db["m_lengths"]          # keep as numpy array for fancy indexing
    database.clip_seq_features = db["clip_seq_features"]
    database.results         = {}    # clear retrieval cache


# =========================================================
# Model config (matches epoch_20.pth weights exactly)
# =========================================================

def _build_model_cfg(db_path: str) -> dict:
    return dict(
        type="MotionDiffusion",
        model=dict(
            type="ReMoDiffuseTransformer",
            input_feats      = INPUT_FEATS,
            max_seq_len      = MAX_SEQ_LEN,
            latent_dim       = LATENT_DIM,
            time_embed_dim   = TIME_EMBED_DIM,
            num_layers       = NUM_LAYERS,
            ca_block_cfg=dict(
                type             = "SemanticsModulatedAttention",
                latent_dim       = LATENT_DIM,
                text_latent_dim  = TEXT_LATENT_DIM,
                num_heads        = NUM_HEADS,
                dropout          = DROPOUT,
                time_embed_dim   = TIME_EMBED_DIM,
            ),
            ffn_cfg=dict(
                latent_dim     = LATENT_DIM,
                ffn_dim        = FF_SIZE,
                dropout        = DROPOUT,
                time_embed_dim = TIME_EMBED_DIM,
            ),
            text_encoder=dict(
                pretrained_model = "clip",
                latent_dim       = TEXT_LATENT_DIM,
                num_layers       = 2,
                ff_size          = 2048,
                dropout          = DROPOUT,
                use_text_proj    = False,
            ),
            retrieval_cfg=dict(
                num_retrieval    = NUM_RETRIEVAL,
                stride           = 4,
                num_layers       = 2,
                num_motion_layers= 2,
                kinematic_coef   = 0.1,
                topk             = TOPK,
                retrieval_file   = db_path,
                latent_dim       = LATENT_DIM,
                output_dim       = LATENT_DIM,
                max_seq_len      = MAX_SEQ_LEN,
                num_heads        = NUM_HEADS,
                ff_size          = FF_SIZE,
                dropout          = DROPOUT,
                ffn_cfg=dict(
                    latent_dim = LATENT_DIM,
                    ffn_dim    = FF_SIZE,
                    dropout    = DROPOUT,
                ),
                sa_block_cfg=dict(
                    type       = "EfficientSelfAttention",
                    latent_dim = LATENT_DIM,
                    num_heads  = NUM_HEADS,
                    dropout    = DROPOUT,
                ),
            ),
            scale_func_cfg=dict(
                coarse_scale = 6.5,
                both_coef    = 0.52351,
                text_coef    = -0.28419,
                retr_coef    = 2.39872,
            ),
        ),
        loss_recon=dict(type="MSELoss", loss_weight=1, reduction="none"),
        diffusion_train=dict(
            beta_scheduler  = "linear",
            diffusion_steps = 1000,
            model_mean_type = "start_x",
            model_var_type  = "fixed_large",
        ),
        diffusion_test=dict(
            beta_scheduler  = "linear",
            diffusion_steps = 1000,
            model_mean_type = "start_x",
            model_var_type  = "fixed_large",
            respace         = "15,15,8,6,6",
        ),
        inference_type="ddim",
    )


# =========================================================
# main
# =========================================================

def main():
    try:
        assert 16 <= motion_length <= MAX_SEQ_LEN, \
            f"motion_length must be in [16, {MAX_SEQ_LEN}], got {motion_length}"

        os.makedirs(output_dir, exist_ok=True)
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[ReMoDiffuse] Input text  : {text}")
        print(f"[ReMoDiffuse] Motion frames: {motion_length}")
        print(f"[ReMoDiffuse] Device       : {device}")

        mean = np.load(mean_path)
        std  = np.load(std_path)

        # ----------------------------------------------------------
        # 1) If real retrieval database does not exist we need to
        #    build it. We bootstrap with a dummy db first so the
        #    model can be constructed, then swap in real data.
        # ----------------------------------------------------------
        need_build_db = not os.path.exists(retrieval_db_path)
        init_db_path  = retrieval_db_path
        if need_build_db:
            dummy_path = retrieval_db_path + ".dummy.npz"
            _create_dummy_db(dummy_path)
            init_db_path = dummy_path
            print("[DB] Real retrieval database not found – will build after model init.")

        # ----------------------------------------------------------
        # 2) Build model and load checkpoint
        # ----------------------------------------------------------
        print(f"[Model] Building ReMoDiffuseTransformer (input_feats={INPUT_FEATS}) ...")
        cfg = _build_model_cfg(init_db_path)
        model = build_architecture(cfg)

        print(f"[Model] Loading checkpoint: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, map_location="cpu", strict=False)
        model.eval()
        model.to(device)

        # ----------------------------------------------------------
        # 3) Build real database if needed
        # ----------------------------------------------------------
        if need_build_db:
            build_retrieval_database(model.model.clip, mean, std, device)
            os.remove(dummy_path)
            _reload_db_into_model(model, retrieval_db_path, device)
        else:
            # Reload to ensure database is on correct device / consistent
            _reload_db_into_model(model, retrieval_db_path, device)

        # ----------------------------------------------------------
        # 4) Inference
        # ----------------------------------------------------------
        ml_tensor = torch.LongTensor([motion_length]).to(device)
        if INPUT_FEATS == 272:
            motion_input = torch.zeros(1, motion_length, INPUT_FEATS).to(device)
        motion_mask = torch.ones(1, motion_length).to(device)

        batch = {
            "motion"       : motion_input,
            "motion_mask"  : motion_mask,
            "motion_length": ml_tensor,
            "motion_metas" : [{"text": text}],
        }

        print("[Inference] Running DDIM sampling ...")
        with torch.no_grad():
            batch["inference_kwargs"] = {}
            result = model(**batch)[0]["pred_motion"]          # (T, 272) normalised

        pred_motion_norm = result.cpu().numpy().astype(np.float32)   # normalised (for saving)
        pred_motion = pred_motion_norm * std + mean                   # denormalised (for visualisation)

        # ----------------------------------------------------------
        # 5) Recover 3-D joint positions and visualise
        # ----------------------------------------------------------
        pred_xyz = recover_from_local_position(pred_motion, 22)   # (T, 22, 3)
        pred_xyz = motion_temporal_filter(pred_xyz, sigma=2.5)     # smooth
        xyz      = pred_xyz.reshape(1, -1, 22, 3)

        timestamp  = time.strftime("%Y%m%d_%H%M%S")
        stem       = f"ReMoDiffuse_{sanitize(text)}_{timestamp}"
        gif_path   = os.path.join(output_dir, f"{stem}.gif")
        npy_path   = os.path.join(output_dir, f"{stem}.npy")

        plot_3d.draw_to_batch(xyz, [text], [gif_path], fps=fps)
        np.save(npy_path, pred_motion_norm)   # save normalised, same as msa/motionstreamer

        print(f"[OK] GIF saved : {gif_path}")
        print(f"[OK] NPY saved : {npy_path}")
        print("[OK] .npy is compatible with output_vis.py --input")

    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()

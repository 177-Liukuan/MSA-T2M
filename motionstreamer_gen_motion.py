import os
import re
import time
import random
import warnings

import numpy as np
import torch

from models.llama_model import LLaMAHF, LLaMAHFConfig
import models.tae as tae
from sentence_transformers import SentenceTransformer
from visualization.recover_visualize import recover_from_local_position
import visualization.plot_3d_global as plot_3d


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================================================
# Hardcoded configuration (edit here directly, no argparse)
# =========================================================
text = "The person was walking forward slowly"

cfg_scale = 4.0
threshold = 0.1
max_length = 300
fps = 30
seed = 123
deterministic = True

# Keep architecture defaults aligned with demo_t2m.py / option_transformer.py
hidden_size = 1024
down_t = 2
stride_t = 2
depth = 3
dilation_growth_rate = 3
latent_dim = 16
num_diffusion_head_layers = 9
unit_length = 4

# Required model/data paths from task description
resume_pth = "Experiments/causal_TAE_t2m_272_h100_20260203/net_last.pth"
resume_trans = "Experiments/MotionStreamer_t2m_272_cached_embeddings_8gpu_bf16/latest.pth"

# Text encoder and auxiliary files (same style as demo_t2m.py)
t5_model_path = "sentencet5-xxl/"
reference_end_latent_path = "reference_end_latent_t2m_272.npy"
mean_path = "humanml3d_272/mean_std/Mean.npy"
std_path = "humanml3d_272/mean_std/Std.npy"

# Output directory
output_dir = "demo_output/MotionStreamer"


def set_reproducibility(seed_value: int, deterministic_mode: bool = True):
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


def sanitize_text_for_filename(raw_text: str) -> str:
    cleaned = raw_text.strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^\w\-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "motion"
    return cleaned[:120]


def load_state_strip_module(state_dict):
    out = {}
    for key, value in state_dict.items():
        if key.split(".")[0] == "module":
            out[".".join(key.split(".")[1:])] = value
        else:
            out[key] = value
    return out


def check_required_paths():
    required_files = [
        ("resume_pth", resume_pth),
        ("resume_trans", resume_trans),
        ("reference_end_latent", reference_end_latent_path),
        ("mean", mean_path),
        ("std", std_path),
    ]

    for name, path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {name} -> {path}")

    if not os.path.isdir(t5_model_path):
        raise FileNotFoundError(f"T5 model path not found: {t5_model_path}")


def main():
    try:
        check_required_paths()
        os.makedirs(output_dir, exist_ok=True)

        set_reproducibility(seed, deterministic_mode=deterministic)
        comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Input text: {text}")
        print(f"Device: {comp_device}")

        # ---------------------------------------------------------
        # 1) Build text encoder (same family as demo_t2m.py)
        # ---------------------------------------------------------
        t5_model = SentenceTransformer(t5_model_path)
        t5_model.eval()
        for p in t5_model.parameters():
            p.requires_grad = False

        # ---------------------------------------------------------
        # 2) Build Causal TAE (VAE decoder path)
        # ---------------------------------------------------------
        clip_range = [-30, 20]
        net = tae.Causal_HumanTAE(
            hidden_size=hidden_size,
            down_t=down_t,
            stride_t=stride_t,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation="relu",
            latent_dim=latent_dim,
            clip_range=clip_range,
        )

        print(f"Loading Causal TAE checkpoint: {resume_pth}")
        ckpt_vae = torch.load(resume_pth, map_location="cpu")
        if "net" not in ckpt_vae:
            raise KeyError("Causal TAE checkpoint must contain key: 'net'")
        net.load_state_dict(ckpt_vae["net"], strict=True)
        net.eval()
        net.to(comp_device)

        # ---------------------------------------------------------
        # 3) Build original MotionStreamer AR+Diffusion model
        # ---------------------------------------------------------
        config = LLaMAHFConfig.from_name("Normal_size")
        config.block_size = 78
        trans_encoder = LLaMAHF(config, num_diffusion_head_layers, latent_dim, comp_device)

        print(f"Loading MotionStreamer checkpoint: {resume_trans}")
        ckpt_trans = torch.load(resume_trans, map_location="cpu")
        if "trans" not in ckpt_trans:
            raise KeyError("MotionStreamer checkpoint must contain key: 'trans'")
        trans_encoder.load_state_dict(load_state_strip_module(ckpt_trans["trans"]), strict=True)
        trans_encoder.eval()
        trans_encoder.to(comp_device)

        # ---------------------------------------------------------
        # 4) Load reference end latent and run AR sampling
        # ---------------------------------------------------------
        reference_end_latent = np.load(reference_end_latent_path)
        reference_end_latent = torch.from_numpy(reference_end_latent).to(comp_device)

        motion_latents = trans_encoder.sample_for_eval_CFG_inference(
            text=text,
            tokenizer=t5_model,
            device=comp_device,
            unit_length=unit_length,
            length=max_length,
            reference_end_latent=reference_end_latent,
            threshold=threshold,
            cfg=cfg_scale,
            temperature=1.0,
        )

        # ---------------------------------------------------------
        # 5) Decode to 272-dim motion (same as demo --mode rot data)
        # ---------------------------------------------------------
        motion_seqs = net.forward_decoder(motion_latents)
        motion = motion_seqs.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # ---------------------------------------------------------
        # 6) Generate stick-figure GIF (same as demo --mode pos)
        # ---------------------------------------------------------
        mean = np.load(mean_path)
        std = np.load(std_path)
        pred_xyz = recover_from_local_position(motion * std + mean, 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        text_token = sanitize_text_for_filename(text)
        stem = f"MotionStreamer_{text_token}_{timestamp}"

        gif_path = os.path.join(output_dir, f"{stem}.gif")
        npy_path = os.path.join(output_dir, f"{stem}.npy")

        plot_3d.draw_to_batch(xyz, [text], [gif_path], fps=fps)

        # Save raw decoded 272-dim motion for output_vis.py compatibility.
        np.save(npy_path, motion)

        print(f"[OK] GIF saved: {gif_path}")
        print(f"[OK] NPY saved: {npy_path}")
        print("[OK] The .npy file is directly compatible with output_vis.py --input")

    except Exception as exc:
        print("[ERROR] Motion generation failed.")
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()

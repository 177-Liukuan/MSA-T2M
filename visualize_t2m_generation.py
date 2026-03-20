"""
Zero-Shot Text-to-Motion Generation Visualization — MSA-VAE

Pipeline ("狸猫换太子"):
  1. Text → CLIP Text Encoder → clip_text_feat  (512d)
  2. clip_text_feat  →  TransformerLatentDecoder (memo = clip_text_feat)
                     →  z_gen  (B, T', latent_dim)
  3. z_gen  →  Causal CNN Decoder  →  x_gen  (B, T, 272)
  4. x_gen * std + mean  →  real-scale 272-dim motion
  5. 272-dim → 3D joint xyz → MP4 skeleton animation

Why this works:
  global_proj = nn.Identity()  →  h_cls IS in the CLIP space
  CosineEmbeddingLoss during training forces h_cls ≈ clip_text_feat direction.
  Substituting clip_text_feat for h_cls seeds the decoder with a semantically
  equivalent vector, allowing the decoder to synthesise matching motion.

Usage:
  # Single text query
  python visualize_t2m_generation.py \
      --resume-pth ./Experiments/MSA_VAEv5_t2m_272_dynamic02/net_last.pth \
      --text "a person walks forward slowly." \
      --target-frames 128 --output-dir viz_output

  # Multiple queries from file (one per line)
  python visualize_t2m_generation.py \
      --resume-pth ./Experiments/xxx/net_best_fid.pth \
      --text-file queries.txt --target-frames 120 --output-dir viz_output

  # Interactive mode (omit --text and --text-file)
  python visualize_t2m_generation.py \
      --resume-pth ./Experiments/xxx/net_best_fid.pth
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import clip

import models.msa_vae as msa_vae
from visualization.recover_visualize import recover_from_local_position
import visualization.plot_3d_global as plot_3d


# ---------------------------------------------------------------------------
#  Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='MSA-VAE Zero-Shot T2M Generation')
    # Model architecture (must match training config)
    p.add_argument('--resume-pth',           type=str,   required=True)
    p.add_argument('--hidden-size',          type=int,   default=1024)
    p.add_argument('--down-t',               type=int,   default=2)
    p.add_argument('--stride-t',             type=int,   default=2)
    p.add_argument('--depth',                type=int,   default=3)
    p.add_argument('--dilation-growth-rate', type=int,   default=3)
    p.add_argument('--latent-dim',           type=int,   default=16)
    p.add_argument('--trans-d-model',        type=int,   default=512)
    p.add_argument('--trans-nhead',          type=int,   default=8)
    p.add_argument('--trans-enc-layers',     type=int,   default=6)
    p.add_argument('--trans-dec-layers',     type=int,   default=6)
    p.add_argument('--trans-ff-size',        type=int,   default=2048)
    p.add_argument('--trans-dropout',        type=float, default=0.1)
    p.add_argument('--clip-version',         type=str,   default='ViT-B/32')
    p.add_argument('--clip-dim',             type=int,   default=512)
    # Data normalization
    p.add_argument('--data-root', type=str,  default='./humanml3d_272',
                   help='Root dir containing mean_std/ for inv_transform')
    # Queries
    p.add_argument('--text',      type=str,  default=None,
                   help='Single text prompt')
    p.add_argument('--text-file', type=str,  default=None,
                   help='File with one text prompt per line')
    # Generation
    p.add_argument('--target-frames', type=int, default=120,
                   help='Desired output motion length in frames (at 30fps)')
    # Output
    p.add_argument('--output-dir', type=str,  default='viz_gen_output')
    p.add_argument('--fps',        type=int,  default=30)
    p.add_argument('--temperature', type=float, default=0.0,
                   help='Temperature for diversity sampling (0=deterministic, >0=stochastic)')
    p.add_argument('--save-npy',   action='store_true',
                   help='Also save raw 272-dim motion as .npy')
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------
def load_model(args, device):
    # ---- Step 1: peek at checkpoint to auto-detect architecture ----
    ckpt  = torch.load(args.resume_pth, map_location='cpu')
    state = ckpt['net'] if (isinstance(ckpt, dict) and 'net' in ckpt) else ckpt

    def _peek(key):
        return state[key] if key in state else None

    # trans_ff_size: encoder FFN linear1 first dim
    _w = _peek('msa_vae.trans_encoder.transformer_encoder.layers.0.linear1.weight')
    if _w is not None:
        detected = _w.shape[0]
        if detected != args.trans_ff_size:
            print(f'[auto-detect] trans_ff_size: {args.trans_ff_size} -> {detected}')
            args.trans_ff_size = detected

    # trans_d_model: CLS token last dim
    _c = _peek('msa_vae.trans_encoder.cls_token')
    if _c is not None:
        detected = _c.shape[2]
        if detected != args.trans_d_model:
            print(f'[auto-detect] trans_d_model: {args.trans_d_model} -> {detected}')
            args.trans_d_model = detected

    # latent_dim: mu_proj output dim
    _m = _peek('msa_vae.cnn_encoder.postnet.mu_proj.weight')
    if _m is not None:
        detected = _m.shape[0]
        if detected != args.latent_dim:
            print(f'[auto-detect] latent_dim: {args.latent_dim} -> {detected}')
            args.latent_dim = detected

    # ---- Step 2: build model with (possibly corrected) args ----
    net = msa_vae.MSA_HumanVAE(
        hidden_size           = args.hidden_size,
        down_t                = args.down_t,
        stride_t              = args.stride_t,
        depth                 = args.depth,
        dilation_growth_rate  = args.dilation_growth_rate,
        activation            = 'relu',
        latent_dim            = args.latent_dim,
        clip_range            = [-30, 20],
        trans_d_model         = args.trans_d_model,
        trans_nhead           = args.trans_nhead,
        trans_enc_layers      = args.trans_enc_layers,
        trans_dec_layers      = args.trans_dec_layers,
        trans_ff_size         = args.trans_ff_size,
        trans_dropout         = args.trans_dropout,
        clip_dim              = args.clip_dim,
    )
    net.load_state_dict(state, strict=True)
    net.eval()
    net.to(device)
    print(f'Model loaded from {args.resume_pth}')
    return net


# ---------------------------------------------------------------------------
#  Core generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_from_text(text, net, clip_model, mean, std, target_frames,
                       unit_length, device, temperature=0.0):
    """
    Zero-shot generation pipeline.

    Args:
        text:          str — the text prompt
        net:           MSA_HumanVAE
        clip_model:    frozen CLIP ViT-B/32
        mean, std:     (272,) arrays for inv_transform
        target_frames: desired motion length at 30fps
        unit_length:   CNN temporal downsampling ratio (e.g. 4 for down_t=2)
        device:        torch.device

    Returns:
        motion_272: (T, 272) float32 numpy array in real-scale
    """
    # --- (1) Text → CLIP feature (512d) ---
    tokens       = clip.tokenize([text], truncate=True).to(device)
    clip_txt_emb = clip_model.encode_text(tokens).float()          # (1, 512)

    # Normalise clip_txt_emb to match the expected scale of h_cls.
    # During training, h_cls is the [CLS] output of TransformerEncoder followed
    # by LayerNorm(d_model), whose expected L2-norm ≈ sqrt(d_model).
    # The alignment loss normalises both sides to unit sphere before computing
    # cosine loss, so it only constrains *direction*, not magnitude.
    # Without this rescaling, the Decoder cross-attention (single memory token →
    # attention weight always = 1.0) receives a memory vector with a different
    # magnitude to what it was trained on, shifting z_gen off-distribution
    # and amplifying CNN decoder artefacts (jitter, drift).
    # d_model = net.msa_vae.trans_d_model
    # clip_txt_emb = F.normalize(clip_txt_emb, dim=-1) * (d_model ** 0.5)

    # --- (2) Derive latent sequence length from target_frames ---
    # CNN downsamples by stride_t for each down_t layer →  T' = T / unit_length
    seq_len = max(1, target_frames // unit_length)

    # --- (3) TransformerDecoder: clip_txt_emb replaces h_cls as memory ---
    #   decode_transformer(h_cls, seq_len) internally does:
    #       memory = h_cls.unsqueeze(1)  →  (B, 1, d_model)
    #       cross-attention over positional queries  →  z_recon (B, T', latent_dim)
    mu_gen = net.msa_vae.decode_transformer(clip_txt_emb, seq_len=seq_len)
    # mu_gen: (1, T', latent_dim) — deterministic semantic centre

    # --- (3b) Temperature sampling for diversity ---
    if temperature > 0:
        epsilon = torch.randn_like(mu_gen)
        z_gen = mu_gen + temperature * epsilon
    else:
        z_gen = mu_gen

    # --- (4) CNN Decoder: z → x (normalized 272-dim) ---
    x_gen = net.msa_vae.decode_cnn(z_gen)                         # (1, T, 272)

    # --- (5) Inverse transform: back to real-scale ---
    motion_norm = x_gen.squeeze(0).cpu().numpy()                   # (T, 272)
    motion_272  = motion_norm * std + mean                         # inv_transform

    return motion_272.astype(np.float32)


# ---------------------------------------------------------------------------
#  Visualization helpers
# ---------------------------------------------------------------------------
def motion_to_xyz(motion_272, num_joints=22):
    """272-dim → (T, num_joints, 3) global xyz."""
    return recover_from_local_position(motion_272, num_joints)


def render_gif(joints_xyz, title, output_path, fps=30):
    """Render (T, 22, 3) skeleton animation to GIF with title watermark."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # draw_to_batch expects (B, T, J, 3)
    joints_batch = joints_xyz[np.newaxis]           # (1, T, 22, 3)
    plot_3d.draw_to_batch(
        joints_batch,
        title_batch=[title],
        outname=[output_path],
        fps=fps,
    )


# ---------------------------------------------------------------------------
#  Batch runner
# ---------------------------------------------------------------------------
def run_queries(queries, net, clip_model, mean, std, args, device):
    unit_length = args.stride_t ** args.down_t      # e.g. 2^2 = 4

    summary = []
    for qi, text in enumerate(queries):
        print(f'\n[{qi+1}/{len(queries)}] Generating: "{text}"')

        motion_272 = generate_from_text(
            text, net, clip_model, mean, std,
            target_frames=args.target_frames,
            unit_length=unit_length,
            device=device,
            temperature=args.temperature,
        )
        T = motion_272.shape[0]
        print(f'  → Generated {T} frames ({T/args.fps:.1f}s @ {args.fps}fps)')

        # File name
        safe = text[:50].replace(' ', '_').replace('/', '_')
        base = os.path.join(args.output_dir, f'gen_{qi:03d}_{safe}')

        # Save .npy
        if args.save_npy:
            np.save(base + '.npy', motion_272)
            print(f'  → Saved .npy: {base}.npy')

        # Render GIF
        joints_xyz = motion_to_xyz(motion_272)
        gif_path   = base + '.gif'
        title_str  = f'Prompt: {text}\n({T} frames, {T/args.fps:.1f}s)'
        render_gif(joints_xyz, title_str, gif_path, fps=args.fps)
        print(f'  → Saved GIF: {gif_path}')

        summary.append({'query': text, 'frames': T, 'gif': gif_path})

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, 'generation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nSummary saved → {summary_path}')


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load normalization stats
    from os.path import join as pjoin
    mean = np.load(pjoin(args.data_root, 'mean_std', 'Mean.npy'))   # (272,)
    std  = np.load(pjoin(args.data_root, 'mean_std', 'Std.npy'))    # (272,)

    # Load models
    net        = load_model(args, device)
    clip_model, _ = clip.load(args.clip_version, device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    print(f'CLIP {args.clip_version} loaded')

    # Collect queries
    queries = []
    if args.text:
        queries.append(args.text)
    if args.text_file:
        with open(args.text_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(line)

    if not queries:
        # Interactive mode
        print('\nNo query provided — entering interactive mode (type "quit" to exit)')
        while True:
            text = input('\nEnter text prompt: ').strip()
            if text.lower() in ('quit', 'exit', 'q'):
                break
            if not text:
                continue
            run_queries([text], net, clip_model, mean, std, args, device)
        return

    run_queries(queries, net, clip_model, mean, std, args, device)


if __name__ == '__main__':
    main()

"""
Demo: Verify MSA-VAE dataset loading for sample 004203 with window [0, 64).

Visualises the motion window as a GIF with:
  - Global text caption on top
  - Local CLIP segment labels displayed per frame
  - Frame counter
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio

# ── project imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.face_z_align_util import rotation_6d_to_matrix

# ─────────────────────────────────────────────────────────
#  1. Raw data for sample 004203
# ─────────────────────────────────────────────────────────
SAMPLE_ID = '004203'
WINDOW_START = 0
WINDOW_SIZE = 64
UNIT_LENGTH = 4  # latent downsampling factor (stride^down_t = 2^2)

# BABEL local annotation (frame-level, 30 fps, 288 frames total)
LOCAL_SEGMENTS = [
    (0,   3,   "Standing"),
    (4,   136, "walking up stairs"),
    (137, 154, "Standing"),
    (155, 263, "walking down stairs"),
    (264, 287, "Standing"),
]

data_root = './humanml3d_272'
motion_path = os.path.join(data_root, 'motion_data', f'{SAMPLE_ID}.npy')
text_path   = os.path.join(data_root, 'texts', f'{SAMPLE_ID}.txt')
clip_path   = os.path.join(data_root, 'clip_enc_single', f'{SAMPLE_ID}.npy')
mean_path   = os.path.join(data_root, 'mean_std', 'Mean.npy')
std_path    = os.path.join(data_root, 'mean_std', 'Std.npy')

# ─────────────────────────────────────────────────────────
#  2. Load & crop -- mimicking dataset_msa_vae.py logic
# ─────────────────────────────────────────────────────────
motion_full = np.load(motion_path)  # (288, 272)
mean = np.load(mean_path)
std  = np.load(std_path)

print(f"Motion shape: {motion_full.shape}")

# ── Global text (same logic as dataset_msa_vae.py) ──
with open(text_path, 'r') as f:
    text_lines = f.readlines()

global_captions = []
for line in text_lines:
    parts = line.strip().split('#')
    if len(parts) >= 4:
        caption = parts[0].strip()
        f_tag  = float(parts[2]) if parts[2] != 'nan' else 0.0
        to_tag = float(parts[3]) if parts[3] != 'nan' else 0.0
        if f_tag == 0.0 and to_tag == 0.0 and caption:
            global_captions.append(caption)

print(f"\n=== Global Captions (from text file) ===")
for i, c in enumerate(global_captions):
    print(f"  [{i}] {c}")

chosen_global = global_captions[0] if global_captions else "N/A"

# ── Crop window [0, 64) ──
idx = WINDOW_START
motion_window = motion_full[idx:idx + WINDOW_SIZE]  # (64, 272)
motion_window_norm = (motion_window - mean) / std

print(f"\nWindow [{idx}, {idx+WINDOW_SIZE}), shape: {motion_window.shape}")

# ── Local CLIP embeddings (same logic as dataset_msa_vae.py) ──
latent_len = WINDOW_SIZE // UNIT_LENGTH  # 64 // 4 = 16

has_local = os.path.exists(clip_path)
print(f"\nLocal CLIP file exists: {has_local}")

if has_local:
    clip_enc_20 = np.load(clip_path)  # (T_20fps, 512)
    T_20 = clip_enc_20.shape[0]
    T_30 = len(motion_full)
    print(f"  clip_enc shape (20fps): {clip_enc_20.shape}")
    print(f"  Motion length (30fps): {T_30}")
    print(f"  Ratio: {T_30}/{T_20} = {T_30/T_20:.2f} (expect ~1.5)")

    # Upsample 20fps -> 30fps via nearest-neighbor
    indices_30 = np.round(np.linspace(0, T_20 - 1, T_30)).astype(int)
    clip_enc_30 = clip_enc_20[indices_30]  # (288, 512)

    # Crop to window
    local_clip_window = clip_enc_30[idx:idx + WINDOW_SIZE]  # (64, 512)

    # Average-pool to latent rate
    def pool_to_latent(clip_window, lat_len):
        T = clip_window.shape[0]
        if T == lat_len:
            return clip_window
        inds = np.linspace(0, T, lat_len + 1).astype(int)
        pooled = np.zeros((lat_len, clip_window.shape[1]), dtype=np.float32)
        for i in range(lat_len):
            pooled[i] = clip_window[inds[i]:inds[i + 1]].mean(axis=0)
        return pooled

    local_clip_latent = pool_to_latent(local_clip_window, latent_len)
    print(f"  local_clip_latent shape: {local_clip_latent.shape}")

# ─────────────────────────────────────────────────────────
#  3. Analyse: what LOCAL label does each frame get?
# ─────────────────────────────────────────────────────────
print(f"\n=== Local Label Analysis for Window [{idx}, {idx+WINDOW_SIZE}) ===")

def get_local_label(frame_idx):
    for start, end, label in LOCAL_SEGMENTS:
        if start <= frame_idx <= end:
            return label
    return "unknown"

frame_labels = []
for f in range(idx, idx + WINDOW_SIZE):
    frame_labels.append(get_local_label(f))

from collections import Counter
label_counts = Counter(frame_labels)
print("  Frame distribution in this window:")
for label, count in label_counts.most_common():
    print(f"    {label}: {count} frames")

# Per-latent-token label (majority vote from 4 frames each)
print(f"\n  Per latent token labels (UNIT_LENGTH={UNIT_LENGTH}):")
latent_labels = []
for t in range(latent_len):
    token_frames = frame_labels[t * UNIT_LENGTH : (t + 1) * UNIT_LENGTH]
    majority = Counter(token_frames).most_common(1)[0][0]
    latent_labels.append(majority)
    print(f"    Token {t:2d} (frames {idx + t*UNIT_LENGTH:3d}-{idx + (t+1)*UNIT_LENGTH - 1:3d}): {majority}")

# ─────────────────────────────────────────────────────────
#  4. CLIP embedding vs label similarity check
# ─────────────────────────────────────────────────────────
print(f"\n=== CLIP Embedding Similarity Check ===")
if has_local:
    try:
        import clip as clip_module
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clip_model, _ = clip_module.load("ViT-B/32", device=device)

        unique_labels = list(set(frame_labels))
        text_tokens = clip_module.tokenize(unique_labels).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        local_clip_tensor = torch.from_numpy(local_clip_latent).to(device)
        local_clip_norm = local_clip_tensor / local_clip_tensor.norm(dim=-1, keepdim=True)

        sim_matrix = (local_clip_norm @ text_features.T).cpu().numpy()
        print(f"  Labels: {unique_labels}")
        for t in range(latent_len):
            best_idx = sim_matrix[t].argmax()
            best_label = unique_labels[best_idx]
            sims_str = ", ".join(f"{unique_labels[j]}:{sim_matrix[t,j]:.3f}" for j in range(len(unique_labels)))
            match = "OK" if best_label == latent_labels[t] else "MISMATCH"
            print(f"    Token {t:2d}: expected={latent_labels[t]:25s}  best_match={best_label:25s} {match}  ({sims_str})")
    except Exception as e:
        print(f"  CLIP check skipped: {e}")

# ─────────────────────────────────────────────────────────
#  5. Recover joint positions from 272d representation
# ─────────────────────────────────────────────────────────
def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    return np.array(R_total)

def recover_from_local_position(final_x, njoint=22):
    if final_x.ndim == 2:
        final_x = final_x[np.newaxis]
        squeeze = True
    else:
        squeeze = False
    bs, nfrm, _ = final_x.shape
    positions_no_heading = final_x[:, :, 8:8 + 3 * njoint].reshape(bs, nfrm, njoint, 3)
    velocities_root_xy = final_x[:, :, :2]
    heading_diff_rot6d = final_x[:, :, 2:8]
    results = []
    for b in range(bs):
        rot6d_torch = torch.from_numpy(heading_diff_rot6d[b])
        rot_matrices = rotation_6d_to_matrix(rot6d_torch).numpy()
        global_heading_rot = accumulate_rotations(rot_matrices)
        inv_heading = np.transpose(global_heading_rot, (0, 2, 1))
        pos = np.matmul(
            np.repeat(inv_heading[:, np.newaxis, :, :], njoint, axis=1),
            positions_no_heading[b][..., np.newaxis]
        ).squeeze(-1)
        vel_xyz = np.zeros((nfrm, 3))
        vel_xyz[:, 0] = velocities_root_xy[b, :, 0]
        vel_xyz[:, 2] = velocities_root_xy[b, :, 1]
        vel_xyz[1:] = np.matmul(inv_heading[:-1], vel_xyz[1:, :, np.newaxis]).squeeze(-1)
        root_trans = np.cumsum(vel_xyz, axis=0)
        pos[:, :, 0] += root_trans[:, 0:1]
        pos[:, :, 2] += root_trans[:, 2:]
        results.append(pos)
    out = np.stack(results, axis=0)
    return out.squeeze(0) if squeeze else out

joints = recover_from_local_position(motion_window, njoint=22)
print(f"\nRecovered joints shape: {joints.shape}")

# ─────────────────────────────────────────────────────────
#  6. Generate GIF with labels
# ─────────────────────────────────────────────────────────
print("\nGenerating GIF...")

kinematic_chain = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
chain_colors = ['red', 'blue', 'black', 'red', 'blue']

data = joints.copy()
MINS = data.min(axis=0).min(axis=0)
MAXS = data.max(axis=0).max(axis=0)
height_offset = MINS[1]
data[:, :, 1] -= height_offset
trajec = data[:, 0, [0, 2]]
data[..., 0] -= data[:, 0:1, 0]
data[..., 2] -= data[:, 0:1, 2]

frames_out = []
for fi in range(WINDOW_SIZE):
    fig = plt.figure(figsize=(10, 10), dpi=96)
    wrapped_title = '\n'.join(wrap(f"Global: {chosen_global}", 55))
    fig.suptitle(wrapped_title, fontsize=14, fontweight='bold', y=0.98)
    local_label = frame_labels[fi]
    abs_frame = idx + fi
    latent_token_idx = fi // UNIT_LENGTH
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.78], projection='3d')
    limits = 2
    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    ax.set_zlim(0, limits)
    ax.view_init(elev=110, azim=-90)
    ax.dist = 7.5
    ax.grid(False)
    ax.set_axis_off()
    verts = [[MINS[0] - trajec[fi, 0], 0, MINS[2] - trajec[fi, 1]],
             [MINS[0] - trajec[fi, 0], 0, MAXS[2] - trajec[fi, 1]],
             [MAXS[0] - trajec[fi, 0], 0, MAXS[2] - trajec[fi, 1]],
             [MAXS[0] - trajec[fi, 0], 0, MINS[2] - trajec[fi, 1]]]
    xz_plane = Poly3DCollection([verts])
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(xz_plane)
    if fi > 1:
        ax.plot3D(trajec[:fi, 0] - trajec[fi, 0],
                  np.zeros(fi),
                  trajec[:fi, 1] - trajec[fi, 1],
                  linewidth=1.0, color='blue')
    for chain, color in zip(kinematic_chain, chain_colors):
        ax.plot3D(data[fi, chain, 0], data[fi, chain, 1], data[fi, chain, 2],
                  linewidth=4.0, color=color)
    fig.text(0.5, 0.06,
             f"Frame {abs_frame} (window frame {fi}/{WINDOW_SIZE})  |  "
             f"Latent token {latent_token_idx}/{latent_len}",
             ha='center', fontsize=12, color='gray')
    fig.text(0.5, 0.02,
             f'Local: "{local_label}"',
             ha='center', fontsize=14, fontweight='bold',
             color='darkgreen' if local_label != "Standing" else 'darkorange')
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    frames_out.append(img.copy())
    plt.close(fig)

out_path = './demo_output/msa_vae_004203_window0_64.gif'
os.makedirs('./demo_output', exist_ok=True)
imageio.mimsave(out_path, frames_out, fps=15, loop=0)
print(f"\nGIF saved to: {out_path}")
print("Done!")

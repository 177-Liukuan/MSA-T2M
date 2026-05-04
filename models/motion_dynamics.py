"""
Motion Dynamics Feature Extractor.

Converts denormalized motion sequences [B, T, 272] into discriminator-ready
dynamics features [B, C, T] using DIFFERENTIABLE torch operations only.

Design rationale
----------------
The full global position recovery (recover_from_local_position in eval_trans.py)
requires per-sample numpy loops + cumulative rotation accumulation, making it
non-differentiable and slow inside a training loop.

Instead, we directly use the LOCAL joint positions stored in the 272-dim vector:
  motion[..., 8 : 8 + 22*3]  →  local joint positions in heading-aligned frame
These already encode all joint configuration dynamics and are differentiable.

Discriminator input features:
  - Local joint positions  : motion[..., 8:74]   → [B, T, 66]
  - Root full (XY vel+rot) : motion[..., 0:8]    → [B, T, 8]   (root dynamics)
  - Joint velocities       : first-order diff of positions → [B, T, 66]
  Total: 66 + 8 + 66 = 140 channels  (default, see DISC_IN_CHANNELS)

All operations are fully differentiable → gradients flow back through the decoder.
"""

import torch


# ── Channel layout of the 272-dim HumanML3D representation ──────────────────
#   [0:2]      root XY velocity (in heading-aligned frame)
#   [2:8]      root heading rotation (6D)
#   [8:74]     local joint positions  (22 joints × 3)   ← primary dynamics cue
#   [74:206]   joint rotations (22 joints × 6D)
#   [206:272]  joint velocities (22 joints × 3)         (also available)
# ────────────────────────────────────────────────────────────────────────────

POS_START = 8
POS_END   = 8 + 22 * 3   # 74
ROOT_VEL_START = 0
ROOT_VEL_END   = 8   # full root: XY velocity(2) + heading rotation 6D(6)

# Default discriminator input channels: pos(66) + root(8) + vel(66) = 140
DISC_IN_CHANNELS = 140


def extract_dynamics_features(motion_denorm: torch.Tensor) -> torch.Tensor:
    """
    Convert denormalized motion to discriminator input features.

    Args:
        motion_denorm: [B, T, 272]  (float32, device-agnostic)

    Returns:
        feats: [B, DISC_IN_CHANNELS, T]  ready for Conv1d discriminator
    """
    # ── Local joint positions ──────────────────────────────────────────────
    pos = motion_denorm[..., POS_START:POS_END]          # [B, T, 66]

    # ── Root representation (XY velocity + heading rotation 6D) ─────────
    root_vel = motion_denorm[..., ROOT_VEL_START:ROOT_VEL_END]  # [B, T, 8]

    # ── Joint velocities (first-order finite difference) ──────────────────
    vel = torch.zeros_like(pos)
    vel[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]     # [B, T, 66]

    # ── Concatenate → [B, T, 140] → permute → [B, 140, T] ────────────────
    feats = torch.cat([pos, root_vel, vel], dim=-1)
    feats = feats.permute(0, 2, 1).contiguous()

    return feats


def feature_matching_loss(
    feats_real_list: list,
    feats_fake_list: list,
) -> torch.Tensor:
    """
    L1 feature matching loss between real and fake discriminator intermediate
    activations (VideoVAE+ / VQGAN style).

    Args:
        feats_real_list: list of tensors from discriminator on real motion
        feats_fake_list: list of tensors from discriminator on reconstructed motion

    Returns:
        scalar FM loss
    """
    import torch.nn.functional as F
    fm = 0.0
    for f_real, f_fake in zip(feats_real_list, feats_fake_list):
        fm = fm + F.l1_loss(f_fake, f_real.detach())
    return fm / max(len(feats_real_list), 1)

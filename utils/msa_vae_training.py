"""Training helpers for variable-length MSA-VAE batches."""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


class MSAVAELossWeights(NamedTuple):
    root: float
    latent: float
    global_align: float
    local_align: float


def valid_mask_from_lengths(lengths, max_len):
    """Return a boolean mask whose True entries are valid timesteps."""
    if max_len < 0:
        raise ValueError('max_len must be nonnegative')
    lengths = lengths.long()
    if torch.any(lengths < 0):
        raise ValueError('lengths must be nonnegative')
    steps = torch.arange(max_len, device=lengths.device)
    return steps.unsqueeze(0) < lengths.unsqueeze(1)


def latent_lengths_from_frames(lengths, stride_t, down_t):
    """Map frame lengths through repeated causal strided convolutions."""
    if stride_t < 1:
        raise ValueError('stride_t must be positive')
    if down_t < 0:
        raise ValueError('down_t must be nonnegative')
    result = lengths.long()
    if torch.any(result < 0):
        raise ValueError('lengths must be nonnegative')
    for _ in range(down_t):
        result = torch.div(result, stride_t, rounding_mode='floor')
    return result


def _expanded_mask(valid_mask, values):
    mask = valid_mask.bool()
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    return mask.expand_as(values)


def _masked_per_sample_mean(values, valid_mask):
    mask = _expanded_mask(valid_mask, values)
    reduce_dims = tuple(range(1, values.dim()))
    counts = mask.sum(dim=reduce_dims)
    valid_samples = counts > 0
    if not torch.any(valid_samples):
        return values.sum() * 0.0
    totals = (values * mask.to(values.dtype)).sum(dim=reduce_dims)
    means = totals[valid_samples] / counts[valid_samples].to(values.dtype)
    return means.mean()


def _masked_global_mean(values, valid_mask):
    mask = _expanded_mask(valid_mask, values)
    count = mask.sum()
    if count.item() == 0:
        return values.sum() * 0.0
    return (values * mask.to(values.dtype)).sum() / count.to(values.dtype)


def masked_mse(pred, target, valid_mask):
    """Per-sample mean squared error over valid timesteps and features."""
    return _masked_per_sample_mean((pred - target).pow(2), valid_mask)


def masked_kl(mu, logvar, valid_mask):
    """Per-sample mean KL divergence over valid latent elements."""
    values = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return _masked_per_sample_mean(values, valid_mask)


def masked_optimal_sigma_nll(pred, target, valid_mask, feature_slice):
    """Optimal-sigma Gaussian NLL normalized per valid sample."""
    selected_pred = pred[..., feature_slice]
    selected_target = target[..., feature_slice]
    squared_error = (selected_target - selected_pred).pow(2)
    mean_squared_error = _masked_global_mean(squared_error, valid_mask)
    sigma = mean_squared_error.sqrt().clamp_min(math.exp(-6))
    log_sigma = sigma.log()
    nll = 0.5 * ((selected_target - selected_pred) / sigma).pow(2)
    nll = nll + log_sigma + 0.5 * math.log(2 * math.pi)
    return _masked_per_sample_mean(nll, valid_mask)


def masked_cosine_alignment(pred, target, valid_mask):
    """Mean cosine distance over valid samples or temporal tokens."""
    losses = 1.0 - F.cosine_similarity(pred, target, dim=-1)
    mask = valid_mask.bool()
    if losses.shape != mask.shape:
        raise ValueError(
            f'alignment mask shape {tuple(mask.shape)} does not match '
            f'feature prefix {tuple(losses.shape)}'
        )
    if not torch.any(mask):
        return pred.sum() * 0.0
    return losses[mask].mean()


def is_window_replay_step(step, interval):
    if interval < 2:
        raise ValueError('replay interval must be at least 2')
    if step < 1:
        raise ValueError('training step must be positive')
    return step % interval == 0


def validate_sequence_training_config(phase, mode, full_batch_size,
                                      replay_interval):
    if mode not in ('window', 'full', 'mixed'):
        raise ValueError(f'unknown sequence mode: {mode}')
    if phase == 1 and mode != 'full':
        raise ValueError('Phase 1 requires full sequence mode')
    if mode == 'mixed' and phase != 2:
        raise ValueError('mixed sequence mode is only valid for Phase 2')
    if mode in ('full', 'mixed') and full_batch_size < 1:
        raise ValueError('full_batch_size must be positive')
    if mode == 'mixed' and replay_interval < 2:
        raise ValueError('replay_interval must be at least 2')


def compute_msa_vae_objective(outputs, targets, phase, batch_kind, weights,
                              stride_t, down_t):
    """Compose masked MSA-VAE losses for one full or replay batch."""
    if phase not in (0, 1, 2):
        raise ValueError(f'unsupported phase: {phase}')
    if batch_kind not in ('full', 'window'):
        raise ValueError(f'unsupported batch_kind: {batch_kind}')

    motion_lengths = targets['motion_lengths']
    latent_lengths = latent_lengths_from_frames(
        motion_lengths,
        stride_t=stride_t,
        down_t=down_t,
    )
    latent_mask = valid_mask_from_lengths(
        latent_lengths, outputs['mu'].size(1)
    )
    local_mask = latent_mask & targets['has_local'].bool().unsqueeze(1)

    latent_loss = masked_mse(
        outputs['mu_recon'], outputs['mu'].detach(), latent_mask
    )
    global_loss = masked_cosine_alignment(
        outputs['clip_global_feat'],
        targets['global_text'],
        targets['has_global'].bool(),
    )
    local_loss = masked_cosine_alignment(
        outputs['clip_local_feat'],
        targets['local_text'],
        local_mask,
    )

    if phase == 1:
        loss_dict = {
            'latent': latent_loss,
            'global_align': global_loss,
            'local_align': local_loss,
        }
        total = (
            weights.latent * latent_loss
            + weights.global_align * global_loss
            + weights.local_align * local_loss
        )
        return total, loss_dict

    frame_mask = valid_mask_from_lengths(
        motion_lengths, targets['motion'].size(1)
    )
    reconstruction = masked_optimal_sigma_nll(
        outputs['x_recon'],
        targets['motion'],
        frame_mask,
        feature_slice=slice(None),
    )
    kl = masked_kl(outputs['mu'], outputs['logvar'], latent_mask)
    root = masked_optimal_sigma_nll(
        outputs['x_recon'],
        targets['motion'],
        frame_mask,
        feature_slice=slice(0, 8),
    )

    if phase == 2 and batch_kind == 'window':
        loss_dict = {
            'recon': reconstruction,
            'kl': kl,
            'root': root,
            'local_align': local_loss,
        }
        zero_semantic = (
            outputs['mu_recon'].sum()
            + outputs['clip_global_feat'].sum()
        ) * 0.0
        total = (
            reconstruction
            + kl
            + weights.root * root
            + weights.local_align * local_loss
            + zero_semantic
        )
        return total, loss_dict

    loss_dict = {
        'recon': reconstruction,
        'kl': kl,
        'root': root,
        'latent': latent_loss,
        'global_align': global_loss,
        'local_align': local_loss,
    }
    total = (
        reconstruction
        + kl
        + weights.root * root
        + weights.latent * latent_loss
        + weights.global_align * global_loss
        + weights.local_align * local_loss
    )
    return total, loss_dict

"""Training helpers for variable-length MSA-VAE batches."""

import math
import json
import os
import re
import tempfile
from collections.abc import Mapping
from typing import NamedTuple

import torch
import torch.nn.functional as F


MOTION_LENGTH_BIN_NAMES = (
    'up_to_64',
    '65_to_128',
    'over_128',
)

TRAINING_IDENTITY_FIELDS = (
    'dataname',
    'batch_size',
    'use_ft_split',
    'length_bucket_size',
    'hidden_size',
    'down_t',
    'stride_t',
    'depth',
    'dilation_growth_rate',
    'latent_dim',
    'trans_d_model',
    'trans_nhead',
    'trans_enc_layers',
    'trans_dec_layers',
    'trans_ff_size',
    'trans_dropout',
    'clip_dim',
    'disable_decoupling',
    'total_iter',
    'warm_up_iter',
    'eval_iter',
    'validation_seed',
    'validation_batch_size',
    'lr',
    'lr_scheduler',
    'gamma',
    'weight_decay',
    'cnn_lr_scale',
    'spotlight_alpha',
    'num_gpus',
    'seed',
    'global_align_weight',
    'local_align_weight',
    'latent_recon_weight',
    'root_loss',
    'exp_name',
    'msa_data_mode',
    'text_encoder_type',
    'text_embed_dim',
    'use_offline_global_text',
    'resume_cnn_pth',
    'resume_cnn_sha256',
    'freeze_phase2_local_proj',
)

MSA_REQUIRED_PHASE_MODULES = (
    'cnn_encoder',
    'cnn_decoder',
    'decode_proj',
    'local_proj',
)
MSA_CNN_MODULES = (
    'cnn_encoder',
    'cnn_decoder',
    'decode_proj',
)


def _msa_core(model):
    core = getattr(model, 'msa_vae', model)
    missing = [
        name
        for name in MSA_REQUIRED_PHASE_MODULES
        if not hasattr(core, name)
    ]
    if missing:
        raise ValueError(
            'MSA-VAE model is missing required modules: '
            + ', '.join(missing)
        )
    return core


def configure_msa_vae_trainability(
        model, phase, freeze_phase2_local_proj=False):
    """Apply the phase trainability contract after checkpoint loading."""
    if phase not in (0, 1, 2):
        raise ValueError(f'unsupported phase: {phase}')
    if freeze_phase2_local_proj and phase != 2:
        raise ValueError(
            'freeze_phase2_local_proj is valid only for Phase 2'
        )
    core = _msa_core(model)
    for parameter in core.parameters():
        parameter.requires_grad = True
    if phase == 1:
        for name in MSA_CNN_MODULES:
            for parameter in getattr(core, name).parameters():
                parameter.requires_grad = False
    elif phase == 2 and freeze_phase2_local_proj:
        for parameter in core.local_proj.parameters():
            parameter.requires_grad = False
    return {
        name: all(
            parameter.requires_grad
            for parameter in module.parameters()
        )
        for name, module in core.named_children()
    }


def build_phase2_optimizer_param_groups(model, lr, cnn_lr_scale):
    """Build trainable-only top/CNN parameter groups for Phase 2."""
    core = _msa_core(model)
    cnn_parameter_ids = {
        id(parameter)
        for name in MSA_CNN_MODULES
        for parameter in getattr(core, name).parameters()
    }
    cnn_params = []
    top_params = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in cnn_parameter_ids:
            cnn_params.append(parameter)
        else:
            top_params.append(parameter)
    if not top_params:
        raise ValueError('Phase 2 top optimizer group has no parameters')
    if not cnn_params:
        raise ValueError('Phase 2 CNN optimizer group has no parameters')
    return [
        {'params': top_params, 'lr': float(lr)},
        {
            'params': cnn_params,
            'lr': float(lr) * float(cnn_lr_scale),
        },
    ]


def motion_length_bin(length):
    """Return the reporting bin for a motion length in frames."""
    length = int(length)
    if length < 0:
        raise ValueError('motion length must be nonnegative')
    if length <= 64:
        return MOTION_LENGTH_BIN_NAMES[0]
    if length <= 128:
        return MOTION_LENGTH_BIN_NAMES[1]
    return MOTION_LENGTH_BIN_NAMES[2]


def summarize_motion_length_bins(lengths):
    """Count motions in the standard short/medium/long reporting bins."""
    counts = {name: 0 for name in MOTION_LENGTH_BIN_NAMES}
    for length in lengths:
        counts[motion_length_bin(length)] += 1
    return counts


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


def _masked_per_sample_values(values, valid_mask):
    mask = _expanded_mask(valid_mask, values)
    reduce_dims = tuple(range(1, values.dim()))
    counts = mask.sum(dim=reduce_dims)
    valid_samples = counts > 0
    totals = (values * mask.to(values.dtype)).sum(dim=reduce_dims)
    means = torch.zeros_like(totals)
    means[valid_samples] = (
        totals[valid_samples] / counts[valid_samples].to(values.dtype)
    )
    return means, valid_samples


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
    mean_squared_error, _ = _masked_per_sample_values(
        squared_error, valid_mask
    )
    sigma = mean_squared_error.sqrt().clamp_min(math.exp(-6))
    sigma = sigma.view(
        sigma.size(0), *([1] * (selected_pred.dim() - 1))
    )
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
    if losses.dim() > 1:
        return _masked_per_sample_mean(losses, mask)
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

    latent_target = outputs.get(
        'trans_latent_target', outputs['mu']
    ).detach()
    latent_loss = masked_mse(
        outputs['mu_recon'], latent_target, latent_mask
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


def select_training_batch(step, mode, full_iter, window_iter,
                          replay_interval):
    """Select a rank-deterministic full or window batch."""
    if mode == 'full':
        return next(full_iter), 'full'
    if mode == 'window':
        return next(window_iter), 'window'
    if mode == 'mixed':
        if is_window_replay_step(step, replay_interval):
            return next(window_iter), 'window'
        return next(full_iter), 'full'
    raise ValueError(f'unknown sequence mode: {mode}')


def build_global_alignment_target(global_text, local_pooled, has_local,
                                  total_frames, window_size, sequence_mode,
                                  spotlight_alpha):
    """Build full-caption targets or preserve legacy window Spotlight."""
    if sequence_mode in ('full', 'mixed'):
        return F.normalize(global_text, dim=-1)
    if sequence_mode != 'window':
        raise ValueError(f'unknown sequence mode: {sequence_mode}')

    if spotlight_alpha < 0:
        alpha = (
            float(window_size) / total_frames.to(global_text.device).float()
        ).clamp(0, 1)
    else:
        alpha = torch.full(
            (global_text.size(0),),
            float(spotlight_alpha),
            device=global_text.device,
        )
    alpha = alpha * has_local.to(global_text.device).float()
    alpha = alpha.unsqueeze(-1)
    mixed = (1 - alpha) * global_text + alpha * local_pooled
    return F.normalize(mixed, dim=-1)


def build_msa_checkpoint_metadata(args):
    """Record sequence-training structure without changing tensor keys."""
    training_args = {
        field: getattr(args, field)
        for field in TRAINING_IDENTITY_FIELDS
    }
    return {
        'format_version': 1,
        'phase': int(args.phase),
        'sequence_mode': str(args.sequence_mode),
        'window_size': int(args.window_size),
        'full_seq_batch_size': int(args.full_seq_batch_size),
        'window_replay_interval': int(args.window_replay_interval),
        'down_t': int(args.down_t),
        'stride_t': int(args.stride_t),
        'unit_length': int(args.stride_t) ** int(args.down_t),
        'latent_dim': int(args.latent_dim),
        'normalized_loss_version': 1,
        'training_args': training_args,
    }


def validate_msa_checkpoint_metadata(metadata, args):
    """Reject structural checkpoint mismatches; accept legacy payloads."""
    if metadata is None:
        return
    if metadata.get('format_version') != 1:
        raise ValueError(
            f'unsupported checkpoint metadata format: '
            f'{metadata.get("format_version")}'
        )
    for field in ('down_t', 'stride_t', 'latent_dim'):
        expected = int(getattr(args, field))
        actual = int(metadata[field])
        if actual != expected:
            raise ValueError(
                f'checkpoint {field}={actual} does not match requested '
                f'{field}={expected}'
            )
    expected_unit_length = int(args.stride_t) ** int(args.down_t)
    actual_unit_length = int(
        metadata.get(
            'unit_length',
            int(metadata['stride_t']) ** int(metadata['down_t']),
        )
    )
    if actual_unit_length != expected_unit_length:
        raise ValueError(
            f'checkpoint unit_length={actual_unit_length} does not match '
            f'requested unit_length={expected_unit_length}'
        )


def validate_phase2_parent_metadata(metadata, args):
    """Require Phase 2 to resume a traceable full-sequence Phase 1 run."""
    if int(args.phase) != 2:
        return
    if not isinstance(metadata, Mapping):
        raise ValueError(
            'Phase 2 requires checkpoint metadata from a fresh Phase 1 run'
        )
    validate_msa_checkpoint_metadata(metadata, args)
    if metadata.get('phase') != 1:
        raise ValueError(
            f'Phase 2 parent checkpoint phase must be 1, '
            f'got {metadata.get("phase")}'
        )
    if metadata.get('sequence_mode') != 'full':
        raise ValueError(
            'Phase 2 parent checkpoint sequence_mode must be full, '
            f'got {metadata.get("sequence_mode")}'
        )
    parent_args = metadata.get('training_args')
    if not isinstance(parent_args, Mapping):
        raise ValueError(
            'Phase 2 parent checkpoint is missing training metadata'
        )
    tae_path = parent_args.get('resume_cnn_pth')
    tae_sha256 = parent_args.get('resume_cnn_sha256')
    if not isinstance(tae_path, str) or not tae_path:
        raise ValueError(
            'Phase 2 parent checkpoint is missing fixed TAE path metadata'
        )
    if (
        not isinstance(tae_sha256, str)
        or re.fullmatch(r'[0-9a-fA-F]{64}', tae_sha256) is None
    ):
        raise ValueError(
            'Phase 2 parent checkpoint is missing valid fixed TAE SHA-256 '
            'metadata'
        )


def validate_phase2_resume_requirement(args):
    """Reject a Phase 2 launch that has no Phase 1 checkpoint to resume."""
    if int(args.phase) == 2 and not getattr(args, 'resume_pth', None):
        raise ValueError(
            'Phase 2 requires --resume-pth from a fresh Phase 1 run'
        )


def inherit_msa_checkpoint_lineage(
        metadata, parent_metadata, parent_checkpoint_path):
    """Carry Phase 1 seed and fixed-TAE identity into Phase 2 metadata."""
    result = dict(metadata)
    lineage = {
        'parent_checkpoint_path': os.fspath(parent_checkpoint_path),
        'parent_checkpoint_metadata': (
            dict(parent_metadata)
            if isinstance(parent_metadata, Mapping)
            else None
        ),
    }
    result['lineage'] = lineage
    if not isinstance(parent_metadata, Mapping):
        return result

    current_args = dict(result.get('training_args') or {})
    parent_args = parent_metadata.get('training_args')
    if not isinstance(parent_args, Mapping):
        return result
    current_seed = current_args.get('seed')
    parent_seed = parent_args.get('seed')
    if (
        current_seed is not None
        and parent_seed is not None
        and current_seed != parent_seed
    ):
        raise ValueError(
            f'Phase 2 seed={current_seed} does not match '
            f'Phase 1 seed={parent_seed}'
        )
    for field in ('resume_cnn_pth', 'resume_cnn_sha256'):
        parent_value = parent_args.get(field)
        current_value = current_args.get(field)
        if current_value and parent_value and current_value != parent_value:
            raise ValueError(
                f'Phase 2 {field}={current_value} does not match '
                f'Phase 1 {field}={parent_value}'
            )
        if not current_value:
            current_args[field] = parent_value
    result['training_args'] = current_args
    return result


def save_msa_checkpoint(path, model, metadata=None):
    """Save a compatible MSA-VAE payload with optional training metadata."""
    payload = {'net': model.state_dict()}
    if metadata is not None:
        payload['metadata'] = dict(metadata)
    torch.save(payload, path)


def checkpoint_signature(path):
    """Return a stable local identity for an extraction checkpoint."""
    resolved = os.path.abspath(os.fspath(path))
    stat = os.stat(resolved)
    return {
        'path': resolved,
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def _validate_extraction_root(root, payload):
    root = os.path.abspath(os.fspath(root))
    if not os.path.exists(root):
        return
    manifest_path = os.path.join(root, 'extraction_metadata.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            existing = json.load(handle)
        if existing != payload:
            raise ValueError(
                f'{root} was prepared from a different checkpoint or '
                f'MSA-VAE configuration'
            )
        return
    if os.listdir(root):
        raise ValueError(
            f'{root} is nonempty but has no extraction manifest'
        )


def _write_extraction_manifest(root, payload):
    root = os.path.abspath(os.fspath(root))
    os.makedirs(root, exist_ok=True)
    manifest_path = os.path.join(root, 'extraction_metadata.json')
    if os.path.exists(manifest_path):
        return
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=root,
                prefix='.extraction_metadata.',
                suffix='.tmp',
                delete=False) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def prepare_extraction_roots(roots, checkpoint_path, checkpoint_metadata,
                             args):
    """Validate all output roots, then atomically publish extraction metadata."""
    metadata = dict(checkpoint_metadata or {})
    payload = {
        'format_version': 1,
        'checkpoint': checkpoint_signature(checkpoint_path),
        'sequence_mode': metadata.get('sequence_mode', 'legacy'),
        'down_t': int(metadata.get('down_t', args.down_t)),
        'stride_t': int(metadata.get('stride_t', args.stride_t)),
        'latent_dim': int(metadata.get('latent_dim', args.latent_dim)),
        'checkpoint_metadata': metadata,
    }
    root_list = [os.path.abspath(os.fspath(root)) for root in roots]
    for root in root_list:
        _validate_extraction_root(root, payload)
    for root in root_list:
        _write_extraction_manifest(root, payload)
    return payload

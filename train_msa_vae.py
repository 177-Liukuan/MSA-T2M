"""
Training script for MSA-VAE (Multi-Scale Semantic Alignment VAE).

Full loss:
  L_total = L_rec + L_KL + λ_root * L_root + λ_latent * L_latent
          + λ_global * L_global_align + λ_local * L_local_align

Where:
  L_rec          : motion reconstruction (Optimal-σ Gaussian NLL)
  L_KL           : KL divergence (posterior vs N(0,I))
  L_root         : root joint reconstruction (first 8 dims)
  L_latent       : Transformer latent reconstruction ||z_local - z_recon||^2
  L_global_align : cosine alignment of h_cls -> CLIP(global caption)
  L_local_align  : cosine alignment of z_i  -> CLIP(local BABEL frame labels)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    import clip
except ImportError:
    clip = None

from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list

import models.msa_vae as msa_vae
import utils.losses as losses
import options.option_msa_vae as option_msa_vae
import utils.utils_model as utils_model
from utils.msa_vae_alignment import (
    distributed_mask_coverage,
    distributed_masked_cosine_alignment,
)
from utils.eval_msa_vae_babel import (
    build_msa_training_loaders,
    evaluate_msa_vae_babel,
    preflight_msa_training_assets,
    prepare_babel_validation_loader,
    validate_msa_assets_after_loader,
)
from humanml3d_272 import dataset_msa_vae
from humanml3d_272.dataset_eval_msa_vae_metrics import (
    MSAVAEMetricsDataset,
    make_msa_vae_metrics_loader,
)
from utils.msa_vae_training import (
    MSAVAELossWeights,
    build_global_alignment_target,
    build_msa_checkpoint_metadata,
    inherit_msa_checkpoint_lineage,
    compute_msa_vae_objective,
    save_msa_checkpoint,
    select_training_batch,
    validate_msa_checkpoint_metadata,
    validate_phase2_parent_metadata,
    validate_phase2_resume_requirement,
    validate_sequence_training_config,
)
from utils.msa_vae_validation import (
    MSAValidationState,
    publish_msa_validation,
    run_deterministic_msa_validation,
)
import sys
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
#   Text Encoders (frozen)
# ---------------------------------------------------------------------------
class FrozenCLIPTextEncoder(nn.Module):
    """Wraps OpenAI CLIP ViT-B/32 text encoder. All params frozen."""
    def __init__(self, clip_version='ViT-B/32', device='cpu'):
        super().__init__()
        if clip is None:
            raise ImportError('clip package is required when text_encoder_type=clip')
        self.clip_model, _ = clip.load(clip_version, device=device, jit=False)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text_list, device):
        tokens = clip.tokenize(text_list, truncate=True).to(device)
        return self.clip_model.encode_text(tokens).float()


class FrozenT5TextEncoder(nn.Module):
    """Wraps SentenceT5 text encoder. All params frozen."""
    def __init__(self, t5_model_path='sentencet5-xxl/', device='cpu', batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(t5_model_path, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text_list, device):
        emb = self.model.encode(
            text_list,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return torch.from_numpy(emb).to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
#   Helpers
# ---------------------------------------------------------------------------
def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    scale = (nb_iter + 1) / (warm_up_iter + 1)
    current_lr = lr * scale
    for param_group in optimizer.param_groups:
        # Respect per-group base LR (set during optimizer init)
        base_lr = param_group.get('initial_lr', lr)
        param_group["lr"] = base_lr * scale
    return optimizer, current_lr


def log_model_params(model, name="Model", accelerator=None):
    """Print model parameter counts and trainable status."""
    if accelerator is not None and not accelerator.is_main_process:
        return

    print(f"\n{'=' * 60}")
    print(f"{name} Parameter Statistics")
    print('=' * 60)

    total_params = 0
    trainable_params = 0
    for n, p in model.named_parameters():
        num = p.numel()
        total_params += num
        if p.requires_grad:
            trainable_params += num
            print(f"  {n:60s} {str(p.shape):20s} trainable")
        else:
            print(f"  {n:60s} {str(p.shape):20s} frozen")

    print(f"Total params      : {total_params:,}")
    print(f"Trainable params  : {trainable_params:,}")
    print(f"Non-trainable     : {total_params - trainable_params:,}")
    print('=' * 60 + '\n')


# ---------------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------------
args = option_msa_vae.get_args_parser()
validate_phase2_resume_requirement(args)
if args.msa_data_mode == 'babel_sparse_global':
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(even_batches=False)
    )
else:
    accelerator = Accelerator()
comp_device = accelerator.device
torch.manual_seed(args.seed)
if args.msa_data_mode == 'humanml_full':
    validate_sequence_training_config(
        phase=args.phase,
        mode=args.sequence_mode,
        full_batch_size=args.full_seq_batch_size,
        replay_interval=args.window_replay_interval,
    )

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok=True)

logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)

# Resolve text encoder setup
default_text_dim = 512 if args.text_encoder_type == 'clip' else 768
if args.text_embed_dim <= 0:
    args.text_embed_dim = default_text_dim

# Force Transformer CLS dim to text embedding dim for alignment compatibility
if args.trans_d_model != args.text_embed_dim:
    logger.info(f'Adjust trans_d_model from {args.trans_d_model} to {args.text_embed_dim} to match text embedding dim')
    args.trans_d_model = args.text_embed_dim

# Keep legacy field in sync
args.clip_dim = args.text_embed_dim

# Resolve offline global embedding directory by encoder type
args.global_embed_dir = args.clip_global_embed_dir if args.text_encoder_type == 'clip' else args.t5_global_embed_dir

logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
logger.info(f"Global text mode: {'offline' if args.use_offline_global_text else 'online'}")
if args.use_offline_global_text:
    logger.info(f'Offline global embed dir: {args.global_embed_dir}')
logger.info(f'Training MSA-VAE on {args.dataname}, motions are with {args.nb_joints} joints')
if args.disable_decoupling:
    logger.info('Using z as Transformer AE input/target (decoupling disabled)')
if args.msa_data_mode == 'babel_sparse_global' and not args.use_offline_global_text:
    raise ValueError('babel_sparse_global requires offline bridge global targets')
logger.info(f'MSA data mode: {args.msa_data_mode}')
logger.info(f'Resolved MSA mean path: {os.path.realpath(args.msa_mean_path)}')
logger.info(f'Resolved MSA std path: {os.path.realpath(args.msa_std_path)}')

##### ---- Dataloader ---- #####
if args.msa_data_mode == 'babel_sparse_global':
    (
        checkpoint_metadata,
        resume_checkpoint,
        validated_cnn_state,
    ) = preflight_msa_training_assets(args, accelerator)
    train_loader, validation_loader, validation_backend = (
        build_msa_training_loaders(args)
    )
    validate_msa_assets_after_loader(args, checkpoint_metadata, accelerator)
    babel_validation_dataset = validation_loader.dataset
    full_loader = None
    window_loader = None
else:
    checkpoint_metadata = build_msa_checkpoint_metadata(args)
    resume_checkpoint = None
    validated_cnn_state = None
    babel_validation_dataset = None
    unit_length = args.stride_t ** args.down_t
    train_dataset = dataset_msa_vae.MSAVAEDataset(
        args.dataname,
        window_size=args.window_size,
        unit_length=unit_length,
        use_ft_split=args.use_ft_split,
        text_encoder_type=args.text_encoder_type,
        clip_embed_dir=args.clip_embed_dir,
        t5_embed_dir=args.t5_embed_dir,
        text_embed_dim=args.text_embed_dim,
        use_offline_global_text=args.use_offline_global_text,
        clip_global_embed_dir=args.clip_global_embed_dir,
        t5_global_embed_dir=args.t5_global_embed_dir,
        sequence_mode=dataset_msa_vae.source_dataset_sequence_mode(
            args.sequence_mode
        ),
    )
    full_loader = None
    window_loader = None
    if args.sequence_mode in ('full', 'mixed'):
        full_loader = dataset_msa_vae.make_loader(
            train_dataset,
            sequence_mode='full',
            batch_size=args.full_seq_batch_size,
            num_workers=8,
            bucket_size=args.length_bucket_size,
            drop_last=True,
            seed=args.seed,
        )
    if args.sequence_mode in ('window', 'mixed'):
        window_loader = dataset_msa_vae.make_loader(
            train_dataset,
            sequence_mode='window',
            batch_size=args.batch_size,
            num_workers=8,
            drop_last=True,
            seed=args.seed,
        )
    if accelerator.is_main_process:
        validation_dataset = MSAVAEMetricsDataset(
            'humanml3d_272',
            split_file='humanml3d_272/split/val.txt',
            unit_length=unit_length,
        )
        validation_loader = make_msa_vae_metrics_loader(
            validation_dataset,
            batch_size=args.validation_batch_size,
            num_workers=8,
            pin_memory=comp_device.type == 'cuda',
        )
        logger.info(
            'Deterministic complete validation: split=val '
            f'samples={len(validation_dataset)} '
            f'sample_hash={validation_dataset.sample_hash} '
            f'seed={args.validation_seed} '
            f'batch_size={args.validation_batch_size}'
        )
    else:
        validation_dataset = None
        validation_loader = None
    validation_backend = 'humanml_complete_deterministic'
logger.info(f'MSA validation backend: {validation_backend}')

##### ---- Network ---- #####
clip_range = [-30, 20]

net = msa_vae.MSA_HumanVAE(
    hidden_size=args.hidden_size,
    down_t=args.down_t,
    stride_t=args.stride_t,
    depth=args.depth,
    dilation_growth_rate=args.dilation_growth_rate,
    activation='relu',
    latent_dim=args.latent_dim,
    clip_range=clip_range,
    trans_d_model=args.trans_d_model,
    trans_nhead=args.trans_nhead,
    trans_enc_layers=args.trans_enc_layers,
    trans_dec_layers=args.trans_dec_layers,
    trans_ff_size=args.trans_ff_size,
    trans_dropout=args.trans_dropout,
    clip_dim=args.text_embed_dim,
    disable_decoupling=args.disable_decoupling,
)

log_model_params(net, name='MSA-HumanVAE (Initial)', accelerator=accelerator)

# ---- Frozen text encoder (only needed for online global text mode) ----
text_encoder = None
if args.global_align_weight > 0 and not args.use_offline_global_text:
    if args.text_encoder_type == 'clip':
        text_encoder = FrozenCLIPTextEncoder(
            clip_version=args.clip_version, device='cpu'
        )
    else:
        text_encoder = FrozenT5TextEncoder(
            t5_model_path=args.t5_model_path, device='cpu', batch_size=args.t5_batch_size
        )

# Optionally load pretrained CNN VAE weights
if args.resume_cnn_pth:
    logger.info(f'Loading pretrained CNN VAE from {args.resume_cnn_pth}')
    if args.msa_data_mode == 'humanml_full':
        cnn_checkpoint = torch.load(
            args.resume_cnn_pth, map_location='cpu'
        )
        cnn_state = (
            cnn_checkpoint
            if not isinstance(cnn_checkpoint, dict)
            or 'net' not in cnn_checkpoint
            else cnn_checkpoint['net']
        )
        validated_cnn_state = {}
        for key, value in cnn_state.items():
            mapped_key = key
            if key.startswith('tae.encoder.'):
                mapped_key = key.replace(
                    'tae.encoder.', 'msa_vae.cnn_encoder.'
                )
            elif key.startswith('tae.decoder.'):
                mapped_key = key.replace(
                    'tae.decoder.', 'msa_vae.cnn_decoder.'
                )
            elif key.startswith('tae.decode_proj.'):
                mapped_key = key.replace(
                    'tae.decode_proj.', 'msa_vae.decode_proj.'
                )
            validated_cnn_state[mapped_key] = value
    missing, unexpected = net.load_state_dict(
        validated_cnn_state, strict=False
    )
    logger.info(f'CNN weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}')
    log_model_params(net, name='MSA-HumanVAE (After CNN weight load)', accelerator=accelerator)

if args.resume_pth:
    logger.info(f'Resuming full MSA-VAE from {args.resume_pth}')
    if args.msa_data_mode == 'babel_sparse_global':
        ckpt = resume_checkpoint
    else:
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        metadata = (
            ckpt.get('metadata') if isinstance(ckpt, dict) else None
        )
        validate_msa_checkpoint_metadata(metadata, args)
        validate_phase2_parent_metadata(metadata, args)
        checkpoint_metadata = inherit_msa_checkpoint_lineage(
            checkpoint_metadata,
            metadata,
            args.resume_pth,
        )
    state = ckpt if not isinstance(ckpt, dict) or 'net' not in ckpt else ckpt['net']
    net.load_state_dict(state, strict=True)

net.to(comp_device)

if text_encoder is not None:
    text_encoder.to(comp_device)
    text_encoder.eval()

##### ---- Phase-aware freeze / unfreeze ---- #####
# Identify CNN params (bottom layer) vs top-layer params
cnn_modules = ['msa_vae.cnn_encoder', 'msa_vae.cnn_decoder', 'msa_vae.decode_proj']
top_modules = ['msa_vae.trans_encoder', 'msa_vae.trans_decoder',
               'msa_vae.global_proj', 'msa_vae.local_proj']

def set_cnn_frozen(model, frozen):
    """Freeze or unfreeze CNN encoder/decoder/decode_proj."""
    for name, param in model.named_parameters():
        if any(name.startswith(m) for m in cnn_modules):
            param.requires_grad = not frozen

if args.phase == 1:
    # Phase 1: freeze CNN, only train Transformer + projections
    set_cnn_frozen(net, frozen=True)
    n_frozen = sum(1 for n, p in net.named_parameters() if not p.requires_grad)
    n_train = sum(1 for n, p in net.named_parameters() if p.requires_grad)
    logger.info(f'Phase 1: CNN frozen ({n_frozen} params frozen, {n_train} trainable)')
elif args.phase == 2:
    # Phase 2: all unfrozen (differential LR set in optimizer)
    set_cnn_frozen(net, frozen=False)
    logger.info(f'Phase 2: all params unfrozen, CNN LR scale = {args.cnn_lr_scale}')
else:
    logger.info(f'Phase 0: legacy mode, all params trainable with uniform LR')

phase_desc = 'Phase 1, CNN frozen' if args.phase == 1 else ('Phase 2, all unfrozen' if args.phase == 2 else 'Phase 0, legacy')
log_model_params(net, name=f'MSA-HumanVAE ({phase_desc})', accelerator=accelerator)

net.train()

##### ---- Evaluator for R_precision / FID ---- #####
evaluator = None
if args.msa_data_mode == 'humanml_full':
    # TMR is deliberately imported and initialized only for HumanML validation.
    sys.path.insert(0, 'Evaluator_272')
    from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
    from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

    evaluator_modelpath = 'Evaluator_272/deps/distilbert-base-uncased'
    eval_textencoder = DistilbertActorAgnosticEncoder(
        evaluator_modelpath, num_layers=4, latent_dim=256
    )
    eval_motionencoder = ActorAgnosticEncoder(
        nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=-1
    )
    evaluator_ckpt_path = 'Evaluator_272/experiments/temos/EXP1/checkpoints/epoch=99.ckpt'
    evaluator_ckpt = torch.load(evaluator_ckpt_path, map_location='cpu')
    for prefix, encoder in [('textencoder', eval_textencoder), ('motionencoder', eval_motionencoder)]:
        state = {
            k.replace(f'{prefix}.', ''): v
            for k, v in evaluator_ckpt['state_dict'].items()
            if k.startswith(f'{prefix}.')
        }
        encoder.load_state_dict(state, strict=True)
        encoder.eval()
        encoder.to(comp_device)
        for p in encoder.parameters():
            p.requires_grad = False
    evaluator = [eval_textencoder, eval_motionencoder]
    logger.info(f'Loaded TMR evaluator from {evaluator_ckpt_path}')
else:
    logger.info('BABEL validation selected: HumanML TMR evaluator is not loaded')

##### ---- Optimizer & Scheduler ---- #####
if args.phase == 2:
    # Differential LR: CNN params get scaled-down LR
    cnn_params, top_params = [], []
    for name, param in net.named_parameters():
        if any(name.startswith(m) for m in cnn_modules):
            cnn_params.append(param)
        else:
            top_params.append(param)
    param_groups = [
        {'params': top_params, 'lr': args.lr},
        {'params': cnn_params, 'lr': args.lr * args.cnn_lr_scale},
    ]
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.99),
                            weight_decay=args.weight_decay)
    logger.info(f'Optimizer: top LR={args.lr}, CNN LR={args.lr * args.cnn_lr_scale}')
else:
    # Phase 0/1: uniform LR on trainable params only
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.99),
                            weight_decay=args.weight_decay)
# Store initial_lr for warm-up scheduling
for pg in optimizer.param_groups:
    pg['initial_lr'] = pg['lr']
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=args.lr_scheduler, gamma=args.gamma,
)

if args.msa_data_mode == 'babel_sparse_global':
    net, optimizer, train_loader = accelerator.prepare(
        net, optimizer, train_loader
    )
    validation_loader = prepare_babel_validation_loader(
        accelerator, validation_loader
    )
    train_loader_iter = dataset_msa_vae.cycle(train_loader)
    full_loader_iter = None
    window_loader_iter = None
else:
    if args.sequence_mode == 'mixed':
        net, optimizer, full_loader, window_loader = accelerator.prepare(
            net, optimizer, full_loader, window_loader,
        )
    elif args.sequence_mode == 'full':
        net, optimizer, full_loader = accelerator.prepare(
            net, optimizer, full_loader,
        )
    else:
        net, optimizer, window_loader = accelerator.prepare(
            net, optimizer, window_loader,
        )
    full_loader_iter = (
        dataset_msa_vae.cycle(full_loader)
        if full_loader is not None
        else None
    )
    window_loader_iter = (
        dataset_msa_vae.cycle(window_loader)
        if window_loader is not None
        else None
    )
    train_loader_iter = None

##### ---- Losses ---- #####
Loss = losses.ReConsLoss(motion_dim=272)
latent_recon_loss_fn = nn.MSELoss()
loss_weights = MSAVAELossWeights(
    root=args.root_loss,
    latent=args.latent_recon_weight,
    global_align=args.global_align_weight,
    local_align=args.local_align_weight,
)
##### ---- Training function (shared by warmup & main) ---- #####
def compute_losses(batch, net_module):
    """Compute fixed-window sparse-supervision BABEL losses.

    Returns: (total_loss, loss_dict)
    """
    gt_motion, captions, global_text_gt, has_global, local_text_gt, has_local, total_frames, local_text_pooled = batch
    gt_motion = gt_motion.to(comp_device).float()
    global_text_gt = global_text_gt.to(comp_device).float()
    has_global = has_global.to(comp_device)
    local_text_gt = local_text_gt.to(comp_device).float()
    has_local = has_local.to(comp_device)
    total_frames = total_frames.to(comp_device)
    local_text_pooled = local_text_pooled.to(comp_device).float()

    out = net_module(gt_motion)

    # --- Reconstruction losses ---
    # Phase 1: skip recon/kl/root (CNN frozen, these produce zero gradients)
    if args.phase == 1:
        zero = torch.tensor(0.0, device=comp_device)
        loss_motion = zero
        loss_kl = zero
        loss_root = zero
    else:
        loss_motion = Loss(out['x_recon'], gt_motion)
        loss_kl = Loss.forward_KL(out['mu'], out['logvar'])
        loss_root = Loss.forward_root(out['x_recon'], gt_motion)
    if args.disable_decoupling:
        loss_latent = latent_recon_loss_fn(out['mu_recon'], out['trans_latent_target'].detach())
    else:
        loss_latent = latent_recon_loss_fn(out['mu_recon'], out['mu'].detach())

    loss_dict = {
        'recon': loss_motion,
        'kl': loss_kl,
        'root': loss_root,
        'latent': loss_latent,
    }

    # --- Global alignment: h_cls vs Spotlight(global text, local pooled text) ---
    global_backward_loss = out['clip_global_feat'].sum() * 0.0
    global_mean = global_backward_loss.detach()
    if args.use_offline_global_text:
        valid_global = has_global
    else:
        valid_global = torch.ones_like(has_local, dtype=torch.bool)
    global_valid_count, global_valid_ratio = distributed_mask_coverage(
        valid_global, tokens_per_sample=1, accelerator=accelerator
    )
    if args.global_align_weight > 0:
        with torch.no_grad():
            if args.use_offline_global_text:
                text_feat = global_text_gt
            else:
                text_feat = text_encoder.encode_text(captions, comp_device)

            # Compute interpolation alpha
            if args.spotlight_alpha < 0:
                alpha = (args.window_size / total_frames.float().to(comp_device)).clamp(0, 1)
            else:
                alpha = torch.full((gt_motion.size(0),), args.spotlight_alpha,
                                   device=comp_device)

            # local pooled text embedding from frame-level labels
            l_pooled = local_text_pooled

            # Zero out alpha for samples without local embeddings
            alpha = alpha * has_local.float()

            # G_mixed = (1 - alpha) * global + alpha * local pooled
            alpha = alpha.unsqueeze(-1)
            g_mixed = (1 - alpha) * text_feat + alpha * l_pooled
            g_target = F.normalize(g_mixed, dim=-1)

        global_alignment = distributed_masked_cosine_alignment(
            out['clip_global_feat'], g_target, valid_global, accelerator
        )
        global_backward_loss = global_alignment.backward_loss
        global_mean = global_alignment.global_mean
        global_valid_count = global_alignment.valid_count
    loss_dict['global_align'] = global_mean
    loss_dict['global_valid_count'] = global_valid_count.float()
    loss_dict['global_valid_ratio'] = global_valid_ratio

    # --- Local alignment: z_i projected vs CLIP(local label) ---
    local_backward_loss = out['clip_local_feat'].sum() * 0.0
    local_mean = local_backward_loss.detach()
    local_valid_count, local_valid_ratio = distributed_mask_coverage(
        has_local,
        tokens_per_sample=local_text_gt.shape[1],
        accelerator=accelerator,
    )
    if args.local_align_weight > 0:
        local_alignment = distributed_masked_cosine_alignment(
            out['clip_local_feat'], local_text_gt, has_local, accelerator
        )
        local_backward_loss = local_alignment.backward_loss
        local_mean = local_alignment.global_mean
        local_valid_count = local_alignment.valid_count
    loss_dict['local_align'] = local_mean
    loss_dict['local_valid_count'] = local_valid_count.float()
    loss_dict['local_valid_ratio'] = local_valid_ratio

    # --- Total loss ---
    if args.phase == 1:
        # Phase 1: only latent + alignment losses
        total_loss = (args.latent_recon_weight * loss_latent
                      + args.global_align_weight * global_backward_loss
                      + args.local_align_weight * local_backward_loss)
    else:
        # Phase 0/2: all losses
        total_loss = (loss_motion
                      + loss_kl
                      + args.root_loss * loss_root
                      + args.latent_recon_weight * loss_latent
                      + args.global_align_weight * global_backward_loss
                      + args.local_align_weight * local_backward_loss)

    return total_loss, loss_dict


def compute_humanml_losses(batch, batch_kind):
    """Compute masked HumanML losses for full or replay batches."""
    (
        gt_motion,
        captions,
        global_text_gt,
        has_global,
        local_text_gt,
        has_local,
        total_frames,
        local_text_pooled,
        motion_lengths,
    ) = batch
    gt_motion = gt_motion.to(comp_device).float()
    global_text_gt = global_text_gt.to(comp_device).float()
    has_global = has_global.to(comp_device)
    local_text_gt = local_text_gt.to(comp_device).float()
    has_local = has_local.to(comp_device)
    total_frames = total_frames.to(comp_device)
    local_text_pooled = local_text_pooled.to(comp_device).float()
    motion_lengths = motion_lengths.to(comp_device)

    out = net(
        gt_motion,
        lengths=motion_lengths,
        semantic_only=args.phase == 1,
    )

    with torch.no_grad():
        if args.use_offline_global_text:
            text_feat = global_text_gt
            valid_global = has_global
        else:
            text_feat = text_encoder.encode_text(captions, comp_device)
            valid_global = torch.ones_like(has_local, dtype=torch.bool)
        global_target = build_global_alignment_target(
            global_text=text_feat,
            local_pooled=local_text_pooled,
            has_local=has_local,
            total_frames=total_frames,
            window_size=args.window_size,
            sequence_mode=args.sequence_mode,
            spotlight_alpha=args.spotlight_alpha,
        )

    targets = {
        'motion': gt_motion,
        'motion_lengths': motion_lengths,
        'global_text': global_target,
        'has_global': valid_global,
        'local_text': local_text_gt,
        'has_local': has_local,
    }
    return compute_msa_vae_objective(
        out,
        targets,
        phase=args.phase,
        batch_kind=batch_kind,
        weights=loss_weights,
        stride_t=args.stride_t,
        down_t=args.down_t,
    )


def select_current_training_batch(step):
    """Route BABEL fixed windows or HumanML full/replay batches."""
    if args.msa_data_mode == 'babel_sparse_global':
        return next(train_loader_iter), 'window'
    return select_training_batch(
        step,
        args.sequence_mode,
        full_loader_iter,
        window_loader_iter,
        args.window_replay_interval,
    )


def run_humanml_complete_validation(iteration, validation_state):
    """Run main-rank complete validation and propagate failures to all ranks."""
    accelerator.wait_for_everyone()
    validation_error = [None]
    caught_error = None
    if accelerator.is_main_process:
        try:
            eval_model = accelerator.unwrap_model(net)
            result = run_deterministic_msa_validation(
                eval_model,
                evaluator,
                validation_loader,
                comp_device,
                seed=args.validation_seed,
            )
            validation_state = publish_msa_validation(
                result=result,
                iteration=iteration,
                out_dir=args.out_dir,
                model=eval_model,
                metadata=checkpoint_metadata,
                state=validation_state,
                logger=logger,
                writer=writer,
                validation_seed=args.validation_seed,
                validation_batch_size=args.validation_batch_size,
            )
        except Exception as exc:
            caught_error = exc
            validation_error[0] = (
                f'{type(exc).__name__}: {exc}'
            )
    broadcast_object_list(validation_error, from_process=0)
    if validation_error[0] is not None:
        message = (
            'deterministic HumanML validation failed on main rank: '
            f'{validation_error[0]}'
        )
        if caught_error is not None:
            raise RuntimeError(message) from caught_error
        raise RuntimeError(message)
    accelerator.wait_for_everyone()
    return validation_state


##### ---- Warm-up ---- #####
loss_log_names = [
    'recon', 'kl', 'root', 'latent', 'global_align', 'local_align',
]
if args.msa_data_mode == 'babel_sparse_global':
    loss_log_names.extend([
        'global_valid_count',
        'local_valid_count',
        'global_valid_ratio',
        'local_valid_ratio',
    ])
avg = {key: 0.0 for key in loss_log_names}

logger.info(f'=== Warm-up: {args.warm_up_iter} iterations ===')
for nb_iter in range(1, args.warm_up_iter):
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    batch, batch_kind = select_current_training_batch(nb_iter)
    if args.msa_data_mode == 'babel_sparse_global':
        total_loss, loss_dict = compute_losses(batch, net)
    else:
        total_loss, loss_dict = compute_humanml_losses(batch, batch_kind)

    optimizer.zero_grad()
    accelerator.backward(total_loss)
    optimizer.step()

    for k in avg:
        if k in loss_dict:
            avg[k] += loss_dict[k].item()

    if nb_iter % args.print_iter == 0:
        if accelerator.is_main_process:
            log_parts = [f"Warmup Iter {nb_iter} : lr {current_lr:.5f}"]
            for k in avg:
                avg[k] /= args.print_iter
                log_parts.append(f"{k} {avg[k]:.5f}")
            logger.info("  ".join(log_parts))
        avg = {k: 0. for k in avg}


##### ---- Training ---- #####
avg = {key: 0.0 for key in loss_log_names}

# Initial eval
best_semantic, best_mpjpe = float('inf'), float('inf')
validation_state = MSAValidationState()
if args.msa_data_mode == 'babel_sparse_global':
    eval_net = accelerator.unwrap_model(net)
    babel_result = evaluate_msa_vae_babel(
        args.out_dir, validation_loader, eval_net, babel_validation_dataset,
        logger, writer, iteration=0, phase=args.phase,
        best_semantic=best_semantic, best_mpjpe=best_mpjpe,
        device=comp_device, accelerator=accelerator,
        metadata=checkpoint_metadata,
    )
    best_semantic = babel_result.best_semantic
    best_mpjpe = babel_result.best_mpjpe
else:
    validation_state = run_humanml_complete_validation(
        iteration=0,
        validation_state=validation_state,
    )

logger.info(f'=== Main training: {args.total_iter} iterations ===')
logger.info(f'  Loss weights: root={args.root_loss}, latent={args.latent_recon_weight}, '
            f'global={args.global_align_weight}, local={args.local_align_weight}')
logger.info(f'  Spotlight alpha: {args.spotlight_alpha} '
            f'({"dynamic: window/total" if args.spotlight_alpha < 0 else "fixed"})')

for nb_iter in range(1, args.total_iter + 1):
    batch, batch_kind = select_current_training_batch(nb_iter)
    if args.msa_data_mode == 'babel_sparse_global':
        total_loss, loss_dict = compute_losses(batch, net)
    else:
        total_loss, loss_dict = compute_humanml_losses(batch, batch_kind)

    optimizer.zero_grad()
    accelerator.backward(total_loss)
    optimizer.step()
    scheduler.step()

    for k in avg:
        try:
            avg[k] += loss_dict[k].item()
        except:
            pass

    if nb_iter % args.print_iter == 0:
        if accelerator.is_main_process:
            log_parts = [f"Train Iter {nb_iter} :"]
            for k in avg:
                avg[k] /= args.print_iter
                writer.add_scalar(f'Train/{k}', avg[k], nb_iter)
                log_parts.append(f"{k} {avg[k]:.5f}")
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Train/LR', current_lr, nb_iter)
            logger.info("  ".join(log_parts))
        avg = {k: 0. for k in avg}

    if nb_iter % args.eval_iter == 0:
        if args.msa_data_mode == 'babel_sparse_global':
            eval_net = accelerator.unwrap_model(net)
            babel_result = evaluate_msa_vae_babel(
                args.out_dir, validation_loader, eval_net,
                babel_validation_dataset, logger, writer,
                iteration=nb_iter, phase=args.phase,
                best_semantic=best_semantic, best_mpjpe=best_mpjpe,
                device=comp_device, accelerator=accelerator,
                metadata=checkpoint_metadata,
            )
            best_semantic = babel_result.best_semantic
            best_mpjpe = babel_result.best_mpjpe
        else:
            validation_state = run_humanml_complete_validation(
                iteration=nb_iter,
                validation_state=validation_state,
            )

accelerator.wait_for_everyone()
if (
    args.msa_data_mode == 'humanml_full'
    and accelerator.is_main_process
):
    final_model = accelerator.unwrap_model(net)
    save_msa_checkpoint(
        os.path.join(args.out_dir, 'net_last.pth'),
        final_model,
        checkpoint_metadata,
    )
    logger.info('Saved final Phase checkpoint to net_last.pth')

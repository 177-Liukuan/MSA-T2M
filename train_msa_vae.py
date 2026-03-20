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
import clip
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

import models.msa_vae as msa_vae
import utils.losses as losses
import options.option_msa_vae as option_msa_vae
import utils.utils_model as utils_model
from humanml3d_272 import dataset_msa_vae, dataset_eval_tae, dataset_eval_t2m
import utils.eval_trans as eval_trans
import sys
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
#   CLIP Text Encoder (frozen)
# ---------------------------------------------------------------------------
class FrozenCLIPTextEncoder(nn.Module):
    """Wraps OpenAI CLIP ViT-B/32 text encoder. All params frozen."""
    def __init__(self, clip_version='ViT-B/32', device='cpu'):
        super().__init__()
        self.clip_model, _ = clip.load(clip_version, device=device, jit=False)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text_list, device):
        """Encode a list of strings -> (B, 512) float tensor."""
        tokens = clip.tokenize(text_list, truncate=True).to(device)
        return self.clip_model.encode_text(tokens).float()


# ---------------------------------------------------------------------------
#   Alignment Losses
# ---------------------------------------------------------------------------
class CLIPAlignmentLoss(nn.Module):
    """Cosine embedding loss for cross-modal alignment."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(margin=0.0)

    def forward(self, feat_a, feat_b, mask=None):
        """
        Args:
            feat_a: (B, D) or (B, T, D)
            feat_b: same shape as feat_a
            mask:   (B,) bool - True means this sample is **valid**
                    If None, all samples are valid.
        Returns:
            scalar loss
        """
        target = torch.ones(feat_a.size(0), device=feat_a.device)

        if feat_a.dim() == 3:
            # Per-token alignment: flatten (B, T, D) -> (B*T, D)
            B, T, D = feat_a.shape
            feat_a = feat_a.reshape(B * T, D)
            feat_b = feat_b.reshape(B * T, D)
            if mask is not None:
                # Expand mask from (B,) -> (B, T) -> (B*T,)
                token_mask = mask.unsqueeze(1).expand(-1, T).reshape(B * T)
                feat_a = feat_a[token_mask]
                feat_b = feat_b[token_mask]
            target = torch.ones(feat_a.size(0), device=feat_a.device)
        else:
            if mask is not None:
                feat_a = feat_a[mask]
                feat_b = feat_b[mask]
                target = torch.ones(feat_a.size(0), device=feat_a.device)

        if feat_a.size(0) == 0:
            return torch.tensor(0.0, device=feat_a.device, requires_grad=True)

        # L2 normalize before cosine loss for stability
        feat_a = F.normalize(feat_a, dim=-1)
        feat_b = F.normalize(feat_b, dim=-1)
        return self.loss_fn(feat_a, feat_b, target)


# ---------------------------------------------------------------------------
#   Eval-compatible wrapper
# ---------------------------------------------------------------------------
class EvalCompat(nn.Module):
    """Thin wrapper: MSA_HumanVAE dict output -> tuple for eval functions."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out['x_recon'], out['mu'], out['logvar']

    def forward_decoder(self, z):
        return self.model.forward_decoder(z)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)


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


# ---------------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------------
accelerator = Accelerator()
comp_device = accelerator.device

args = option_msa_vae.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok=True)

logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
logger.info(f'Training MSA-VAE on {args.dataname}, motions are with {args.nb_joints} joints')

##### ---- Dataloader ---- #####
train_loader = dataset_msa_vae.DATALoader(
    args.dataname, args.batch_size,
    window_size=args.window_size, unit_length=2 ** args.down_t,
    use_ft_split=args.use_ft_split,
)
val_loader = dataset_eval_tae.DATALoader(
    args.dataname, False, 32, unit_length=2 ** args.down_t,
)

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
    clip_dim=args.clip_dim,
)

# ---- Frozen CLIP text encoder ----
clip_text_encoder = FrozenCLIPTextEncoder(
    clip_version=args.clip_version, device='cpu'
)

# Optionally load pretrained CNN VAE weights
if args.resume_cnn_pth:
    logger.info(f'Loading pretrained CNN VAE from {args.resume_cnn_pth}')
    ckpt = torch.load(args.resume_cnn_pth, map_location='cpu')
    cnn_state = ckpt if not isinstance(ckpt, dict) or 'net' not in ckpt else ckpt['net']
    mapped = {}
    for k, v in cnn_state.items():
        new_key = k
        if k.startswith('tae.encoder.'):
            new_key = k.replace('tae.encoder.', 'msa_vae.cnn_encoder.')
        elif k.startswith('tae.decoder.'):
            new_key = k.replace('tae.decoder.', 'msa_vae.cnn_decoder.')
        elif k.startswith('tae.decode_proj.'):
            new_key = k.replace('tae.decode_proj.', 'msa_vae.decode_proj.')
        mapped[new_key] = v
    missing, unexpected = net.load_state_dict(mapped, strict=False)
    logger.info(f'CNN weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}')

if args.resume_pth:
    logger.info(f'Resuming full MSA-VAE from {args.resume_pth}')
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    state = ckpt if not isinstance(ckpt, dict) or 'net' not in ckpt else ckpt['net']
    net.load_state_dict(state, strict=True)

net.to(comp_device)

clip_text_encoder.to(comp_device)
clip_text_encoder.eval()

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

net.train()
net_eval = EvalCompat(net)

##### ---- Evaluator for R_precision / FID ---- #####
# Text-aware val loader (NOT prepared by accelerator, runs on main only)
val_loader_t2m = dataset_eval_t2m.DATALoader(
    args.dataname, False, 32, unit_length=2 ** args.down_t,
)
# Load TMR/TEMOS evaluator encoders
sys.path.insert(0, 'Evaluator_272')
from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

evaluator_modelpath = 'Evaluator_272/deps/distilbert-base-uncased'
eval_textencoder = DistilbertActorAgnosticEncoder(evaluator_modelpath, num_layers=4, latent_dim=256)
eval_motionencoder = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)

evaluator_ckpt_path = 'Evaluator_272/experiments/temos/EXP1/checkpoints/epoch=99.ckpt'
evaluator_ckpt = torch.load(evaluator_ckpt_path, map_location='cpu')
for prefix, encoder in [('textencoder', eval_textencoder), ('motionencoder', eval_motionencoder)]:
    state = {k.replace(f'{prefix}.', ''): v for k, v in evaluator_ckpt['state_dict'].items() if k.startswith(f'{prefix}.')}
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    encoder.to(comp_device)
    for p in encoder.parameters():
        p.requires_grad = False
evaluator = [eval_textencoder, eval_motionencoder]
logger.info(f'Loaded TMR evaluator from {evaluator_ckpt_path}')

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

net, optimizer, train_loader, val_loader = accelerator.prepare(
    net, optimizer, train_loader, val_loader,
)
train_loader_iter = dataset_msa_vae.cycle(train_loader)

##### ---- Losses ---- #####
Loss = losses.ReConsLoss(motion_dim=272)
latent_recon_loss_fn = nn.MSELoss()
global_align_loss_fn = CLIPAlignmentLoss()
local_align_loss_fn = CLIPAlignmentLoss()

##### ---- Training function (shared by warmup & main) ---- #####
def compute_losses(batch, net_module):
    """Compute all MSA-VAE losses from one batch.

    Returns: (total_loss, loss_dict)
    """
    gt_motion, captions, local_clip_gt, has_local, total_frames, local_clip_pooled = batch

    gt_motion = gt_motion.to(comp_device).float()
    local_clip_gt = local_clip_gt.to(comp_device).float()
    has_local = has_local.to(comp_device)
    total_frames = total_frames.to(comp_device)
    local_clip_pooled = local_clip_pooled.to(comp_device).float()

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
    loss_latent = latent_recon_loss_fn(out['mu_recon'], out['mu'].detach())

    loss_dict = {
        'recon': loss_motion,
        'kl': loss_kl,
        'root': loss_root,
        'latent': loss_latent,
    }

    # --- Global alignment: h_cls vs Spotlight(CLIP caption, local) ---
    loss_global = torch.tensor(0.0, device=comp_device, requires_grad=True)
    if args.global_align_weight > 0:
        with torch.no_grad():
            clip_text_feat = clip_text_encoder.encode_text(captions, comp_device)
            # clip_text_feat: (B, 512) — original global text vector G_origin

            # --- Spotlight: dynamic context label ---
            # Compute interpolation alpha
            if args.spotlight_alpha < 0:
                # Dynamic: alpha = window_size / total_frames (per sample)
                alpha = (args.window_size / total_frames.float().to(comp_device)).clamp(0, 1)
            else:
                alpha = torch.full((gt_motion.size(0),), args.spotlight_alpha,
                                   device=comp_device)

            # L_pooled: pre-computed mean of 64 raw frame-level CLIP features -> (B, 512)
            l_pooled = local_clip_pooled

            # Zero out alpha for samples without local CLIP embeddings
            alpha = alpha * has_local.float()

            # G_mixed = (1 - alpha) * G_origin + alpha * L_pooled
            alpha = alpha.unsqueeze(-1)  # (B, 1)
            g_mixed = (1 - alpha) * clip_text_feat + alpha * l_pooled

            # L2 normalize -> project back to CLIP unit hypersphere
            g_target = F.normalize(g_mixed, dim=-1)

        loss_global = global_align_loss_fn(out['clip_global_feat'], g_target)
    loss_dict['global_align'] = loss_global

    # --- Local alignment: z_i projected vs CLIP(local label) ---
    loss_local = torch.tensor(0.0, device=comp_device, requires_grad=True)
    if args.local_align_weight > 0 and has_local.any():
        loss_local = local_align_loss_fn(
            out['clip_local_feat'], local_clip_gt, mask=has_local
        )
    loss_dict['local_align'] = loss_local

    # --- Total loss ---
    if args.phase == 1:
        # Phase 1: only latent + alignment losses
        total_loss = (args.latent_recon_weight * loss_latent
                      + args.global_align_weight * loss_global
                      + args.local_align_weight * loss_local)
    else:
        # Phase 0/2: all losses
        total_loss = (loss_motion
                      + loss_kl
                      + args.root_loss * loss_root
                      + args.latent_recon_weight * loss_latent
                      + args.global_align_weight * loss_global
                      + args.local_align_weight * loss_local)

    return total_loss, loss_dict


##### ---- Warm-up ---- #####
avg = {k: 0. for k in ['recon', 'kl', 'root', 'latent', 'global_align', 'local_align']}

logger.info(f'=== Warm-up: {args.warm_up_iter} iterations ===')
for nb_iter in range(1, args.warm_up_iter):
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    batch = next(train_loader_iter)
    net_module = net.module if args.num_gpus > 1 else net
    total_loss, loss_dict = compute_losses(batch, net_module)

    optimizer.zero_grad()
    accelerator.backward(total_loss)
    optimizer.step()

    for k in avg:
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
avg = {k: 0. for k in ['recon', 'kl', 'root', 'latent', 'global_align', 'local_align']}

# Initial eval
eval_net = net.module if args.num_gpus > 1 else net
net_eval.model = eval_net
best_iter, best_fid, best_mpjpe, writer, logger = eval_trans.evaluation_msa_vae_multi(
    args.out_dir, val_loader_t2m, net_eval, logger, writer, 0,
    best_iter=0, best_fid=1e6, best_mpjpe=1000,
    evaluator=evaluator, device=comp_device, accelerator=accelerator,
)

logger.info(f'=== Main training: {args.total_iter} iterations ===')
logger.info(f'  Loss weights: root={args.root_loss}, latent={args.latent_recon_weight}, '
            f'global={args.global_align_weight}, local={args.local_align_weight}')
logger.info(f'  Spotlight alpha: {args.spotlight_alpha} '
            f'({"dynamic: window/total" if args.spotlight_alpha < 0 else "fixed"})')

for nb_iter in range(1, args.total_iter + 1):
    batch = next(train_loader_iter)
    net_module = net.module if args.num_gpus > 1 else net
    total_loss, loss_dict = compute_losses(batch, net_module)

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
        eval_net = net.module if args.num_gpus > 1 else net
        net_eval.model = eval_net
        best_iter, best_fid, best_mpjpe, writer, logger = eval_trans.evaluation_msa_vae_multi(
            args.out_dir, val_loader_t2m, net_eval, logger, writer, nb_iter,
            best_iter, best_fid, best_mpjpe,
            evaluator=evaluator, device=comp_device, accelerator=accelerator,
        )

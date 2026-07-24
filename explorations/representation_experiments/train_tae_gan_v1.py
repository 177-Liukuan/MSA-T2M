"""
TAE-GAN-v1 Training Script.

Training strategy
-----------------
Phase 0  [0 .. disc_start)   : reconstruction-only warmup.
Phase 1  [disc_start .. end] : interleaved generator + discriminator updates.

Generator update  (per step):
  loss_G = loss_recon + loss_KL + root_loss
           + d_weight * loss_adv_G        (adversarial)
           + fm_weight * loss_fm          (feature matching)

Discriminator update (per step):
  loss_D = hinge_d_loss(real, fake)

Two separate AdamW optimizers:
  opt_G  →  decoder parameters only  (lr = args.lr)
  opt_D  →  discriminator parameters (lr = args.lr_disc)
"""

import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import warnings

import models.tae_gan_v1 as tae_gan
import utils.losses as losses
import options.option_tae_gan_v1 as option_tae_gan_v1
import utils.utils_model as utils_model
from humanml3d_272 import dataset_tae, dataset_eval_tae
import utils.eval_trans as eval_trans
from models.motion_dynamics import feature_matching_loss

warnings.filterwarnings('ignore')

# ── Accelerator ──────────────────────────────────────────────────────────────
accelerator = Accelerator()
comp_device = accelerator.device


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for pg in optimizer.param_groups:
        pg['lr'] = current_lr
    return optimizer, current_lr


# ── Args ─────────────────────────────────────────────────────────────────────
args = option_tae_gan_v1.get_args_parser()
args = args.parse_args()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, args.exp_name)
os.makedirs(args.out_dir, exist_ok=True)

# ── Logger ────────────────────────────────────────────────────────────────────
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# ── Normalisation stats (for dynamics features) ───────────────────────────────
if args.dataname == 't2m_272':
    meta_dir = './humanml3d_272/mean_std'
elif args.dataname == 't2m_babel_272':
    meta_dir = './babel_272/t2m_babel_mean_std'
else:
    raise ValueError(f'Unknown dataname: {args.dataname}')

mean_np = np.load(os.path.join(meta_dir, 'Mean.npy'))   # [272]
std_np  = np.load(os.path.join(meta_dir, 'Std.npy'))    # [272]

# ── Dataloaders ───────────────────────────────────────────────────────────────
train_loader = dataset_tae.DATALoader(
    args.dataname,
    args.batch_size,
    window_size=args.window_size,
    unit_length=2 ** args.down_t,
)
val_loader = dataset_eval_tae.DATALoader(
    args.dataname, False, 32, unit_length=2 ** args.down_t,
)

# ── Model ─────────────────────────────────────────────────────────────────────
clip_range = [-30, 20]
net = tae_gan.TAEGANV1(
    hidden_size=args.hidden_size,
    down_t=args.down_t,
    stride_t=args.stride_t,
    depth=args.depth,
    dilation_growth_rate=args.dilation_growth_rate,
    latent_dim=args.latent_dim,
    clip_range=clip_range,
    disc_start=args.disc_start,
    disc_weight=args.disc_weight,
    fm_weight=args.fm_weight,
    disc_ndf=args.disc_ndf,
    disc_n_layers=args.disc_n_layers,
)

# Load pretrained TAE backbone
if accelerator.is_main_process:
    logger.info(f'Loading TAE checkpoint from {args.tae_ckpt}')
missing, unexpected = net.load_tae_checkpoint(args.tae_ckpt, strict=False)
if accelerator.is_main_process and missing:
    logger.info(f'Missing keys (expected for new GAN params): {missing[:5]} ...')

# Optionally resume a previous TAE-GAN-v1 run
if args.resume_pth:
    if accelerator.is_main_process:
        logger.info(f'Resuming TAE-GAN-v1 from {args.resume_pth}')
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt.get('net', ckpt), strict=False)

net.train()
net.to(comp_device)

# Move normalisation tensors to device
mean_t = torch.from_numpy(mean_np).float().to(comp_device)
std_t  = torch.from_numpy(std_np).float().to(comp_device)

# ── Optimizers ────────────────────────────────────────────────────────────────
# Decoder params only (encoder is frozen)
decoder_params = [p for p in net.tae.tae.decoder.parameters() if p.requires_grad]
decoder_params += [p for p in net.tae.tae.decode_proj.parameters() if p.requires_grad]

optimizer_G = optim.AdamW(decoder_params, lr=args.lr,
                           betas=(0.9, 0.99), weight_decay=args.weight_decay)
optimizer_D = optim.AdamW(net.discriminator.parameters(), lr=args.lr_disc,
                           betas=(0.9, 0.99), weight_decay=args.weight_decay)

scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_G, milestones=args.lr_scheduler, gamma=args.gamma)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_D, milestones=args.lr_scheduler, gamma=args.gamma)

# ── Prepare with Accelerate ───────────────────────────────────────────────────
net, optimizer_G, optimizer_D, train_loader, val_loader = accelerator.prepare(
    net, optimizer_G, optimizer_D, train_loader, val_loader
)
train_loader_iter = dataset_tae.cycle(train_loader)

# ── Loss ──────────────────────────────────────────────────────────────────────
Loss = losses.ReConsLoss(motion_dim=272)

# ── Helper: unwrap net safely for multi-GPU ──────────────────────────────────
def unwrap(model):
    return model.module if hasattr(model, 'module') else model

# ── Logging accumulators ──────────────────────────────────────────────────────
avg_recons = avg_kl = avg_root = avg_g = avg_d = avg_fm = 0.

# ── Warm-up (reconstruction only) ─────────────────────────────────────────────
for nb_iter in range(1, args.warm_up_iter + 1):
    optimizer_G, current_lr = update_lr_warm_up(
        optimizer_G, nb_iter, args.warm_up_iter, args.lr)

    gt_motion = next(train_loader_iter).to(comp_device).float()

    net_m = unwrap(net)
    pred_motion, mu, logvar = net_m(gt_motion)

    loss_recon = Loss(pred_motion, gt_motion)
    loss_kl    = Loss.forward_KL(mu, logvar)
    loss_root  = Loss.forward_root(pred_motion, gt_motion)
    loss = loss_recon + loss_kl + args.root_loss * loss_root

    optimizer_G.zero_grad()
    accelerator.backward(loss)
    optimizer_G.step()

    avg_recons += loss_recon.item()
    avg_kl     += loss_kl.item()
    avg_root   += loss_root.item()

    if nb_iter % args.print_iter == 0 and accelerator.is_main_process:
        avg_recons /= args.print_iter
        avg_kl     /= args.print_iter
        avg_root   /= args.print_iter
        logger.info(
            f'Warmup Iter {nb_iter}: lr={current_lr:.2e} '
            f'Recons={avg_recons:.5f} KL={avg_kl:.5f} Root={avg_root:.5f}'
        )
        avg_recons = avg_kl = avg_root = 0.

# ── Initial evaluation ────────────────────────────────────────────────────────
net_m = unwrap(net)
best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_tae_multi(
    args.out_dir, val_loader, net_m, logger, writer, 0,
    best_iter=0, best_mpjpe=1000, device=comp_device, accelerator=accelerator,
)

# ── Main training loop ────────────────────────────────────────────────────────
avg_recons = avg_kl = avg_root = avg_g = avg_d = avg_fm = 0.
current_lr = args.lr

for nb_iter in range(1, args.total_iter + 1):

    gt_motion = next(train_loader_iter).to(comp_device).float()
    net_m = unwrap(net)

    # ── Forward (encoder frozen → only decoder in graph) ─────────────────
    pred_motion, mu, logvar = net_m(gt_motion)

    # ── Reconstruction losses ─────────────────────────────────────────────
    loss_recon = Loss(pred_motion, gt_motion)
    loss_kl    = Loss.forward_KL(mu, logvar)
    loss_root  = Loss.forward_root(pred_motion, gt_motion)
    loss_base  = loss_recon + loss_kl + args.root_loss * loss_root

    # ── Generator update ──────────────────────────────────────────────────
    g_loss, feats_fake_list, d_weight, g_log = net_m.generator_step(
        motion_pred=pred_motion,
        nll_loss=loss_base,
        mean=mean_t,
        std=std_t,
        global_step=nb_iter,
    )

    if g_loss is not None:
        # Feature matching: get real feats (detached, no_grad)
        feats_real = net_m.real_dynamics_features(gt_motion, mean_t, std_t)
        _, feats_real_list = net_m.discriminator(feats_real, return_features=True)
        fm_loss = feature_matching_loss(feats_real_list, feats_fake_list)

        loss_G = loss_base + d_weight * g_loss + args.fm_weight * fm_loss
    else:
        loss_G = loss_base
        fm_loss = torch.tensor(0.0)

    optimizer_G.zero_grad()
    accelerator.backward(loss_G)
    optimizer_G.step()
    scheduler_G.step()

    # ── Discriminator update ──────────────────────────────────────────────
    d_loss, d_log = net_m.discriminator_step(
        motion_gt=gt_motion,
        motion_pred=pred_motion,
        mean=mean_t,
        std=std_t,
        global_step=nb_iter,
    )

    if d_loss is not None and (nb_iter % args.disc_freq == 0):
        optimizer_D.zero_grad()
        accelerator.backward(d_loss)
        if args.disc_clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                net_m.discriminator.parameters(), args.disc_clip_grad)
        optimizer_D.step()
    elif d_loss is None:
        pass  # before disc_start
    # Step scheduler_D every iter (mirrors scheduler_G), so milestones fire
    # at the same absolute iteration regardless of disc_start warmup offset.
    scheduler_D.step()

    # ── Accumulate logging ────────────────────────────────────────────────
    try:
        avg_recons += loss_recon.item()
        avg_kl     += loss_kl.item()
        avg_root   += loss_root.item()
        avg_g      += g_loss.item()    if g_loss  is not None else 0.
        avg_d      += d_loss.item()    if d_loss  is not None else 0.
        avg_fm     += fm_loss.item()   if fm_loss is not None else 0.
    except Exception:
        continue

    if nb_iter % args.print_iter == 0:
        if accelerator.is_main_process:
            n = args.print_iter
            avg_recons /= n; avg_kl /= n; avg_root /= n
            avg_g /= n;      avg_d  /= n; avg_fm   /= n

            for key, val in [
                ('Train/Recon_loss', avg_recons),
                ('Train/KL',         avg_kl),
                ('Train/Root_loss',  avg_root),
                ('Train/G_loss',     avg_g),
                ('Train/D_loss',     avg_d),
                ('Train/FM_loss',    avg_fm),
            ]:
                writer.add_scalar(key, val, nb_iter)

            gan_active = '✓ GAN ON' if nb_iter >= args.disc_start else f'○ GAN in {args.disc_start - nb_iter}'
            logger.info(
                f'[{gan_active}] Iter {nb_iter}: '
                f'Recons={avg_recons:.4f} KL={avg_kl:.4f} Root={avg_root:.4f} '
                f'G={avg_g:.4f} D={avg_d:.4f} FM={avg_fm:.4f}'
            )
            avg_recons = avg_kl = avg_root = avg_g = avg_d = avg_fm = 0.

    if nb_iter % args.eval_iter == 0:
        net_m = unwrap(net)
        best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_tae_multi(
            args.out_dir, val_loader, net_m, logger, writer, nb_iter,
            best_iter, best_mpjpe, device=comp_device, accelerator=accelerator,
        )

accelerator.wait_for_everyone()

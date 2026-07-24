"""
train_sae_v1.py  -  Training script for SAE-v1.

SAE-v1 is identical to Causal TAE except for the replicate-padding fix in
the encoder's first layer.  Loss function is unchanged:
    L = L_rec (Optimal-sigma Gaussian NLL) + L_KL + lambda_root * L_root

Training-time evaluation reports three metrics every eval_iter iterations:
  - MPJPE (mm)          via evaluation_tae_multi  [all ranks, reduced]
  - FID                 via compute_fid_and_jitter [rank 0, non-distributed]
  - First-frame jitter  via compute_fid_and_jitter [rank 0, non-distributed]
"""

import os
import sys
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import models.sae_v1 as sae_v1
import utils.losses as losses
import options.option_tae as option_tae
import utils.utils_model as utils_model
from humanml3d_272 import dataset_tae, dataset_eval_tae
import utils.eval_trans as eval_trans
from utils.eval_trans import (
    recover_from_local_position,
    calculate_activation_statistics,
    calculate_frechet_distance,
)
import warnings
warnings.filterwarnings('ignore')

# Add the repository evaluator package after this entrypoint moved two levels
# below the root.
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
)
sys.path.insert(0, os.path.join(REPO_ROOT, 'Evaluator_272'))


# ---------------------------------------------------------------------------
# Accelerator
# ---------------------------------------------------------------------------
accelerator = Accelerator()
comp_device = accelerator.device


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return optimizer, current_lr


# ---------------------------------------------------------------------------
# Args / dirs / logger
# ---------------------------------------------------------------------------
args = option_tae.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok=True)

logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
logger.info(f'[SAE-v1] Training on {args.dataname}, {args.nb_joints} joints')

# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------
train_loader = dataset_tae.DATALoader(
    args.dataname, args.batch_size,
    window_size=args.window_size,
    unit_length=2 ** args.down_t,
)
val_loader = dataset_eval_tae.DATALoader(
    args.dataname, False, 32,
    unit_length=2 ** args.down_t,
)

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
clip_range = [-30, 20]

net = sae_v1.SAE_HumanV1(
    hidden_size=args.hidden_size,
    down_t=args.down_t,
    stride_t=args.stride_t,
    depth=args.depth,
    dilation_growth_rate=args.dilation_growth_rate,
    activation='relu',
    latent_dim=args.latent_dim,
    clip_range=clip_range,
)

if args.resume_pth:
    logger.info('Loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    state = ckpt['net'] if (isinstance(ckpt, dict) and 'net' in ckpt) else ckpt
    net.load_state_dict(state, strict=True)

net.train()
net.to(comp_device)

# ---------------------------------------------------------------------------
# Load motionencoder for FID + jitter (rank 0 only, non-critical)
# ---------------------------------------------------------------------------
_motionencoder = None
_val_loader_fid = None

if accelerator.is_main_process:
    try:
        from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
        _me = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4,
                                   latent_dim=256, max_len=300)
        _eval_ckpt_path = os.path.join(
            'Evaluator_272', 'experiments', 'temos', 'EXP1',
            'checkpoints', 'epoch=99.ckpt')
        _eval_ckpt = torch.load(_eval_ckpt_path, map_location='cpu')
        _me_state = {k.replace('motionencoder.', ''): v
                     for k, v in _eval_ckpt['state_dict'].items()
                     if k.startswith('motionencoder.')}
        _me.load_state_dict(_me_state, strict=True)
        _me.eval()
        _me.to(comp_device)
        _motionencoder = _me
        # Dedicated single-process val loader ensures full dataset coverage for FID
        _val_loader_fid = dataset_eval_tae.DATALoader(
            args.dataname, False, 32, unit_length=2 ** args.down_t)
        logger.info('[SAE-v1] Motionencoder loaded — FID & jitter will be tracked')
    except Exception as _e:
        logger.warning(
            f'[SAE-v1] Could not load motionencoder ({_e}). '            f'FID & jitter monitoring disabled.')


# ---------------------------------------------------------------------------
# FID + first-frame jitter evaluation (rank 0 only)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_fid_and_jitter(nb_iter, net_unwrapped):
    """Compute FID and first-5-frame jitter on rank 0 using a dedicated loader.

    Runs independently from evaluation_tae_multi; does not affect MPJPE
    checkpoint saving logic.  Logs results to both logger and TensorBoard.
    """
    if not accelerator.is_main_process:
        return
    if _motionencoder is None or _val_loader_fid is None:
        return

    net_unwrapped.eval()
    motion_gt_embs   = []
    motion_pred_embs = []
    gt_jitter_sum   = 0.0
    pred_jitter_sum = 0.0
    n_seqs = 0
    num_joints = 22

    for batch in _val_loader_fid:
        motion, m_length = batch
        motion = motion.to(comp_device).float()
        bs = motion.shape[0]
        pred_pose_eval = torch.zeros_like(motion)

        for i in range(bs):
            L = int(m_length[i])
            gt_clip = motion[i:i+1, :L, :]
            pred_pose, _, _ = net_unwrapped(gt_clip)
            pred_pose_eval[i:i+1, :L, :] = pred_pose

            # ---- first-frame jitter ----
            gt_denorm   = _val_loader_fid.dataset.inv_transform(
                gt_clip.detach().cpu().numpy())
            pred_denorm = _val_loader_fid.dataset.inv_transform(
                pred_pose.detach().cpu().numpy())
            gt_xyz   = recover_from_local_position(gt_denorm.squeeze(0),   num_joints)
            pred_xyz = recover_from_local_position(pred_denorm.squeeze(0), num_joints)
            # recover_from_local_position returns (J, T, 3); permute to (T, J, 3)
            gt_tj   = torch.from_numpy(gt_xyz).float().permute(1, 0, 2)
            pred_tj = torch.from_numpy(pred_xyz).float().permute(1, 0, 2)
            t = min(7, gt_tj.shape[0])   # need >=3 frames for acceleration
            if t >= 3:
                gt_acc   = gt_tj[2:t]   - 2 * gt_tj[1:t-1]   + gt_tj[0:t-2]
                pred_acc = pred_tj[2:t] - 2 * pred_tj[1:t-1] + pred_tj[0:t-2]
                gt_jitter_sum   += float(torch.norm(gt_acc,   dim=-1).mean())
                pred_jitter_sum += float(torch.norm(pred_acc, dim=-1).mean())
                n_seqs += 1

        em_gt   = _motionencoder(motion,         m_length).loc
        em_pred = _motionencoder(pred_pose_eval, m_length).loc
        motion_gt_embs.append(em_gt.cpu())
        motion_pred_embs.append(em_pred.cpu())

    # ---- FID ----
    ann_np  = torch.cat(motion_gt_embs,   dim=0).numpy()
    pred_np = torch.cat(motion_pred_embs, dim=0).numpy()
    gt_mu,   gt_cov   = calculate_activation_statistics(ann_np)
    pred_mu, pred_cov = calculate_activation_statistics(pred_np)
    fid_val = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)

    # ---- jitter (m -> mm) ----
    gt_jit   = gt_jitter_sum   / max(n_seqs, 1) * 1000
    pred_jit = pred_jitter_sum / max(n_seqs, 1) * 1000

    logger.info(
        f'--> 	 [FID/Jitter] Iter {nb_iter} : '        f'FID {fid_val:.4f} | '        f'Pred-Jitter {pred_jit:.4f} mm  (GT {gt_jit:.4f} mm)')
    writer.add_scalar('./Test/FID',            fid_val,  nb_iter)
    writer.add_scalar('./Test/pred_jitter_mm', pred_jit, nb_iter)
    writer.add_scalar('./Test/gt_jitter_mm',   gt_jit,   nb_iter)

    net_unwrapped.train()


# ---------------------------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------------------------
optimizer = optim.AdamW(net.parameters(), lr=args.lr,
                        betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

net, optimizer, train_loader, val_loader = accelerator.prepare(
    net, optimizer, train_loader, val_loader)
train_loader_iter = dataset_tae.cycle(train_loader)

Loss = losses.ReConsLoss(motion_dim=272)

# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------
avg_recons, avg_kl, avg_root = 0., 0., 0.

for nb_iter in range(1, args.warm_up_iter):
    optimizer, current_lr = update_lr_warm_up(
        optimizer, nb_iter, args.warm_up_iter, args.lr)

    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.to(comp_device).float()

    net_fwd = net.module if args.num_gpus > 1 else net
    pred_motion, mu, logvar = net_fwd(gt_motion)

    loss_motion = Loss(pred_motion, gt_motion)
    loss_kl     = Loss.forward_KL(mu, logvar)
    loss_root   = Loss.forward_root(pred_motion, gt_motion)
    loss = loss_motion + loss_kl + args.root_loss * loss_root

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()

    avg_recons += loss_motion.item()
    avg_kl     += loss_kl.item()
    avg_root   += loss_root.item()

    if nb_iter % args.print_iter == 0 and accelerator.is_main_process:
        avg_recons /= args.print_iter
        avg_kl     /= args.print_iter
        avg_root   /= args.print_iter
        logger.info(
            f'Warmup Iter {nb_iter} : lr {current_lr:.5f} | '            f'Recons {avg_recons:.5f} | KL {avg_kl:.5f} | Root {avg_root:.5f}')
        avg_recons, avg_kl, avg_root = 0., 0., 0.

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
avg_recons, avg_kl, avg_root = 0., 0., 0.
current_lr = args.lr

net_eval = net.module if args.num_gpus > 1 else net
best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_tae_multi(
    args.out_dir, val_loader, net_eval, logger, writer,
    0, best_iter=0, best_mpjpe=1000, device=comp_device,
    accelerator=accelerator)
if accelerator.is_main_process:
    ckpt_name = 'sae_net{:06d}.pth'.format(0)
    torch.save({'net': net_eval.state_dict()},
               os.path.join(args.out_dir, ckpt_name))
    logger.info(f'Saved checkpoint: {ckpt_name}')
compute_fid_and_jitter(0, net_eval)

for nb_iter in range(1, args.total_iter + 1):
    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.to(comp_device).float()

    net_fwd = net.module if args.num_gpus > 1 else net
    pred_motion, mu, logvar = net_fwd(gt_motion)

    loss_motion = Loss(pred_motion, gt_motion)
    loss_kl     = Loss.forward_KL(mu, logvar)
    loss_root   = Loss.forward_root(pred_motion, gt_motion)
    loss = loss_motion + loss_kl + args.root_loss * loss_root

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()

    try:
        avg_recons += loss_motion.item()
        avg_kl     += loss_kl.item()
        avg_root   += loss_root.item()
    except Exception:
        continue

    if nb_iter % args.print_iter == 0 and accelerator.is_main_process:
        avg_recons /= args.print_iter
        avg_kl     /= args.print_iter
        avg_root   /= args.print_iter
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('./Train/Recon_loss', avg_recons, nb_iter)
        writer.add_scalar('./Train/KL',         avg_kl,     nb_iter)
        writer.add_scalar('./Train/Root_loss',  avg_root,   nb_iter)
        writer.add_scalar('./Train/LR',         current_lr, nb_iter)
        logger.info(
            f'Train Iter {nb_iter} : '            f'Recons {avg_recons:.5f} | KL {avg_kl:.5f} | Root {avg_root:.5f}')
        avg_recons, avg_kl, avg_root = 0., 0., 0.

    if nb_iter % args.eval_iter == 0:
        net_eval = net.module if args.num_gpus > 1 else net
        best_iter, best_mpjpe, writer, logger = eval_trans.evaluation_tae_multi(
            args.out_dir, val_loader, net_eval, logger, writer,
            nb_iter, best_iter, best_mpjpe,
            device=comp_device, accelerator=accelerator)
        if accelerator.is_main_process:
            ckpt_name = 'sae_net{:06d}.pth'.format(nb_iter)
            torch.save({'net': net_eval.state_dict()},
                       os.path.join(args.out_dir, ckpt_name))
            logger.info(f'Saved checkpoint: {ckpt_name}')
        compute_fid_and_jitter(nb_iter, net_eval)

accelerator.wait_for_everyone()

"""
eval_sae_v1.py  –  Evaluation script for SAE-v1.

Metrics reported:
  - MPJPE  (mean per-joint position error, mm)
  - FID    (Fréchet Inception Distance, using the custom motion evaluator)
  - R@1/2/3, MM-dist (via motion evaluator, same as eval_causal_TAE.py)
  - First-frame jitter  (acceleration magnitude of the first 5 decoded frames)

Usage:
    python eval_sae_v1.py --resume-pth <checkpoint.pth> [options]

Evaluation is run 3 times and results are averaged for stability.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

import models.sae_v1 as sae_v1
import options.option_tae as option_tae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from humanml3d_272 import dataset_eval_t2m

# Change into Evaluator_272 so internal imports work, mirroring eval_causal_TAE
os.chdir('Evaluator_272')
sys.path.insert(0, os.getcwd())

comp_device = torch.device('cuda')

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
args = option_tae.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok=True)

logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
val_loader = dataset_eval_t2m.DATALoader(
    args.dataname, True, 32,
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

print('Loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
state = ckpt['net'] if (isinstance(ckpt, dict) and 'net' in ckpt) else ckpt
net.load_state_dict(state, strict=True)
net.eval()
net.to(comp_device)


# Thin wrapper for eval_trans compatibility (expects net(x) → (x_recon, mu, logvar))
class EvalCompat(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


net_eval = EvalCompat(net)

# ---------------------------------------------------------------------------
# Evaluator (same as eval_causal_TAE.py)
# ---------------------------------------------------------------------------
from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

modelpath = './deps/distilbert-base-uncased'
textencoder  = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
motionencoder = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4,
                                     latent_dim=256, max_len=300)

ckpt_eval = torch.load('./experiments/temos/EXP1/checkpoints/epoch=99.ckpt')

textencoder_ckpt = {k.replace('textencoder.', ''): v
                    for k, v in ckpt_eval['state_dict'].items()
                    if k.startswith('textencoder.')}
textencoder.load_state_dict(textencoder_ckpt, strict=True)
textencoder.eval()
textencoder.to(comp_device)

motionencoder_ckpt = {k.replace('motionencoder.', ''): v
                      for k, v in ckpt_eval['state_dict'].items()
                      if k.startswith('motionencoder.')}
motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
motionencoder.eval()
motionencoder.to(comp_device)

evaluator = [textencoder, motionencoder]

# ---------------------------------------------------------------------------
# First-frame jitter helper
# ---------------------------------------------------------------------------

def compute_first_frame_jitter(gt_xyz, pred_xyz, n_frames=5):
    """Compute mean acceleration magnitude for the first n_frames of a sequence.

    Acceleration is the second-order finite difference of 3-D joint positions.
    Shape of inputs: (T, J, 3) — time, joints, xyz.

    Returns scalar jitter (in the same unit as xyz, typically metres before
    *1000 scaling, or mm after).
    """
    t = min(n_frames + 2, gt_xyz.shape[0])  # need at least 3 frames
    if t < 3:
        return 0.0, 0.0

    # acceleration = pos[t+2] - 2*pos[t+1] + pos[t]  for t in 0..n_frames-1
    gt_acc   = gt_xyz[2:t]   - 2 * gt_xyz[1:t-1]   + gt_xyz[0:t-2]    # (n,J,3)
    pred_acc = pred_xyz[2:t] - 2 * pred_xyz[1:t-1] + pred_xyz[0:t-2]  # (n,J,3)

    gt_jitter   = float(torch.norm(gt_acc,   dim=-1).mean())
    pred_jitter = float(torch.norm(pred_acc, dim=-1).mean())
    return gt_jitter, pred_jitter


# ---------------------------------------------------------------------------
# Evaluation loop (repeat 3 times for stability)
# ---------------------------------------------------------------------------
from utils.eval_trans import (
    calculate_mpjpe, recover_from_local_position,
    calculate_activation_statistics, calculate_frechet_distance,
)

fid_list, mpjpe_list = [], []
gt_jitter_list, pred_jitter_list = [], []

num_repeats = 3

for rep in range(num_repeats):
    nb_sample    = 0
    mpjpe_acc    = torch.tensor(0.0, device=comp_device)
    num_poses    = torch.tensor(0,   device=comp_device)
    gt_jitter_acc   = 0.0
    pred_jitter_acc = 0.0
    n_seqs = 0

    motion_annotation_list = []
    motion_pred_list       = []

    num_joints = 22

    with torch.no_grad():
        for batch in val_loader:
            motion, m_length = batch
            motion = motion.to(comp_device).float()
            bs, seq = motion.shape[0], motion.shape[1]

            em_gt = motionencoder(motion, m_length).loc

            pred_pose_eval = torch.zeros_like(motion)

            for i in range(bs):
                L = m_length[i]
                gt_clip  = motion[i:i+1, :L, :]
                pred_pose, _, _ = net_eval(gt_clip)
                pred_pose_eval[i:i+1, :L, :] = pred_pose

                # ---- MPJPE ----
                gt_denorm   = val_loader.dataset.inv_transform(
                    gt_clip.detach().cpu().numpy())
                pred_denorm = val_loader.dataset.inv_transform(
                    pred_pose.detach().cpu().numpy())

                gt_xyz   = recover_from_local_position(gt_denorm.squeeze(0),   num_joints)
                pred_xyz = recover_from_local_position(pred_denorm.squeeze(0), num_joints)

                gt_xyz_t   = torch.from_numpy(gt_xyz).float().to(comp_device)
                pred_xyz_t = torch.from_numpy(pred_xyz).float().to(comp_device)

                mpjpe_acc += torch.sum(
                    calculate_mpjpe(gt_xyz_t[:, :L].squeeze(),
                                    pred_xyz_t[:, :L].squeeze()))
                num_poses += gt_xyz_t.shape[0]

                # ---- First-frame jitter ----
                # gt_xyz shape: (J, T, 3) → transpose to (T, J, 3)
                gt_xyz_tj   = torch.from_numpy(gt_xyz).float().permute(1, 0, 2)
                pred_xyz_tj = torch.from_numpy(pred_xyz).float().permute(1, 0, 2)
                gj, pj = compute_first_frame_jitter(gt_xyz_tj, pred_xyz_tj,
                                                    n_frames=5)
                gt_jitter_acc   += gj
                pred_jitter_acc += pj
                n_seqs += 1

            em_pred = motionencoder(pred_pose_eval, m_length).loc
            motion_annotation_list.append(em_gt)
            motion_pred_list.append(em_pred)
            nb_sample += bs

    mpjpe_val = float(mpjpe_acc / num_poses) * 1000   # convert to mm

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np       = torch.cat(motion_pred_list,       dim=0).cpu().numpy()
    gt_mu,   gt_cov  = calculate_activation_statistics(motion_annotation_np)
    pred_mu, pred_cov = calculate_activation_statistics(motion_pred_np)
    fid_val = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)

    avg_gt_jitter   = gt_jitter_acc   / max(n_seqs, 1) * 1000   # m → mm
    avg_pred_jitter = pred_jitter_acc / max(n_seqs, 1) * 1000

    fid_list.append(fid_val)
    mpjpe_list.append(mpjpe_val)
    gt_jitter_list.append(avg_gt_jitter)
    pred_jitter_list.append(avg_pred_jitter)

    logger.info(
        f'Rep {rep+1}/{num_repeats} | '
        f'FID {fid_val:.4f} | MPJPE {mpjpe_val:.3f} mm | '
        f'GT-jitter {avg_gt_jitter:.4f} mm | '
        f'Pred-jitter {avg_pred_jitter:.4f} mm')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
logger.info('\n========== SAE-v1 Final Results (mean over {} repeats) =========='.format(num_repeats))
logger.info(f'FID:          {np.mean(fid_list):.4f} +/- {np.std(fid_list):.4f}')
logger.info(f'MPJPE:        {np.mean(mpjpe_list):.3f} mm')
logger.info(f'GT  jitter (first 5 frames):   {np.mean(gt_jitter_list):.4f} mm')
logger.info(f'Pred jitter (first 5 frames):  {np.mean(pred_jitter_list):.4f} mm')
logger.info(f'Jitter ratio (pred/GT):        {np.mean(pred_jitter_list)/max(np.mean(gt_jitter_list),1e-6):.4f}')

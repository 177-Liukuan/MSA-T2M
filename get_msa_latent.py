"""
Extract motion latents from trained MSA-VAE model.

This script processes raw 272-dim motion sequences through the CNN encoder
of MSA-VAE and persists the latent representations for efficient T2M training.

Output: ./humanml3d_272/t2m_latents_msa_vae/<experiment_name>/
Each sample saved as <name>.npy with reference_end_latent appended at the end.
"""

import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import json
import models.msa_vae as msa_vae
import options.option_msa_vae as option_msa_vae
import utils.utils_model as utils_model
from utils.msa_vae_training import prepare_extraction_roots
from humanml3d_272 import dataset_tae_tokenizer
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

##### ---- Setup ---- #####
args = option_msa_vae.get_args_parser()
torch.manual_seed(args.seed)

# If latent_dir not provided via CLI, use a default based on exp name
if not hasattr(args, 'latent_dir') or args.latent_dir is None:
    args.latent_dir = pjoin('./humanml3d_272/t2m_latents_msa_vae',
                            args.exp_name if hasattr(args, 'exp_name') else 'default')

# Initialize h_cls_dir for global semantic features
if not hasattr(args, 'h_cls_dir') or args.h_cls_dir is None:
    args.h_cls_dir = pjoin('./humanml3d_272/h_cls_latents_msa_vae',
                           args.exp_name if hasattr(args, 'exp_name') else 'default')

# Initialize mu_latent_dir for deterministic latent means (used as local RAG tokens)
if not hasattr(args, 'mu_latent_dir') or args.mu_latent_dir is None:
    args.mu_latent_dir = pjoin('./humanml3d_272/mu_latents_msa_vae',
                               args.exp_name if hasattr(args, 'exp_name') else 'default')

args.out_dir = os.path.join(args.out_dir, 'get_msa_latent_log') if hasattr(args, 'out_dir') else './Experiments/get_msa_latent_log'
os.makedirs(args.out_dir, exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
logger.info(f"MSA-VAE Motion Latent Extraction")
logger.info(f"Resume checkpoint: {args.resume_pth if hasattr(args, 'resume_pth') else 'None'}")
logger.info(f"Output latent dir: {args.latent_dir}")
logger.info(f"Output h_cls dir: {args.h_cls_dir}")
logger.info(f"Output mu latent dir: {args.mu_latent_dir}")
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
train_loader = dataset_tae_tokenizer.DATALoader(args.dataname)

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
    trans_d_model=args.trans_d_model if hasattr(args, 'trans_d_model') else 768,
    trans_nhead=args.trans_nhead if hasattr(args, 'trans_nhead') else 8,
    trans_enc_layers=args.trans_enc_layers if hasattr(args, 'trans_enc_layers') else 6,
    trans_dec_layers=args.trans_dec_layers if hasattr(args, 'trans_dec_layers') else 6,
    trans_ff_size=args.trans_ff_size if hasattr(args, 'trans_ff_size') else 2048,
    trans_dropout=args.trans_dropout if hasattr(args, 'trans_dropout') else 0.1,
    clip_dim=args.clip_dim if hasattr(args, 'clip_dim') else 768,
)

# Load model checkpoint
checkpoint_path = args.resume_pth if hasattr(args, 'resume_pth') else None
if not checkpoint_path:
    raise ValueError('--resume-pth is required for MSA latent extraction')

logger.info(f'Loading MSA-VAE checkpoint from {checkpoint_path}')
ckpt = torch.load(checkpoint_path, map_location='cpu')
# Handle both direct and wrapped model state dicts
if 'net' in ckpt:
    net.load_state_dict(ckpt['net'], strict=True)
else:
    net.load_state_dict(ckpt, strict=True)
checkpoint_metadata = (
    ckpt.get('metadata') if isinstance(ckpt, dict) else None
)
extraction_metadata = prepare_extraction_roots(
    [args.latent_dir, args.h_cls_dir, args.mu_latent_dir],
    checkpoint_path,
    checkpoint_metadata,
    args,
)
logger.info(
    f'Extraction metadata: {json.dumps(extraction_metadata, sort_keys=True)}'
)

net.eval()
net.cuda()

logger.info(f"Model loaded. Total parameters: {sum(p.numel() for p in net.parameters()):,}")

##### ---- Extract reference end latent ---- #####
# Create "impossible pose" prior: all zeros (physically unrealistic)
reference_end_pose = torch.zeros(1, 4, 272).cuda()
with torch.no_grad():
    z_local, mu, logvar, h_cls = net.encode(reference_end_pose)
    # Use z_local (physical track) for consistency with CNN decoder
    # Shape: (1, T', latent_dim) where T' = 4 / (stride^down_t) = 4 / 4 = 1 timestep
    # Squeeze batch dim: (1, 1, latent_dim) → (1, latent_dim)
    reference_end_latent = z_local.squeeze(0)  # (1, latent_dim)

reference_end_latent_np = reference_end_latent.cpu().detach().numpy()
logger.info(f"Reference end latent shape: {reference_end_latent_np.shape}")

os.makedirs(args.latent_dir, exist_ok=True)
os.makedirs(args.h_cls_dir, exist_ok=True)
os.makedirs(args.mu_latent_dir, exist_ok=True)

# Save reference latent for inference
ref_latent_path = pjoin(args.latent_dir, f'reference_end_latent_msa_vae_{args.dataname}.npy')
np.save(ref_latent_path, reference_end_latent_np)
logger.info(f"✓ Reference end latent saved to: {ref_latent_path}")


##### ---- Extract motion latents ---- #####
logger.info(f"Starting latent extraction for {args.dataname}...")
processed_count = 0
skipped_count = 0

with torch.no_grad():
    for batch in tqdm(train_loader, desc="Extracting latents"):
        try:
            pose, name = batch
            bs, seq = pose.shape[0], pose.shape[1]
            pose = pose.cuda().float()

            # Encode through MSA-VAE CNN encoder
            # z_local shape: (bs, T', latent_dim) where T' = seq / (stride^down_t)
            # mu shape: (bs, T', latent_dim) - deterministic mean, used as local RAG tokens
            # h_cls shape: (bs, ...) - global semantic features
            z_local, mu, logvar, h_cls = net.encode(pose)

            # Process each sample in the batch individually
            for i in range(bs):
                # Extract local latent for sample i
                # z_local[i] shape: (T', latent_dim)
                z_local_i = z_local[i:i+1]  # (1, T', latent_dim) - keep time dim for cat

                # Squeeze batch dimension: (1, T', latent_dim) → (T', latent_dim)
                z_local_i = z_local_i.squeeze(0)  # (T', latent_dim)

                # Append reference end latent along time dimension
                # reference_end_latent: (1, latent_dim)
                # z_local_i: (T', latent_dim)
                latent = torch.cat([z_local_i, reference_end_latent], dim=0)  # (T'+1, latent_dim)
                latent_np = latent.cpu().detach().numpy()

                # Save local latent (z_local + reference)
                sample_name = name[i] if isinstance(name, (list, tuple)) else name
                latent_save_path = pjoin(args.latent_dir, sample_name + '.npy')
                np.save(latent_save_path, latent_np)

                # Extract and save global semantic feature (h_cls)
                # h_cls[i] shape depends on model architecture; typically (h_cls_dim,)
                h_cls_i = h_cls[i].cpu().detach().numpy()
                h_cls_save_path = pjoin(args.h_cls_dir, sample_name + '.npy')
                np.save(h_cls_save_path, h_cls_i)

                # Extract and save deterministic mu latent (for local RAG tokens)
                # mu[i]: (T', latent_dim) — deterministic, no noise, no end token needed
                mu_i = mu[i].cpu().detach().numpy()
                mu_save_path = pjoin(args.mu_latent_dir, sample_name + '.npy')
                np.save(mu_save_path, mu_i)

                processed_count += 1

        except Exception as e:
            logger.warning(f"Error processing batch: {str(e)}")
            skipped_count += bs
            continue

##### ---- Summary ---- #####
logger.info("")
logger.info("=" * 70)
logger.info("✓ Motion Latent Extraction Complete")
logger.info("=" * 70)
logger.info(f"Processed: {processed_count} samples")
logger.info(f"Skipped: {skipped_count} samples")
logger.info(f"Local latent (z_local) saved to: {args.latent_dir}")
logger.info(f"Global semantic (h_cls) saved to: {args.h_cls_dir}")
logger.info(f"Mu latent (deterministic) saved to: {args.mu_latent_dir}")
logger.info(f"Reference latent: {ref_latent_path}")
logger.info("")

"""
Complete Workflow Guide for MSA-VAE T2M + CLIP Text Embeddings

This document provides step-by-step instructions for:
  1. Extracting MSA-VAE motion latents
  2. Extracting CLIP text embeddings
  3. Training T2M models with MSA-VAE (improved) vs Baseline (TAE)
  4. Running ablation studies

# ============================================================================
# WORKFLOW OVERVIEW
# ============================================================================
#
# Stage 1: Feature Extraction (Offline)
#   ├─ get_msa_latent.py      → Motion latents from MSA-VAE
#   ├─ get_latent.py          → Motion latents from TAE (baseline)
#   └─ get_text_latent_clip.py → CLIP text embeddings (shared)
#
# Stage 2: Training T2M Models (Online)
#   ├─ train_t2m_msa.py        + TRAIN_t2m_msa.sh       → MSA-VAE + CLIP
#   └─ train_t2m_baseline_clip.py + TRAIN_t2m_baseline_clip.sh → TAE + CLIP
#
# ============================================================================


# ============================================================================
# STEP 1: RUN SMOKE TESTS
# ============================================================================

cd /share/home/tm878032203900000/a878044490/MotionStreamer

# Validate all new scripts
python smoke_test.py

# Expected output:
#   ✓ SMOKE TESTS PASSED


# ============================================================================
# STEP 2: EXTRACT MOTION LATENTS
# ============================================================================

# ---- 2.1: Extract MSA-VAE Motion Latents ----
# Assume your trained MSA-VAE checkpoint is at:
#   Experiments/MSA_VAEv5_phase2_t2m_272_iter2000/net_best.pth

python get_msa_latent.py \
  --resume-pth Experiments/MSA_VAEv5_phase2_t2m_272_iter2000/net_best.pth \
  --latent_dir ./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv5_phase2_t2m_272_iter2000 \
  --dataname t2m_272 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --latent_dim 16 \
  --hidden_size 1024 \
  --trans_d_model 512 \
  --trans_nhead 8 \
  --trans_enc_layers 4 \
  --trans_dec_layers 4 \
  --trans_ff_size 1024 \
  --trans_dropout 0.1 \
  --clip_dim 512

# Expected files:
#   humanml3d_272/t2m_latents_msa_vae/MSA_VAEv5_phase2_t2m_272_iter2000/
#   ├─ reference_end_latent_msa_vae_t2m_272.npy
#   ├─ 000000.npy
#   ├─ 000002.npy
#   ├─ ...
#   └─ [~7056 sample files]


# ---- 2.2: Verify TAE Motion Latents Exist (for Baseline) ----
# These should already exist from original MotionStreamer training

ls -lh humanml3d_272/t2m_latents/causal_TAE_t2m_272_h100_20260203/

# Expected output:
#   reference_end_latent_t2m_272.npy
#   000000.npy
#   000002.npy
#   ... [~7056 files]


# ============================================================================
# STEP 3: EXTRACT CLIP TEXT EMBEDDINGS
# ============================================================================

# This generates CLIP embeddings for ALL samples
# Used by both MSA-VAE and Baseline models (shared)

python get_text_latent_clip.py

# Expected files:
#   humanml3d_272/text_latents_clip/
#   ├─ empty_cfg_text_clip.npy  (512,) - for CFG masking
#   ├─ 000000.npy
#   ├─ 000002.npy
#   ├─ ...
#   └─ [~7056 sample files]

# Time estimate: ~20-30 minutes on single GPU
# Output size: ~23 GB for 23,384 captions × 512d


# ============================================================================
# STEP 4: VERIFY FEATURE EXTRACTION
# ============================================================================

# Verify MSA-VAE latents
python -c "
import numpy as np
import os
from os.path import join as pjoin

latent_dir = './humanml3d_272/t2m_latents_msa_vae/MSA_VAEv5_phase2_t2m_272_iter2000'
npy_files = [f for f in os.listdir(latent_dir) if f.endswith('.npy')]
print(f'Total MSA-VAE latent files: {len(npy_files)}')

# Check one sample
sample = np.load(pjoin(latent_dir, '000000.npy'))
print(f'Sample motion latent shape: {sample.shape}')
print(f'  - Frames (including reference): {sample.shape[0]}')
print(f'  - Latent dimension: {sample.shape[1]}')
"

# Verify CLIP embeddings
python -c "
import numpy as np
import os
from os.path import join as pjoin

text_dir = './humanml3d_272/text_latents_clip'
npy_files = [f for f in os.listdir(text_dir) if f.endswith('.npy')]
print(f'Total CLIP text embedding files: {len(npy_files)}')

# Check empty CFG embedding
empty_cfg = np.load(pjoin(text_dir, 'empty_cfg_text_clip.npy'))
print(f'Empty CFG embedding shape: {empty_cfg.shape}')

# Check one sample
sample = np.load(pjoin(text_dir, '000000.npy'))
print(f'Sample text embedding shape: {sample.shape}')
print(f'  - Number of captions: {sample.shape[0]}')
print(f'  - CLIP dimension: {sample.shape[1]}')
"


# ============================================================================
# STEP 5: TRAIN MSA-VAE T2M MODEL
# ============================================================================

# Launch training on 4 GPUs (or adjust NUM_GPUS as needed)

bash TRAIN_t2m_msa.sh 4 \
  MSA_VAEv5_phase2_t2m_272_iter2000 \
  T2M_MSA_CLIP_exp1

# This will:
#   1. Verify motion latents exist
#   2. Verify CLIP embeddings exist
#   3. Launch training with accelerate
#   4. Use bf16 mixed precision for memory efficiency
#   5. Save checkpoints every 10k iterations

# Expected output:
#   Experiments/T2M_MSA_CLIP_exp1/
#   ├─ ckpt_iter_10000.pth
#   ├─ ckpt_iter_20000.pth
#   ├─ ...
#   ├─ net_best.pth
#   ├─ events.out.tfevents.*
#   └─ log.txt


# ============================================================================
# STEP 6: TRAIN BASELINE (TAE+CLIP) MODEL
# ============================================================================

# For ablation study: compare MSA-VAE+CLIP vs TAE+CLIP

bash TRAIN_t2m_baseline_clip.sh 4 \
  humanml3d_272/t2m_latents/causal_TAE_t2m_272_h100_20260203 \
  Baseline_TAE_CLIP_exp1

# This will:
#   1. Use original TAE motion latents
#   2. Use same CLIP text embeddings (shared)
#   3. Same training configuration as MSA-VAE model

# Expected output:
#   Experiments/Baseline_TAE_CLIP_exp1/
#   ├─ ckpt_iter_10000.pth
#   ├─ ckpt_iter_20000.pth
#   ├─ ...
#   ├─ net_best.pth
#   ├─ events.out.tfevents.*
#   └─ log.txt


# ============================================================================
# STEP 7: COMPARE RESULTS
# ============================================================================

# Compare training losses
python -c "
import os
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

exp_dirs = {
    'MSA-VAE + CLIP': 'Experiments/T2M_MSA_CLIP_exp1',
    'Baseline (TAE + CLIP)': 'Experiments/Baseline_TAE_CLIP_exp1',
}

for name, exp_dir in exp_dirs.items():
    ea = EventAccumulator(exp_dir)
    ea.Reload()
    
    if 'Loss/train' in ea.Tags()['scalars']:
        losses = ea.Scalars('Loss/train')
        print(f'{name}:')
        print(f'  Final loss: {losses[-1].value:.5f}')
        print(f'  Loss at 10k iter: {losses[10000].value:.5f}' if len(losses) > 10000 else '')
        print()
"

# Compare with original T5-based MotionStreamer (if available)
# You can also run the original train_t2m.py with CLIP embeddings for another comparison


# ============================================================================
# STEP 8: INFERENCE / GENERATION
# ============================================================================

# After training, you can use the trained models for generation
# Create a visualization script similar to visualize_t2m_generation.py:

python -c "
import torch
import numpy as np
import clip
from models.llama_model import LLaMAHF, LLaMAHFConfig

# Load MSA-VAE T2M checkpoint
ckpt = torch.load('Experiments/T2M_MSA_CLIP_exp1/net_best.pth')

# Load models and weights
config = LLaMAHFConfig.from_name('Normal_size')
config.block_size = 78
trans_encoder = LLaMAHF(config, num_diffusion_head_layers=8, latent_dim=16, device='cuda')
trans_encoder.load_state_dict(ckpt['trans'])

# Load projector
from torch.nn import Linear
text_projector = Linear(512, 768)
text_projector.load_state_dict(ckpt['text_projector'])

print('✓ Models loaded successfully')
print('  Ready for generation / inference')
"


# ============================================================================
# KEY CONFIGURATION PARAMETERS
# ============================================================================

# Motion Latent Dimensions:
#   - MSA-VAE       : (T', 16) where T' = T/4 + 1 (one reference frame appended)
#   - TAE (Baseline): (T', 16) where T' = T/4 + 1 (one reference frame appended)
#
# Text Embedding Dimensions:
#   - CLIP ViT-B/32: 512d → projected to 768d (model input)
#   - Original T5  : already 768d
#
# Transformer Model:
#   - Config: Normal_size (8 layers of LLaMA blocks)
#   - Block size: 78 (max sequence length)
#   - Diffusion heads: 8
#   - Latent dim: 16
#
# Training:
#   - Mixed precision: bf16 (important for 4090 memory efficiency)
#   - Scheduler: Warmup (10% of total iters) + Cosine annealing
#   - CFG masking: 10% of batch (classifier-free guidance preparation)
#   - Two-forward strategy: Gradually mix predictions with ground truth


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Error: "ModuleNotFoundError: No module named 'clip'"
#   Solution: pip install openai-clip

# Error: "CUDA out of memory"
#   Solution: Reduce batch size or use smaller num_gpus in bash script

# Error: "FileNotFoundError: Motion latent directory not found"
#   Solution: Ensure get_msa_latent.py completed successfully

# Error: "FileNotFoundError: Text latent directory not found"
#   Solution: Ensure get_text_latent_clip.py completed successfully

# Missing checkpoint files
#   Solution: Use the provided TRAIN_*.sh scripts which verify paths


# ============================================================================
# NEXT STEPS FOR INFERENCE & EVALUATION
# ============================================================================

# 1. Implement text-to-motion generation function
# 2. Generate motions from trained model
# 3. Evaluate with TMR/TEMOS metric suite
# 4. Compare FID, MPJPE, R@1/2/3 scores
# 5. Perform human evaluation if possible
"""

print(__doc__)

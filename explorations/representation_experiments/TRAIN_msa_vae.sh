#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
# -----------------------------------------------------------
#  MSA-VAE training script (Phase 2: full multi-scale alignment)
#
#  Usage:
#    bash TRAIN_msa_vae.sh [NUM_GPUS] [DATASET_NAME]
#    e.g.  bash TRAIN_msa_vae.sh 1 t2m_272
#
#  Optional env vars:
#    CNN_CKPT   - path to pretrained Causal CNN VAE checkpoint
#    RESUME     - path to MSA-VAE checkpoint to resume from
# -----------------------------------------------------------

NUM_GPUS=${1:-1}
dataset_name=${2:-t2m_272}

BATCH_SIZE=$((128 / NUM_GPUS))

echo "========== MSA-VAE Training (Phase 2: Full Alignment) =========="
echo "GPUs            : $NUM_GPUS"
echo "Dataset         : $dataset_name"
echo "Batch per GPU   : $BATCH_SIZE"
echo "================================================================"

# Build optional resume flags
RESUME_FLAGS=""
if [ -n "$CNN_CKPT" ]; then
    echo "CNN checkpoint  : $CNN_CKPT"
    RESUME_FLAGS="$RESUME_FLAGS --resume-cnn-pth $CNN_CKPT"
fi
if [ -n "$RESUME" ]; then
    echo "Resume from     : $RESUME"
    RESUME_FLAGS="$RESUME_FLAGS --resume-pth $RESUME"
fi

accelerate launch --num_processes $NUM_GPUS \
  train_msa_vae.py \
  --batch-size $BATCH_SIZE \
  --lr 0.00005 \
  --total-iter 2000000 \
  --warm-up-iter 1000 \
  --lr-scheduler 1900000 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --out-dir Experiments \
  --dataname $dataset_name \
  --exp-name MSA_VAEv5_${dataset_name}_dynamic02_closefp16 \
  --root_loss 7.0 \
  --latent_dim 16 \
  --hidden_size 1024 \
  --trans_d_model 512 \
  --trans_nhead 8 \
  --trans_enc_layers 6 \
  --trans_dec_layers 6 \
  --trans_ff_size 2048 \
  --trans_dropout 0.1 \
  --clip_dim 512 \
  --clip_version ViT-B/32 \
  --latent_recon_weight 1.0 \
  --global_align_weight 0.5 \
  --local_align_weight 0.2 \
  --spotlight_alpha 0 \
  --use_ft_split \
  --num_gpus $NUM_GPUS \
  $RESUME_FLAGS

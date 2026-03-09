#!/bin/bash
# -----------------------------------------------------------
#  MSA-VAE Phase 2: Unfreeze CNN, differential LR, all 6 losses
#
#  CNN params use lr * cnn_lr_scale (default 0.1)
#  Top params (Transformer + proj) use lr
#
#  Usage:
#    PHASE1_DIR=Experiments/MSA_VAE_phase1_t2m_272 bash TRAIN_msa_vae_phase2.sh [NUM_GPUS] [DATASET]
# -----------------------------------------------------------

NUM_GPUS=${1:-1}
dataset_name=${2:-t2m_272}
BATCH_SIZE=$((128 / NUM_GPUS))

# Phase 1 output directory (must contain net_last.pth)
PHASE1_DIR=${PHASE1_DIR:?"ERROR: set PHASE1_DIR to Phase 1 output directory"}
RESUME_PTH="${PHASE1_DIR}/net_best_fid.pth"

echo "========== MSA-VAE Phase 2: Full Fine-tune =========="
echo "GPUs            : $NUM_GPUS"
echo "Dataset         : $dataset_name"
echo "Batch per GPU   : $BATCH_SIZE"
echo "Phase1 ckpt     : $RESUME_PTH"
echo "CNN LR scale    : 0.1"
echo "====================================================="

accelerate launch --num_processes $NUM_GPUS --mixed_precision bf16 \
  train_msa_vae.py \
  --phase 2 \
  --cnn_lr_scale 0.1 \
  --batch-size $BATCH_SIZE \
  --lr 5e-5 \
  --total-iter 500000 \
  --warm-up-iter 1000 \
  --lr-scheduler 400000 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --out-dir Experiments \
  --dataname $dataset_name \
  --exp-name MSA_VAEv2_phase2_${dataset_name}_phase1_step300000_weightdown \
  --root_loss 7.0 \
  --latent_dim 16 \
  --hidden_size 1024 \
  --trans_d_model 512 \
  --trans_nhead 8 \
  --trans_enc_layers 4 \
  --trans_dec_layers 4 \
  --trans_ff_size 1024 \
  --trans_dropout 0.1 \
  --clip_dim 512 \
  --clip_version ViT-B/32 \
  --latent_recon_weight 1.0 \
  --global_align_weight 0.1 \
  --local_align_weight 0.01 \
  --use_ft_split \
  --num_gpus $NUM_GPUS \
  --resume-pth $RESUME_PTH \
  --eval-iter 1000 \
  --print-iter 200

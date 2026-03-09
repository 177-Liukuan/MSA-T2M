#!/bin/bash
# -----------------------------------------------------------
#  MSA-VAE Phase 1: Freeze CNN, train Transformer + projections
#
#  Losses: L_latent + L_global_align + L_local_align
#  CNN encoder/decoder/decode_proj are FROZEN (loaded from pretrained TAE)
#
#  Usage:
#    CNN_CKPT=Causal_TAE/net_last.pth bash TRAIN_msa_vae_phase1.sh [NUM_GPUS] [DATASET]
# -----------------------------------------------------------

NUM_GPUS=${1:-1}
dataset_name=${2:-t2m_272}
BATCH_SIZE=$((128 / NUM_GPUS))

# CNN checkpoint is required for Phase 1
CNN_CKPT=${CNN_CKPT:?"ERROR: set CNN_CKPT to pretrained TAE checkpoint path"}

echo "========== MSA-VAE Phase 1: Freeze CNN =========="
echo "GPUs            : $NUM_GPUS"
echo "Dataset         : $dataset_name"
echo "Batch per GPU   : $BATCH_SIZE"
echo "CNN checkpoint  : $CNN_CKPT"
echo "=================================================="

accelerate launch --num_processes $NUM_GPUS --mixed_precision bf16 \
  train_msa_vae.py \
  --phase 1 \
  --batch-size $BATCH_SIZE \
  --lr 1e-4 \
  --total-iter 300000 \
  --warm-up-iter 1000 \
  --lr-scheduler 250000 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --out-dir Experiments \
  --dataname $dataset_name \
  --exp-name MSA_VAEv2_phase1_${dataset_name} \
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
  --global_align_weight 0.5 \
  --local_align_weight 0.2 \
  --use_ft_split \
  --num_gpus $NUM_GPUS \
  --resume-cnn-pth $CNN_CKPT \
  --eval-iter 10000 \
  --print-iter 200

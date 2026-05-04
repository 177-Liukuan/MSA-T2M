#!/bin/bash
# -----------------------------------------------------------
#  MSA-VAE Phase 2: Unfreeze CNN, differential LR, all 6 losses
#
#  CNN params use lr * cnn_lr_scale (default 0.1)
#  Top params (Transformer + proj) use lr
#
#  Usage:
#    PHASE1_DIR=Experiments/MSA_VAE_phase1_t2m_272 bash TRAIN_msa_vae_phase2.sh [NUM_GPUS] [DATASET]
#
#  Optional switch:
#    TEXT_ENCODER_TYPE=t5|clip
# -----------------------------------------------------------

NUM_GPUS=${1:-1}
dataset_name=${2:-t2m_272}
BATCH_SIZE=$((128 / NUM_GPUS))

TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-t5}
T5_EMBED_DIR=${T5_EMBED_DIR:-./humanml3d_272/t5_enc_single}
CLIP_EMBED_DIR=${CLIP_EMBED_DIR:-./humanml3d_272/clip_enc_single}
T5_MODEL_PATH=${T5_MODEL_PATH:-sentencet5-xxl/}
PHASE1_DIR=${PHASE1_DIR:-Experiments/MSA_VAEv6_phase1_t2m_272_t5_alpha0_662048_fulldb}

if [ "$TEXT_ENCODER_TYPE" = "t5" ]; then
  TEXT_EMBED_DIM=768
else
  TEXT_EMBED_DIM=512
fi

# Phase 1 output directory (must contain net_last.pth)
PHASE1_DIR=${PHASE1_DIR:?"ERROR: set PHASE1_DIR to Phase 1 output directory"}
RESUME_PTH="${PHASE1_DIR}/net_best_fid.pth"

echo "========== MSA-VAE Phase 2: Full Fine-tune =========="
echo "GPUs            : $NUM_GPUS"
echo "Dataset         : $dataset_name"
echo "Batch per GPU   : $BATCH_SIZE"
echo "Text encoder    : $TEXT_ENCODER_TYPE"
echo "Text dim        : $TEXT_EMBED_DIM"
echo "Phase1 ckpt     : $RESUME_PTH"
echo "CNN LR scale    : 0.1"
echo "====================================================="

# Use full HumanML3D train split; L_local is auto-masked by has_local
accelerate launch --num_processes $NUM_GPUS --mixed_precision bf16 \
  train_msa_vae.py \
  --phase 2 \
  --cnn_lr_scale 0.1 \
  --batch-size $BATCH_SIZE \
  --lr 5e-5 \
  --total-iter 50000 \
  --warm-up-iter 1000 \
  --lr-scheduler 50000 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --out-dir Experiments \
  --dataname $dataset_name \
  --exp-name MSA_VAEv6_phase2_${dataset_name}_phase1_alpha0_${TEXT_ENCODER_TYPE}_trans662048_fulldb_right \
  --root_loss 7.0 \
  --latent_dim 16 \
  --hidden_size 1024 \
  --trans_d_model $TEXT_EMBED_DIM \
  --trans_nhead 8 \
  --trans_enc_layers 6 \
  --trans_dec_layers 6 \
  --trans_ff_size 2048 \
  --trans_dropout 0.1 \
  --clip_dim $TEXT_EMBED_DIM \
  --text_encoder_type $TEXT_ENCODER_TYPE \
  --text_embed_dim $TEXT_EMBED_DIM \
  --clip_embed_dir $CLIP_EMBED_DIR \
  --t5_embed_dir $T5_EMBED_DIR \
  --t5_model_path $T5_MODEL_PATH \
  --clip_version ViT-B/32 \
  --latent_recon_weight 1.0 \
  --global_align_weight 0.1 \
  --local_align_weight 0.001 \
  --no_ft_split \
  --num_gpus $NUM_GPUS \
  --resume-pth $RESUME_PTH \
  --eval-iter 500 \
  --print-iter 200 \
  --spotlight_alpha 0.0

  # 0.1 0.01

#!/bin/bash
# -----------------------------------------------------------
#  SAE-v1 training script
#  (Causal TAE + replicate-padding on encoder first layer)
#
#  Usage:
#    bash TRAIN_sae_v1.sh [NUM_GPUS] [DATASET_NAME]
#    e.g.  bash TRAIN_sae_v1.sh 1 t2m_272
#          bash TRAIN_sae_v1.sh 4 t2m_272
#
#  Optional env vars:
#    RESUME   - path to a SAE-v1 checkpoint to resume from
# -----------------------------------------------------------

NUM_GPUS=${1:-1}
dataset_name=${2:-t2m_272}

BATCH_SIZE=$((128 / NUM_GPUS))

echo "========== SAE-v1 Training =========="
echo "GPUs            : $NUM_GPUS"
echo "Dataset         : $dataset_name"
echo "Batch per GPU   : $BATCH_SIZE"
echo "====================================="

RESUME_FLAGS=""
if [ -n "$RESUME" ]; then
    echo "Resume from     : $RESUME"
    RESUME_FLAGS="--resume-pth $RESUME"
fi

accelerate launch --num_processes $NUM_GPUS \
  train_sae_v1.py \
  --batch-size $BATCH_SIZE \
  --lr 0.00001 \
  --total-iter 2000000 \
  --warm-up-iter 1000 \
  --lr-scheduler 200000 800000 \
  --gamma 0.2 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --out-dir Experiments \
  --dataname $dataset_name \
  --exp-name SAE_v1_${dataset_name} \
  --root_loss 7.0 \
  --latent_dim 16 \
  --hidden_size 1024 \
  --num_gpus $NUM_GPUS \
  --resume-pth Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth \
  $RESUME_FLAGS

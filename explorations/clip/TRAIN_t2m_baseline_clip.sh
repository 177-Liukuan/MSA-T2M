#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
NUM_GPUS=${1:-1}  # default: 1 GPU

BATCH_SIZE=$((256 / NUM_GPUS))

echo "Using $NUM_GPUS GPUs, each with a batch size of $BATCH_SIZE"

accelerate launch --num_processes $NUM_GPUS \
--mixed_precision bf16 \
-m explorations.clip.train_t2m_baseline_clip \
--batch-size $BATCH_SIZE \
--lr 0.0001 \
--total-iter 100000 \
--out-dir Experiments/explorations/clip \
--exp-name MotionStreamer_t2m_272_baseline_clip \
--dataname t2m_272 \
--latent_dir humanml3d_272/t2m_latents/causal_TAE_t2m_272_h100_20260203 \
--num_gpus $NUM_GPUS

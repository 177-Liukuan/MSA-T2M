#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# Usage:
#   bash Train_t2m_rag_rf.sh [NUM_GPUS]

NUM_GPUS=${1:-1}
BATCH_SIZE=$((256 / NUM_GPUS))

DATASET=${DATASET:-t2m_272}
TEXT_LATENT_DIR=${TEXT_LATENT_DIR:-./humanml3d_272/text_latents_t5}
HCLS_DIR=${HCLS_DIR:-./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048}
MOTION_LATENT_DIR=${MOTION_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048}
EMPTY_TEXT_PATH=${EMPTY_TEXT_PATH:-./humanml3d_272/text_latents_t5/empty_text_embedding.npy}
TEXT_EMBED_DIM=${TEXT_EMBED_DIM:-768}

# RF-oriented defaults (can be overridden by environment variables)
GENERATIVE_HEAD_TYPE=${GENERATIVE_HEAD_TYPE:-rectified_flow}
LR=${LR:-0.0001}
TOTAL_ITER=${TOTAL_ITER:-100000}
CFG_DROPOUT_PROB=${CFG_DROPOUT_PROB:-0.1}
NUM_FLOW_STEPS=${NUM_FLOW_STEPS:-50}
FLOW_SOLVER=${FLOW_SOLVER:-euler}
RF_TIME_SAMPLING=${RF_TIME_SAMPLING:-uniform}
RF_LOSS_TYPE=${RF_LOSS_TYPE:-mse}
EMA_DECAY=${EMA_DECAY:-0.9999}
EMA_UPDATE_EVERY=${EMA_UPDATE_EVERY:-1}

EXP_NAME=${EXP_NAME:-MotionStreamer_t2m_272_msa_rag_t5_trans662048_rf_100000Iter_addEMA}

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate mgpt
fi

echo "=========================================="
echo "MotionStreamer Stage-2 RAG Training (RF)"
echo "=========================================="
echo "GPU count       : $NUM_GPUS"
echo "Batch per GPU   : $BATCH_SIZE"
echo "Total batch     : 256"
echo "Dataset         : $DATASET"
echo "Text latents    : $TEXT_LATENT_DIR"
echo "h_cls latents   : $HCLS_DIR"
echo "Motion latents  : $MOTION_LATENT_DIR"
echo "Text embed dim  : $TEXT_EMBED_DIM"
echo "Head type       : $GENERATIVE_HEAD_TYPE"
echo "LR              : $LR"
echo "Total iter      : $TOTAL_ITER"
echo "CFG dropout     : $CFG_DROPOUT_PROB"
echo "Flow steps      : $NUM_FLOW_STEPS"
echo "Flow solver     : $FLOW_SOLVER"
echo "RF time sample  : $RF_TIME_SAMPLING"
echo "RF loss type    : $RF_LOSS_TYPE"
echo "EMA decay       : $EMA_DECAY"
echo "EMA update every: $EMA_UPDATE_EVERY"
echo "Exp name        : $EXP_NAME"

if [ ! -d "$TEXT_LATENT_DIR" ]; then
  echo "ERROR: text latent dir not found: $TEXT_LATENT_DIR"
  exit 1
fi
if [ ! -d "$HCLS_DIR" ]; then
  echo "ERROR: h_cls dir not found: $HCLS_DIR"
  exit 1
fi
if [ ! -d "$MOTION_LATENT_DIR" ]; then
  echo "ERROR: motion latent dir not found: $MOTION_LATENT_DIR"
  exit 1
fi

accelerate launch --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    train_t2m_rag.py \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --total-iter $TOTAL_ITER \
    --out-dir Experiments \
    --exp-name $EXP_NAME \
    --dataname $DATASET \
    --latent_dir $MOTION_LATENT_DIR \
    --text_latent_dir $TEXT_LATENT_DIR \
    --hcls_dir $HCLS_DIR \
    --empty_text_path $EMPTY_TEXT_PATH \
    --text_embed_dim $TEXT_EMBED_DIM \
    --num_gpus $NUM_GPUS \
    --retrieval_topk 3 \
    --cfg_dropout_prob $CFG_DROPOUT_PROB \
    --generative_head_type $GENERATIVE_HEAD_TYPE \
    --num_flow_steps $NUM_FLOW_STEPS \
    --flow_solver $FLOW_SOLVER \
    --rf_time_sampling $RF_TIME_SAMPLING \
    --rf_loss_type $RF_LOSS_TYPE \
    --ema_decay $EMA_DECAY \
    --ema_update_every $EMA_UPDATE_EVERY

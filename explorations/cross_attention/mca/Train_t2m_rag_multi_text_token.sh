#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

# Train MotionStreamer Stage-2 with RAG + Multi-Text Cross-Attention (MCA)
#
# Usage:
#   bash Train_t2m_rag_multi_text_token.sh [NUM_GPUS]
#
# Key differences from Train_t2m_rag_rf.sh:
#   - Uses -m explorations.cross_attention.mca.train_t2m_rag_multi_text_token
#   - Adds TEXT_TOKEN_LATENT_DIR, CA_EVERY_N_LAYERS, CA_N_HEAD env vars

NUM_GPUS=${1:-1}
BATCH_SIZE=$((256 / NUM_GPUS))

DATASET=${DATASET:-t2m_272}
TEXT_LATENT_DIR=${TEXT_LATENT_DIR:-./humanml3d_272/text_latents_t5}
TEXT_TOKEN_LATENT_DIR=${TEXT_TOKEN_LATENT_DIR:-./humanml3d_272/text_token_latents_t5}
HCLS_DIR=${HCLS_DIR:-./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
MOTION_LATENT_DIR=${MOTION_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
EMPTY_TEXT_PATH=${EMPTY_TEXT_PATH:-./humanml3d_272/text_latents_t5/empty_text_embedding.npy}
TEXT_EMBED_DIM=${TEXT_EMBED_DIM:-768}

# Cross-attention hyper-parameters
CA_N_HEAD=${CA_N_HEAD:-0}   # 0 = auto (same as backbone)
CA_EVERY_N_LAYERS=${CA_EVERY_N_LAYERS:-1}   # Insert one cross-attention block every N layers (e.g. 4 → layers [3,7,11] for 12-layer backbone)
TEXT_TOKEN_DIM=${TEXT_TOKEN_DIM:-1024}  # dim of token-level T5 embeddings (1024 for sentence-t5-xxl encoder)

# RF-oriented defaults (can be overridden by environment variables)
GENERATIVE_HEAD_TYPE=${GENERATIVE_HEAD_TYPE:-ddpm}
LR=${LR:-0.0001}    # 1e-4: standard LR for full scratch training (backbone + MCA jointly)
TOTAL_ITER=${TOTAL_ITER:-100000}
CFG_DROPOUT_PROB=${CFG_DROPOUT_PROB:-0.1}
NUM_FLOW_STEPS=${NUM_FLOW_STEPS:-50}
FLOW_SOLVER=${FLOW_SOLVER:-euler}
RF_TIME_SAMPLING=${RF_TIME_SAMPLING:-uniform}
RF_LOSS_TYPE=${RF_LOSS_TYPE:-mse}
EMA_DECAY=${EMA_DECAY:-0.9999}
EMA_UPDATE_EVERY=${EMA_UPDATE_EVERY:-1}

FREEZE_BACKBONE=${FREEZE_BACKBONE:-false}  # false: train backbone from scratch; true: Flamingo-style freeze (requires pretrained backbone via --resume-trans)
NUM_WORKERS=${NUM_WORKERS:-4}              # dataloader worker processes; >0 hides npz I/O latency behind GPU compute
USE_GATED_CA=${USE_GATED_CA:-false}        # true = Branch B: Flamingo tanh-gate; false = Branch A: gate-free zero-init out_proj
EXP_NAME=${EXP_NAME:-MotionStreamer_t2m_272_msa_rag_t5_trans662048_mca_every${CA_EVERY_N_LAYERS}layer_ddpm_scratch_Flamingo_gateclose_ca12}

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate mgpt
fi

echo "=========================================="
echo "MotionStreamer Stage-2 RAG+MCA Training"
echo "=========================================="
echo "GPU count            : $NUM_GPUS"
echo "Batch per GPU        : $BATCH_SIZE"
echo "Total batch          : 256"
echo "Dataset              : $DATASET"
echo "Text latents         : $TEXT_LATENT_DIR"
echo "Text token latents   : $TEXT_TOKEN_LATENT_DIR"
echo "h_cls latents        : $HCLS_DIR"
echo "Motion latents       : $MOTION_LATENT_DIR"
echo "Text embed dim       : $TEXT_EMBED_DIM"
echo "CA heads             : $CA_N_HEAD (0=auto)"
echo "CA every N layers    : $CA_EVERY_N_LAYERS"
echo "Token emb dim         : $TEXT_TOKEN_DIM"
echo "Head type            : $GENERATIVE_HEAD_TYPE"
echo "LR                   : $LR"
echo "Total iter           : $TOTAL_ITER"
echo "CFG dropout          : $CFG_DROPOUT_PROB"
echo "Flow steps           : $NUM_FLOW_STEPS"
echo "Flow solver          : $FLOW_SOLVER"
echo "RF time sample       : $RF_TIME_SAMPLING"
echo "RF loss type         : $RF_LOSS_TYPE"
echo "EMA decay            : $EMA_DECAY"
echo "EMA update every     : $EMA_UPDATE_EVERY"
echo "Freeze backbone      : $FREEZE_BACKBONE"
echo "Num workers          : $NUM_WORKERS"
echo "Use gated CA         : $USE_GATED_CA  (false=Branch A zero-init, true=Branch B Flamingo gate)"
echo "Exp name             : $EXP_NAME"
echo "=========================================="

# Pre-flight checks
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
    -m explorations.cross_attention.mca.train_t2m_rag_multi_text_token \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --total-iter $TOTAL_ITER \
    --out-dir Experiments/explorations/cross_attention/mca \
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
    --ema_update_every $EMA_UPDATE_EVERY \
    --ca_every_n_layers $CA_EVERY_N_LAYERS \
    --ca_n_head $CA_N_HEAD \
    --text_token_latent_dir $TEXT_TOKEN_LATENT_DIR \
    --text_token_dim $TEXT_TOKEN_DIM \
    --num_workers $NUM_WORKERS \
    $([ "$FREEZE_BACKBONE" = "true" ] && echo "--freeze-backbone") \
    $([ "$USE_GATED_CA" = "true" ] && echo "--use_gated_ca")

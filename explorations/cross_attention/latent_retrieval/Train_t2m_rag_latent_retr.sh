#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
# Train MotionStreamer Stage-2: global h_cls RAG + local motion-latent retrieval CA
#
# Usage:
#   bash Train_t2m_rag_latent_retr.sh [NUM_GPUS]
#
# Offline library cache (recommended):
#   1. Pre-build once:
#      python build_latent_retr_library.py \
#          --motion_latent_dir $MOTION_LATENT_DIR \
#          --text_latent_dir   $TEXT_LATENT_DIR   \
#          --hcls_dir          $HCLS_DIR           \
#          --output_cache_dir  $LIBRARY_CACHE_DIR
#   2. Set LIBRARY_CACHE_DIR before calling this script.
#      Subsequent runs load the cache in ~2 s instead of rebuilding from ~42k files.

NUM_GPUS=${1:-1}
BATCH_SIZE=$((256 / NUM_GPUS))

DATASET=${DATASET:-t2m_272}
TEXT_LATENT_DIR=${TEXT_LATENT_DIR:-./humanml3d_272/text_latents_t5}
HCLS_DIR=${HCLS_DIR:-./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
MOTION_LATENT_DIR=${MOTION_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
EMPTY_TEXT_PATH=${EMPTY_TEXT_PATH:-./humanml3d_272/text_latents_t5/empty_text_embedding.npy}
TEXT_EMBED_DIM=${TEXT_EMBED_DIM:-768}

# Local-RAG library cache (set to empty string "" to disable and rebuild each run)
LIBRARY_CACHE_DIR=${LIBRARY_CACHE_DIR:-./humanml3d_272/latent_retr_library_cache/$(basename "${MOTION_LATENT_DIR%/}")}

# Pre-computed retrieval lookup (eliminates per-sample CPU matmul, ~1.5-2x speedup)
# Build once: python precompute_latent_retr_lookup.py --library_cache_dir $LIBRARY_CACHE_DIR --output_dir $PRECOMPUTED_RETR_DIR
PRECOMPUTED_RETR_DIR=${PRECOMPUTED_RETR_DIR:-./humanml3d_272/latent_retr_lookup/$(basename "${MOTION_LATENT_DIR%/}")_top${LATENT_RETR_TOPK:-3}}

# Cross-attention hyper-parameters
CA_N_HEAD=${CA_N_HEAD:-0}             # 0 = auto (same as backbone)
CA_EVERY_N_LAYERS=${CA_EVERY_N_LAYERS:-2}  # Insert one cross-attention block every N layers (e.g. 4 → layers [3,7,11] for 12-layer backbone)
# CA insertion mode (controls CA/SA order and which layers get CA):
#   before_sa    (A, default) — CA→SA, all layers  [original Flamingo]
#   after_sa     (B)          — SA→CA, all layers  [text-first]
#   late_after_sa(C)          — SA-only first half, SA→CA second half
#   When using mode C, set CA_EVERY_N_LAYERS=1 or 2 for more CA blocks in second half.
CA_INSERTION_MODE=${CA_INSERTION_MODE:-after_sa}
LATENT_RETR_TOPK=${LATENT_RETR_TOPK:-3}   # Retrieved motion latents per query
LATENT_DIM=${LATENT_DIM:-16}              # Motion latent dimension

# Training hyper-parameters
GENERATIVE_HEAD_TYPE=${GENERATIVE_HEAD_TYPE:-ddpm}
LR=${LR:-0.0001}
TOTAL_ITER=${TOTAL_ITER:-100000}
CFG_DROPOUT_PROB=${CFG_DROPOUT_PROB:-0.1}
RETR_CFG_DROP_PROB=${RETR_CFG_DROP_PROB:-0.1}   # Independent retrieval dropout (decoupled from text CFG)
NUM_FLOW_STEPS=${NUM_FLOW_STEPS:-50}
FLOW_SOLVER=${FLOW_SOLVER:-euler}
RF_TIME_SAMPLING=${RF_TIME_SAMPLING:-uniform}
RF_LOSS_TYPE=${RF_LOSS_TYPE:-mse}
EMA_DECAY=${EMA_DECAY:-0.9999}
EMA_UPDATE_EVERY=${EMA_UPDATE_EVERY:-1}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-false}
NUM_WORKERS=${NUM_WORKERS:-4}
USE_GATED_CA=${USE_GATED_CA:-false}
DISABLE_LATENT_RETR=${DISABLE_LATENT_RETR:-false}

EXP_NAME=${EXP_NAME:-MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_${CA_INSERTION_MODE}_every${CA_EVERY_N_LAYERS}layer_top${LATENT_RETR_TOPK}_ddpm_cfg_saca_dropout01}
# EXP_NAME=${EXP_NAME:tasete}
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate mgpt
fi

echo "=========================================="
echo "MotionStreamer Stage-2: RAG + Latent Retr CA"
echo "=========================================="
echo "GPU count            : $NUM_GPUS"
echo "Batch per GPU        : $BATCH_SIZE"
echo "Dataset              : $DATASET"
echo "Text latents         : $TEXT_LATENT_DIR"
echo "h_cls latents        : $HCLS_DIR"
echo "Motion latents       : $MOTION_LATENT_DIR"
echo "Library cache dir    : ${LIBRARY_CACHE_DIR:-(none, rebuild each run)}"
echo "Precomputed retr dir : ${PRECOMPUTED_RETR_DIR:-(none, online matmul fallback)}"
echo "Text embed dim       : $TEXT_EMBED_DIM"
echo "CA heads             : $CA_N_HEAD (0=auto)"
echo "CA every N layers    : $CA_EVERY_N_LAYERS"
echo "CA insertion mode    : $CA_INSERTION_MODE"
echo "Latent retr topk     : $LATENT_RETR_TOPK"
echo "Latent dim           : $LATENT_DIM"
echo "Head type            : $GENERATIVE_HEAD_TYPE"
echo "LR                   : $LR"
echo "Total iter           : $TOTAL_ITER"
echo "CFG dropout          : $CFG_DROPOUT_PROB"
echo "Retr CFG dropout     : $RETR_CFG_DROP_PROB"
echo "EMA decay            : $EMA_DECAY"
echo "Freeze backbone      : $FREEZE_BACKBONE"
echo "Num workers          : $NUM_WORKERS"
echo "Use gated CA         : $USE_GATED_CA"
echo "Disable latent retr  : $DISABLE_LATENT_RETR"
echo "Exp name             : $EXP_NAME"
echo "=========================================="

# Pre-flight checks
for DIR in "$TEXT_LATENT_DIR" "$HCLS_DIR" "$MOTION_LATENT_DIR"; do
  if [ ! -d "$DIR" ]; then
    echo "ERROR: directory not found: $DIR"
    exit 1
  fi
done

# Build extra flags
EXTRA_FLAGS=""
[ "$FREEZE_BACKBONE" = "true" ]    && EXTRA_FLAGS="$EXTRA_FLAGS --freeze_backbone"
[ "$USE_GATED_CA" = "true" ]       && EXTRA_FLAGS="$EXTRA_FLAGS --use_gated_ca"
[ "$DISABLE_LATENT_RETR" = "true" ] && EXTRA_FLAGS="$EXTRA_FLAGS --disable_latent_retr"
[ -n "$LIBRARY_CACHE_DIR" ]        && EXTRA_FLAGS="$EXTRA_FLAGS --library_cache_dir $LIBRARY_CACHE_DIR"
[ -n "$PRECOMPUTED_RETR_DIR" ] && [ -d "$PRECOMPUTED_RETR_DIR" ] && EXTRA_FLAGS="$EXTRA_FLAGS --precomputed_retr_dir $PRECOMPUTED_RETR_DIR"

accelerate launch --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    -m explorations.cross_attention.latent_retrieval.train_t2m_rag_latent_retr \
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
    --latent_retr_topk $LATENT_RETR_TOPK \
    --latent_dim $LATENT_DIM \
    --cfg_dropout_prob $CFG_DROPOUT_PROB \
    --retr_cfg_drop_prob $RETR_CFG_DROP_PROB \
    --generative_head_type $GENERATIVE_HEAD_TYPE \
    --num_flow_steps $NUM_FLOW_STEPS \
    --flow_solver $FLOW_SOLVER \
    --rf_time_sampling $RF_TIME_SAMPLING \
    --rf_loss_type $RF_LOSS_TYPE \
    --ema_decay $EMA_DECAY \
    --ema_update_every $EMA_UPDATE_EVERY \
    --ca_every_n_layers $CA_EVERY_N_LAYERS \
    --ca_insertion_mode $CA_INSERTION_MODE \
    --ca_n_head $CA_N_HEAD \
    --num_workers $NUM_WORKERS \
    $EXTRA_FLAGS

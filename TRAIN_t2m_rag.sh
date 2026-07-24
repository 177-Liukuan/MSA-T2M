#!/bin/bash
set -euo pipefail

# python get_msa_latent.py \
#   --resume-pth Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/net_best_mpjpe.pth \
#   --latent_dir ./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right \
#   --dataname t2m_272 \
#   --trans_enc_layers 6 \
#   --trans_dec_layers 6 \
#   --trans_nhead 8 \
#   --trans_ff_size 2048

NUM_GPUS=${1:-1}
BATCH_SIZE=$((256 / NUM_GPUS))

DATASET=${DATASET:-t2m_272}
TEXT_LATENT_DIR=${TEXT_LATENT_DIR:-./humanml3d_272/text_latents_t5}
HCLS_DIR=${HCLS_DIR:-./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
MOTION_LATENT_DIR=${MOTION_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
EMPTY_TEXT_PATH=${EMPTY_TEXT_PATH:-./humanml3d_272/text_latents_t5/empty_text_embedding.npy}
TEXT_EMBED_DIM=${TEXT_EMBED_DIM:-768}
RETRIEVAL_TOPK=${RETRIEVAL_TOPK:-5}
NUM_WORKERS=${NUM_WORKERS:-4}

RAG_CACHE_MODE=${RAG_CACHE_MODE:-packed}
MOTION_EXPERIMENT=$(basename "${MOTION_LATENT_DIR%/}")
RAG_CACHE_DIR=${RAG_CACHE_DIR:-./humanml3d_272/msa_rag_cache/${MOTION_EXPERIMENT}-top${RETRIEVAL_TOPK}}
REBUILD_RAG_CACHE=${REBUILD_RAG_CACHE:-false}
PYTHON_BIN=${PYTHON_BIN:-python}
ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}

GENERATIVE_HEAD_TYPE=${GENERATIVE_HEAD_TYPE:-ddpm}
NUM_FLOW_STEPS=${NUM_FLOW_STEPS:-20}
FLOW_SOLVER=${FLOW_SOLVER:-euler}
RF_TIME_SAMPLING=${RF_TIME_SAMPLING:-uniform}
RF_LOSS_TYPE=${RF_LOSS_TYPE:-mse}

echo "=========================================="
echo "MotionStreamer Stage-2 RAG Training"
echo "=========================================="
echo "GPU count       : $NUM_GPUS"
echo "Batch per GPU   : $BATCH_SIZE"
echo "Total batch     : 256"
echo "Dataset         : $DATASET"
echo "Text latents    : $TEXT_LATENT_DIR"
echo "h_cls latents   : $HCLS_DIR"
echo "Motion latents  : $MOTION_LATENT_DIR"
echo "Text embed dim  : $TEXT_EMBED_DIM"
echo "Retrieval top-K : $RETRIEVAL_TOPK"
echo "Workers         : $NUM_WORKERS"
echo "Cache mode      : $RAG_CACHE_MODE"
echo "Cache dir       : $RAG_CACHE_DIR"
echo "Head type       : $GENERATIVE_HEAD_TYPE"
echo "Flow steps      : $NUM_FLOW_STEPS"
echo "Flow solver     : $FLOW_SOLVER"

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

case "$RAG_CACHE_MODE" in
  packed)
    CACHE_FLAGS=()
    if [ "$REBUILD_RAG_CACHE" = "true" ]; then
      CACHE_FLAGS+=(--force)
    fi
    "$PYTHON_BIN" build_msa_rag_cache.py \
      --dataset-name "$DATASET" \
      --motion-latent-dir "$MOTION_LATENT_DIR" \
      --text-latent-dir "$TEXT_LATENT_DIR" \
      --hcls-dir "$HCLS_DIR" \
      --cache-dir "$RAG_CACHE_DIR" \
      --topk "$RETRIEVAL_TOPK" \
      --text-embed-dim "$TEXT_EMBED_DIM" \
      "${CACHE_FLAGS[@]}"
    ;;
  reference)
    ;;
  *)
    echo "ERROR: RAG_CACHE_MODE must be packed or reference, got: $RAG_CACHE_MODE"
    exit 2
    ;;
esac

"$ACCELERATE_BIN" launch --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    train_t2m_rag.py \
    --batch-size "$BATCH_SIZE" \
    --lr 0.0001 \
    --total-iter 100000 \
    --out-dir Experiments \
    --exp-name MotionStreamer_t2m_272_msa_rag_t5_trans662048_vaefulldb_k5_ema_worker4 \
    --dataname "$DATASET" \
    --latent_dir "$MOTION_LATENT_DIR" \
    --text_latent_dir "$TEXT_LATENT_DIR" \
    --hcls_dir "$HCLS_DIR" \
    --empty_text_path "$EMPTY_TEXT_PATH" \
    --text_embed_dim "$TEXT_EMBED_DIM" \
    --num_gpus "$NUM_GPUS" \
    --retrieval_topk "$RETRIEVAL_TOPK" \
    --num_workers "$NUM_WORKERS" \
    --cache_mode "$RAG_CACHE_MODE" \
    --cache_dir "$RAG_CACHE_DIR" \
    --generative_head_type "$GENERATIVE_HEAD_TYPE" \
    --num_flow_steps "$NUM_FLOW_STEPS" \
    --flow_solver "$FLOW_SOLVER" \
    --rf_time_sampling "$RF_TIME_SAMPLING" \
    --rf_loss_type "$RF_LOSS_TYPE"

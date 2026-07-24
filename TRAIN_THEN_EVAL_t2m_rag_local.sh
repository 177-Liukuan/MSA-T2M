#!/bin/bash
# Train MotionStreamer stage-2 with global + local RAG tokens,
# then automatically evaluate the last saved checkpoint.
#
# Usage:
#   bash TRAIN_THEN_EVAL_t2m_rag_local.sh [NUM_GPUS]
#
# All variables can be overridden via environment, e.g.:
#   TOTAL_ITER=50000 bash TRAIN_THEN_EVAL_t2m_rag_local.sh 4

set -e  # abort on any error

NUM_GPUS=${1:-1}
BATCH_SIZE=$((256 / NUM_GPUS))

# -----------------------------------------------------------------------
# Shared paths (used by both training and evaluation)
# -----------------------------------------------------------------------
DATASET=${DATASET:-t2m_272}
TEXT_LATENT_DIR=${TEXT_LATENT_DIR:-./humanml3d_272/text_latents_t5}
HCLS_DIR=${HCLS_DIR:-./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
MOTION_LATENT_DIR=${MOTION_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
Z_LATENT_DIR=${Z_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
EMPTY_TEXT_PATH=${EMPTY_TEXT_PATH:-./humanml3d_272/text_latents_t5/empty_text_embedding.npy}
TEXT_EMBED_DIM=${TEXT_EMBED_DIM:-768}

# -----------------------------------------------------------------------
# Local RAG hyper-parameters (must be consistent across train and eval)
# -----------------------------------------------------------------------
L_LOCAL=${L_LOCAL:-4}
LOCAL_RAG_DIM=${LOCAL_RAG_DIM:-16}
RETRIEVAL_TOPK=${RETRIEVAL_TOPK:-3}

# -----------------------------------------------------------------------
# Training-specific parameters
# -----------------------------------------------------------------------
TOTAL_ITER=${TOTAL_ITER:-100000}
OUT_DIR=${OUT_DIR:-Experiments}
EXP_NAME=${EXP_NAME:-MotionStreamer_t2m_272_msa_rag_local_L${L_LOCAL}_k${RETRIEVAL_TOPK}_crossattn}

GENERATIVE_HEAD_TYPE=${GENERATIVE_HEAD_TYPE:-ddpm}
NUM_FLOW_STEPS=${NUM_FLOW_STEPS:-20}
FLOW_SOLVER=${FLOW_SOLVER:-euler}
RF_TIME_SAMPLING=${RF_TIME_SAMPLING:-uniform}
RF_LOSS_TYPE=${RF_LOSS_TYPE:-mse}

# -----------------------------------------------------------------------
# Evaluation-specific parameters
# -----------------------------------------------------------------------
MSA_VAE_CKPT=${MSA_VAE_CKPT:-Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/net_best_mpjpe.pth}
TEXT_SOURCE=${TEXT_SOURCE:-online_t5}
T5_MODEL_PATH=${T5_MODEL_PATH:-sentencet5-xxl/}
STOP_THRESHOLD=${STOP_THRESHOLD:-0.1}
CFG_SCALE=${CFG_SCALE:-4.0}

# -----------------------------------------------------------------------
# Derived paths (do not modify)
# -----------------------------------------------------------------------
TRAIN_OUT_DIR="${OUT_DIR}/${EXP_NAME}"
LAST_CKPT="${TRAIN_OUT_DIR}/net_Iter$(printf '%06d' ${TOTAL_ITER}).pth"
EVAL_EXP_NAME="${EXP_NAME}_eval"
REFERENCE_END_LATENT="humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/reference_end_latent_msa_vae_t2m_272.npy"

# -----------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------
for DIR in "$TEXT_LATENT_DIR" "$HCLS_DIR" "$MOTION_LATENT_DIR" "$Z_LATENT_DIR"; do
  if [ ! -d "$DIR" ]; then
    echo "ERROR: directory not found: $DIR"
    exit 1
  fi
done

echo "=========================================="
echo "  MotionStreamer  Train -> Eval  Pipeline  "
echo "=========================================="
echo "GPU count       : $NUM_GPUS"
echo "Batch per GPU   : $BATCH_SIZE"
echo "Total iterations: $TOTAL_ITER"
echo "Experiment name : $EXP_NAME"
echo "Train output    : $TRAIN_OUT_DIR"
echo "Last checkpoint : $LAST_CKPT"
echo "L_local         : $L_LOCAL  (block_size = $((78 + L_LOCAL)))"
echo "local_rag_dim   : $LOCAL_RAG_DIM"
echo "Retrieval top-K : $RETRIEVAL_TOPK"
echo "Head type       : $GENERATIVE_HEAD_TYPE"
echo "=========================================="

# -----------------------------------------------------------------------
# Phase 1: Training
# -----------------------------------------------------------------------
echo ""
echo "[Phase 1] Starting training ..."
echo ""

accelerate launch --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    train_t2m_rag_local.py \
    --batch-size $BATCH_SIZE \
    --lr 0.0001 \
    --total-iter $TOTAL_ITER \
    --out-dir $OUT_DIR \
    --exp-name $EXP_NAME \
    --dataname $DATASET \
    --latent_dir $MOTION_LATENT_DIR \
    --text_latent_dir $TEXT_LATENT_DIR \
    --hcls_dir $HCLS_DIR \
    --z_latent_dir $Z_LATENT_DIR \
    --empty_text_path $EMPTY_TEXT_PATH \
    --text_embed_dim $TEXT_EMBED_DIM \
    --num_gpus $NUM_GPUS \
    --retrieval_topk $RETRIEVAL_TOPK \
    --L_local $L_LOCAL \
    --local_rag_dim $LOCAL_RAG_DIM \
    --generative_head_type $GENERATIVE_HEAD_TYPE \
    --num_flow_steps $NUM_FLOW_STEPS \
    --flow_solver $FLOW_SOLVER \
    --rf_time_sampling $RF_TIME_SAMPLING \
    --rf_loss_type $RF_LOSS_TYPE

echo ""
echo "[Phase 1] Training finished."

# -----------------------------------------------------------------------
# Phase 2: Evaluate last checkpoint
# -----------------------------------------------------------------------
echo ""
echo "[Phase 2] Starting evaluation ..."
echo "          RAG checkpoint : $LAST_CKPT"
echo ""

if [ ! -f "$LAST_CKPT" ]; then
  echo "ERROR: Last checkpoint not found: $LAST_CKPT"
  echo "       Training may have ended early or the checkpoint was not saved."
  exit 1
fi

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate mgpt
fi

python eval_msa_t2m_rag_local.py \
  --resume-pth "$MSA_VAE_CKPT" \
  --resume-trans "$LAST_CKPT" \
  --latent_dir "$MOTION_LATENT_DIR" \
  --text_latent_dir "$TEXT_LATENT_DIR" \
  --hcls_dir "$HCLS_DIR" \
  --z_latent_dir "$Z_LATENT_DIR" \
  --empty_text_path "$EMPTY_TEXT_PATH" \
  --text_embed_dim "$TEXT_EMBED_DIM" \
  --text_source "$TEXT_SOURCE" \
  --t5_model_path "$T5_MODEL_PATH" \
  --trans_d_model "$TEXT_EMBED_DIM" \
  --clip_dim "$TEXT_EMBED_DIM" \
  --cfg_scale "$CFG_SCALE" \
  --stop_threshold "$STOP_THRESHOLD" \
  --retrieval_topk "$RETRIEVAL_TOPK" \
  --L_local "$L_LOCAL" \
  --local_rag_dim "$LOCAL_RAG_DIM" \
  --exp-name "$EVAL_EXP_NAME" \
  --reference_end_latent_path "$REFERENCE_END_LATENT" \
  --disable_ema

echo ""
echo "[Phase 2] Evaluation finished."
echo "          Results saved under: ${OUT_DIR}/${EVAL_EXP_NAME}/"
echo "=========================================="

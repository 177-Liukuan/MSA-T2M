#!/bin/bash
set -euo pipefail
# -----------------------------------------------------------
#  MSA-VAE Phase 1: Freeze CNN, train Transformer + projections
#
#  Losses: L_latent + L_global_align + L_local_align
#  CNN encoder/decoder/decode_proj are FROZEN (loaded from pretrained TAE)
#
#  Usage:
#    CNN_CKPT=Causal_TAE/net_last.pth bash TRAIN_msa_vae_phase1.sh [NUM_GPUS] [DATASET]
#
#  Optional overrides:
#    TEXT_ENCODER_TYPE=t5|clip EXP_NAME=... SEED=...
#    GLOBAL_ALIGN_WEIGHT=... LOCAL_ALIGN_WEIGHT=...
#    TOTAL_ITER=... WARM_UP_ITER=... EVAL_ITER=... OUT_DIR=...
# -----------------------------------------------------------

NUM_GPUS=${1:-1}
dataset_name=${2:-t2m_272}
BATCH_SIZE=$((128 / NUM_GPUS))
DEFAULT_FULL_SEQ_BATCH_SIZE=$((32 / NUM_GPUS))
if [ "$DEFAULT_FULL_SEQ_BATCH_SIZE" -lt 1 ]; then
  DEFAULT_FULL_SEQ_BATCH_SIZE=1
fi
FULL_SEQ_BATCH_SIZE=${FULL_SEQ_BATCH_SIZE:-$DEFAULT_FULL_SEQ_BATCH_SIZE}
LENGTH_BUCKET_SIZE=${LENGTH_BUCKET_SIZE:-256}
TOTAL_ITER=${TOTAL_ITER:-50000}
WARM_UP_ITER=${WARM_UP_ITER:-500}
EVAL_ITER=${EVAL_ITER:-2500}
VALIDATION_SEED=${VALIDATION_SEED:-123}
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-32}
OUT_DIR=${OUT_DIR:-Experiments}
SEED=${SEED:-123}
GLOBAL_ALIGN_WEIGHT=${GLOBAL_ALIGN_WEIGHT:-0.5}
LOCAL_ALIGN_WEIGHT=${LOCAL_ALIGN_WEIGHT:-0.2}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-}

TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-t5}
T5_EMBED_DIR=${T5_EMBED_DIR:-./humanml3d_272/t5_enc_single}
CLIP_EMBED_DIR=${CLIP_EMBED_DIR:-./humanml3d_272/clip_enc_single}
T5_MODEL_PATH=${T5_MODEL_PATH:-sentencet5-xxl/}
CNN_CKPT=${CNN_CKPT:-Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth}
CNN_CKPT_SHA256=${CNN_CKPT_SHA256:-}

if [[ ${EXP_NAME+x} && -z "$EXP_NAME" ]]; then
  echo "ERROR: EXP_NAME must not be empty" >&2
  exit 2
fi
EXP_NAME=${EXP_NAME:-MSA_VAEv7_phase1_fullseq_${dataset_name}_${TEXT_ENCODER_TYPE}_fulldb}

validate_nonnegative_weight() {
  local name=$1
  local value=$2
  if ! awk -v value="$value" \
    'BEGIN {
      valid = value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/
      exit !(valid && value + 0 >= 0)
    }'; then
    echo "ERROR: ${name} must be a non-negative number, got '${value}'" >&2
    exit 2
  fi
}

validate_nonnegative_weight "GLOBAL_ALIGN_WEIGHT" "$GLOBAL_ALIGN_WEIGHT"
validate_nonnegative_weight "LOCAL_ALIGN_WEIGHT" "$LOCAL_ALIGN_WEIGHT"
ACCELERATE_PORT_ARGS=()
if [[ -n "$MAIN_PROCESS_PORT" ]]; then
  if [[ ! "$MAIN_PROCESS_PORT" =~ ^[0-9]+$ ]] \
      || (( MAIN_PROCESS_PORT < 1 || MAIN_PROCESS_PORT > 65535 )); then
    echo "ERROR: MAIN_PROCESS_PORT must be an integer from 1 to 65535" >&2
    exit 2
  fi
  ACCELERATE_PORT_ARGS=(--main_process_port "$MAIN_PROCESS_PORT")
fi

if [[ ! -f "$CNN_CKPT" ]]; then
  echo "ERROR: CNN_CKPT is not a file: $CNN_CKPT" >&2
  exit 2
fi
if [[ ! "$CNN_CKPT_SHA256" =~ ^[[:xdigit:]]{64}$ ]]; then
  if [[ -n "$CNN_CKPT_SHA256" ]]; then
    echo "ERROR: CNN_CKPT_SHA256 must contain 64 hexadecimal characters" >&2
    exit 2
  fi
fi
ACTUAL_CNN_CKPT_SHA256=$(sha256sum -- "$CNN_CKPT" | awk '{print $1}')
if [[ -n "$CNN_CKPT_SHA256" \
      && "${CNN_CKPT_SHA256,,}" != "$ACTUAL_CNN_CKPT_SHA256" ]]; then
  echo "ERROR: CNN_CKPT_SHA256 does not match CNN_CKPT contents" >&2
  exit 2
fi
CNN_CKPT_SHA256=$ACTUAL_CNN_CKPT_SHA256

if [ "$TEXT_ENCODER_TYPE" = "t5" ]; then
  TEXT_EMBED_DIM=768
else
  TEXT_EMBED_DIM=512
fi

# CNN checkpoint is required for Phase 1
CNN_CKPT=${CNN_CKPT:?"ERROR: set CNN_CKPT to pretrained TAE checkpoint path"}

echo "========== MSA-VAE Phase 1: Freeze CNN =========="
echo "GPUs            : $NUM_GPUS"
echo "Dataset         : $dataset_name"
echo "Batch per GPU   : $BATCH_SIZE"
echo "Full batch/GPU  : $FULL_SEQ_BATCH_SIZE"
echo "Length bucket   : $LENGTH_BUCKET_SIZE"
echo "Text encoder    : $TEXT_ENCODER_TYPE"
echo "Text dim        : $TEXT_EMBED_DIM"
echo "CNN checkpoint  : $CNN_CKPT"
echo "CNN SHA-256     : $CNN_CKPT_SHA256"
echo "Experiment      : $EXP_NAME"
echo "Output root     : $OUT_DIR"
echo "Seed            : $SEED"
echo "Iterations      : $TOTAL_ITER (warm-up $WARM_UP_ITER)"
echo "Eval interval   : $EVAL_ITER"
echo "Validation      : seed=$VALIDATION_SEED batch=$VALIDATION_BATCH_SIZE"
echo "Align weights   : global=$GLOBAL_ALIGN_WEIGHT local=$LOCAL_ALIGN_WEIGHT"
echo "=================================================="

# Use full HumanML3D train split; L_local is auto-masked by has_local
accelerate launch --num_processes "$NUM_GPUS" \
  "${ACCELERATE_PORT_ARGS[@]}" --mixed_precision bf16 \
  train_msa_vae.py \
  --phase 1 \
  --batch-size "$BATCH_SIZE" \
  --sequence_mode full \
  --full-seq-batch-size "$FULL_SEQ_BATCH_SIZE" \
  --length-bucket-size "$LENGTH_BUCKET_SIZE" \
  --lr 1e-4 \
  --total-iter "$TOTAL_ITER" \
  --warm-up-iter "$WARM_UP_ITER" \
  --lr-scheduler 50000 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --out-dir "$OUT_DIR" \
  --dataname "$dataset_name" \
  --exp-name "$EXP_NAME" \
  --root_loss 7.0 \
  --latent_dim 16 \
  --hidden_size 1024 \
  --trans_d_model "$TEXT_EMBED_DIM" \
  --trans_nhead 8 \
  --trans_enc_layers 6 \
  --trans_dec_layers 6 \
  --trans_ff_size 2048 \
  --trans_dropout 0.1 \
  --clip_dim "$TEXT_EMBED_DIM" \
  --text_encoder_type "$TEXT_ENCODER_TYPE" \
  --text_embed_dim "$TEXT_EMBED_DIM" \
  --clip_embed_dir "$CLIP_EMBED_DIR" \
  --t5_embed_dir "$T5_EMBED_DIR" \
  --use_offline_global_text \
  --clip_global_embed_dir ./humanml3d_272/text_latents_clip \
  --t5_global_embed_dir ./humanml3d_272/text_latents_t5 \
  --t5_model_path "$T5_MODEL_PATH" \
  --clip_version ViT-B/32 \
  --latent_recon_weight 1.0 \
  --global_align_weight "$GLOBAL_ALIGN_WEIGHT" \
  --local_align_weight "$LOCAL_ALIGN_WEIGHT" \
  --no_ft_split \
  --num_gpus "$NUM_GPUS" \
  --resume-cnn-pth "$CNN_CKPT" \
  --resume-cnn-sha256 "$CNN_CKPT_SHA256" \
  --eval-iter "$EVAL_ITER" \
  --validation-seed "$VALIDATION_SEED" \
  --validation-batch-size "$VALIDATION_BATCH_SIZE" \
  --print-iter 200 \
  --seed "$SEED"

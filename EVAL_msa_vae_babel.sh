#!/usr/bin/env bash
# Evaluate BABEL sparse-global reconstruction and local semantic alignment.
# It deliberately does not invoke the HumanML3D TMR or text-to-motion protocol.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
cd "$REPO_ROOT"

BABEL_CKPT=${BABEL_CKPT:?"ERROR: set BABEL_CKPT to a BABEL sparse-global phase-2 net_best_mpjpe.pth checkpoint"}
PYTHON_BIN=${PYTHON_BIN:-python}
T5_MODEL_PATH=${T5_MODEL_PATH:-sentencet5-xxl/}
BATCH_SIZE=${BATCH_SIZE:-128}

MSA_MEAN_PATH=${MSA_MEAN_PATH:-babel_272/t2m_babel_mean_std/Mean.npy}
MSA_STD_PATH=${MSA_STD_PATH:-babel_272/t2m_babel_mean_std/Std.npy}
BRIDGE_SPLIT_FILE=${BRIDGE_SPLIT_FILE:-humanml3d_272/split/train_ft.txt}
BRIDGE_MOTION_DIR=${BRIDGE_MOTION_DIR:-humanml3d_272/motion_data}
BRIDGE_TEXT_DIR=${BRIDGE_TEXT_DIR:-humanml3d_272/texts}
BRIDGE_GLOBAL_EMBED_DIR=${BRIDGE_GLOBAL_EMBED_DIR:-humanml3d_272/text_latents_t5}
BRIDGE_LOCAL_EMBED_DIR=${BRIDGE_LOCAL_EMBED_DIR:-humanml3d_272/t5_enc_single}

BABEL_T5_ROOT=${BABEL_T5_ROOT:-babel_272_stream/t5_enc_single}
BABEL_TRAIN_MOTION_DIR=${BABEL_TRAIN_MOTION_DIR:-babel_272_stream/train_stream}
BABEL_TRAIN_TEXT_DIR=${BABEL_TRAIN_TEXT_DIR:-babel_272_stream/train_stream_text}
BABEL_TRAIN_CACHE_DIR=${BABEL_TRAIN_CACHE_DIR:-$BABEL_T5_ROOT/train}
BABEL_TRAIN_MANIFEST=${BABEL_TRAIN_MANIFEST:-$BABEL_TRAIN_CACHE_DIR/manifest.json}
BABEL_VAL_MOTION_DIR=${BABEL_VAL_MOTION_DIR:-babel_272_stream/val_stream}
BABEL_VAL_TEXT_DIR=${BABEL_VAL_TEXT_DIR:-babel_272_stream/val_stream_text}
BABEL_VAL_CACHE_DIR=${BABEL_VAL_CACHE_DIR:-$BABEL_T5_ROOT/val}
BABEL_VAL_MANIFEST=${BABEL_VAL_MANIFEST:-$BABEL_VAL_CACHE_DIR/manifest.json}
EXP_NAME=${EXP_NAME:-MSA_VAEv6_babel_sparse_global_eval}

"$PYTHON_BIN" eval_msa_vae.py \
  --phase 2 \
  --resume-pth "$BABEL_CKPT" \
  --out-dir Experiments \
  --exp-name "$EXP_NAME" \
  --dataname t2m_babel_272 \
  --batch-size "$BATCH_SIZE" \
  --msa_data_mode babel_sparse_global \
  --msa_mean_path "$MSA_MEAN_PATH" \
  --msa_std_path "$MSA_STD_PATH" \
  --bridge_split_file "$BRIDGE_SPLIT_FILE" \
  --bridge_motion_dir "$BRIDGE_MOTION_DIR" \
  --bridge_text_dir "$BRIDGE_TEXT_DIR" \
  --bridge_global_embed_dir "$BRIDGE_GLOBAL_EMBED_DIR" \
  --bridge_local_embed_dir "$BRIDGE_LOCAL_EMBED_DIR" \
  --babel_train_motion_dir "$BABEL_TRAIN_MOTION_DIR" \
  --babel_train_text_dir "$BABEL_TRAIN_TEXT_DIR" \
  --babel_train_t5_cache_dir "$BABEL_TRAIN_CACHE_DIR" \
  --babel_train_cache_manifest "$BABEL_TRAIN_MANIFEST" \
  --babel_val_motion_dir "$BABEL_VAL_MOTION_DIR" \
  --babel_val_text_dir "$BABEL_VAL_TEXT_DIR" \
  --babel_val_t5_cache_dir "$BABEL_VAL_CACHE_DIR" \
  --babel_val_cache_manifest "$BABEL_VAL_MANIFEST" \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --latent_dim 16 \
  --hidden_size 1024 \
  --trans_d_model 768 \
  --trans_nhead 8 \
  --trans_enc_layers 6 \
  --trans_dec_layers 6 \
  --trans_ff_size 2048 \
  --trans_dropout 0.1 \
  --clip_dim 768 \
  --text_encoder_type t5 \
  --text_embed_dim 768 \
  --t5_embed_dir "$BRIDGE_LOCAL_EMBED_DIR" \
  --use_offline_global_text \
  --t5_global_embed_dir "$BRIDGE_GLOBAL_EMBED_DIR" \
  --t5_model_path "$T5_MODEL_PATH" \
  --global_align_weight 0.1 \
  --local_align_weight 0.001

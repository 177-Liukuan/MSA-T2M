#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

# Evaluate MotionStreamer Stage-2 with global h_cls RAG + local motion latent retrieval CA.
#
# Usage:
#   bash EVAL_t2m_rag_latent_retr.sh
#
# Differences from EVAL_t2m_rag_mca.sh:
#   - Uses -m explorations.cross_attention.latent_retrieval.eval_msa_t2m_rag_latent_retr
#   - Requires --library_cache_dir pointing to the 5-file pre-built cache
#     (built by build_latent_retr_library.py).
#   - Passes --latent_retr_topk and --latent_dim instead of --text_token_dim.
#   - CA KV comes from retrieved motion latents, not word-level T5 tokens.

MSA_VAE_CKPT=${MSA_VAE_CKPT:-Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/net_best_mpjpe.pth}
RAG_CKPT=${RAG_CKPT:-Experiments/explorations/cross_attention/latent_retrieval/MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_6layer_top3_ddpm/net_Iter100000.pth}
MOTION_LATENT_DIR=${MOTION_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
TEXT_LATENT_DIR=${TEXT_LATENT_DIR:-./humanml3d_272/text_latents_t5}
HCLS_DIR=${HCLS_DIR:-./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
EMPTY_TEXT_PATH=${EMPTY_TEXT_PATH:-./humanml3d_272/text_latents_t5/empty_text_embedding.npy}
TEXT_EMBED_DIM=${TEXT_EMBED_DIM:-768}
TEXT_SOURCE=${TEXT_SOURCE:-online_t5}
T5_MODEL_PATH=${T5_MODEL_PATH:-sentencet5-xxl/}
STOP_THRESHOLD=${STOP_THRESHOLD:-0.1}

# Local-RAG library cache (auto-derived from MOTION_LATENT_DIR if not set)
EXP=$(basename "${MOTION_LATENT_DIR%/}")
LIBRARY_CACHE_DIR=${LIBRARY_CACHE_DIR:-./humanml3d_272/latent_retr_library_cache/${EXP}}

# Local-RAG hyper-parameters  ── must match training config
LATENT_RETR_TOPK=${LATENT_RETR_TOPK:-3}
LATENT_DIM=${LATENT_DIM:-16}

# CA architecture hyper-parameters  ── must match training config
CA_N_HEAD=${CA_N_HEAD:-0}               # 0 = auto (same as backbone)
CA_EVERY_N_LAYERS=${CA_EVERY_N_LAYERS:-2}  # Must match training (e.g. 1=every layer)
CA_INSERTION_MODE=${CA_INSERTION_MODE:-after_sa}   # before_sa | after_sa | late_after_sa
# CFG_SCALE_RETR is no longer used: inference uses 2-forward velocity-space CFG
# (joint dropout mode). The retrieval signal is baked into z_cond via CA blocks.

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate mgpt
fi

echo "=========================================="
echo "MSA-T2M RAG + Local Latent Retrieval Evaluation"
echo "=========================================="
echo "MSA-VAE ckpt         : $MSA_VAE_CKPT"
echo "RAG ckpt             : $RAG_CKPT"
echo "Motion latents       : $MOTION_LATENT_DIR"
echo "Text latents         : $TEXT_LATENT_DIR"
echo "h_cls latents        : $HCLS_DIR"
echo "Library cache dir    : $LIBRARY_CACHE_DIR"
echo "Text embed dim       : $TEXT_EMBED_DIM"
echo "Text source          : $TEXT_SOURCE"
echo "T5 model path        : $T5_MODEL_PATH"
echo "Stop threshold       : $STOP_THRESHOLD"
echo "Latent retr topk     : $LATENT_RETR_TOPK"
echo "Latent dim           : $LATENT_DIM"
echo "CA every N layers    : $CA_EVERY_N_LAYERS"
echo "CA insertion mode    : $CA_INSERTION_MODE"
echo "CA heads  (train cfg): $CA_N_HEAD"
echo "CFG mode             : 2-forward velocity-space (joint dropout)"
echo "=========================================="

# Pre-flight: verify library cache exists
if [ ! -f "${LIBRARY_CACHE_DIR}/lib_text_embs.npy" ]; then
  echo "[ERROR] Library cache not found at: ${LIBRARY_CACHE_DIR}"
  echo "  Run build_latent_retr_library.py first:"
  echo "    python -m explorations.cross_attention.latent_retrieval.build_latent_retr_library \\"
  echo "      --motion_latent_dir ${MOTION_LATENT_DIR} \\"
  echo "      --text_latent_dir ${TEXT_LATENT_DIR} \\"
  echo "      --output_cache_dir ${LIBRARY_CACHE_DIR}"
  exit 1
fi

python -m explorations.cross_attention.latent_retrieval.eval_msa_t2m_rag_latent_retr \
  --resume-pth $MSA_VAE_CKPT \
  --resume-trans $RAG_CKPT \
  --latent_dir $MOTION_LATENT_DIR \
  --text_latent_dir $TEXT_LATENT_DIR \
  --hcls_dir $HCLS_DIR \
  --library_cache_dir $LIBRARY_CACHE_DIR \
  --empty_text_path $EMPTY_TEXT_PATH \
  --text_embed_dim $TEXT_EMBED_DIM \
  --text_source $TEXT_SOURCE \
  --t5_model_path $T5_MODEL_PATH \
  --trans_d_model $TEXT_EMBED_DIM \
  --clip_dim $TEXT_EMBED_DIM \
  --cfg_scale 4.0 \
  --stop_threshold $STOP_THRESHOLD \
  --latent_retr_topk $LATENT_RETR_TOPK \
  --latent_dim $LATENT_DIM \
  --retrieval_topk 3 \
  --exp-name MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_after_sa_every1layer_top3_ddpm_cfg_saca_dropout01 \
  --reference_end_latent_path humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/reference_end_latent_msa_vae_t2m_272.npy \
  --disable_ema \
  --ca_every_n_layers $CA_EVERY_N_LAYERS \
  --ca_insertion_mode $CA_INSERTION_MODE \
  --ca_n_head $CA_N_HEAD \
  --cfg_scale_retr 1.0

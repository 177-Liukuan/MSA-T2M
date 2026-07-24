#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

# Evaluate MotionStreamer Stage-2 with RAG + Multi-Text Cross-Attention (MCA)
#
# Usage:
#   bash EVAL_t2m_rag_mca.sh
#
# Differences from EVAL_t2m_rag_t5.sh:
#   - Uses -m explorations.cross_attention.mca.eval_msa_t2m_rag_mca
#   - Passes --ca_n_layers, --ca_n_head, --text_token_dim (must match training)
#   - MCA token encoding is always online (T5 model already loaded for sentence emb)

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

# MCA hyper-parameters  ── must match training config
CA_N_LAYERS=${CA_N_LAYERS:-6}
CA_N_HEAD=${CA_N_HEAD:-0}            # 0 = auto (same as backbone)
CA_EVERY_N_LAYERS=${CA_EVERY_N_LAYERS:-4}  # insertion interval (4 = layers [3,7,11] for 12-layer backbone)
TEXT_TOKEN_DIM=${TEXT_TOKEN_DIM:-1024}  # 1024 for sentence-t5-xxl encoder
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate mgpt
fi

echo "=========================================="
echo "MSA-T2M RAG+MCA Evaluation"
echo "=========================================="
echo "MSA-VAE ckpt         : $MSA_VAE_CKPT"
echo "RAG ckpt             : $RAG_CKPT"
echo "Motion latents       : $MOTION_LATENT_DIR"
echo "Text latents         : $TEXT_LATENT_DIR"
echo "h_cls latents        : $HCLS_DIR"
echo "Text embed dim       : $TEXT_EMBED_DIM"
echo "Text source          : $TEXT_SOURCE"
echo "T5 model path        : $T5_MODEL_PATH"
echo "Stop threshold       : $STOP_THRESHOLD"
echo "CA layers (train cfg): $CA_N_LAYERS"
echo "CA every N layers    : $CA_EVERY_N_LAYERS"
echo "CA heads  (train cfg): $CA_N_HEAD"
echo "Token emb dim        : $TEXT_TOKEN_DIM"
echo "=========================================="

python -m explorations.cross_attention.mca.eval_msa_t2m_rag_mca \
  --resume-pth $MSA_VAE_CKPT \
  --resume-trans $RAG_CKPT \
  --latent_dir $MOTION_LATENT_DIR \
  --text_latent_dir $TEXT_LATENT_DIR \
  --hcls_dir $HCLS_DIR \
  --empty_text_path $EMPTY_TEXT_PATH \
  --text_embed_dim $TEXT_EMBED_DIM \
  --text_source $TEXT_SOURCE \
  --t5_model_path $T5_MODEL_PATH \
  --trans_d_model $TEXT_EMBED_DIM \
  --clip_dim $TEXT_EMBED_DIM \
  --cfg_scale 4.0 \
  --stop_threshold $STOP_THRESHOLD \
  --exp-name MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_6layer_top3_ddpm \
  --reference_end_latent_path humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/reference_end_latent_msa_vae_t2m_272.npy \
  --retrieval_topk 3 \
  --disable_ema \
  --ca_n_layers $CA_N_LAYERS \
  --ca_every_n_layers $CA_EVERY_N_LAYERS \
  --ca_n_head $CA_N_HEAD \
  --text_token_dim $TEXT_TOKEN_DIM

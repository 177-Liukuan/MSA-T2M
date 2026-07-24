#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
# Evaluate MotionStreamer Stage-2 with global + local RAG tokens.
# Usage:
#   bash EVAL_t2m_rag_local.sh [NUM_GPUS]

NUM_GPUS=${1:-1}

MSA_VAE_CKPT=${MSA_VAE_CKPT:-Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/net_best_mpjpe.pth}
RAG_CKPT=${RAG_CKPT:-Experiments/explorations/cross_attention/local_rag/MotionStreamer_t2m_272_msa_rag_local_L16_k3_sa_ca/net_Iter100000.pth}
MOTION_LATENT_DIR=${MOTION_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
TEXT_LATENT_DIR=${TEXT_LATENT_DIR:-./humanml3d_272/text_latents_t5}
HCLS_DIR=${HCLS_DIR:-./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
Z_LATENT_DIR=${Z_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
EMPTY_TEXT_PATH=${EMPTY_TEXT_PATH:-./humanml3d_272/text_latents_t5/empty_text_embedding.npy}
TEXT_EMBED_DIM=${TEXT_EMBED_DIM:-768}
TEXT_SOURCE=${TEXT_SOURCE:-online_t5}
T5_MODEL_PATH=${T5_MODEL_PATH:-sentencet5-xxl/}
STOP_THRESHOLD=${STOP_THRESHOLD:-0.1}

# Local RAG hyper-parameters — must match the trained checkpoint.
ADD_SELFATTEN=${ADD_SELFATTEN:-1}  # set to 1 to enable SA encoding before cross-attn (must match training)
L_LOCAL=${L_LOCAL:-16}
LOCAL_RAG_DIM=${LOCAL_RAG_DIM:-16}
RETRIEVAL_TOPK=${RETRIEVAL_TOPK:-3}

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate mgpt
fi

echo "=========================================="
echo "MSA-T2M Local-RAG Evaluation"
echo "=========================================="
echo "MSA-VAE ckpt    : $MSA_VAE_CKPT"
echo "RAG ckpt        : $RAG_CKPT"
echo "Motion latents  : $MOTION_LATENT_DIR"
echo "Text latents    : $TEXT_LATENT_DIR"
echo "h_cls latents   : $HCLS_DIR"
echo "Z latents      : $Z_LATENT_DIR"
echo "Text embed dim  : $TEXT_EMBED_DIM"
echo "Text source     : $TEXT_SOURCE"
echo "T5 model path   : $T5_MODEL_PATH"
echo "Stop threshold  : $STOP_THRESHOLD"
echo "L_local         : $L_LOCAL  (block_size = $((78 + L_LOCAL)))"
echo "local_rag_dim   : $LOCAL_RAG_DIM"
echo "Retrieval top-K : $RETRIEVAL_TOPK"
echo "add_selfatten   : $ADD_SELFATTEN"

# Validate required directories
for DIR in "$HCLS_DIR" "$Z_LATENT_DIR" "$MOTION_LATENT_DIR" "$TEXT_LATENT_DIR"; do
  if [ ! -d "$DIR" ]; then
    echo "ERROR: directory not found: $DIR"
    exit 1
  fi
done

python -m explorations.cross_attention.local_rag.eval_msa_t2m_rag_local \
  --resume-pth "$MSA_VAE_CKPT" \
  --resume-trans "$RAG_CKPT" \
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
  --cfg_scale 4.0 \
  --stop_threshold "$STOP_THRESHOLD" \
  --retrieval_topk "$RETRIEVAL_TOPK" \
  --L_local "$L_LOCAL" \
  --local_rag_dim "$LOCAL_RAG_DIM" \
  --exp-name "MotionStreamer_t2m_272_msa_rag_local_L${L_LOCAL}_k${RETRIEVAL_TOPK}__self_cross" \
  --reference_end_latent_path "humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/reference_end_latent_msa_vae_t2m_272.npy" \
  $([ "$ADD_SELFATTEN" = "1" ] && echo "--add_selfatten")

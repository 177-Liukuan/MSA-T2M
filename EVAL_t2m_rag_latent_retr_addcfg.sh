#!/bin/bash

# Evaluate MotionStreamer Stage-2 with 3-forward ADDITIVE velocity-space CFG.
#
# Uses NEW model trained with independent retrieval dropout (retr_cfg_drop_prob=0.3).
# Decouples text and retrieval guidance via separate scales s_t and s_r:
#
#   v_guided = v_nn + s_t*(v_tn - v_nn) + s_r*(v_tr - v_tn)
#
# All 3 z-vectors are in-distribution — no OOD hidden-space extrapolation.
#
# Usage:
#   bash EVAL_t2m_rag_latent_retr_addcfg.sh
# Override scales:
#   CFG_SCALE_T=7.0 CFG_SCALE_R=2.5 bash EVAL_t2m_rag_latent_retr_addcfg.sh

MSA_VAE_CKPT=${MSA_VAE_CKPT:-Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/net_best_mpjpe.pth}
RAG_CKPT=${RAG_CKPT:-Experiments/MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_after_sa_every2layer_top3_ddpm_cfg_saca_dropout01/net_Iter100000.pth}
MOTION_LATENT_DIR=${MOTION_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
TEXT_LATENT_DIR=${TEXT_LATENT_DIR:-./humanml3d_272/text_latents_t5}
HCLS_DIR=${HCLS_DIR:-./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
EMPTY_TEXT_PATH=${EMPTY_TEXT_PATH:-./humanml3d_272/text_latents_t5/empty_text_embedding.npy}
TEXT_EMBED_DIM=${TEXT_EMBED_DIM:-768}
TEXT_SOURCE=${TEXT_SOURCE:-online_t5}
T5_MODEL_PATH=${T5_MODEL_PATH:-sentencet5-xxl/}
STOP_THRESHOLD=${STOP_THRESHOLD:-0.1}

# Local-RAG library cache
EXP=$(basename "${MOTION_LATENT_DIR%/}")
LIBRARY_CACHE_DIR=${LIBRARY_CACHE_DIR:-./humanml3d_272/latent_retr_library_cache/${EXP}}

# Local-RAG hyper-parameters — must match training config
LATENT_RETR_TOPK=${LATENT_RETR_TOPK:-3}
LATENT_DIM=${LATENT_DIM:-16}

# CA architecture — must match training config
CA_N_HEAD=${CA_N_HEAD:-0}
CA_EVERY_N_LAYERS=${CA_EVERY_N_LAYERS:-2}
CA_INSERTION_MODE=${CA_INSERTION_MODE:-after_sa}   # before_sa | after_sa | late_after_sa

# Additive CFG scales (key difference from 2-forward script)
CFG_SCALE_T=${CFG_SCALE_T:-6.0}   # text guidance strength s_t
CFG_SCALE_R=${CFG_SCALE_R:-2.0}   # retrieval guidance strength s_r

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate mgpt
fi

echo "=========================================="
echo "MSA-T2M RAG + Additive CFG Evaluation"
echo "=========================================="
echo "MSA-VAE ckpt         : $MSA_VAE_CKPT"
echo "RAG ckpt             : $RAG_CKPT"
echo "Motion latents       : $MOTION_LATENT_DIR"
echo "Library cache dir    : $LIBRARY_CACHE_DIR"
echo "Text source          : $TEXT_SOURCE"
echo "Stop threshold       : $STOP_THRESHOLD"
echo "CA every N layers    : $CA_EVERY_N_LAYERS"
echo "CA insertion mode    : $CA_INSERTION_MODE"
echo "CFG mode             : 3-forward additive velocity-space"
echo "  s_t (text)         : $CFG_SCALE_T"
echo "  s_r (retrieval)    : $CFG_SCALE_R"
echo "=========================================="

# Pre-flight: verify library cache
if [ ! -f "${LIBRARY_CACHE_DIR}/lib_text_embs.npy" ]; then
  echo "[ERROR] Library cache not found at: ${LIBRARY_CACHE_DIR}"
  echo "  Run build_latent_retr_library.py first."
  exit 1
fi

python eval_msa_t2m_rag_latent_retr_addcfg.py \
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
  --cfg_scale_t $CFG_SCALE_T \
  --cfg_scale_r $CFG_SCALE_R \
  --stop_threshold $STOP_THRESHOLD \
  --latent_retr_topk $LATENT_RETR_TOPK \
  --latent_dim $LATENT_DIM \
  --retrieval_topk 3 \
  --exp-name MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_after_sa_every2layer_top3_ddpm_cfg_saca_dropout01 \
  --reference_end_latent_path humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/reference_end_latent_msa_vae_t2m_272.npy \
  --disable_ema \
  --ca_every_n_layers $CA_EVERY_N_LAYERS \
  --ca_insertion_mode $CA_INSERTION_MODE \
  --ca_n_head $CA_N_HEAD

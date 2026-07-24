#!/bin/bash
# Train MotionStreamer stage-2 with global + local RAG tokens.
# Usage: bash TRAIN_t2m_rag_local.sh [NUM_GPUS]

NUM_GPUS=${1:-1}
BATCH_SIZE=$((256 / NUM_GPUS))

DATASET=${DATASET:-t2m_272}
TEXT_LATENT_DIR=${TEXT_LATENT_DIR:-./humanml3d_272/text_latents_t5}
HCLS_DIR=${HCLS_DIR:-./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
MOTION_LATENT_DIR=${MOTION_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
Z_LATENT_DIR=${Z_LATENT_DIR:-./humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right}
EMPTY_TEXT_PATH=${EMPTY_TEXT_PATH:-./humanml3d_272/text_latents_t5/empty_text_embedding.npy}
TEXT_EMBED_DIM=${TEXT_EMBED_DIM:-768}

# Local RAG hyper-parameters.
L_LOCAL=${L_LOCAL:-16}
LOCAL_RAG_DIM=${LOCAL_RAG_DIM:-16}
RETRIEVAL_TOPK=${RETRIEVAL_TOPK:-3}

GENERATIVE_HEAD_TYPE=${GENERATIVE_HEAD_TYPE:-ddpm}
ADD_SELFATTEN=${ADD_SELFATTEN:-1}  # set to 1 to enable SA encoding before cross-attn
NUM_FLOW_STEPS=${NUM_FLOW_STEPS:-20}
FLOW_SOLVER=${FLOW_SOLVER:-euler}
RF_TIME_SAMPLING=${RF_TIME_SAMPLING:-uniform}
RF_LOSS_TYPE=${RF_LOSS_TYPE:-mse}

echo "=========================================="
echo "MotionStreamer Stage-2 Local-RAG Training"
echo "=========================================="
echo "GPU count       : $NUM_GPUS"
echo "Batch per GPU   : $BATCH_SIZE"
echo "Total batch     : 256"
echo "Dataset         : $DATASET"
echo "Text latents    : $TEXT_LATENT_DIR"
echo "h_cls latents   : $HCLS_DIR"
echo "Motion latents  : $MOTION_LATENT_DIR"
echo "Z latents      : $Z_LATENT_DIR"
echo "Text embed dim  : $TEXT_EMBED_DIM"
echo "L_local         : $L_LOCAL  (block_size = $((78 + L_LOCAL)))"
echo "local_rag_dim   : $LOCAL_RAG_DIM"
echo "Retrieval top-K : $RETRIEVAL_TOPK"
echo "Head type       : $GENERATIVE_HEAD_TYPE"
echo "add_selfatten   : $ADD_SELFATTEN"

for DIR in "$TEXT_LATENT_DIR" "$HCLS_DIR" "$MOTION_LATENT_DIR" "$Z_LATENT_DIR"; do
  if [ ! -d "$DIR" ]; then
    echo "ERROR: directory not found: $DIR"
    exit 1
  fi
done

accelerate launch --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    train_t2m_rag_local.py \
    --batch-size $BATCH_SIZE \
    --lr 0.0001 \
    --total-iter 100000 \
    --out-dir Experiments \
    --exp-name MotionStreamer_t2m_272_msa_rag_local_L${L_LOCAL}_k${RETRIEVAL_TOPK}_sa_ca \
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
    --rf_loss_type $RF_LOSS_TYPE \
    $([ "$ADD_SELFATTEN" = "1" ] && echo "--add_selfatten")

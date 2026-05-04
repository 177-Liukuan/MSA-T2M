#!/bin/bash
# ============================================================
# TAE-GAN-v1: Decoder-only adversarial fine-tuning of Causal TAE
# Usage:
#   bash TRAIN_tae_gan_v1.sh [NUM_GPUS]   (default: 1)
#
# Key env vars (override before calling):
#   TAE_CKPT        path to pretrained TAE checkpoint
#   DISC_START      steps before GAN loss activates  (default: 10000)
#   DATASET         t2m_272 | t2m_babel_272
# ============================================================

NUM_GPUS=${1:-1}
BATCH_SIZE=$((64 / NUM_GPUS))

DATASET=${DATASET:-t2m_272}

# ── Pretrained TAE checkpoint (REQUIRED) ────────────────────────────────────
TAE_CKPT=${TAE_CKPT:-./Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth}

# ── GAN hyper-params ─────────────────────────────────────────────────────────
DISC_START=${DISC_START:-10000}      # steps of recon-only warmup
DISC_WEIGHT=${DISC_WEIGHT:-0.2}     # adaptive weight scale (lowered: 0.5->0.2 to reduce adversarial gradient)
FM_WEIGHT=${FM_WEIGHT:-10.0}        # feature matching weight
DISC_NDF=${DISC_NDF:-64}            # discriminator base channels
DISC_N_LAYERS=${DISC_N_LAYERS:-3}   # discriminator depth

# ── Learning rates ────────────────────────────────────────────────────────────
LR=${LR:-1e-5}                      # decoder lr (conservative — fine-tune)
LR_DISC=${LR_DISC:-5e-6}            # discriminator lr (lowered: 2e-5->5e-6, must be < decoder lr)

DISC_FREQ=${DISC_FREQ:-3}            # update D every 3 G-steps (G trains 3x faster than D)
DISC_CLIP_GRAD=${DISC_CLIP_GRAD:-1.0} # max grad norm for D (0=disable)

echo "=========================================="
echo "  TAE-GAN-v1 Training"
echo "=========================================="
echo "  GPUs            : $NUM_GPUS"
echo "  Batch/GPU        : $BATCH_SIZE"
echo "  Dataset          : $DATASET"
echo "  TAE checkpoint   : $TAE_CKPT"
echo "  disc_start       : $DISC_START"
echo "  disc_weight      : $DISC_WEIGHT"
echo "  fm_weight        : $FM_WEIGHT"
echo "  lr (decoder)     : $LR"
echo "  lr (disc)        : $LR_DISC"
echo "=========================================="

if [ ! -f "$TAE_CKPT" ]; then
    echo "ERROR: TAE checkpoint not found: $TAE_CKPT"
    echo "Set TAE_CKPT env var to the correct path."
    exit 1
fi

conda run -n mgpt --no-capture-output \
accelerate launch --num_processes $NUM_GPUS \
    train_tae_gan_v1.py \
    --batch-size $BATCH_SIZE \
    --dataname $DATASET \
    --tae-ckpt $TAE_CKPT \
    --lr $LR \
    --lr-disc $LR_DISC \
    --total-iter 200000 \
    --warm-up-iter 1000 \
    --lr-scheduler 150000 \
    --gamma 0.1 \
    --down-t 2 \
    --depth 3 \
    --dilation-growth-rate 3 \
    --latent_dim 16 \
    --hidden_size 1024 \
    --root_loss 7.0 \
    --disc-start $DISC_START \
    --disc-weight $DISC_WEIGHT \
    --fm-weight $FM_WEIGHT \
    --disc-ndf $DISC_NDF \
    --disc-n-layers $DISC_N_LAYERS \
    --disc-freq $DISC_FREQ \
    --disc-clip-grad $DISC_CLIP_GRAD \
    --out-dir Experiments/TAE_GAN_Loss_ \
    --exp-name tae_gan_v1_t2m_272_v2 \
    --print-iter 200 \
    --eval-iter 10000 \
    --num_gpus $NUM_GPUS

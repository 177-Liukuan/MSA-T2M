#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
# TRAIN_qformer_rag.sh — Q-Former RAG 训练启动脚本
#
# 用法（全新训练）:
#   bash TRAIN_qformer_rag.sh [NUM_GPUS] [EXP_NAME]
#   bash TRAIN_qformer_rag.sh 1 QFormer_t2m_272_v3
#
# 用法（断点续训）:
#   RESUME=Experiments/QFormer_t2m_272_v2/net_best_r1.pth \
#   bash TRAIN_qformer_rag.sh 1 QFormer_t2m_272_v3
#
# ── v3 相对 v2 的改动 ──────────────────────────────────────────────
#   --w-mtg 0.0           【核心】完全关闭 MTG 损失
#                          MTG raw loss 在 v2 iter14000 仍=1.508，
#                          即使 w=0.05 也占用梯度带宽，导致 MTC/MTM 收敛变慢。
#   --lr-sched-type cosine 【核心】余弦退火代替 MultiStepLR
#                          v2 iter10500 LR 骤降 3e-5→9e-6 后 R@1 不升反降；
#                          余弦平滑衰减避免突然失去动量。
#   --early-stop-patience 20 【核心】从 8 增到 20
#                          val R@1 波动约 ±0.015（1338 样本计数噪声），
#                          patience=8 在 v2 iter6000 最优后 8k 无改善即触发，
#                          patience=20 给模型足够时间从波谷恢复。
#   --batch-size 512       更多 in-batch negative（511 vs 255），
#                          训练时负样本数更接近 val 评估时的 1337 个
#   --total-iter 80000     给余弦调度足够的衰减空间
#   （已移除 --lr-scheduler / --gamma，cosine 不需要）
#   val deterministic      代码层面已修复：val __getitem__ 固定 cap_idx=idx%n_cap，
#                          消除每次评估因随机 caption 导致的 R@1 波动

set -e

NUM_GPUS=${1:-1}
EXP_NAME=${2:-QFormer_t2m_272_v4}
TAE_CKPT="Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth"

[ -f "${TAE_CKPT}" ] || { echo "[ERROR] TAE checkpoint not found: ${TAE_CKPT}"; exit 1; }
[ -d "humanml3d_272/text_latents_t5" ] || { echo "[ERROR] Text latents not found. Run: bash PREPARE_text_embeddings.sh"; exit 1; }

RESUME_FLAGS=""
[ -n "${RESUME}" ] && RESUME_FLAGS="--resume-pth ${RESUME}"

echo "======================================="
echo "  Q-Former RAG 训练  [${EXP_NAME}]"
echo "  GPUs: ${NUM_GPUS}   Effective batch: $((256 * NUM_GPUS))"
[ -n "${RESUME}" ] && echo "  Resume: ${RESUME}"
echo "======================================="

accelerate launch --num_processes ${NUM_GPUS} \
  -m explorations.qformer.train_qformer_rag \
  --dataname          t2m_272 \
  --data-root         ./humanml3d_272 \
  --text-latent-dir   ./humanml3d_272/text_latents_t5 \
  --tae-ckpt          ${TAE_CKPT} \
  --queue-size 4096 \
  --num-queries       16 \
  --query-dim         768 \
  --motion-dim        1024 \
  --num-layers        6 \
  --num-heads         8 \
  --dropout           0.1 \
  --text-emb-dim      768 \
  --t5-model-path     sentencet5-xxl/ \
  --max-text-len      64 \
  --max-motion-len    300 \
  --batch-size        256 \
  --num-workers       8 \
  --grad-accum        1 \
  --t5-cache-batch    256 \
  --lr                3e-5 \
  --weight-decay      1e-4 \
  --warm-up-iter      500 \
  --total-iter        80000 \
  --lr-sched-type     cosine \
  --lr-eta-min-ratio  0.01 \
  --w-mtc             2.0 \
  --w-mtm             1.0 \
  --w-mtg             0.1 \
  --early-stop-patience 20 \
  --out-dir           Experiments \
  --exp-name          ${EXP_NAME} \
  --print-iter        100 \
  --eval-iter         1000 \
  --save-iter         10000 \
  --num-gpus          ${NUM_GPUS} \
  ${RESUME_FLAGS}

echo "======================================="
echo "  训练完成: Experiments/${EXP_NAME}/"
echo "======================================="

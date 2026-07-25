#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# =====================================================
# 使用预计算文本嵌入的T2M训练脚本
# 
# 使用方法:
#   bash TRAIN_t2m_cached.sh [GPU_数量]
#   
# 例子:
#   bash TRAIN_t2m_cached.sh 8        # 使用8个GPU
#   bash TRAIN_t2m_cached.sh          # 默认1个GPU
# =====================================================

NUM_GPUS=${1:-1}  # default: 1 GPU
BATCH_SIZE=$((256 / NUM_GPUS))

echo "=========================================="
echo "T2M 训练 (使用预计算文本嵌入)"
echo "=========================================="
echo "GPU数量: $NUM_GPUS"
echo "每个GPU的批大小: $BATCH_SIZE"
echo "总批大小: 256"
echo ""

# 检查预计算的文本嵌入是否存在
TEXT_LATENT_DIR="./humanml3d_272/text_latents_t5"
if [ ! -d "$TEXT_LATENT_DIR" ]; then
    echo "❌ 错误: 文本嵌入目录不存在: $TEXT_LATENT_DIR"
    echo ""
    echo "请先运行准备脚本:"
    echo "  python get_text_latent_t5.py"
    echo ""
    exit 1
fi

# 检查文本嵌入文件数量
TEXT_FILES=$(find "$TEXT_LATENT_DIR" -name "*.npy" | wc -l)
echo "✓ 找到 $TEXT_FILES 个文本嵌入文件"
echo ""

accelerate launch --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    -m explorations.motionstreamer_baselines.train_t2m_cached \
    --batch-size $BATCH_SIZE \
    --lr 0.0001 \
    --total-iter 100000 \
    --out-dir Experiments/explorations/motionstreamer_baselines \
    --exp-name MotionStreamer_t2m_272_cached_embeddings_8gpu_bf16 \
    --dataname t2m_272 \
    --latent_dir humanml3d_272/t2m_latents/causal_TAE_t2m_272_h100_20260203 \
    --num_gpus $NUM_GPUS

#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# =====================================================
# 准备文本嵌入脚本
# 
# 功能:
#   - 加载T5模型
#   - 处理所有训练数据的文本
#   - 计算并保存文本嵌入到磁盘
# 
# 用时: ~10-20分钟（取决于数据大小）
# 输出: ./humanml3d_272/text_latents/
# =====================================================

echo "=========================================="
echo "准备文本嵌入"
echo "=========================================="
echo ""

# 检查T5模型是否存在
if [ ! -d "sentencet5-xxl" ]; then
    echo "❌ 错误: 未找到T5模型目录: sentencet5-xxl/"
    echo ""
    echo "请确保T5模型已下载到: sentencet5-xxl/"
    echo ""
    exit 1
fi

echo "✓ 检测到T5模型"
echo ""
echo "开始处理文本嵌入..."
echo ""

python prepare_text_embeddings.py \
    --dataset-name t2m_272 \
    --output-dir ./humanml3d_272/text_latents_t5 \
    --t5-model-path sentencet5-xxl/ \
    --batch-size 64

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 文本嵌入准备完成!"
    echo "=========================================="
    echo ""
    echo "现在可以运行训练:"
    echo "  bash TRAIN_t2m_cached.sh 8"
    echo ""
else
    echo ""
    echo "❌ 文本嵌入准备失败!"
    echo ""
    exit 1
fi

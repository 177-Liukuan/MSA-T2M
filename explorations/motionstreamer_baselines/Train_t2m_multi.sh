#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# ================= 配置区域 =================
# 1. 设置总卡数 (4节点 * 2卡 = 8)
TOTAL_NUM_GPUS=8
# 2. 设置节点数 (Replicas)
NUM_MACHINES=4
# 3. 设置单节点卡数
GPUS_PER_NODE=2
# ===========================================

echo ">>> [Init] 分布式环境自动配置中..."

# --- 1. 获取 Master IP (主节点地址) ---
if [ -z "$VC_WORKER_HOSTS" ]; then
    echo "Error: 环境变量 VC_WORKER_HOSTS 未找到，请确保 Role 名称为 'worker'"
    exit 1
fi
MASTER_ADDR=$(echo $VC_WORKER_HOSTS | awk -F , '{print $1}')

# 端口固定 (建议注释单独写一行，防止解析错误)
MASTER_PORT=29500

# --- 2. 获取 Rank (当前节点序号) ---
NODE_RANK=$VC_TASK_INDEX

if [ -z "$NODE_RANK" ]; then
    echo "Error: 环境变量 VC_TASK_INDEX 未找到"
    exit 1
fi

echo "------------------------------------------------"
echo "Master Addr  : $MASTER_ADDR"
echo "Master Port  : $MASTER_PORT"
echo "Machine Rank : $NODE_RANK (of $NUM_MACHINES)"
echo "Total GPUs   : $TOTAL_NUM_GPUS"
echo "Mixed Prec   : bf16 (H100 Optimized)"
echo "------------------------------------------------"

# --- 3. 计算 Batch Size ---
# 逻辑：总 Batch 256 / 总卡数 8 = 单卡 32
BATCH_SIZE=$((256 / TOTAL_NUM_GPUS))

# --- 4. 启动 Accelerate ---
# 修正点：
# 1. 使用 --mixed_precision="bf16" 规范格式
# 2. 确保每行末尾的反斜杠后没有任何空格
# 3. 显式指定 python 解释器 (可选，但更稳妥)

accelerate launch \
    --num_processes $TOTAL_NUM_GPUS \
    --num_machines $NUM_MACHINES \
    --machine_rank $NODE_RANK \
    --mixed_precision="bf16" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --num_cpu_threads_per_process 8 \
    -m explorations.motionstreamer_baselines.train_t2m \
    --batch-size $BATCH_SIZE \
    --lr 0.0001 \
    --total-iter 100000 \
    --out-dir Experiments/explorations/motionstreamer_baselines \
    --exp-name MotionStreamer_8gpus_distributed_mp \
    --dataname t2m_272 \
    --latent_dir humanml3d_272/t2m_latents/causal_TAE_t2m_272_h100_20260203 \
    --num_gpus $TOTAL_NUM_GPUS

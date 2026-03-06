# NUM_GPUS=${1:-1}  # default: 1 GPU
# dataset_name=${2:-t2m_272} # default: t2m_272, options: t2m_272, t2m_babel_272

# BATCH_SIZE=$((128 / NUM_GPUS))

# echo "Using $NUM_GPUS GPUs, each with a batch size of $BATCH_SIZE"

# # accelerate launch --num_processes $NUM_GPUS -m ipdb train_causal_TAE.py \
# accelerate launch --num_processes $NUM_GPUS --mixed_precision bf16 train_causal_TAE.py \
# --batch-size $BATCH_SIZE \
# --lr 0.00005 \
# --total-iter 2000000 \
# --lr-scheduler 1900000 \
# --down-t 2 \
# --depth 3 \
# --dilation-growth-rate 3 \
# --out-dir Experiments \
# --dataname $dataset_name \
# --exp-name causal_TAE_${dataset_name}_h100_20260205 \
# --root_loss 7.0 \
# --latent_dim 16 \
# --hidden_size 1024 \
# --num_gpus $NUM_GPUS


#!/bin/bash

NUM_GPUS=${1:-1}
NUM_NODES=2
GPUS_PER_NODE=4

dataset_name=${2:-t2m_272}

BATCH_SIZE=$((128 / NUM_GPUS))

echo "========== Distributed Training =========="
echo "World size      : $NUM_GPUS"
echo "Nodes           : $NUM_NODES"
echo "GPU per node    : $GPUS_PER_NODE"
echo "Batch per GPU   : $BATCH_SIZE"
echo "========================================="

accelerate launch \
  --num_processes $NUM_GPUS \
  --num_machines $NUM_NODES \
  --machine_rank $VC_TASK_INDEX \
  --main_process_ip $VC_WORKER_HOSTS_0 \
  --main_process_port 29500 \
  --mixed_precision bf16 \
  train_causal_TAE.py \
  --batch-size $BATCH_SIZE \
  --lr 0.00005 \
  --total-iter 2000000 \
  --lr-scheduler 1900000 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --out-dir Experiments \
  --dataname $dataset_name \
  --exp-name causal_TAE_${dataset_name}_8gpu \
  --root_loss 7.0 \
  --latent_dim 16 \
  --hidden_size 1024 \
  --num_gpus $NUM_GPUS

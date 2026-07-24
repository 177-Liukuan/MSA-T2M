#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
# Training launcher script

NUM_GPUS=${1:-4}
EXP_NAME=${2:-exp1}

echo "Starting training with $NUM_GPUS GPUs"
python train_t2m_msa.py --num_gpus $NUM_GPUS --exp_name $EXP_NAME

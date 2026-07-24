#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
# -----------------------------------------------------------
#  SAE-v1 evaluation script
#
#  Usage:
#    bash EVAL_sae_v1.sh <checkpoint_path>
#    e.g.  bash EVAL_sae_v1.sh ../Experiments/SAE_v1_t2m_272/net_best_mpjpe.pth
#
#  Metrics: FID, MPJPE (mm), first-5-frame jitter (mm)
# -----------------------------------------------------------
# Create symlinks expected by the evaluator (idempotent)
ln -sf ../utils        ./Evaluator_272/utils        2>/dev/null || true
ln -sf ../humanml3d_272 ./Evaluator_272/humanml3d_272 2>/dev/null || true
ln -sf ../options      ./Evaluator_272/options      2>/dev/null || true
ln -sf ../models       ./Evaluator_272/models       2>/dev/null || true
ln -sf ../visualization ./Evaluator_272/visualization 2>/dev/null || true

python -m explorations.representation_experiments.eval_sae_v1 \
  --resume-pth ../Experiments/SAE_v1_t2m_272/net_best_mpjpe.pth \
  --out-dir Evaluator_272/output \
  --exp-name SAE_v1_eval \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --latent_dim 16 \
  --hidden_size 1024

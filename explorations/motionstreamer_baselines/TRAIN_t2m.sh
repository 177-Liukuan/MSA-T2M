SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

NUM_GPUS=${1:-1}  # default: 1 GPU

BATCH_SIZE=$((256 / NUM_GPUS))

echo "Using $NUM_GPUS GPUs, each with a batch size of $BATCH_SIZE"

accelerate launch --num_processes $NUM_GPUS -m explorations.motionstreamer_baselines.train_t2m \
--batch-size $BATCH_SIZE \
--lr 0.0001 \
--total-iter 100000 \
--out-dir Experiments \
--exp-name MotionStreamer_vae_causal_TAE_t2m_272_h100_20260203_t2m_h100_20260206 \
--dataname t2m_272 \
--latent_dir humanml3d_272/t2m_latents/causal_TAE_t2m_272_h100_20260203 \
--num_gpus $NUM_GPUS


# python get_latent.py --resume-pth Experiments/causal_TAE_t2m_272_h100_20260203/net_last.pth --latent_dir humanml3d_272/t2m_latents/causal_TAE_t2m_272_h100_20260203
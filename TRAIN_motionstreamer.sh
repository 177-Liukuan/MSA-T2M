NUM_GPUS=${1:-1}  # default: 1 GPU

BATCH_SIZE=$((256 / NUM_GPUS))

echo "Using $NUM_GPUS GPUs, each with a batch size of $BATCH_SIZE"

accelerate launch --num_processes $NUM_GPUS train_motionstreamer.py \
--batch-size $BATCH_SIZE \
--lr 0.0001 \
--total-iter 100000 \
--out-dir Experiments \
--exp-name motionstreamer_model_causal_TAE_t2m_babel_272_h100_20260205_20260209 \
--dataname t2m_babel_272 \
--latent_dir babel_272_stream/t2m_babel_latents/causal_TAE_t2m_babel_272_h100_20260205 \
--num_gpus $NUM_GPUS


# python get_latent.py --resume-pth Experiments/causal_TAE_t2m_babel_272_h100_20260205/net_last.pth --latent_dir babel_272_stream/t2m_babel_latents/causal_TAE_t2m_babel_272_h100_20260205 --dataname t2m_babel_272
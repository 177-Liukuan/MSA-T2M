ln -sf ../utils ./Evaluator_272/
ln -sf ../humanml3d_272 ./Evaluator_272/
ln -sf ../options ./Evaluator_272/
ln -sf ../models ./Evaluator_272/
ln -sf ../visualization ./Evaluator_272/

# Usage: bash EVAL_msa_vae.sh <checkpoint_path_relative_to_Evaluator_272>
# Example:
# bash EVAL_msa_vae.sh ../Experiments/MotionStreamer_vae_xxx/net_best_fid.pth
python eval_msa_vae.py \
--resume-pth ../Experiments/MSA_VAEv5_phase2_t2m_272_iter2000_α0/net_best_mpjpe.pth \
--trans_enc_layers 6 \
--trans_dec_layers 6 \
--trans_ff_size 2048 \

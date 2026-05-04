ln -s ../utils ./Evaluator_272/
ln -s ../humanml3d_272 ./Evaluator_272/
ln -s ../options ./Evaluator_272/
ln -s ../models ./Evaluator_272/
ln -s ../visualization ./Evaluator_272/
# python eval_causal_TAE.py --resume-pth ../Experiments/causal_TAE_t2m_272_h100*1/net_best_mpjpe.pth # FID. 0.4800, mpjpe. 20.95000 (mm)

python eval_causal_TAE.py --resume-pth ../Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth  # FID. 0.5004, mpjpe. 21.96341 (mm)

# python eval_causal_TAE.py --resume-pth ../Experiments/TAE_GAN_Loss_/tae_gan_v1_t2m_272_op/net_best_mpjpe.pth  # FID. 0.4960, mpjpe. 22.13312 (mm)
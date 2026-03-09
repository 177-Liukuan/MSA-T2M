ln -s ../utils ./Evaluator_272/
ln -s ../humanml3d_272 ./Evaluator_272/
ln -s ../options ./Evaluator_272/
ln -s ../models ./Evaluator_272/
ln -s ../visualization ./Evaluator_272/
# python eval_causal_TAE.py --resume-pth ../Causal_TAE/net_last.pth # FID. 0.6462, mpjpe. 22.68629 (mm)
# python eval_causal_TAE.py --resume-pth ../Experiments/causal_TAE_t2m_272_4090*4/net_last.pth # FID. 0.6676, mpjpe. 23.75903 (mm)
# python eval_causal_TAE.py --resume-pth ../Experiments/causal_TAE_t2m_272_4090*4/net_best_mpjpe.pth # FID. 0.7171, mpjpe. 23.29394 (mm)
# python eval_causal_TAE.py --resume-pth ../Experiments/causal_TAE_t2m_272_h100*1/net_best_mpjpe.pth # FID. 0.4800, mpjpe. 20.95000 (mm)
# python eval_causal_TAE.py --resume-pth ../Experiments/causal_TAE_t2m_272_h100*1/net_last.pth # FID. 0.5048, mpjpe. 20.81398 (mm)
# python eval_causal_TAE.py --resume-pth ../Experiments/MSA_VAE_phase2_t2m_272/net_last.pth
# python eval_causal_TAE.py --resume-pth ../Experiments/MSA_VAE_phase2_t2m_272/net_best_mpjpe.pth

python eval_causal_TAE.py --resume-pth ../Experiments/MSA_VAEv2_phase2_t2m_272_phase1_step300000_weightdown/net_last.pth
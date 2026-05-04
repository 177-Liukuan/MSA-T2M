ln -s ../utils ./Evaluator_272/ 2>/dev/null || true
ln -s ../humanml3d_272 ./Evaluator_272/ 2>/dev/null || true
ln -s ../options ./Evaluator_272/ 2>/dev/null || true
ln -s ../models ./Evaluator_272/ 2>/dev/null || true
ln -s ../visualization ./Evaluator_272/ 2>/dev/null || true
ln -s ../Causal_TAE ./Evaluator_272/ 2>/dev/null || true
python eval_t2m_rag.py \
  --trans_nhead 8 \
  --trans_enc_layers 6 \
  --trans_dec_layers 6 \
  --trans_ff_size 2048 \
  --resume-pth ../Experiments/MSA_VAEv5_phase2_t2m_272_iter2000_α0/net_last.pth \
  --resume-trans ../Experiments/MotionStreamer_t2m_272_msa_rag/latest.pth \
  --exp-name MotionStreamer_t2m_272_msa_rag_eval

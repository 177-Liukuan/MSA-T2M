ln -s ../utils ./Evaluator_272/ 2>/dev/null || true
ln -s ../humanml3d_272 ./Evaluator_272/ 2>/dev/null || true
ln -s ../options ./Evaluator_272/ 2>/dev/null || true
ln -s ../models ./Evaluator_272/ 2>/dev/null || true
ln -s ../visualization ./Evaluator_272/ 2>/dev/null || true
ln -s ../Causal_TAE ./Evaluator_272/ 2>/dev/null || true
python eval_t2m_clip_baseline.py \
  --resume-pth ../Experiments/causal_TAE_t2m_272_h100_20260203/net_last.pth \
  --resume-trans ../Experiments/MotionStreamer_t2m_272_baseline_clip/latest.pth \
  --exp-name MotionStreamer_t2m_272_baseline_clip_eval

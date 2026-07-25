SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

ln -s ../utils ./Evaluator_272/
ln -s ../humanml3d_272 ./Evaluator_272/
ln -s ../options ./Evaluator_272/
ln -s ../models ./Evaluator_272/
ln -s ../visualization ./Evaluator_272/
ln -s ../Causal_TAE ./Evaluator_272/
python -m explorations.motionstreamer_baselines.eval_t2m --resume-pth ../Experiments/causal_TAE_t2m_272_h100_20260203/net_last.pth --resume-trans Experiments/explorations/clip/MotionStreamer_t2m_272_baseline_clip/latest.pth --exp-name MotionStreamer_t2m_272_baseline_clip

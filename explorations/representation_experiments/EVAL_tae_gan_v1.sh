SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

ln -s ../utils ./Evaluator_272/
ln -s ../humanml3d_272 ./Evaluator_272/
ln -s ../options ./Evaluator_272/
ln -s ../models ./Evaluator_272/
ln -s ../visualization ./Evaluator_272/

CKPT=${1:-Experiments/explorations/representation_experiments/TAE_GAN_Loss_/tae_gan_v1_t2m_272_v2/net_best_mpjpe.pth}

python eval_causal_TAE.py --resume-pth $CKPT

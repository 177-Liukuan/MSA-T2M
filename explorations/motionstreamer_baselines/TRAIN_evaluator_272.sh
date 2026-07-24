SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export HF_ENDPOINT=https://hf-mirror.com
cd Evaluator_272
huggingface-cli download --resume-download distilbert/distilbert-base-uncased --local-dir ./deps/distilbert-base-uncased
# ln -s ../humanml3d_272 ./datasets/humanml3d_272
# 关键修改：增加 ../ 并加上 -f 强制覆盖
ln -sf ../../humanml3d_272 ./datasets/humanml3d_272
python -m train --cfg configs/configs_evaluator_272/H3D-TMR.yaml --cfg_assets configs/assets.yaml --batch_size 256 --nodebug
cd ..
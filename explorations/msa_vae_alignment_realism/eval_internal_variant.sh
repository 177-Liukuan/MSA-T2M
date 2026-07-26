#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 SLUG GPU CHECKPOINT" >&2
    exit 2
fi

SLUG=$1
EVALUATION_GPU=$2
CHECKPOINT=$3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PILOT_ROOT=${PILOT_ROOT:-"${REPO_ROOT}/Experiments/msa_vae_alignment_realism_pilot_s123_20260726"}
OUTPUT_DIR="${PILOT_ROOT}/evaluation_internal/${SLUG}"
STATUS_DIR="${PILOT_ROOT}/status"
STATUS_FILE="${STATUS_DIR}/${SLUG}.internal_evaluation.status"

mkdir -p "$STATUS_DIR"

write_status() {
    local state=$1
    local detail=${2:-}
    local temporary="${STATUS_FILE}.tmp.$$"
    {
        printf 'state=%s\n' "$state"
        printf 'variant=%s\n' "$SLUG"
        printf 'gpu=%s\n' "$EVALUATION_GPU"
        printf 'pid=%s\n' "$$"
        printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
        printf 'detail=%s\n' "$detail"
        printf 'checkpoint=%s\n' "$CHECKPOINT"
        printf 'output_dir=%s\n' "$OUTPUT_DIR"
    } > "$temporary"
    mv -f "$temporary" "$STATUS_FILE"
}

on_exit() {
    local code=$?
    if [[ $code -ne 0 ]]; then
        write_status \
            "internal_evaluation_failed" \
            "runner exited with code ${code}"
    fi
}
trap on_exit EXIT

if [[ ! -s "$CHECKPOINT" ]]; then
    echo "Internal evaluation checkpoint is missing: ${CHECKPOINT}" >&2
    exit 10
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Fresh internal evaluation output already exists: ${OUTPUT_DIR}" >&2
    exit 11
fi

export CUDA_VISIBLE_DEVICES="$EVALUATION_GPU"
write_status "internal_evaluation_running"
cd "$REPO_ROOT"

if [[ -n ${EVAL_COMMAND:-} ]]; then
    command=("$EVAL_COMMAND")
elif [[ ${CONDA_DEFAULT_ENV:-} == mgpt ]]; then
    command=(python "${REPO_ROOT}/eval_msa_vae_alignment.py")
else
    command=(
        conda run --no-capture-output -n mgpt
        python "${REPO_ROOT}/eval_msa_vae_alignment.py"
    )
fi

"${command[@]}" \
    "$CHECKPOINT" \
    --data-root humanml3d_272 \
    --split-file humanml3d_272/split/test.txt \
    --global-text-embed-dir humanml3d_272/text_latents_t5 \
    --local-split-file humanml3d_272/split/train_ft.txt \
    --local-text-embed-dir humanml3d_272/t5_enc_single \
    --local-target-scope in-sample \
    --evaluator-root Evaluator_272 \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --batch-size 32 \
    --num-workers 8 \
    --seed 123

if [[ ! -s "${OUTPUT_DIR}/metrics.json" ]]; then
    echo "Internal metrics.json is missing for ${SLUG}" >&2
    exit 12
fi
write_status "internal_evaluation_complete"
echo "MSA-VAE pilot internal evaluation complete: ${SLUG}"

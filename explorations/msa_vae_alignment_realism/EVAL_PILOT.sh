#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PILOT_ROOT=${PILOT_ROOT:-"${REPO_ROOT}/Experiments/msa_vae_alignment_realism_pilot_s123_20260726"}
SCREEN_BIN=${SCREEN_BIN:-screen}
NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-nvidia-smi}
PILOT_DRY_RUN=${PILOT_DRY_RUN:-0}
PILOT_CLI="${SCRIPT_DIR}/pilot.py"
RUNNER="${SCRIPT_DIR}/eval_variant.sh"

cd "$REPO_ROOT"

pilot_python() {
    if [[ ${CONDA_DEFAULT_ENV:-} == mgpt ]]; then
        python "$PILOT_CLI" --output-root "$PILOT_ROOT" "$@"
    else
        conda run -n mgpt python "$PILOT_CLI" \
            --output-root "$PILOT_ROOT" "$@"
    fi
}

weight_name() {
    local value=$1
    printf '%s' "${value//./p}"
}

mapfile -t contract_lines < <(pilot_python contract --format tsv)
pilot_python verify >/dev/null
screen_listing=$("$SCREEN_BIN" -ls 2>&1 || true)

for line in "${contract_lines[@]}"; do
    IFS=$'\t' read -r slug label gpu_pair eval_gpu training_session \
        p1_global p1_local p2_global p2_local <<< "$line"
    status_file="${PILOT_ROOT}/status/${slug}.status"
    if [[ ! -f "$status_file" ]] || ! grep -qx 'state=complete' "$status_file"; then
        echo "Training is not complete for ${slug}" >&2
        exit 3
    fi
    eval_session="msa_eval_${slug}_s123"
    if grep -Eq "[.]${eval_session}([[:space:]]|$)" <<< "$screen_listing"; then
        echo "Evaluation Screen already exists: ${eval_session}" >&2
        exit 3
    fi
    if [[ -e "${PILOT_ROOT}/evaluation/${slug}" ]]; then
        echo "Evaluation output already exists for ${slug}" >&2
        exit 3
    fi
done

compute_processes=$(
    "$NVIDIA_SMI_BIN" \
        --query-compute-apps=gpu_uuid,pid,used_memory \
        --format=csv,noheader,nounits
)
if [[ -n "${compute_processes//[[:space:]]/}" ]]; then
    echo "GPU compute process detected; refusing evaluation launch" >&2
    exit 4
fi

if [[ "$PILOT_DRY_RUN" == 1 ]]; then
    echo "Dry run: no evaluation sessions will be created."
fi

export PILOT_ROOT
for line in "${contract_lines[@]}"; do
    IFS=$'\t' read -r slug label gpu_pair eval_gpu training_session \
        p1_global p1_local p2_global p2_local <<< "$line"
    p2_name="${slug}_s123_phase2_25k_g$(weight_name "$p2_global")_l$(weight_name "$p2_local")"
    checkpoint="${PILOT_ROOT}/${p2_name}/net_last.pth"
    eval_session="msa_eval_${slug}_s123"
    log_file="${PILOT_ROOT}/logs/${slug}.evaluation.screen.log"
    if [[ "$PILOT_DRY_RUN" == 1 ]]; then
        echo "SCREEN ${eval_session} GPU=${eval_gpu}: ${RUNNER} ${slug} ${eval_gpu} ${checkpoint}"
    else
        "$SCREEN_BIN" -L -Logfile "$log_file" -dmS "$eval_session" \
            bash "$RUNNER" "$slug" "$eval_gpu" "$checkpoint"
        echo "Started ${eval_session}: ${label}, GPU ${eval_gpu}"
    fi
done

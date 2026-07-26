#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PILOT_ROOT=${PILOT_ROOT:-"${REPO_ROOT}/Experiments/msa_vae_alignment_realism_pilot_s123_20260726"}
SCREEN_BIN=${SCREEN_BIN:-screen}
NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-nvidia-smi}
PILOT_DRY_RUN=${PILOT_DRY_RUN:-0}
PILOT_CLI="${SCRIPT_DIR}/pilot.py"
RUNNER="${SCRIPT_DIR}/eval_internal_variant.sh"

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

internal_session() {
    case "$1" in
        no_align)
            printf '%s' "msa_internal_eval_no_align_s123"
            ;;
        global_only)
            printf '%s' "msa_internal_eval_global_only_s123"
            ;;
        local_only)
            printf '%s' "msa_internal_eval_local_only_s123"
            ;;
        global_local)
            printf '%s' "msa_internal_eval_global_local_s123"
            ;;
        *)
            echo "Unknown pilot variant: $1" >&2
            return 2
            ;;
    esac
}

expected_eval_gpu() {
    case "$1" in
        no_align) printf '%s' "0" ;;
        global_only) printf '%s' "2" ;;
        local_only) printf '%s' "4" ;;
        global_local) printf '%s' "6" ;;
        *) return 2 ;;
    esac
}

if ! command -v "$SCREEN_BIN" >/dev/null 2>&1; then
    echo "GNU Screen is unavailable: ${SCREEN_BIN}" >&2
    exit 2
fi
if ! command -v "$NVIDIA_SMI_BIN" >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable: ${NVIDIA_SMI_BIN}" >&2
    exit 2
fi

mapfile -t contract_lines < <(
    pilot_python contract --format tsv | sed '/^[[:space:]]*$/d'
)
if [[ ${#contract_lines[@]} -ne 4 ]]; then
    echo "Pilot contract must contain exactly four variants" >&2
    exit 2
fi
for line in "${contract_lines[@]}"; do
    if [[ $(awk -F '\t' '{print NF}' <<< "$line") -ne 10 ]]; then
        echo "Pilot contract row must contain exactly ten fields" >&2
        exit 2
    fi
done

# This validates both Phase-1/Phase-2 metadata and the exact parent lineage.
pilot_python verify >/dev/null
screen_listing=$("$SCREEN_BIN" -ls 2>&1 || true)
for line in "${contract_lines[@]}"; do
    IFS=$'\t' read -r slug label gpu_pair eval_gpu training_session \
        main_process_port p1_global p1_local p2_global p2_local <<< "$line"
    expected_gpu=$(expected_eval_gpu "$slug")
    if [[ "$eval_gpu" != "$expected_gpu" ]]; then
        echo "Evaluation GPU mapping mismatch for ${slug}" >&2
        exit 2
    fi
    session=$(internal_session "$slug")
    if grep -Eq "[.]${session}([[:space:]]|$)" <<< "$screen_listing"; then
        echo "Internal evaluation Screen already exists: ${session}" >&2
        exit 3
    fi
    if [[ -e "${PILOT_ROOT}/evaluation_internal/${slug}" ]]; then
        echo "Internal evaluation output already exists for ${slug}" >&2
        exit 3
    fi
done

compute_processes=$(
    "$NVIDIA_SMI_BIN" \
        --query-compute-apps=gpu_uuid,pid,used_memory \
        --format=csv,noheader,nounits
)
if [[ -n "${compute_processes//[[:space:]]/}" ]]; then
    echo "GPU compute process detected; refusing internal evaluation" >&2
    exit 4
fi

gpu_count=0
while IFS=',' read -r index memory_used memory_total utilization; do
    index=${index//[[:space:]]/}
    memory_used=${memory_used//[[:space:]]/}
    [[ -z "$index" ]] && continue
    gpu_count=$((gpu_count + 1))
    if [[ "$memory_used" -gt 100 ]]; then
        echo "GPU ${index} is busy: ${memory_used} MiB used" >&2
        exit 4
    fi
done < <(
    "$NVIDIA_SMI_BIN" \
        --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
)
if [[ $gpu_count -ne 8 ]]; then
    echo "Expected exactly eight visible GPUs, found ${gpu_count}" >&2
    exit 4
fi

if [[ "$PILOT_DRY_RUN" == 1 ]]; then
    echo "Dry run: no internal evaluation sessions will be created."
    for line in "${contract_lines[@]}"; do
        IFS=$'\t' read -r slug label gpu_pair eval_gpu training_session \
            main_process_port p1_global p1_local p2_global p2_local <<< "$line"
        p2_name="${slug}_s123_phase2_25k_g$(weight_name "$p2_global")_l$(weight_name "$p2_local")"
        checkpoint="${PILOT_ROOT}/${p2_name}/net_last.pth"
        session=$(internal_session "$slug")
        echo "SCREEN ${session} GPU=${eval_gpu}: ${RUNNER} ${slug} ${eval_gpu} ${checkpoint}"
    done
    exit 0
fi

mkdir -p "$PILOT_ROOT/logs" "$PILOT_ROOT/status"
export PILOT_ROOT
launch_failures=0
for line in "${contract_lines[@]}"; do
    IFS=$'\t' read -r slug label gpu_pair eval_gpu training_session \
        main_process_port p1_global p1_local p2_global p2_local <<< "$line"
    p2_name="${slug}_s123_phase2_25k_g$(weight_name "$p2_global")_l$(weight_name "$p2_local")"
    checkpoint="${PILOT_ROOT}/${p2_name}/net_last.pth"
    session=$(internal_session "$slug")
    log_file="${PILOT_ROOT}/logs/${slug}.internal_evaluation.screen.log"
    if "$SCREEN_BIN" -L -Logfile "$log_file" -dmS "$session" \
            bash "$RUNNER" "$slug" "$eval_gpu" "$checkpoint"; then
        echo "Started ${session}: ${label}, GPU ${eval_gpu}"
    else
        echo "Failed to start ${session}: ${label}" >&2
        launch_failures=$((launch_failures + 1))
    fi
done
if [[ $launch_failures -ne 0 ]]; then
    echo "${launch_failures} internal evaluation Screen session(s) failed" >&2
    exit 5
fi

echo "Status: bash ${SCRIPT_DIR}/STATUS_PILOT.sh"
echo "Collect: conda run -n mgpt python ${PILOT_CLI} collect-internal"

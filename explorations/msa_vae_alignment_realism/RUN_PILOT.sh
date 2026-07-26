#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PILOT_ROOT=${PILOT_ROOT:-"${REPO_ROOT}/Experiments/msa_vae_alignment_realism_pilot_s123_20260726"}
TAE_CHECKPOINT=${TAE_CHECKPOINT:-"${REPO_ROOT}/Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth"}
TAE_SHA256=${TAE_SHA256:-7c92115aeb36c71f93baa381869ae35f391e7d4dc2b51fe2b8c6761bf352bdd8}
SCREEN_BIN=${SCREEN_BIN:-screen}
NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-nvidia-smi}
SHA256SUM_BIN=${SHA256SUM_BIN:-sha256sum}
PILOT_DRY_RUN=${PILOT_DRY_RUN:-0}
RUNNER="${SCRIPT_DIR}/run_variant.sh"
PILOT_CLI="${SCRIPT_DIR}/pilot.py"

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

if ! command -v "$SCREEN_BIN" >/dev/null 2>&1; then
    echo "GNU Screen is unavailable: ${SCREEN_BIN}" >&2
    exit 2
fi
if ! command -v "$NVIDIA_SMI_BIN" >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable: ${NVIDIA_SMI_BIN}" >&2
    exit 2
fi
if [[ ! -f "$TAE_CHECKPOINT" ]]; then
    echo "Fixed TAE checkpoint is missing: ${TAE_CHECKPOINT}" >&2
    exit 2
fi
actual_tae_sha=$("$SHA256SUM_BIN" -- "$TAE_CHECKPOINT" | awk '{print $1}')
if [[ "$actual_tae_sha" != "$TAE_SHA256" ]]; then
    echo "Fixed TAE SHA-256 mismatch" >&2
    exit 2
fi

screen_listing=$("$SCREEN_BIN" -ls 2>&1 || true)
mapfile -t contract_lines < <(pilot_python contract --format tsv)
if [[ ${#contract_lines[@]} -ne 4 ]]; then
    echo "Pilot contract must contain exactly four variants" >&2
    exit 2
fi

for line in "${contract_lines[@]}"; do
    IFS=$'\t' read -r slug label gpu_pair eval_gpu session \
        p1_global p1_local p2_global p2_local <<< "$line"
    if grep -Eq "[.]${session}([[:space:]]|$)" <<< "$screen_listing"; then
        echo "Screen session already exists: ${session}" >&2
        exit 3
    fi
    p1_name="${slug}_s123_phase1_25k_g$(weight_name "$p1_global")_l$(weight_name "$p1_local")"
    p2_name="${slug}_s123_phase2_25k_g$(weight_name "$p2_global")_l$(weight_name "$p2_local")"
    if [[ -e "${PILOT_ROOT}/${p1_name}" || -e "${PILOT_ROOT}/${p2_name}" ]]; then
        echo "Fresh-run target already exists for ${slug}" >&2
        exit 3
    fi
done

compute_processes=$(
    "$NVIDIA_SMI_BIN" \
        --query-compute-apps=gpu_uuid,pid,used_memory \
        --format=csv,noheader,nounits
)
if [[ -n "${compute_processes//[[:space:]]/}" ]]; then
    echo "GPU compute process detected; refusing to launch:" >&2
    echo "$compute_processes" >&2
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
    echo "Dry run: no directories or Screen sessions will be created."
    for line in "${contract_lines[@]}"; do
        IFS=$'\t' read -r slug label gpu_pair eval_gpu session \
            p1_global p1_local p2_global p2_local <<< "$line"
        echo "SCREEN ${session} GPUs=${gpu_pair}: ${RUNNER} ${slug} ${gpu_pair} ${p1_global} ${p1_local} ${p2_global} ${p2_local}"
    done
    exit 0
fi

mkdir -p "$PILOT_ROOT/logs" "$PILOT_ROOT/status"
contract_tmp="${PILOT_ROOT}/contract.json.tmp.$$"
pilot_python contract --format json > "$contract_tmp"
mv -f "$contract_tmp" "${PILOT_ROOT}/contract.json"

export PILOT_ROOT TAE_CHECKPOINT TAE_SHA256
for line in "${contract_lines[@]}"; do
    IFS=$'\t' read -r slug label gpu_pair eval_gpu session \
        p1_global p1_local p2_global p2_local <<< "$line"
    log_file="${PILOT_ROOT}/logs/${slug}.screen.log"
    "$SCREEN_BIN" -L -Logfile "$log_file" -dmS "$session" \
        bash "$RUNNER" "$slug" "$gpu_pair" \
        "$p1_global" "$p1_local" "$p2_global" "$p2_local"
    echo "Started ${session}: ${label}, GPUs ${gpu_pair}"
done

echo "Status: bash ${SCRIPT_DIR}/STATUS_PILOT.sh"
echo "Attach: screen -r <session>; detach with Ctrl-A then D"

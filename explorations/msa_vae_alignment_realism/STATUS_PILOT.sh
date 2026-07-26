#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PILOT_ROOT=${PILOT_ROOT:-"${REPO_ROOT}/Experiments/msa_vae_alignment_realism_pilot_s123_20260726"}
SCREEN_BIN=${SCREEN_BIN:-screen}
NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-nvidia-smi}

cd "$REPO_ROOT"

echo "=== Screen sessions ==="
"$SCREEN_BIN" -ls 2>&1 || true
echo
echo "=== Variant status ==="
for slug in no_align global_only local_only global_local; do
    echo "--- ${slug} ---"
    if [[ -f "${PILOT_ROOT}/status/${slug}.status" ]]; then
        cat "${PILOT_ROOT}/status/${slug}.status"
    else
        echo "state=not_started"
    fi
    if [[ -f "${PILOT_ROOT}/status/${slug}.evaluation.status" ]]; then
        cat "${PILOT_ROOT}/status/${slug}.evaluation.status"
    fi
    log_file="${PILOT_ROOT}/logs/${slug}.screen.log"
    if [[ -f "$log_file" ]]; then
        echo "log_tail:"
        tail -n 8 "$log_file"
    fi
done
echo
echo "=== GPUs ==="
"$NVIDIA_SMI_BIN" \
    --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
echo
echo "=== Disk ==="
df -h "$(dirname "$PILOT_ROOT")"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 SLUG GPU_PAIR P1_GLOBAL P1_LOCAL P2_GLOBAL P2_LOCAL" >&2
    exit 2
fi

SLUG=$1
GPU_PAIR=$2
P1_GLOBAL=$3
P1_LOCAL=$4
P2_GLOBAL=$5
P2_LOCAL=$6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PILOT_ROOT=${PILOT_ROOT:-"${REPO_ROOT}/Experiments/msa_vae_alignment_realism_pilot_s123_20260726"}
TAE_CHECKPOINT=${TAE_CHECKPOINT:-"${REPO_ROOT}/Experiments/causal_TAE_t2m_272_h100_20260203/net_best_mpjpe.pth"}
TAE_SHA256=${TAE_SHA256:-7c92115aeb36c71f93baa381869ae35f391e7d4dc2b51fe2b8c6761bf352bdd8}
PHASE1_LAUNCHER=${PHASE1_LAUNCHER:-"${REPO_ROOT}/TRAIN_msa_vae_phase1.sh"}
PHASE2_LAUNCHER=${PHASE2_LAUNCHER:-"${REPO_ROOT}/TRAIN_msa_vae_phase2.sh"}

cd "$REPO_ROOT"

weight_name() {
    local value=$1
    printf '%s' "${value//./p}"
}

P1_NAME="${SLUG}_s123_phase1_25k_g$(weight_name "$P1_GLOBAL")_l$(weight_name "$P1_LOCAL")"
P2_NAME="${SLUG}_s123_phase2_25k_g$(weight_name "$P2_GLOBAL")_l$(weight_name "$P2_LOCAL")"
P1_DIR="${PILOT_ROOT}/${P1_NAME}"
P2_DIR="${PILOT_ROOT}/${P2_NAME}"
STATUS_DIR="${PILOT_ROOT}/status"
STATUS_FILE="${STATUS_DIR}/${SLUG}.status"

mkdir -p "$STATUS_DIR"

write_status() {
    local state=$1
    local detail=${2:-}
    local temporary="${STATUS_FILE}.tmp.$$"
    {
        printf 'state=%s\n' "$state"
        printf 'variant=%s\n' "$SLUG"
        printf 'gpu_pair=%s\n' "$GPU_PAIR"
        printf 'pid=%s\n' "$$"
        printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
        printf 'detail=%s\n' "$detail"
        printf 'phase1_dir=%s\n' "$P1_DIR"
        printf 'phase2_dir=%s\n' "$P2_DIR"
    } > "$temporary"
    mv -f "$temporary" "$STATUS_FILE"
}

on_exit() {
    local code=$?
    if [[ $code -ne 0 ]]; then
        write_status "failed" "runner exited with code ${code}"
    fi
}
trap on_exit EXIT

if [[ -e "$P1_DIR" || -e "$P2_DIR" ]]; then
    echo "Fresh-run violation: target phase directory already exists" >&2
    exit 3
fi

export CUDA_VISIBLE_DEVICES="$GPU_PAIR"

write_status "phase1_running"
if ! env \
    OUT_DIR="$PILOT_ROOT" \
    EXP_NAME="$P1_NAME" \
    SEED=123 \
    TOTAL_ITER=25000 \
    EVAL_ITER=5000 \
    VALIDATION_SEED=123 \
    VALIDATION_BATCH_SIZE=32 \
    GLOBAL_ALIGN_WEIGHT="$P1_GLOBAL" \
    LOCAL_ALIGN_WEIGHT="$P1_LOCAL" \
    CNN_CKPT="$TAE_CHECKPOINT" \
    CNN_CKPT_SHA256="$TAE_SHA256" \
    TEXT_ENCODER_TYPE=t5 \
    bash "$PHASE1_LAUNCHER" 2 t2m_272; then
    echo "Phase 1 launcher failed for ${SLUG}" >&2
    exit 20
fi
if [[ ! -s "${P1_DIR}/net_last.pth" ]]; then
    echo "Phase 1 net_last.pth is missing for ${SLUG}" >&2
    exit 21
fi
write_status "phase1_complete"

write_status "phase2_running"
if ! env \
    OUT_DIR="$PILOT_ROOT" \
    PHASE1_DIR="$P1_DIR" \
    EXP_NAME="$P2_NAME" \
    SEED=123 \
    TOTAL_ITER=25000 \
    EVAL_ITER=5000 \
    VALIDATION_SEED=123 \
    VALIDATION_BATCH_SIZE=32 \
    GLOBAL_ALIGN_WEIGHT="$P2_GLOBAL" \
    LOCAL_ALIGN_WEIGHT="$P2_LOCAL" \
    CNN_CKPT="$TAE_CHECKPOINT" \
    CNN_CKPT_SHA256="$TAE_SHA256" \
    TEXT_ENCODER_TYPE=t5 \
    bash "$PHASE2_LAUNCHER" 2 t2m_272; then
    echo "Phase 2 launcher failed for ${SLUG}" >&2
    exit 30
fi
if [[ ! -s "${P2_DIR}/net_last.pth" ]]; then
    echo "Phase 2 net_last.pth is missing for ${SLUG}" >&2
    exit 31
fi

write_status "complete"
echo "MSA-VAE pilot variant complete: ${SLUG}"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "Usage: bash EVAL_msa_vae_metrics.sh <checkpoint.pth> [evaluation options]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" == "mgpt" ]]; then
  exec python "$SCRIPT_DIR/eval_msa_vae_metrics.py" "$@"
fi

exec conda run -n mgpt python "$SCRIPT_DIR/eval_msa_vae_metrics.py" "$@"

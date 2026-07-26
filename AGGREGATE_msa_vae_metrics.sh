#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" == "mgpt" ]]; then
    exec python "${SCRIPT_DIR}/aggregate_msa_vae_metrics.py" "$@"
fi

exec conda run -n mgpt python "${SCRIPT_DIR}/aggregate_msa_vae_metrics.py" "$@"

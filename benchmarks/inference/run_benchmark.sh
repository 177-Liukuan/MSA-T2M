#!/usr/bin/env bash
set -euo pipefail

# Usage: bash benchmarks/inference/run_benchmark.sh --method msa_t2m [options]
# MSA-T2M/MotionStreamer use mgpt; ReMoDiffuse uses mogen.
method=""
for arg in "$@"; do
  case "$arg" in
    --method=*) method="${arg#*=}" ;;
  esac
done
if [[ -z "$method" ]]; then
  echo "--method is required (msa_t2m, motionstreamer, remodiffuse)" >&2
  exit 2
fi
env_name="mgpt"
[[ "$method" == "remodiffuse" ]] && env_name="mogen"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec conda run -n "$env_name" python -m benchmarks.inference.run_benchmark "$@" \
  --output "${BENCHMARK_OUTPUT:-$repo_root/benchmark_results/inference}"

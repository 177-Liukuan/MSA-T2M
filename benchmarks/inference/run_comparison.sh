#!/usr/bin/env bash
set -euo pipefail

# Runs the three methods with identical deterministic captions and lengths.
# ReMoDiffuse is intentionally run in mogen; the two MSA-family models use mgpt.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
common=(--data-root "${BENCHMARK_DATA_ROOT:-$root/humanml3d_272}" \
  --output "${BENCHMARK_OUTPUT:-$root/benchmark_results/inference}" \
  --device "${BENCHMARK_DEVICE:-cuda:0}" --num-runs "${BENCHMARK_RUNS:-20}" \
  --warmups "${BENCHMARK_WARMUPS:-2}" --seed "${BENCHMARK_SEED:-42}")

conda run -n mgpt python -m benchmarks.inference.run_benchmark --method msa_t2m "${common[@]}"
conda run -n mgpt python -m benchmarks.inference.run_benchmark --method motionstreamer "${common[@]}"
conda run -n mogen python -m benchmarks.inference.run_benchmark --method remodiffuse "${common[@]}"

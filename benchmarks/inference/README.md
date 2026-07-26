# Inference-cost benchmark

Run the matched comparison from the repository root:

```bash
bash benchmarks/inference/run_comparison.sh
```

The launcher uses `mgpt` for MSA-T2M and MotionStreamer and `mogen` for
ReMoDiffuse. Defaults are one RTX 4090 (`cuda:0`), seed 42, two warmups and 20
measured prompts at 60/120/196 frames. MSA-T2M and MotionStreamer also receive
the 300-frame extension; ReMoDiffuse is capped at 196 frames.

Each method writes `samples.jsonl`, `summary.json`, `summary.csv`,
`table_main.tex`, and `table_streaming.tex` under a timestamped directory.
Model/database loading, rendering, and file I/O are outside the timed region.
Use `--manifest-only` to inspect the deterministic prompt manifest before a
GPU run and `--allow-busy-gpu` only when a shared GPU run is intentional.

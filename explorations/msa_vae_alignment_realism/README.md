# MSA-VAE Alignment–Realism Pilot

This exploration runs the approved single-seed reviewer pilot:

| Variant | Phase 1 global/local | Phase 2 global/local | Training GPUs |
|---|---:|---:|---|
| No Alignment | 0 / 0 | 0 / 0 | 0,1 |
| Global Only | 0.2 / 0 | 0.05 / 0 | 2,3 |
| Local Only | 0 / 0.2 | 0 / 0.05 | 4,5 |
| Global + Local | 0.2 / 0.2 | 0.05 / 0.05 | 6,7 |

Every phase uses 25,000 iterations, seed 123, complete deterministic
validation every 5,000 iterations, the fixed TAE, and exactly two GPUs per
training group.

## Launch and monitor

```bash
PILOT_DRY_RUN=1 \
bash explorations/msa_vae_alignment_realism/RUN_PILOT.sh

bash explorations/msa_vae_alignment_realism/RUN_PILOT.sh

bash explorations/msa_vae_alignment_realism/STATUS_PILOT.sh
```

If a Screen command failed before creating that variant's phase directory,
retry only the missing variant after the GPUs become idle:

```bash
PILOT_ONLY_VARIANT=global_only \
bash explorations/msa_vae_alignment_realism/RUN_PILOT.sh
```

Launch attempts are appended to `launch_status.tsv`. Commands, lifecycle
timestamps, exit codes, and final checkpoint hashes are stored in
`run_manifest.json`.

Attach to one training session:

```bash
screen -r msa_pilot_global_s123
```

Detach without stopping it using `Ctrl-A`, then `D`.

## Evaluate and collect

After all four training status files report `state=complete`:

```bash
bash explorations/msa_vae_alignment_realism/EVAL_PILOT.sh

bash explorations/msa_vae_alignment_realism/STATUS_PILOT.sh

conda run -n mgpt python \
  explorations/msa_vae_alignment_realism/pilot.py collect
```

The table is written under:

```text
Experiments/msa_vae_alignment_realism_pilot_s123_20260726/summary/
```

This pilot has one seed and must not be reported with a standard deviation.

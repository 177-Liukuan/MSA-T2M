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

## Evaluate internal alignment and realism

After all four training status files report `state=complete`:

```bash
PILOT_DRY_RUN=1 \
bash explorations/msa_vae_alignment_realism/EVAL_INTERNAL_PILOT.sh

bash explorations/msa_vae_alignment_realism/EVAL_INTERNAL_PILOT.sh

bash explorations/msa_vae_alignment_realism/STATUS_PILOT.sh

conda run -n mgpt python \
  explorations/msa_vae_alignment_realism/pilot.py collect-internal
```

The internal table, No-Alignment deltas, and four Pareto plots are written
under:

```text
Experiments/msa_vae_alignment_realism_pilot_s123_20260726/summary/
```

The main internal protocol measures `global_proj(h_cls)` and `local_proj(mu)`
directly against the SentenceT5 targets used for training. Reconstruction uses
the posterior mean (`mu`) and reports FID, MPJPE, P-MPJPE, ACCEL, and skating.

The current local cache covers `train_ft.txt`, so
`in_sample_local_cosine` is a training-set diagnostic and is not held-out. It
must not be placed in the final paper's held-out `Local Cosine` column. The
pilot has one seed and must not be reported with a standard deviation or a
claim of statistical significance.

## Supplementary external-TMR preservation

The older frozen-TMR retrieval table remains a supplementary test of whether
the reconstructed motion is recognized by an external model. It is not the
internal MSA-VAE alignment metric. Its historical artifacts under
`evaluation/` are preserved and can still be collected with:

```bash
bash explorations/msa_vae_alignment_realism/EVAL_PILOT.sh

conda run -n mgpt python \
  explorations/msa_vae_alignment_realism/pilot.py collect
```

New internal artifacts are isolated under `evaluation_internal/`; neither
workflow overwrites the other.

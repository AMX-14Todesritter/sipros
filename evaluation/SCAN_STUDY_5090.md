# Time-bounded RTX 5090 scan-scaling task

Prerequisites: converted X1/X2/X3 under output/evaluation/inputs/ecoli_x{1,2,3},
full Marine FASTA, original Regular.cfg, and the verified cache-off multi reader
in output/evaluation/build/gpu_multi/bin/sipros_mvh_cuda. That build must target
CUDA architecture 120; the historical 52 build is never used here.

Synchronize these runtime files before running:

- evaluation/scripts/scan_study_5090.py
- evaluation/scripts/scaling_inputs.py
- evaluation/scripts/batch_study.py

The task fixes generated-peptide batch size at 4M and uses real, nested,
source-stratified samples of 20k, 40k, 60k, 80k, 100k, 120k and 140,297 MS2 scans.
Every level contains proportional X1/X2/X3 contributions. Only S-line scan IDs
are remapped; peak lines and precursor lines are unchanged. Maps retain source,
original ID and original/remapped record hashes. Seed is fixed at 20261028.

Source verification and input preparation precede the measurement budget.
Prepared nested inputs occupy roughly 10–12 GB in addition to original FT2s;
results/PSMs require additional space. Inputs are reused only after hash checks.

From the project root, inside tmux:

```bash
docker compose run --rm -e NVIDIA_DRIVER_CAPABILITIES=all \
  sipros python3 -B evaluation/scripts/scan_study_5090.py \
    --id scan_5090_multi_marine_4m_v1 \
    --hours 6.5 --repeats 3 --batch 4000000 \
    --backends cuda rt-instanced rt-custom
```

There are at most 7 sizes × 3 backends × 3 repeats = 63 measured processes.
There are no added warmup processes and no profiler. Each round covers every
sample size before another round begins, with rotated sample/backend order.
This is initial scaling coverage with three repetitions, not a definitive
precision estimate for small speedups.

The code stops dispatch at the UTC deadline and lets the active native process
finish; total wall time can exceed 6.5 hours by preparation plus one process.
No exact completion-time guarantee is implied. Device free memory below 3 GiB
or host process RSS above 55 GiB triggers the child runner guard and stops the
entire task. Sampling cannot guarantee interception of transient memory peaks.
Failures are preserved and no automatic batch reduction occurs.

Global live reports:

```text
output/evaluation/scan_studies/scan_5090_multi_marine_4m_v1/progress.md
output/evaluation/scan_studies/scan_5090_multi_marine_4m_v1/summary.csv
output/evaluation/scan_studies/scan_5090_multi_marine_4m_v1/measurements.csv
output/evaluation/scan_studies/scan_5090_multi_marine_4m_v1/manifest.json
```

Child progress is under output/evaluation/batch_studies/<ID>_s<count>_r<round>/;
raw measurements are under output/evaluation/results/scan_scaling/.

Resume a deadline pause with another time window:

```bash
docker compose run --rm -e NVIDIA_DRIVER_CAPABILITIES=all \
  sipros python3 -B evaluation/scripts/scan_study_5090.py \
    --id scan_5090_multi_marine_4m_v1 --resume --hours 6.5
```

Completed cases do not rerun. Experimental settings come from the saved plan;
use a new ID to change them. Failed cases require inspection; do not resume them
blindly. For manual safe pausing, create PAUSE in the master study directory to
finish the active child study (up to three backends), or in its active child
batch-study directory to finish only the current backend before pausing.

## Inspected infrastructure failures

After verifying GPU access in a fresh container, `--resume --retry-failed` on
`scan_study_5090.py` permits explicit retry of failed child jobs. Completed
measurements are retained; new attempts use new directories. Do not change
`run.py`, inputs or binaries merely to recover a container/NVML failure, because
content fingerprint changes intentionally block reuse of a frozen study.

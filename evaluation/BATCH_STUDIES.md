# Configurable batch studies

Run `evaluation/scripts/batch_study.py` inside the project container. It delegates
measurements to `run.py`; it never builds or changes matching/scoring algorithms.
The chosen binary determines single- versus multi-precursor input semantics.
`--precursor-mode multi` requires matching build metadata and binary hash from
`prepare_batch_binary.py`; the flag itself does not change reader behavior.
Theoretical-ion caching must be disabled. Cache counters, workload totals, and
same-backend PSM hashes are checked; scientific validation failures always stop.

## Example: single-precursor capacity, 16M to 24M

First finish the small migration smoke test. Use a fresh study ID for each new
study. Run from the project root on the measurement machine:

```bash
docker compose run --rm \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  sipros python3 -B evaluation/scripts/batch_study.py \
    --id capacity_5090_primary_16m_24m \
    --scans output/evaluation/inputs/ecoli_x1/Pan_062822_X1iso5.FT2 \
    --fasta raw/Marine_fw_3rev.fasta \
    --config experiments/Regular.cfg \
    --gpu-binary output/evaluation/build/gpu/bin/sipros_mvh_cuda \
    --batches 16000000 20000000 24000000 \
    --backends cuda rt-custom \
    --repeats 1 --warmups 0 \
    --min-device-free-gib 3 \
    --max-host-rss-gib 55 \
    --failure-policy stop
```

Batch sizes count generated peptide entries. `24000000` means 24 million, not
2400. Defaults are CUDA and RT-custom, one measured repetition, zero warmups,
ascending batch order, a 3-GiB free-device-memory guard, and no host RSS guard.
No CPU or profiler is launched. Resource guards are sampled; they cannot prevent
every transient allocation failure or system crash. A guard-triggered process is
terminated and its measurement is failed, rather than reported as successful.
For unattended capacity checks, retain conservative guards and a finite list.

## Options and order

- `--batches`: positive, unique values; ascending mode sorts them numerically.
- `--backends`: CUDA, RT-instanced, RT-custom; default excludes RT-instanced.
- `--repeats`: independent measured processes per batch/backend.
- `--warmups`: excluded independent processes before the first measurement of
  each cell; these are not same-context hot-kernel warmups.
- `--order ascending`: finish smaller batches first; rotate backend order across
  repetitions. Recommended for capacity checks.
- `--order round-robin`: visit every batch once per round and rotate batch and
  backend order. Useful for repeated performance studies.
- `--failure-policy stop`: stop immediately after a failed case (default).
- `--failure-policy continue`: skip future repeats of the failed batch/backend
  cell, then continue other cells. Do not use for unattended pressure escalation.
- `--failure-policy skip-larger`: after a child capacity/run failure, skip all
  remaining jobs at that batch size and larger, across backends. It does not
  diagnose an OOM from generic errors; inspect saved native logs for the cause.
- `--dry-run`: create the plan and reports without GPU execution. Start that plan
  later using `--resume`.
- `--authorize-paused`: if the legacy pause manifest exists, authorize only each
  exact child command in this explicit study. Unrelated suites remain paused.

Use `--help` for full argument descriptions. Do not run two coordinators
concurrently. The shared coordinator lock prevents this script from doing so;
each child also uses the existing measurement lock.

## Results and live progress

The coordinator updates these files at job transitions and approximately every
five seconds while waiting for a child:

```text
output/evaluation/batch_studies/<ID>/
  manifest.json        settings, fingerprints, statuses, commands, attempts
  progress.md          live job status, partial batch count, memory sample, table
  summary.csv          median, mean, min, max, sample stdev, memory and PSM checks
  measurements.csv     one row per successful measured process
  <child-ID>.log       child runner output, including infrastructure errors
```

Detailed logs, memory JSONL, copied scripts/configuration, PSMs and run manifests
remain under `output/evaluation/results/batch_size/<child-ID>/`. The coordinator
manifest links every attempt to that directory. Only validated completed jobs
enter aggregate tables. Warmups and failed/partial processes are excluded.

Complete scoring is the sum of the same run's host packing, GPU service and host
restoration. RT setup is already included. Fused scoring uses ordinary CUDA
event intervals, not profiler-isolated hot-kernel latency. GPU memory values are
sampled whole-device usage. A one-run capacity check is not a robust performance
comparison; increase repetitions separately for small timing differences.

## Safe pause and resume

Create a pause request from the remote project root:

```bash
touch output/evaluation/batch_studies/capacity_5090_primary_16m_24m/PAUSE
```

The active child case finishes (including its configured warmups), then dispatch
stops. SIGINT/SIGTERM sent to the coordinator itself also request stopping after
the active case; force-stopping Docker can still interrupt children. Prefer the
pause file and wait until `progress.md` says paused.

Remove the pause request and resume with saved settings:

```bash
rm output/evaluation/batch_studies/capacity_5090_primary_16m_24m/PAUSE
docker compose run --rm -e NVIDIA_DRIVER_CAPABILITIES=all \
  sipros python3 -B evaluation/scripts/batch_study.py \
    --id capacity_5090_primary_16m_24m --resume
```

Completed jobs never rerun. Interrupted jobs retry in new `_a2`, `_a3`, ...
directories, retaining earlier evidence. A still-live previous child blocks
resume. Failed jobs remain failed by default; use `--resume --retry-failed` to
retry them explicitly, with the saved resource guards. Pending jobs may resume
without retrying failures. Jobs marked skipped by a failure policy stay skipped;
create a new study to revisit them. Resume uses saved experiment settings;
changing input paths, batch values, backends, repetition counts or resource
limits requires a new study ID. Inputs, configuration, binary and runner content
fingerprints must match. Existing study directories are never overwritten.

## GPU-free coordinator checks

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s evaluation/tests -p test_batch_study.py
```

These tests use a synthetic child process to exercise orchestration, not matching
algorithms or GPU performance.

## Both precursor modes, 2M through 24M

Build isolated binaries once. The multi build restores only the three historical
FT2 reader method bodies from commit `82522a4`; it preserves current signatures,
matching/scoring kernels, geometry, and the compile-time cache-off baseline.
The full Git history must be available. Build metadata records source and binary
hashes and the selected mode. Existing build directories are not overwritten.

```bash
docker compose run --rm -e NVIDIA_DRIVER_CAPABILITIES=all sipros bash -lc '
  set -e
  python3 -B evaluation/scripts/prepare_batch_binary.py --precursor-mode primary
  python3 -B evaluation/scripts/prepare_batch_binary.py --precursor-mode multi
'
```

After small input validation, execute the two distinct studies sequentially:

```bash
docker compose run --rm -e NVIDIA_DRIVER_CAPABILITIES=all sipros bash -lc '
  set -e
  for mode in primary multi; do
    python3 -B evaluation/scripts/batch_study.py \
      --id "capacity_5090_${mode}_2m_24m" \
      --precursor-mode "$mode" \
      --scans output/evaluation/inputs/ecoli_x1/Pan_062822_X1iso5.FT2 \
      --fasta raw/Marine_fw_3rev.fasta \
      --config experiments/Regular.cfg \
      --gpu-binary "output/evaluation/build/gpu_${mode}/bin/sipros_mvh_cuda" \
      --batches 2000000 4000000 6000000 8000000 10000000 12000000 \
                14000000 16000000 18000000 20000000 22000000 24000000 \
      --backends cuda rt-custom --repeats 1 --warmups 0 \
      --min-device-free-gib 3 --max-host-rss-gib 55 --failure-policy stop
  done
'
```

There are 24 measured jobs per mode, 48 total if all fit. Any failure stops the
outer shell too; larger multi-precursor cases are not guaranteed to fit. Each
mode has its own `progress.md`. Old laptop 12M cancellation remains in force for
that old study; this example is a separately authorized 5090 capacity study.

## Controlled CUDA compilation-target comparison

Use `prepare_batch_binary.py --precursor-mode multi --cuda-architectures 52
--build-name gpu_multi_sm52` to create a separate build without replacing the
existing `gpu_multi` architecture-120 build. Check reader, engine and protected
source hashes match before timing. Architecture 52 is a historical compilation
control, not a recommended native target for Blackwell. The main CUDA objects
change compilation target; OptiX shader PTX retains its existing target and is
compiled by the driver. Hold source, inputs, options, cache policy and event
boundaries constant. Use distinct study IDs and compare both targets on the same
GPU before attributing cross-machine speedup changes solely to hardware.

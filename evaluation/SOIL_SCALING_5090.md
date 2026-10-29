# Unattended Soil scaling: all X1+X2 scans, fixed 8M batch

Sync `evaluation/scripts/soil_scaling_5090.py` before starting. Existing run.py
must include cross-study PSM deduplication and the disk reserve check. Reuse the
verified `output/evaluation/build/gpu_multi` sm120/multi/cache-off binary.
The existing batch_study.py must support category database_scaling.

Fixed workload: every original X1 and X2 MS2 spectrum, 46,066 + 46,862 = 92,928
physical scans. The script creates `output/evaluation/inputs/x1_x2_full/` with
combined FT2, source/original/global scan mapping and record hashes. Peak and
precursor lines are preserved; S-line IDs are globally remapped. This is a
new fixed scan workload and must not be pooled directly with earlier 140,297
scan measurements.

Matrix: Soil family fractions 1, 5, 10, 25, 50, 100%; CUDA and RT-custom;
three rounds, 36 measurements total. The first round covers every fraction;
later rounds rotate fraction/backend order. These are all new measurements,
not reuse of old X1+X2+X3 pilots. Cache off, sm120, no profiler or CPU, zero extra
warmup processes. Batch counts eight million generated peptide entries.

There is NO wall-clock or dispatch deadline. The queue continues until all jobs
finish or an error/resource guard stops it. It does not require intermediate
operator input. Six to eight hours is only a planning estimate from prior
pilots, not a completion guarantee. Full Soil behavior is not established by
1%/5% pilots, so longer runtime is possible.

On the remote project root:

```bash
docker compose run -d --name sipros-5090-soil-x1x2-v1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  sipros bash -lc '
    set -e
    nvidia-smi
    exec python3 -u -B evaluation/scripts/soil_scaling_5090.py \
      --id soil_5090_x1x2_8m_v1 --repeats 3 \
      > output/evaluation/soil_5090_x1x2_8m_v1.log 2>&1
  '
```

One global report, including all cells, active child details and errors:

```text
output/evaluation/soil_studies/soil_5090_x1x2_8m_v1/progress.md
output/evaluation/soil_studies/soil_5090_x1x2_8m_v1/measurements.csv
output/evaluation/soil_studies/soil_5090_x1x2_8m_v1/summary.csv
output/evaluation/soil_studies/soil_5090_x1x2_8m_v1/manifest.json
```

Guards retain 3 GiB device memory, cap sampled host RSS at 55 GiB and require
10 GiB free disk before every process. Sampling cannot intercept every transient
allocation. Any failure stops future dispatch without silently reducing batch
size. Native failures, logs and attempts are retained. Never launch another GPU
experiment concurrently.

The original Soil input and target/decoy-preserving nested subsets are verified
and reused. The original full FASTA supplies the 100% cell. Content-identical
PSMs share disk blocks across processes but every original result path remains.
Inputs, binary, config and runner fingerprints are frozen; changing them requires
a new study ID. Reports check candidate totals and same-backend PSM hashes across
rounds within each fraction. Three rounds do not establish statistical significance
of small differences; selected points can be extended in later studies.

After an external interruption, resume with `--id soil_5090_x1x2_8m_v1 --resume`
in a fresh container. Completed cases are not repeated. After inspecting an actual
failure, explicitly add `--retry-failed`; it creates new attempts and retains old
failed logs. This is optional recovery after returning, not a required mid-run
operation. For manual safe pause, create PAUSE in the master directory or its
active child directory; no pause file is created by default.

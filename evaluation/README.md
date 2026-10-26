# SiprosRT evaluation suite

Source and reusable configuration live here. **All generated files must live under
`output/evaluation/`**, including inventories, manifests, logs, prepared inputs,
build directories, profiler captures, tables and figures. Run inside the existing
`sipros-sipros-1` container at `/workspace/sipros`.

Primary comparison: fused scoring for `cuda`, RT V1 `rt-instanced`, RT V2
`rt-custom`. No matching-only experiments or rt-triangle benchmarks. CPU is a
correctness/application reference; exclude it from Soil real-world runs and both
scaling experiments. Do not compare CPU search time against GPU kernel time.

Start with `python3 -B evaluation/scripts/inventory.py`. See `experiment_plan.md`.
Existing production algorithms are not changed by this suite. Diagnostic runs
and profiler replay are excluded from baseline performance statistics.

## Commands

```sh
python3 -B evaluation/scripts/inventory.py
python3 -B evaluation/scripts/prepare.py recover ecoli_x2 ecoli_x3 human
python3 -B evaluation/scripts/prepare.py pilot --scans 256
python3 -B evaluation/scripts/scaling_inputs.py soil --seed 20261026
python3 -B evaluation/scripts/scaling_inputs.py scans --seed 20261026
python3 -B evaluation/scripts/run.py --id UNIQUE_ID --category pilot \
  --scans output/evaluation/inputs/ecoli_x1_first256/spectra.FT2 \
  --fasta raw/Ecoli.fasta --backends cpu cuda rt-instanced rt-custom \
  --batch 2000000 --warmups 1 --repeats 5
python3 -B evaluation/scripts/summarize.py
```

The runner rejects CPU for Soil and scaling and uses an exclusive lock. Do not
run manual GPU jobs concurrently. `--verify`/`--score-impact` runs are diagnostics,
not performance measurements. A strict verification failure is retained and
must not be hidden by deleting/overwriting its directory.

`MVH_ENABLE_EVALUATION_TIMING=ON` is an optional build flag added for this suite.
`fused_device_seconds` uses CUDA events after RT Params upload and before/after
OptiX launch. It still includes launch submission gaps on the GPU timeline;
profiler kernel timestamps are a distinct measurement. The old `kernel_seconds`
interval is preserved. Diagnostic/audit modes can execute extra reference kernels
and their timings are excluded from baseline statistics.

Main experiment entry point (inspect a plan before executing):

```sh
python3 -B evaluation/scripts/suite.py --stage real_world --workloads ecoli_x1 \
  --id UNIQUE_SUITE --batch 2000000
# Add --execute to run. Other stages: batch_size, database_scaling, scan_scaling.
```

Soil scaling execution is gated by a documented resource pilot. No Soil CPU runs
are generated. Exact-equal PSM exports within a run share hard links to save disk;
outputs are immutable after completion. Do not edit individual exports in place.
For plotting use `PYTHONPATH=output/evaluation/dependencies` and
`MPLCONFIGDIR=output/evaluation/cache/matplotlib`.

Rebuild reproducibly inside the container:

```sh
python3 -B evaluation/scripts/build.py --kind timing
python3 -B evaluation/scripts/build.py --kind cpu
# Separate diagnostic/profiler binaries; their timings are excluded:
python3 -B evaluation/scripts/build.py --kind flow
python3 -B evaluation/scripts/build.py --kind profile
```

Avoid builds/data preparation during measured runs. OptiX SDK and runtime are
currently reused from build/mvh_rt; their paths are recorded in build/run manifests.

## Complete persistent serial queue

The authoritative complete matrix and live status are
`output/evaluation/manifests/all_experiments.json` and
`output/evaluation/reports/queue_progress.md`.

```sh
# Create a fresh plan (refuses to overwrite an existing plan).
python3 -B evaluation/scripts/full_queue.py
# Start it, or resume after the controlling process has stopped.
python3 -B evaluation/scripts/full_queue.py --resume --execute
```

The queue reuses validated X1/X2/X3 results, runs family-preserving full-spectrum
resource probes, then the remaining real-world, batch, Soil DB, scan-scaling,
independent work-count and fused peak/score diagnostics, NSYS/NCU and analysis.
Marine2M is reused as batch B1; Soil100% is reused for real-world Soil. These are
explicit aliases of the same five measurements, not extra independent repeats.

A failed backend does not prevent other backends or independent jobs. Failed
resource probes remain evidence; excluded backends/reasons are recorded. No batch
value is silently reduced. The configured host-RSS guard is 18 GiB and the free
storage reserve is 32 GiB. There is no arbitrary default runtime cutoff for the
large Soil cases. Source/templates are frozen; changing them stops the queue
before mixing versions. Resume never overwrites completed or interrupted runs.

Peak identity diagnostics execute in the original fused scoring path, using an
isolated source overlay under output/evaluation/build/peak_audit_source. Added
observers never supply the class used by scoring. Exact mismatch counts cover
all queries; examples are capped at 128 per batch and include original FT2 peak
indices/spectrum records after mapping. Boundary tests also execute MVH; none of
these are standalone matching performance experiments. Diagnostic timings are
excluded from all baseline statistics. FDR-filtered biological accuracy remains
outside claims unless a comparable downstream pipeline is validated.

## Active scan policy: x1_fixed_v2

All fixed-load experiments use Pan X1 (46,066 scans), including Human/Marine/Soil
FASTA comparisons, batch sensitivity, Soil database scaling, profiling and their
diagnostics. Pan X2/X3 are used only to construct the scan-scaling series together
with X1. Native Human/Marine/Soil spectra are excluded. Legacy runs and reports
remain archived; active summaries enforce this scan policy.

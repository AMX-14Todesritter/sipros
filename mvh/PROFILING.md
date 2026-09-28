# CPU baseline profiling

Run from WSL, using the existing container:

```bash
bash mvh/run_profile.sh --dataset marine --threads 4
```

The wrapper runs the existing `build/mvh/bin/sipros_mvh` without rebuilding or
changing its computation. Four threads match the CPU setting in the benchmark
suite; the baseline executable itself defaults to one thread when `-t` is absent.
The CPU baseline batches **2,000,000 assigned peptides**, which is different from
the GPU program's batch limit on generated peptides.

Options: `--dataset smoke|ecoli|marine`, `--threads N`, `--frequency HZ` (default
100), and `--output NEW_CONTAINER_DIRECTORY`. Default results go to a timestamped
directory under `output/benchmarks/cpu_profiling/`.

## Prerequisites and isolation

The current container has `/usr/lib/x86_64-linux-gnu/libprofiler.so.0` from
gperftools 2.9.1. Its matching official pprof script is stored in
`build/tools/cpu_profile/pprof`, downloaded from:

https://raw.githubusercontent.com/gperftools/gperftools/gperftools-2.9.1/src/pprof

Only the child search process receives `LD_PRELOAD` and `CPUPROFILE` variables.
The standalone profiler library does not replace the allocator with tcmalloc.
Ordinary baseline runs do not load it. No root privilege or perf hardware event
support is needed by the sampler. The current container does not support
`perf_event_open` for NSYS CPU sampling.

## Reports

- `ANALYSIS.md` and `analysis.json`: wall-time summary and disjoint sampled-stack
  categories. Regenerate with `python3 mvh/analyze_cpu_profile.py OUTPUT_DIRECTORY`.

- `results/run_summary.tsv`: application wall times for load, preprocessing,
  search and export; thread count, batch limit and PSM count.
- `flat.txt`: sampled CPU time attributed to each function itself.
- `cumulative.txt`: sampled CPU time including callees. Parent and child rows
  overlap and must not be added together.
- `stacks.folded`: sampled stacks, suitable for flame graph generation and
  attribution of allocation/free work to its callers.
- `callgrind.out`: sampled call graph for compatible viewers. These are sampled
  costs, not actual function invocation counts.
- `process_samples.csv`: approximately one-second observations of cumulative
  process CPU seconds, RSS, VmHWM and thread count. CPU seconds include workers
  and may exceed elapsed wall time. Sampling can miss short-lived memory peaks.
- `manifest.json`: inputs/binary/tool hashes, exact command, report exit codes,
  observed process resource use and final PSM hash. `complete` is set only after
  successful search and report generation.
- `search.log` and `cpu.prof`: original console output and raw sampling profile.

Function percentages describe the distribution of **sampled active CPU time**,
not percentages of total elapsed wall time. Waiting and scheduling delays need
separate analysis. Optimized/inlined code can be attributed to its enclosing
function. Profiling has some overhead; use an ordinary run for absolute speed
comparisons. A short smoke run can legitimately collect zero CPU samples.

The 2026-09-25 validation used a separate four-thread busy-loop probe to confirm
worker sampling and symbol resolution, followed by an end-to-end smoke run.

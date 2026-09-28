# Clean RT/CUDA build and timing contract

The `rtMVH` branch contains no application NVTX ranges, fine-grained batch/RT
clocks or CUDA timing events in the RT/CUDA search implementation. Coarse run
timings and the existing spectrum-loading summary remain. Detailed instrumentation
belongs on the separate profiling branch. The old `MVH_ENABLE_PROFILING` CMake
option, NVTX dependency and `gpu_bridge/run_profile.sh` / `profile.py` entry points
have been removed here. Old build directories and binaries are not automatically
cleaned or rebuilt; do not use an old profiling binary to benchmark this source.

## Preserved interfaces

Normal CLI search, all matching backends, peptide batch size, theoretical-ion
cache, host/device spectrum cache, explicit verification and score-impact
analysis remain available. Batch/setup counters and progress logs remain for
validation and benchmark reporting. The CPU reference project is unchanged.

`run_summary.tsv` retains `config_and_load_seconds`, `preprocess_seconds`,
`search_seconds` and `export_seconds`. It also records `total_seconds`, measured
from configuration through completed PSM export, excluding final object teardown
and summary writing. Search includes database preparation, all batches, RT setup
and search-local cleanup. Console output prints search and total time after export.
For full process time, use the benchmark wrapper's `wall_seconds`.

`run_benchmark.sh`, `run_suite.py`, `benchmark.py`, validation commands and saved
report rendering remain supported. New benchmark JSON represents unavailable
fine-grained timings as `null`; Markdown displays an em dash. Historical reports
with numeric timings remain readable. Missing timings are never reported as zero.

## Build and validate

Use the project's compatible CUDA 12.8 / OptiX 9 toolchain. In the usual project
container (paths below refer to its project mount):

```bash
cmake -S /workspace/sipros/mvh_cuda \
  -B /workspace/sipros/build/mvh_rt/gpu_integration \
  -DCMAKE_BUILD_TYPE=Release -DMVH_CUDA_ENABLE_RT=ON \
  -DOPTIX_ROOT=/workspace/sipros/build/mvh_rt/optix_sdk
cmake --build /workspace/sipros/build/mvh_rt/gpu_integration -j4
LD_LIBRARY_PATH=/workspace/sipros/build/mvh_rt/optix_runtime:/usr/local/cuda/lib64 \
  ctest --test-dir /workspace/sipros/build/mvh_rt/gpu_integration --output-on-failure
```

Existing compute synchronization remains. The scoring completion event wait is
replaced by a normal completion/error check; synchronization optimization is a
separate algorithm change. No GPU kernel, scoring formula, candidate ordering or
cache policy is changed by this cleanup.

External profilers can still capture CUDA activity, but this branch does not emit
application stage ranges. Use the profiling branch for detailed attribution.

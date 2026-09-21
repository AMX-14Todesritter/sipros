#!/usr/bin/env bash
# Use staged OptiX libraries for this process only; never replace WSL libcuda.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_dir="$project_root/build/mvh_rt/optix_runtime"
build_dir="$project_root/build/mvh_rt/optix_example"
if [[ $# -ne 1 ]]; then
    echo "Usage (inside existing container): bash MVH_RT/scripts/run_optix.sh NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
if [[ ! -f "$runtime_dir/libnvoptix.so.1" ]]; then
    echo "Missing staged runtime: $runtime_dir. See MVH_RT/README.md." >&2
    exit 2
fi
if [[ "${LD_LIBRARY_PATH:-}" == *'/stubs'* ]]; then
    echo "Remove CUDA stubs from runtime LD_LIBRARY_PATH before running this example." >&2
    exit 2
fi
# A command-scoped assignment does not change the caller's environment.
LD_LIBRARY_PATH="$runtime_dir:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$build_dir/bin/optix_spheres" "$build_dir/device_programs.ptx" "$1"

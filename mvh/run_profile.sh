#!/usr/bin/env bash
# Launch the CPU-only baseline profiler inside the existing container.
set -euo pipefail
exec docker exec -i "${SIPROS_CONTAINER:-sipros-sipros-1}" \
  python3 /workspace/sipros/mvh/profile_cpu.py "$@"

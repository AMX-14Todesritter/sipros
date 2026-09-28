#!/usr/bin/env bash
# WSL entry point: run profiling inside the existing project container.
set -euo pipefail
exec docker exec -i "${SIPROS_CONTAINER:-sipros-sipros-1}" \
  python3 /workspace/sipros/MVH_RT/gpu_bridge/profile.py "$@"

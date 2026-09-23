#!/usr/bin/env bash
# Run from the WSL host; all builds and measurements happen in the existing container.
set -euo pipefail
exec docker exec -i "${SIPROS_CONTAINER:-sipros-sipros-1}" \
  python3 /workspace/sipros/MVH_RT/gpu_bridge/run_suite.py "$@"

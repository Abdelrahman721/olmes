#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_tulu3_unseen_2gpu.sh <slot:0|1> [physical_gpu_id]
exec "$(dirname "$0")/run_tulu3_unseen_split.sh" 2 "$@"

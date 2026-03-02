#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_tulu3_unseen_8gpu.sh <slot:0..7> [physical_gpu_id]
exec "$(dirname "$0")/run_tulu3_unseen_split.sh" 8 "$@"

#!/usr/bin/env bash
set -euo pipefail

# Launch all tulu_3_unseen split slots in background and report failures.
#
# Usage:
#   ./run_tulu3_unseen_launch.sh <setup: 2|4|8> [gpu_id0 gpu_id1 ...]
#
# Examples:
#   ./run_tulu3_unseen_launch.sh 2
#   ./run_tulu3_unseen_launch.sh 4 0 1 2 3
#   ./run_tulu3_unseen_launch.sh 8 0 1 2 3 4 5 6 7

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <setup: 2|4|8> [gpu_id0 gpu_id1 ...]"
  exit 1
fi

SETUP="$1"
shift

case "$SETUP" in
  2|4|8) ;;
  *)
    echo "Error: setup must be one of 2, 4, 8"
    exit 1
    ;;
esac

SLOTS="$SETUP"
GPU_INPUT=("$@")

if [[ ${#GPU_INPUT[@]} -gt 0 && ${#GPU_INPUT[@]} -ne "$SLOTS" ]]; then
  echo "Error: expected either 0 GPU ids or exactly $SLOTS GPU ids, got ${#GPU_INPUT[@]}"
  exit 1
fi

SPLIT_SCRIPT="./run_tulu3_unseen_split.sh"
if [[ ! -x "$SPLIT_SCRIPT" ]]; then
  echo "Error: required script not found or not executable: $SPLIT_SCRIPT"
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-workspace/splits}"
SETUP_TAG="${SETUP}gpu_tulu3_unseen"
LOG_DIR="${OUTPUT_ROOT}/${SETUP_TAG}/run_${RUN_ID}/launcher_logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Launching tulu_3_unseen on ${SLOTS} slots"
echo "Run id: ${RUN_ID}"
echo "Logs: ${LOG_DIR}"
echo "============================================================"

declare -a PIDS=()
declare -a LOGS=()
declare -a GPUS=()

for ((slot=0; slot<SLOTS; slot++)); do
  if [[ ${#GPU_INPUT[@]} -eq 0 ]]; then
    gpu_id="$slot"
  else
    gpu_id="${GPU_INPUT[$slot]}"
  fi

  GPUS[slot]="$gpu_id"
  log_file="${LOG_DIR}/slot_${slot}_gpu_${gpu_id}.log"
  LOGS[slot]="$log_file"

  echo "[launch] slot=${slot} gpu=${gpu_id} log=${log_file}"
  (
    echo "[start] $(date -Iseconds) slot=${slot} gpu=${gpu_id}"
    "$SPLIT_SCRIPT" "$SETUP" "$slot" "$gpu_id"
  ) >"$log_file" 2>&1 &
  PIDS[slot]=$!
done

echo
echo "All slots started in background. Monitoring..."
echo

any_failure=0
for ((slot=0; slot<SLOTS; slot++)); do
  pid="${PIDS[slot]}"
  gpu="${GPUS[slot]}"
  log="${LOGS[slot]}"

  if wait "$pid"; then
    echo "[ok]   slot=${slot} gpu=${gpu}"
  else
    status=$?
    any_failure=1
    echo "[fail] slot=${slot} gpu=${gpu} exit=${status}"
    echo "       log: ${log}"
    echo "       last 60 lines:"
    sed -n '1,$p' "$log" | tail -n 60
    echo
  fi
done

if [[ "$any_failure" -ne 0 ]]; then
  echo "Completed with failures. See logs in ${LOG_DIR}"
  exit 1
fi

echo "All slots completed successfully."

#!/usr/bin/env bash
set -euo pipefail

# Launch all split slots in background and report failures.
# Usage:
#   ./run_split_launch.sh <setup: 2|4|8> [gpu_id0 gpu_id1 ...]
#
# Examples:
#   ./run_split_launch.sh 2
#   ./run_split_launch.sh 4 0 1 2 3
#   ./run_split_launch.sh 8 0 1 2 3 4 5 6 7
#
# Notes:
# - If GPU ids are omitted, logical slot index is used.
# - Logs are written per slot under workspace/splits/<setup>gpu/run_<RUN_ID>/launcher_logs/.

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
SPLIT_SCRIPT="./run_split_${SETUP}gpu.sh"

if [[ ! -x "$SPLIT_SCRIPT" ]]; then
  echo "Error: required script not found or not executable: $SPLIT_SCRIPT"
  exit 1
fi

if [[ $# -gt 0 && $# -ne "$SLOTS" ]]; then
  echo "Error: expected either 0 GPU ids or exactly $SLOTS GPU ids, got $#"
  exit 1
fi
GPU_INPUT=("$@")

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-workspace/splits}"
LAUNCHER_LOG_DIR="${OUTPUT_ROOT}/${SETUP}gpu/run_${RUN_ID}/launcher_logs"
mkdir -p "$LAUNCHER_LOG_DIR"

echo "============================================================"
echo "Launching ${SLOTS} slots using ${SPLIT_SCRIPT}"
echo "Run id: ${RUN_ID}"
echo "Launcher logs: ${LAUNCHER_LOG_DIR}"
echo "============================================================"

declare -a PIDS=()
declare -a LOGS=()
declare -a GPU_IDS=()

for ((slot=0; slot<SLOTS; slot++)); do
  if [[ $# -eq 0 ]]; then
    gpu_id="$slot"
  else
    gpu_id="${GPU_INPUT[$slot]}"
  fi
  GPU_IDS[slot]="$gpu_id"
  log_file="${LAUNCHER_LOG_DIR}/slot_${slot}_gpu_${gpu_id}.log"
  LOGS[slot]="$log_file"

  echo "[launch] slot=${slot} gpu=${gpu_id} log=${log_file}"
  (
    echo "[start] $(date -Iseconds) slot=${slot} gpu=${gpu_id}"
    "$SPLIT_SCRIPT" "$slot" "$gpu_id"
  ) >"$log_file" 2>&1 &

  PIDS[slot]=$!
done

echo
echo "All slots started in background."
echo "Monitoring until completion..."
echo

any_failure=0

for ((slot=0; slot<SLOTS; slot++)); do
  pid="${PIDS[slot]}"
  gpu_id="${GPU_IDS[slot]}"
  log_file="${LOGS[slot]}"

  if wait "$pid"; then
    echo "[ok]   slot=${slot} gpu=${gpu_id}"
  else
    status=$?
    any_failure=1
    echo "[fail] slot=${slot} gpu=${gpu_id} exit=${status}"
    echo "       log: ${log_file}"
    echo "       last 40 lines:"
    sed -n '1,$p' "$log_file" | tail -n 40
    echo
  fi
done

if [[ "$any_failure" -ne 0 ]]; then
  echo "Completed with failures. Review logs in: ${LAUNCHER_LOG_DIR}"
  exit 1
fi

echo "All slots completed successfully."

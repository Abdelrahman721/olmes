#!/usr/bin/env bash
set -euo pipefail

# Balanced 8-GPU split.
# At 8 GPUs we can isolate each very heavy benchmark family onto its own GPU.

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <slot: 0..7> [physical_gpu_id]"
  exit 1
fi

SLOT="$1"
if [[ "$SLOT" != "0" && "$SLOT" != "1" && "$SLOT" != "2" && "$SLOT" != "3" && \
      "$SLOT" != "4" && "$SLOT" != "5" && "$SLOT" != "6" && "$SLOT" != "7" ]]; then
  echo "Error: slot must be one of 0,1,2,3,4,5,6,7"
  exit 1
fi

PHYSICAL_GPU="${2:-$SLOT}"
SETUP_TAG="8gpu"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-workspace/splits}"

MODEL_PATH="${MODEL_PATH:-./model}"
BATCH_SIZE="${BATCH_SIZE:-96}"
DEFAULT_MODEL_ARGS='{"trust_remote_code":"true", "max_length":4096}'
MODEL_ARGS="${MODEL_ARGS:-$DEFAULT_MODEL_ARGS}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MODEL_ARGS_JSON="$(python3 - "${MODEL_ARGS}" "${GPU_MEMORY_UTILIZATION}" <<'PY'
import json
import sys

model_args = json.loads(sys.argv[1])
if "gpu_memory_utilization" not in model_args:
    model_args["gpu_memory_utilization"] = float(sys.argv[2])
print(json.dumps(model_args))
PY
)"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

TASKS=()
case "$SLOT" in
  0) TASKS=("minerva_math::tulu") ;;
  1) TASKS=("ifeval::tulu") ;;
  2) TASKS=("codex_humaneval::tulu") ;;
  3) TASKS=("codex_humanevalplus::tulu") ;;
  4) TASKS=("bbh:cot-v1::tulu") ;;
  5) TASKS=("mmlu:mc::tulu") ;;
  6) TASKS=("drop::llama3" "gsm8k::tulu") ;;
  7) TASKS=("popqa::tulu" "truthfulqa::tulu") ;;
esac

echo "Running ${SETUP_TAG} slot=${SLOT} on CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}"
echo "Run id: ${RUN_ID}"
echo "Tasks: ${TASKS[*]}"

SLOT_OUTPUT_DIR="${OUTPUT_ROOT}/${SETUP_TAG}/run_${RUN_ID}/gpu${SLOT}"
mkdir -p "$SLOT_OUTPUT_DIR"

echo "============================================================"
echo "Output: ${SLOT_OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" olmes --model "${MODEL_PATH}" \
  --task "${TASKS[@]}" \
  --output-dir "${SLOT_OUTPUT_DIR}" \
  --model-type vllm \
  --model-args "${MODEL_ARGS_JSON}" \
  --batch-size "${BATCH_SIZE}"

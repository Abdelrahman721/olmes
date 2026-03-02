#!/usr/bin/env bash
set -euo pipefail

# Balanced 2-GPU split for:
# minerva_math, coding (humaneval/humanevalplus), ifeval, and non-coding benchmarks.
# Runtime-informed notes:
# - codex_humaneval::tulu and codex_humanevalplus::tulu use repeats=20 (very heavy).
# - minerva_math::tulu expands to 7 MATH subsets (5000 test rows total).
# - ifeval::tulu uses max_gen_toks=2048 and is typically one of the slowest non-coding tasks.

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <slot: 0|1> [physical_gpu_id]"
  exit 1
fi

SLOT="$1"
if [[ "$SLOT" != "0" && "$SLOT" != "1" ]]; then
  echo "Error: slot must be 0 or 1"
  exit 1
fi

PHYSICAL_GPU="${2:-$SLOT}"
SETUP_TAG="2gpu"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-workspace/splits}"

MODEL_PATH="${MODEL_PATH:-./model_clean}"
BATCH_SIZE="${BATCH_SIZE:-96}"
DEFAULT_MODEL_ARGS='{"trust_remote_code":"true", "max_length":4096}'
MODEL_ARGS="${MODEL_ARGS:-$DEFAULT_MODEL_ARGS}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

TASKS=()
case "$SLOT" in
  0)
    TASKS=(
      "minerva_math::tulu"
      "mmlu:mc::tulu"
      "bbh:cot-v1::tulu"
      "drop::llama3"
      "popqa::tulu"
    )
    ;;
  1)
    TASKS=(
      "codex_humaneval::tulu"
      "codex_humanevalplus::tulu"
      "ifeval::tulu"
      "gsm8k::tulu"
      "truthfulqa::tulu"
    )
    ;;
esac

echo "Running ${SETUP_TAG} slot=${SLOT} on CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}"
echo "Run id: ${RUN_ID}"
echo "Tasks: ${TASKS[*]}"

for TASK in "${TASKS[@]}"; do
  SAFE_TASK="$(echo "$TASK" | sed -e 's/::/-/g' -e 's/:/_/g' -e 's|/|_|g')"
  TASK_OUTPUT_DIR="${OUTPUT_ROOT}/${SETUP_TAG}/run_${RUN_ID}/gpu${SLOT}/${SAFE_TASK}"
  mkdir -p "$TASK_OUTPUT_DIR"

  echo "============================================================"
  echo "Task: ${TASK}"
  echo "Output: ${TASK_OUTPUT_DIR}"

  CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" olmes --model "${MODEL_PATH}" \
    --task "${TASK}" \
    --output-dir "${TASK_OUTPUT_DIR}" \
    --model-type vllm \
    --model-args "${MODEL_ARGS}" \
    --batch-size "${BATCH_SIZE}"
done

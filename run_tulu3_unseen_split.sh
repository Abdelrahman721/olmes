#!/usr/bin/env bash
set -euo pipefail

# Run one shard of tulu_3_unseen expanded atomic tasks.
# Usage:
#   ./run_tulu3_unseen_split.sh <setup: 2|4|8> <slot> [physical_gpu_id]
#
# This expands task-suite components into concrete tasks and shards them using
# a weighted greedy assignment (runtime proxy) for the requested setup.

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <setup: 2|4|8> <slot> [physical_gpu_id]"
  exit 1
fi

SETUP="$1"
SLOT="$2"
PHYSICAL_GPU="${3:-$SLOT}"

case "$SETUP" in
  2|4|8) ;;
  *)
    echo "Error: setup must be one of 2, 4, 8"
    exit 1
    ;;
esac

if ! [[ "$SLOT" =~ ^[0-9]+$ ]]; then
  echo "Error: slot must be numeric"
  exit 1
fi
if (( SLOT < 0 || SLOT >= SETUP )); then
  echo "Error: slot must be in [0, $((SETUP-1))]"
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-./model_clean}"
BATCH_SIZE="${BATCH_SIZE:-96}"
DEFAULT_MODEL_ARGS='{"trust_remote_code":"true", "max_length":4096}'
MODEL_ARGS="${MODEL_ARGS:-$DEFAULT_MODEL_ARGS}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MODEL_ARGS_JSON="$(python3 - "${MODEL_ARGS}" "${GPU_MEMORY_UTILIZATION}" <<'PY'
import json
import sys

model_args = json.loads(sys.argv[1])
if "gpu_memory_utilization" not in model_args:
    model_args["gpu_memory_utilization"] = float(sys.argv[2])
print(json.dumps(model_args))
PY
)"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-workspace/splits}"
SETUP_TAG="${SETUP}gpu_tulu3_unseen"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

TASKS_FILE="$(mktemp)"

python3 - "$SETUP" "$SLOT" > "$TASKS_FILE" <<'PY'
import sys
from collections import defaultdict

setup = int(sys.argv[1])
slot = int(sys.argv[2])

# Runtime proxy data (derived from repo configs + dataset sizes):
# - AGI-Eval local jsonl counts
# - DeepMind Math 58 categories, each limited to 100 for 0shot_cot::tulu3
# - MMLU-Pro total test 12032 split over 14 categories (~859 each)
# - GPQA ~448 train examples
# - BigCodeBench / hard row counts from HF metadata
agi_counts = {
    "lsat-ar": 230,
    "lsat-lr": 510,
    "lsat-rc": 269,
    "logiqa-en": 651,
    "sat-math": 220,
    "sat-en": 206,
    "aqua-rat": 254,
    "gaokao-english": 306,
}

deepmind_categories = [
    "algebra__linear_1d",
    "algebra__linear_1d_composed",
    "algebra__linear_2d",
    "algebra__linear_2d_composed",
    "algebra__polynomial_roots",
    "algebra__polynomial_roots_composed",
    "algebra__sequence_next_term",
    "algebra__sequence_nth_term",
    "arithmetic__add_or_sub",
    "arithmetic__add_or_sub_in_base",
    "arithmetic__add_sub_multiple",
    "arithmetic__div",
    "arithmetic__mixed",
    "arithmetic__mul",
    "arithmetic__mul_div_multiple",
    "arithmetic__nearest_integer_root",
    "arithmetic__simplify_surd",
    "calculus__differentiate",
    "calculus__differentiate_composed",
    "comparison__closest",
    "comparison__closest_composed",
    "comparison__kth_biggest",
    "comparison__kth_biggest_composed",
    "comparison__pair",
    "comparison__pair_composed",
    "comparison__sort",
    "comparison__sort_composed",
    "measurement__conversion",
    "measurement__time",
    "numbers__base_conversion",
    "numbers__div_remainder",
    "numbers__div_remainder_composed",
    "numbers__gcd",
    "numbers__gcd_composed",
    "numbers__is_factor",
    "numbers__is_factor_composed",
    "numbers__is_prime",
    "numbers__is_prime_composed",
    "numbers__lcm",
    "numbers__lcm_composed",
    "numbers__list_prime_factors",
    "numbers__list_prime_factors_composed",
    "numbers__place_value",
    "numbers__place_value_composed",
    "numbers__round_number",
    "numbers__round_number_composed",
    "polynomials__add",
    "polynomials__coefficient_named",
    "polynomials__collect",
    "polynomials__compose",
    "polynomials__evaluate",
    "polynomials__evaluate_composed",
    "polynomials__expand",
    "polynomials__simplify_power",
    "probability__swr_p_level_set",
    "probability__swr_p_sequence",
]

mmlu_pro_categories = [
    "math",
    "health",
    "physics",
    "business",
    "biology",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "philosophy",
    "other",
    "history",
    "psychology",
    "law",
]

tasks = []

# agi_eval_english:0shot_cot::tulu3 expansion
for t, n in agi_counts.items():
    task = f"agi_eval_{t}:0shot_cot::tulu3"
    weight = n * 2048
    tasks.append((task, weight))

# deepmind_math:0shot_cot::tulu3 expansion (limit=100 per category)
for cat in deepmind_categories:
    task = f"deepmind_math_{cat}:0shot_cot::tulu3"
    weight = 100 * 2048
    tasks.append((task, weight))

# mmlu_pro variants expansion
per_mmlu_pro_cat = 12032 / 14.0
for cat in mmlu_pro_categories:
    t1 = f"mmlu_pro_{cat}:0shot_cot::tulu3"
    w1 = per_mmlu_pro_cat * 2048
    tasks.append((t1, w1))

    t2 = f"mmlu_pro_{cat}:cot::llama3.1"
    w2 = per_mmlu_pro_cat * 1024
    tasks.append((t2, w2))

# gpqa tasks
tasks.append(("gpqa:0shot_cot::tulu3", 448 * 2048))
tasks.append(("gpqa:0shot_cot::llama3.1", 448 * 2048))

# code tasks (execution-heavy, weight multiplier for runtime)
tasks.append(("bigcodebench::tulu", 1140 * 1280 * 4))
tasks.append(("bigcodebench_hard::tulu", 148 * 1280 * 4))

# Greedy number partitioning by weight
tasks_sorted = sorted(tasks, key=lambda x: x[1], reverse=True)
buckets = [[] for _ in range(setup)]
bucket_weight = [0.0 for _ in range(setup)]
for task, w in tasks_sorted:
    idx = min(range(setup), key=lambda i: bucket_weight[i])
    buckets[idx].append(task)
    bucket_weight[idx] += float(w)

for t in buckets[slot]:
    print(t)
PY

mapfile -t TASKS < "$TASKS_FILE"
rm -f "$TASKS_FILE"

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "No tasks assigned for setup=${SETUP} slot=${SLOT}"
  exit 1
fi

echo "Running ${SETUP_TAG} slot=${SLOT} on CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}"
echo "Run id: ${RUN_ID}"
echo "Task count: ${#TASKS[@]}"

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

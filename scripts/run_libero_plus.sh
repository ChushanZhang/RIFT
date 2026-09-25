#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: LIBERO_PLUS_ROOT=/path/to/LIBERO-plus $0 CHECKPOINT DATASET_STATS [OUTPUT_DIR]" >&2
  exit 2
fi
if [[ -z "${LIBERO_PLUS_ROOT:-}" ]]; then
  echo "LIBERO_PLUS_ROOT must point to the official LIBERO-Plus checkout" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$script_dir/.." && pwd)
checkpoint=$(realpath "$1")
dataset_stats=$(realpath "$2")
libero_plus_root=$(realpath "$LIBERO_PLUS_ROOT")
output_dir=${3:-"$project_root/evaluate_results/libero_plus/$(basename "${checkpoint%.*}")"}
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")

python_bin=${PYTHON:-python}
task_config=${TASK_CONFIG:-libero_rift_2cam224_1e-4}
gpu_ids_csv=${GPU_IDS:-0}
workers_per_gpu=${WORKERS_PER_GPU:-1}
global_task_ids=${GLOBAL_TASK_IDS:-}
catalog=${LIBERO_PLUS_CATALOG:-"$output_dir/task_catalog.json"}

[[ -s "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
[[ -s "$dataset_stats" ]] || { echo "missing dataset stats: $dataset_stats" >&2; exit 2; }
[[ -f "$libero_plus_root/libero/libero/__init__.py" ]] || { echo "invalid LIBERO_PLUS_ROOT: $libero_plus_root" >&2; exit 2; }
[[ "$workers_per_gpu" =~ ^[1-9][0-9]*$ ]] || { echo "WORKERS_PER_GPU must be positive" >&2; exit 2; }
if [[ -n "$global_task_ids" && ! "$global_task_ids" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GLOBAL_TASK_IDS must be comma-separated non-negative integers" >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<<"$gpu_ids_csv"
declare -A seen_gpu_ids=()
for gpu in "${gpu_ids[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "GPU_IDS must contain integers" >&2; exit 2; }
  [[ -z "${seen_gpu_ids[$gpu]:-}" ]] || { echo "duplicate GPU ID: $gpu" >&2; exit 2; }
  seen_gpu_ids[$gpu]=1
done
num_gpus=${#gpu_ids[@]}
num_shards=$((num_gpus * workers_per_gpu))

export PYTHONNOUSERSITE=1
export PYTHONPATH="$libero_plus_root:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_STRICT_DET=${LIBERO_STRICT_DET:-1}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export HYDRA_FULL_ERROR=1

catalog_args=(
  --output "$catalog"
  --libero-plus-root "$libero_plus_root"
)
if [[ -n "${LIBERO_PLUS_REVISION:-}" ]]; then
  catalog_args+=(--libero-plus-revision "$LIBERO_PLUS_REVISION")
fi
if [[ -n "${LIBERO_PLUS_ASSETS_ARCHIVE:-}" ]]; then
  catalog_args+=(--assets-archive "$LIBERO_PLUS_ASSETS_ARCHIVE")
fi
if [[ ! -s "$catalog" ]]; then
  (cd "$project_root" && "$python_bin" experiments/libero/libero_plus_catalog.py "${catalog_args[@]}")
fi

checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
mkdir -p "$output_dir/logs"
exec {manager_lock}>"$output_dir/.manager.lock"
if ! flock -n "$manager_lock"; then
  echo "another LIBERO-Plus manager owns $output_dir" >&2
  exit 3
fi

pids=()
cleanup_children() {
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
    pids=()
  fi
}
remove_pid() {
  local completed=$1
  local live=()
  local pid
  for pid in "${pids[@]}"; do
    [[ "$pid" == "$completed" ]] || live+=("$pid")
  done
  pids=("${live[@]}")
}
trap 'cleanup_children; exit 130' INT TERM HUP
trap cleanup_children EXIT

cd "$project_root"
for ((shard=0; shard<num_shards; shard++)); do
  physical_gpu=${gpu_ids[$((shard % num_gpus))]}
  worker_args=(
    "task=$task_config"
    "ckpt=$checkpoint"
    "gpu_id=0"
    "EVALUATION.output_dir=$output_dir"
    "EVALUATION.num_trials=1"
    "EVALUATION.save_video=false"
    "EVALUATION.dataset_stats_path=$dataset_stats"
    "+EVALUATION.catalog_path=$catalog"
    "+EVALUATION.libero_plus_root=$libero_plus_root"
    "+EVALUATION.checkpoint_sha256=$checkpoint_sha"
    "+EVALUATION.shard_index=$shard"
    "+EVALUATION.num_shards=$num_shards"
    "+EVALUATION.continue_on_error=true"
    "+EVALUATION.max_task_errors=${MAX_TASK_ERRORS:-10}"
    "+EVALUATION.task_timeout_seconds=${TASK_TIMEOUT_SECONDS:-1800}"
  )
  if [[ -n "$global_task_ids" ]]; then
    worker_args+=("+EVALUATION.global_task_ids=[$global_task_ids]")
  fi
  CUDA_VISIBLE_DEVICES=$physical_gpu MUJOCO_EGL_DEVICE_ID=$physical_gpu \
    "$python_bin" experiments/libero/eval_libero_plus_shard.py "${worker_args[@]}" \
    >"$output_dir/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
while [[ ${#pids[@]} -gt 0 ]]; do
  completed_pid=""
  status=0
  if wait -n -p completed_pid "${pids[@]}"; then
    status=0
  else
    status=$?
  fi
  if [[ -z "$completed_pid" || $status -ne 0 ]]; then
    echo "LIBERO-Plus worker failed; inspect $output_dir/logs" >&2
    failed=1
    cleanup_children
    break
  fi
  remove_pid "$completed_pid"
done
pids=()
[[ $failed -eq 0 ]] || exit 1

summary_args=(--run-dir "$output_dir" --catalog "$catalog")
if [[ -n "$global_task_ids" ]]; then
  summary_args+=(--allow-partial)
fi
"$python_bin" experiments/libero/summarize_libero_plus.py "${summary_args[@]}"

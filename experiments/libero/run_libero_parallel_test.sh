#!/usr/bin/env bash

# Task-level multi-GPU launcher for the standalone LIBERO worker.
set -u

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <task_file>" >&2
  exit 2
fi

task_file=$1
root_dir=${ROOT_DIR:-"$(pwd)"}
output_dir=${OUTPUT_DIR:-"$root_dir/evaluate_results/libero"}
checkpoint=${CKPT:-}
config=${CONFIG:-libero_rift_2cam224_1e-4}
num_trials=${NUM_TRIALS:-50}
max_tasks_per_gpu=${MAX_TASKS_PER_GPU:-1}
poll_seconds=${MONITORING_INTERVAL:-2}

if [[ ! -f "$task_file" ]]; then
  echo "Task file does not exist: $task_file" >&2
  exit 2
fi
if [[ -z "$checkpoint" ]]; then
  echo "CKPT must point to the released weights file." >&2
  exit 2
fi
if [[ ! "$max_tasks_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TASKS_PER_GPU must be a positive integer." >&2
  exit 2
fi

config=${config#configs/}
config=${config#task/}
config=${config%.yaml}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
else
  num_gpus=${NUM_GPUS:-}
  if [[ ! "$num_gpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "Set NUM_GPUS or CUDA_VISIBLE_DEVICES." >&2
    exit 2
  fi
  gpu_ids=()
  for ((gpu = 0; gpu < num_gpus; gpu++)); do
    gpu_ids+=("$gpu")
  done
fi

mkdir -p "$output_dir/task_logs" "$output_dir/task_status"
cp "$task_file" "$output_dir/tasks.txt"
: > "$output_dir/failed_tasks.txt"

declare -a slot_gpus=()
for gpu in "${gpu_ids[@]}"; do
  for ((slot = 0; slot < max_tasks_per_gpu; slot++)); do
    slot_gpus+=("$gpu")
  done
done
declare -a slot_pids=()
declare -a slot_tasks=()
for ((slot = 0; slot < ${#slot_gpus[@]}; slot++)); do
  slot_pids+=("")
  slot_tasks+=("")
done

extra_overrides=()
if [[ -n "${EXTRA_ARGS_FILE:-}" ]]; then
  if [[ ! -f "$EXTRA_ARGS_FILE" ]]; then
    echo "EXTRA_ARGS_FILE does not exist: $EXTRA_ARGS_FILE" >&2
    exit 2
  fi
  mapfile -t extra_overrides < "$EXTRA_ARGS_FILE"
fi
failed=0

reap_finished() {
  local block=${1:-false}
  while true; do
    local active=0
    local reaped=0
    for ((slot = 0; slot < ${#slot_pids[@]}; slot++)); do
      local pid=${slot_pids[$slot]}
      [[ -z "$pid" ]] && continue
      active=1
      if ! kill -0 "$pid" 2>/dev/null; then
        local rc=0
        wait "$pid" || rc=$?
        if [[ $rc -ne 0 ]]; then
          echo "${slot_tasks[$slot]},gpu=${slot_gpus[$slot]},rc=$rc" \
            >> "$output_dir/failed_tasks.txt"
          failed=1
        fi
        slot_pids[$slot]=""
        slot_tasks[$slot]=""
        reaped=1
      fi
    done
    [[ "$block" != true || $active -eq 0 || $reaped -eq 1 ]] && return
    sleep "$poll_seconds"
  done
}

find_free_slot() {
  while true; do
    reap_finished false
    for ((slot = 0; slot < ${#slot_pids[@]}; slot++)); do
      if [[ -z "${slot_pids[$slot]}" ]]; then
        free_slot=$slot
        return
      fi
    done
    reap_finished true
  done
}

while IFS=',' read -r suite task_id; do
  [[ -z "$suite" ]] && continue
  free_slot=-1
  find_free_slot
  slot=$free_slot
  gpu=${slot_gpus[$slot]}
  log_file="$output_dir/task_logs/${suite}_task${task_id}_gpu${gpu}.log"
  status_file="$output_dir/task_status/${suite}_task${task_id}.status"
  (
    cd "$root_dir" || exit 2
    if CUDA_VISIBLE_DEVICES="$gpu" python experiments/libero/eval_libero_single.py \
        "task=$config" \
        "ckpt=$checkpoint" \
        "gpu_id=$gpu" \
        "EVALUATION.task_suite_name=$suite" \
        "EVALUATION.task_id=$task_id" \
        "EVALUATION.num_trials=$num_trials" \
        "EVALUATION.output_dir=$output_dir" \
        "${extra_overrides[@]}"; then
      echo "DONE|$gpu|$BASHPID|$log_file" > "$status_file"
    else
      rc=$?
      echo "FAILED|$gpu|$BASHPID|$log_file|rc=$rc" > "$status_file"
      exit "$rc"
    fi
  ) > "$log_file" 2>&1 &
  pid=$!
  slot_pids[$slot]=$pid
  slot_tasks[$slot]="$suite,$task_id"
  echo "RUNNING|$gpu|$pid|$log_file" > "$status_file"
  echo "Started $suite task $task_id on GPU $gpu (pid $pid)"
done < "$task_file"

while true; do
  reap_finished false
  active=0
  for pid in "${slot_pids[@]}"; do
    [[ -n "$pid" ]] && active=1
  done
  [[ $active -eq 0 ]] && break
  sleep "$poll_seconds"
done

if [[ $failed -ne 0 ]]; then
  echo "One or more LIBERO tasks failed; see $output_dir/failed_tasks.txt." >&2
  exit 1
fi

python "$root_dir/experiments/libero/summarize_results.py" --output_dir "$output_dir"

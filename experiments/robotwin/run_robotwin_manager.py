"""Schedule the official RoboTwin clean and randomized benchmarks."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(path_str)))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_robotwin_root(path_value: Any) -> Path:
    configured = None if path_value is None else str(path_value).strip()
    raw_path = configured or os.environ.get("ROBOTWIN_ROOT", "").strip()
    if not raw_path:
        raise ValueError(
            "RoboTwin is external to this repository. Set ROBOTWIN_ROOT or pass "
            "EVALUATION.robotwin_root=/path/to/RoboTwin."
        )
    root = _resolve_path(raw_path, base=PROJECT_ROOT)
    if not (root / "script" / "eval_policy.py").is_file():
        raise FileNotFoundError(f"RoboTwin evaluator not found under: {root}")
    return root


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 < len(parts):
            return f"{parts[runs_idx + 1]}_{parts[runs_idx + 2]}"
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.robotwin_root",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [
        override
        for override in HydraConfig.get().overrides.task
        if not _is_blocked_override(override)
    ]


def _load_all_tasks(robotwin_root: Path, task_list_path: Any = None) -> list[str]:
    if task_list_path is None or str(task_list_path).strip().lower() in {
        "",
        "none",
        "null",
    }:
        config_path = robotwin_root / "task_config" / "_eval_step_limit.yml"
    else:
        config_path = _resolve_path(str(task_list_path), base=PROJECT_ROOT)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"RoboTwin task list not found: {config_path}. Set EVALUATION.task_list_path "
            "for a nonstandard checkout."
        )
    with config_path.open("r", encoding="utf-8") as file:
        task_map = yaml.safe_load(file)
    if not isinstance(task_map, dict) or not task_map:
        raise ValueError(f"Invalid RoboTwin task list: {config_path}")
    return list(dict.fromkeys(str(task) for task in task_map))


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.is_file():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    last_value: float | None = None
    for line in result_file.read_text(encoding="utf-8").splitlines():
        try:
            last_value = float(line.strip())
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"No success rate found in: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else float(sum(present) / len(present))


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str
    process: subprocess.Popen[str]


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml"
)
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.is_file():
        raise FileNotFoundError(f"Single-task evaluator not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    robotwin_root = _resolve_robotwin_root(cfg.EVALUATION.robotwin_root)

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if num_gpus <= 0 or max_tasks_per_gpu <= 0:
        raise ValueError("MULTIRUN.num_gpus and max_tasks_per_gpu must be positive.")
    gpu_ids = list(range(num_gpus))

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_output_dir = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / _resolve_ckpt_tag(ckpt_path)
        / output_dir.name
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    task_name_cfg = cfg.EVALUATION.task_name
    tasks = (
        _load_all_tasks(robotwin_root, cfg.EVALUATION.task_list_path)
        if task_name_cfg is None or not str(task_name_cfg).strip()
        else [str(task_name_cfg)]
    )
    extra_overrides = _collect_worker_overrides()
    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running: list[RunningState] = []

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def build_cmd(task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = "demo_clean" if phase == "clean" else "demo_randomized"
        return [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={ckpt_path}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.robotwin_root={robotwin_root}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={output_dir}",
            *extra_overrides,
        ]

    def launch(task_name: str, gpu_id: int, phase: str) -> RunningState:
        cmd = build_cmd(task_name, gpu_id, phase)
        log(f"launch task={task_name} phase={phase} gpu={gpu_id}")
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=subprocess.Popen(cmd, cwd=PROJECT_ROOT, text=True),
        )

    def running_count(gpu_id: int) -> int:
        return sum(
            state.gpu_id == gpu_id and state.process.poll() is None for state in running
        )

    def fill_gpu(gpu_id: int) -> None:
        while pending_tasks and running_count(gpu_id) < max_tasks_per_gpu:
            running.append(launch(pending_tasks.popleft(), gpu_id, "clean"))

    def terminate_all() -> None:
        active = [state for state in running if state.process.poll() is None]
        for state in active:
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in active:
            try:
                state.process.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                state.process.kill()
                state.process.wait()

    def write_outputs() -> None:
        clean_mean = _mean([task_rates[task]["clean"] for task in tasks])
        random_mean = _mean([task_rates[task]["random"] for task in tasks])
        with summary_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow(
                    [task, task_rates[task]["clean"], task_rates[task]["random"]]
                )
            writer.writerow(["__overall__", clean_mean, random_mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": task_rates[task]["clean"],
                    "random_success_rate": task_rates[task]["random"],
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": clean_mean,
                "random_mean_success_rate": random_mean,
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with failed_tasks_file.open("w", encoding="utf-8") as file:
            for record in failed_records:
                file.write(
                    f"{record['task_name']},{record['phase']},gpu={record['gpu_id']},"
                    f"return_code={record['return_code']},reason={record['reason']}\n"
                )

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu}"
    )
    for gpu_id in gpu_ids:
        fill_gpu(gpu_id)

    failure_message: str | None = None
    while running and failure_message is None:
        progressed = False
        for state in list(running):
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running.remove(state)
            if return_code != 0:
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={state.gpu_id}, return_code={return_code}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": state.gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                terminate_all()
                running.clear()
                break

            result_file = (
                run_output_dir / state.task_name / _phase_result_filename(state.phase)
            )
            try:
                rate = _parse_success_rate(result_file)
            except Exception as exc:
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"error={exc!r}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": state.gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                terminate_all()
                running.clear()
                break

            task_rates[state.task_name][state.phase] = rate
            log(
                f"done task={state.task_name} phase={state.phase} "
                f"gpu={state.gpu_id} success_rate={rate:.4f}"
            )
            if state.phase == "clean":
                running.append(launch(state.task_name, state.gpu_id, "random"))
            else:
                fill_gpu(state.gpu_id)
        if not progressed and failure_message is None:
            time.sleep(POLL_INTERVAL_SEC)

    if failure_message is not None:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "phase": "not_started",
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")
    if failure_message is not None:
        raise RuntimeError(failure_message)
    log("manager finished successfully")


if __name__ == "__main__":
    main()

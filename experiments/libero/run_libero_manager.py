"""Create the official LIBERO task matrix and run it across available GPUs."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import subprocess

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _benchmark_dict():
    try:
        from libero.libero import benchmark
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LIBERO is required to enumerate benchmark tasks. Install the official "
            "LIBERO environment before running this manager."
        ) from exc
    return benchmark.get_benchmark_dict()


def create_task_file(output_file: Path, task_suite_names: list[str]) -> Path:
    benchmarks = _benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for suite_name in task_suite_names:
        if suite_name not in benchmarks:
            raise ValueError(f"Unknown LIBERO task suite: {suite_name}")
        task_suite = benchmarks[suite_name]()
        rows.extend(f"{suite_name},{task_id}" for task_id in range(task_suite.n_tasks))
    output_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Created {len(rows)} LIBERO tasks at {output_file}")
    return output_file


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    return key in {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.output_dir",
    } or key.startswith(("MULTIRUN.", "hydra."))


def collect_worker_overrides() -> list[str]:
    return [
        override
        for override in HydraConfig.get().overrides.task
        if not _is_blocked_override(override)
    ]


def _task_choice() -> str:
    choice = HydraConfig.get().runtime.choices.get("task")
    if not choice:
        raise ValueError("Hydra did not resolve a task recipe.")
    return str(choice)


def run_evaluation(
    *,
    task_file: Path,
    task_choice: str,
    checkpoint: str,
    num_gpus: int,
    num_trials: int,
    max_tasks_per_gpu: int,
    output_dir: Path,
    extra_overrides: list[str],
) -> None:
    script_path = PROJECT_ROOT / "experiments/libero/run_libero_parallel_test.sh"
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing LIBERO worker launcher: {script_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides_file = output_dir / "worker_overrides.txt"
    overrides_file.write_text("\n".join(extra_overrides), encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "ROOT_DIR": str(PROJECT_ROOT),
            "OUTPUT_DIR": str(output_dir),
            "CKPT": checkpoint,
            "CONFIG": task_choice,
            "NUM_GPUS": str(num_gpus),
            "NUM_TRIALS": str(num_trials),
            "MAX_TASKS_PER_GPU": str(max_tasks_per_gpu),
            "EXTRA_ARGS_FILE": str(overrides_file),
        }
    )
    subprocess.run(
        ["bash", str(script_path), str(task_file)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("Pass ckpt=/path/to/rift_step021700.pt.")
    manager = cfg.MULTIRUN
    base_output = Path(
        os.path.expandvars(os.path.expanduser(str(cfg.EVALUATION.output_dir)))
    )
    output_dir = base_output / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    configured_task_file = manager.get("task_file")
    if configured_task_file:
        task_file = Path(
            os.path.expandvars(os.path.expanduser(str(configured_task_file)))
        )
        if not task_file.is_file():
            raise FileNotFoundError(f"Task file does not exist: {task_file}")
    else:
        task_file = create_task_file(
            output_dir / "tasks.txt", list(manager.task_suite_names)
        )

    OmegaConf.save(cfg, output_dir / "manager_config.yaml", resolve=True)
    if bool(manager.get("create_only", False)):
        return
    run_evaluation(
        task_file=task_file,
        task_choice=_task_choice(),
        checkpoint=str(cfg.ckpt),
        num_gpus=int(manager.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(manager.max_tasks_per_gpu),
        output_dir=output_dir,
        extra_overrides=collect_worker_overrides(),
    )


if __name__ == "__main__":
    main()

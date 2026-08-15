"""Run one RIFT policy task through RoboTwin's official evaluator.

RoboTwin is installed separately. This entrypoint links the local policy adapter
into ``<RoboTwin>/policy/rift_policy`` and invokes ``script/eval_policy.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "rift_policy"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_robotwin_root(path_value: Any) -> Path:
    configured = None if path_value is None else str(path_value).strip()
    raw_path = configured or os.environ.get("ROBOTWIN_ROOT", "").strip()
    if not raw_path:
        raise ValueError(
            "RoboTwin is external to this repository. Set ROBOTWIN_ROOT or pass "
            "EVALUATION.robotwin_root=/path/to/RoboTwin."
        )
    return _resolve_path(raw_path, base=PROJECT_ROOT)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(
        cfg.EVALUATION.dataset_stats_path,
        base=PROJECT_ROOT,
    )
    candidates = [explicit] if explicit is not None else []
    candidates.extend(
        parent / "dataset_stats.json" for parent in list(ckpt_path.parents)[:4]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate dataset_stats.json. Set "
        "EVALUATION.dataset_stats_path=/path/to/dataset_stats.json. "
        f"Checked: {attempted}"
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 < len(parts):
            return f"{parts[runs_idx + 1]}_{parts[runs_idx + 2]}"
    return ckpt_path.stem


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(
            f"RoboTwin policy directory not found: {policy_root}. "
            "Set ROBOTWIN_ROOT to a compatible RoboTwin checkout."
        )

    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()
    if not policy_target.exists() and not policy_target.is_symlink():
        policy_target.symlink_to(source_resolved, target_is_directory=True)
        return policy_target

    if policy_target.is_symlink() and policy_target.resolve() == source_resolved:
        return policy_target

    if policy_target.is_symlink():
        raise RuntimeError(
            f"Policy symlink conflict: {policy_target} -> {policy_target.resolve()}, "
            f"expected {source_resolved}"
        )
    raise RuntimeError(
        f"Policy path already exists and is not a symlink: {policy_target}. "
        "Remove or rename it before running evaluation."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(
    overrides: list[str],
    key: str,
    value: Any,
    *,
    skip_none: bool = True,
) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def _build_robotwin_command(
    cfg: DictConfig, *, ckpt_path: Path, stats_path: Path
) -> list[str]:
    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")
    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", POLICY_NAME)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(stats_path))
    _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
    _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
    _append_override(
        overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps
    )
    _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
    _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
    _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
    _append_override(overrides, "rand_device", cfg.EVALUATION.rand_device)
    _append_override(overrides, "tiled", cfg.EVALUATION.tiled)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)

    return [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]


def _phase_result_filename(task_config: str) -> str:
    if task_config == "demo_clean":
        return "_result_clean.txt"
    if task_config == "demo_randomized":
        return "_result_random.txt"
    raise ValueError(
        "RoboTwin manager supports task_config=demo_clean or demo_randomized, "
        f"got {task_config!r}."
    )


def _collect_result(
    *,
    cfg: DictConfig,
    robotwin_root: Path,
    destination_dir: Path,
    started_at: float,
) -> Path:
    """Normalize patched and stock RoboTwin result layouts."""
    task_config = str(cfg.EVALUATION.task_config)
    destination = destination_dir / _phase_result_filename(task_config)
    if destination.is_file():
        return destination

    stock_root = (
        robotwin_root
        / "eval_result"
        / str(cfg.EVALUATION.task_name)
        / POLICY_NAME
        / task_config
    )
    candidates = (
        [
            path
            for path in stock_root.rglob("_result.txt")
            if path.is_file() and path.stat().st_mtime >= started_at - 2.0
        ]
        if stock_root.is_dir()
        else []
    )
    if not candidates:
        raise FileNotFoundError(
            "RoboTwin exited successfully but produced no result file under either "
            f"{destination} or {stock_root}."
        )
    source = max(candidates, key=lambda path: path.stat().st_mtime)
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml"
)
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")
    if str(cfg.EVALUATION.policy_name) != POLICY_NAME:
        raise ValueError(f"EVALUATION.policy_name must be {POLICY_NAME!r}.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    robotwin_root = _resolve_robotwin_root(cfg.EVALUATION.robotwin_root)
    if not (robotwin_root / "script" / "eval_policy.py").is_file():
        raise FileNotFoundError(
            f"Compatible RoboTwin evaluator not found under: {robotwin_root}. "
            "Set ROBOTWIN_ROOT or EVALUATION.robotwin_root."
        )

    policy_source_dir = PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME
    _ensure_policy_symlink(robotwin_root, policy_source_dir)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_output_dir = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / _resolve_ckpt_tag(ckpt_path)
        / output_dir.name
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / (
        f"eval_{cfg.EVALUATION.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    cmd = _build_robotwin_command(cfg, ckpt_path=ckpt_path, stats_path=stats_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    started_at = time.time()
    with log_file.open("w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=robotwin_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}"
        )

    result_file = _collect_result(
        cfg=cfg,
        robotwin_root=robotwin_root,
        destination_dir=run_output_dir / str(cfg.EVALUATION.task_name),
        started_at=started_at,
    )

    OmegaConf.save(
        config=cfg,
        f=run_output_dir / f"eval_config_{cfg.EVALUATION.task_name}.yaml",
    )
    print(f"Evaluation finished. Result: {result_file}. Log: {log_file}")


if __name__ == "__main__":
    main()

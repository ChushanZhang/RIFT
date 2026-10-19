"""Persistent-worker RIFT evaluation on the official LIBERO-Plus suite."""

from __future__ import annotations

import contextlib
import copy
import fcntl
import io
import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_TEXT = str(PROJECT_ROOT)
if PROJECT_ROOT_TEXT in sys.path:
    sys.path.remove(PROJECT_ROOT_TEXT)
sys.path.insert(0, PROJECT_ROOT_TEXT)

from experiments.checkpoint_utils import load_rift_checkpoint_exact  # noqa: E402
from experiments.libero.eval_libero_single import (  # noqa: E402
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    run_single_episode,
)
from experiments.libero.libero_plus_catalog import (  # noqa: E402
    SUITE_ORDER,
    TOTAL_TASKS,
    bootstrap_libero_plus,
    canonical_sha256,
    file_sha256,
    git_revision,
    load_catalog_file,
    write_json_atomic,
)
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env  # noqa: E402
from rift.datasets.lerobot.processors.rift_processor import RIFTProcessor  # noqa: E402
from rift.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from rift.utils.pytorch_utils import set_global_seed  # noqa: E402


RECEIPT_SCHEMA = "rift.libero-plus-task-result.v1"
MANIFEST_SCHEMA = "rift.libero-plus-run-manifest.v1"


def _semantic_config(cfg: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved Hydra config is not a mapping")
    value = copy.deepcopy(value)
    value.pop("gpu_id", None)
    value.pop("ckpt", None)
    evaluation = value.get("EVALUATION", {})
    for key in (
        "output_dir",
        "catalog_path",
        "libero_plus_root",
        "checkpoint_sha256",
        "shard_index",
        "num_shards",
        "task_suite_name",
        "task_id",
        "continue_on_error",
        "preflight_only",
    ):
        evaluation.pop(key, None)
    return value


def _build_manifest(
    cfg: DictConfig,
    *,
    catalog: dict[str, Any],
    checkpoint: Path,
    dataset_stats_path: Path,
    selected_global_task_ids: list[int],
) -> dict[str, Any]:
    configured_hash = cfg.EVALUATION.get("checkpoint_sha256")
    checkpoint_hash = (
        str(configured_hash) if configured_hash else file_sha256(checkpoint)
    )
    checkpoint_stat = checkpoint.stat()
    semantic = {
        "schema": MANIFEST_SCHEMA,
        "protocol": {
            "benchmark": "LIBERO-Plus",
            "official_task_count": TOTAL_TASKS,
            "selected_task_count": len(selected_global_task_ids),
            "selected_global_task_ids": selected_global_task_ids,
            "trials_per_task": 1,
            "initial_state_index": 0,
            "prompt_mode": "pinned_task_language",
            "rng_reset": "same_fixed_seed_before_each_task",
        },
        "catalog_sha256": catalog["catalog_sha256"],
        "libero_plus": catalog["identity"],
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
            "size": checkpoint_stat.st_size,
            "mtime_ns": checkpoint_stat.st_mtime_ns,
        },
        "dataset_stats": {
            "path": str(dataset_stats_path),
            "sha256": file_sha256(dataset_stats_path),
        },
        "config": _semantic_config(cfg),
        "seed": int(cfg.seed),
    }
    return {**semantic, "manifest_sha256": canonical_sha256(semantic)}


def _publish_or_validate_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    path = output_dir / "run_manifest.json"
    try:
        write_json_atomic(path, manifest, exclusive=True)
    except FileExistsError:
        with path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != manifest:
            raise RuntimeError(f"existing run manifest differs from this worker: {path}")


def _valid_existing_receipt(
    path: Path,
    *,
    task_row: dict[str, Any],
    manifest: dict[str, Any],
) -> bool:
    try:
        with path.open("r", encoding="utf-8") as stream:
            receipt = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid existing receipt {path}: {exc}") from exc
    expected = {
        "schema": RECEIPT_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "catalog_sha256": manifest["catalog_sha256"],
        "suite": task_row["suite"],
        "runtime_task_id": task_row["runtime_task_id"],
        "classification_id": task_row["classification_id"],
        "global_task_id": task_row["global_task_id"],
        "task_name": task_row["name"],
        "category": task_row["category"],
        "difficulty_level": task_row["difficulty_level"],
        "prompt_mode": "pinned_task_language",
        "logical_bddl_file": task_row["bddl_file"],
        "resolved_bddl_path": task_row["resolved_bddl_path"],
        "resolved_init_path": task_row["resolved_init_path"],
        "prompt": task_row["task_language"],
        "seed": manifest["seed"],
        "initial_state_index": 0,
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            raise RuntimeError(
                f"existing receipt {path} has {key}={receipt.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    if type(receipt.get("success")) is not bool:
        raise RuntimeError(f"existing receipt {path} has non-boolean success")
    return True


def _assert_plain_protocol(cfg: DictConfig) -> None:
    if int(cfg.EVALUATION.get("num_trials", 1)) != 1:
        raise ValueError("LIBERO-Plus requires EVALUATION.num_trials=1")
    if bool(cfg.EVALUATION.get("save_video", False)):
        raise ValueError("LIBERO-Plus persistent workers require EVALUATION.save_video=false")
    if cfg.get("seed") is None:
        raise ValueError("LIBERO-Plus requires a fixed seed")


@contextlib.contextmanager
def _task_timeout(seconds: int):
    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"LIBERO-Plus task exceeded {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _validate_live_plus_identity(
    catalog: dict[str, Any], benchmark: Any, get_libero_path: Any
) -> None:
    identity = catalog["identity"]
    plus_root = Path(identity["libero_plus_root"]).resolve()
    benchmark_file = Path(benchmark.__file__).resolve()
    if not benchmark_file.is_relative_to(plus_root):
        raise RuntimeError(f"imported vanilla LIBERO from {benchmark_file}")
    recorded_revision = identity.get("libero_plus_revision")
    if recorded_revision and git_revision(plus_root) != recorded_revision:
        raise RuntimeError("live LIBERO-Plus revision differs from the catalog")
    if file_sha256(benchmark_file) != identity["benchmark_sha256"]:
        raise RuntimeError("live LIBERO-Plus benchmark source differs from the catalog")
    classification_file = Path(identity["classification_file"]).resolve()
    if file_sha256(classification_file) != identity["classification_sha256"]:
        raise RuntimeError("live task_classification.json differs from the catalog")
    configured_roots = {
        "bddl_root": Path(get_libero_path("bddl_files")).resolve(),
        "init_root": Path(get_libero_path("init_states")).resolve(),
        "assets_root": Path(get_libero_path("assets")).resolve(),
    }
    for key, actual in configured_roots.items():
        if actual != Path(identity[key]).resolve():
            raise RuntimeError(f"live {key} differs from the catalog")
    config_file = identity.get("libero_config_file")
    if config_file and file_sha256(config_file) != identity["libero_config_sha256"]:
        raise RuntimeError("live LIBERO config differs from the catalog")
    archive = identity.get("assets_archive")
    if archive:
        archive_stat = Path(archive["path"]).resolve().stat()
        if (
            archive_stat.st_size != int(archive["size"])
            or archive_stat.st_mtime_ns != int(archive["mtime_ns"])
        ):
            raise RuntimeError("LIBERO-Plus assets archive differs from the catalog")


def _make_suite_cache(benchmark: Any) -> dict[str, Any]:
    benchmark_dict = benchmark.get_benchmark_dict()
    cache: dict[str, Any] = {}
    for suite_name in SUITE_ORDER:
        with contextlib.redirect_stdout(io.StringIO()):
            cache[suite_name] = benchmark_dict[suite_name]()
    return cache


def _evaluate_task(
    *,
    task_row: dict[str, Any],
    suite: Any,
    model: torch.nn.Module,
    processor: RIFTProcessor,
    cfg: DictConfig,
    action_horizon: int,
    model_device: str,
) -> tuple[bool, str, float]:
    task_id = int(task_row["runtime_task_id"])
    task = suite.get_task(task_id)
    if task.name != task_row["name"]:
        raise RuntimeError(f"runtime task mismatch: {task.name!r}")
    if str(task.language) != str(task_row["task_language"]):
        raise RuntimeError(f"runtime prompt mismatch: {task.language!r}")
    with open_dict(cfg):
        cfg.EVALUATION.task_suite_name = task_row["suite"]
        cfg.EVALUATION.task_id = task_id

    fixed_seed = int(cfg.seed)
    set_global_seed(fixed_seed, get_worker_init_fn=False)
    if os.environ.get("LIBERO_STRICT_DET") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    initial_states = suite.get_task_init_states(task_id)
    if len(initial_states) < 1:
        raise RuntimeError(f"{task_row['suite']}/{task.name}: no initial state")

    env = None
    started = time.time()
    with _task_timeout(int(cfg.EVALUATION.get("task_timeout_seconds", 1800))):
        try:
            env, task_description = get_libero_env(
                task, LIBERO_ENV_RESOLUTION, fixed_seed
            )
            success, _ = run_single_episode(
                env=env,
                initial_state=initial_states[0],
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                episode_idx=0,
                action_horizon=action_horizon,
                model_device=model_device,
                collect_rollout_images=False,
                show_progress=False,
            )
        finally:
            if env is not None:
                env.close()
    return bool(success), str(task_description), time.time() - started


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero")
def evaluate_shard(cfg: DictConfig) -> None:
    _assert_plain_protocol(cfg)
    if cfg.ckpt is None:
        raise ValueError("Pass ckpt=/path/to/rift_step021700.pt")

    shard_index = int(cfg.EVALUATION.get("shard_index", 0))
    num_shards = int(cfg.EVALUATION.get("num_shards", 1))
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")

    plus_root = bootstrap_libero_plus(str(cfg.EVALUATION.libero_plus_root))
    from libero.libero import benchmark, get_libero_path

    catalog_path = Path(str(cfg.EVALUATION.catalog_path)).expanduser().resolve()
    output_dir = Path(str(cfg.EVALUATION.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog_file(catalog_path)
    if Path(catalog["identity"]["libero_plus_root"]).resolve() != plus_root:
        raise RuntimeError("catalog LIBERO-Plus root differs from evaluator root")
    _validate_live_plus_identity(catalog, benchmark, get_libero_path)

    checkpoint = Path(str(cfg.ckpt)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    requested_task_ids = cfg.EVALUATION.get("global_task_ids")
    if requested_task_ids is None:
        selected_global_task_ids = list(range(TOTAL_TASKS))
    else:
        selected_global_task_ids = sorted(int(value) for value in requested_task_ids)
        if len(selected_global_task_ids) != len(set(selected_global_task_ids)):
            raise ValueError("EVALUATION.global_task_ids contains duplicates")
        if not selected_global_task_ids or any(
            value < 0 or value >= TOTAL_TASKS for value in selected_global_task_ids
        ):
            raise ValueError("EVALUATION.global_task_ids is empty or out of range")

    manifest = _build_manifest(
        cfg,
        catalog=catalog,
        checkpoint=checkpoint,
        dataset_stats_path=dataset_stats_path,
        selected_global_task_ids=selected_global_task_ids,
    )
    _publish_or_validate_manifest(output_dir, manifest)

    selected_set = set(selected_global_task_ids)
    tasks = [
        row
        for row in catalog["tasks"]
        if int(row["global_task_id"]) in selected_set
        and int(row["global_task_id"]) % num_shards == shard_index
    ]
    pending: list[tuple[dict[str, Any], Path]] = []
    for row in tasks:
        receipt_path = (
            output_dir
            / "receipts"
            / row["suite"]
            / f"{int(row['runtime_task_id']):04d}.json"
        )
        if receipt_path.exists():
            _valid_existing_receipt(receipt_path, task_row=row, manifest=manifest)
        else:
            pending.append((row, receipt_path))
    if not pending:
        print(f"shard {shard_index}/{num_shards}: all {len(tasks)} receipts are valid")
        return
    if bool(cfg.EVALUATION.get("preflight_only", False)):
        print(
            f"shard {shard_index}/{num_shards}: preflight passed; "
            f"assigned={len(tasks)} pending={len(pending)}"
        )
        return

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    load_rift_checkpoint_exact(model, checkpoint)
    model = model.to(model_device).eval()

    processor: RIFTProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(str(dataset_stats_path))
    )
    configured_horizon = cfg.EVALUATION.get("action_horizon")
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if configured_horizon is None
        else int(configured_horizon)
    )
    if action_horizon <= 0:
        raise ValueError("EVALUATION.action_horizon must be positive")
    suite_cache = _make_suite_cache(benchmark)

    completed = 0
    errors_this_attempt = 0
    for row, receipt_path in pending:
        lock_path = (
            output_dir
            / "task_locks"
            / row["suite"]
            / f"{int(row['runtime_task_id']):04d}.lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            if receipt_path.exists():
                _valid_existing_receipt(receipt_path, task_row=row, manifest=manifest)
                continue
            try:
                success, prompt, duration = _evaluate_task(
                    task_row=row,
                    suite=suite_cache[row["suite"]],
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    action_horizon=action_horizon,
                    model_device=model_device,
                )
                receipt = {
                    "schema": RECEIPT_SCHEMA,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "catalog_sha256": catalog["catalog_sha256"],
                    "suite": row["suite"],
                    "runtime_task_id": row["runtime_task_id"],
                    "classification_id": row["classification_id"],
                    "global_task_id": row["global_task_id"],
                    "task_name": row["name"],
                    "category": row["category"],
                    "difficulty_level": row["difficulty_level"],
                    "prompt_mode": "pinned_task_language",
                    "logical_bddl_file": row["bddl_file"],
                    "resolved_bddl_path": row["resolved_bddl_path"],
                    "resolved_init_path": row["resolved_init_path"],
                    "prompt": prompt,
                    "success": success,
                    "duration_seconds": duration,
                    "seed": int(cfg.seed),
                    "initial_state_index": 0,
                    "checkpoint_loader": "load_rift_checkpoint_exact",
                }
                write_json_atomic(receipt_path, receipt, exclusive=True)
                completed += 1
                print(
                    f"[{completed}/{len(pending)}] {row['suite']}:"
                    f"{row['runtime_task_id']} success={int(success)} "
                    f"duration={duration:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                errors_this_attempt += 1
                error_path = (
                    output_dir
                    / "errors"
                    / row["suite"]
                    / f"{int(row['runtime_task_id']):04d}.json"
                )
                write_json_atomic(
                    error_path,
                    {
                        "schema": "rift.libero-plus-task-error.v1",
                        "manifest_sha256": manifest["manifest_sha256"],
                        "suite": row["suite"],
                        "runtime_task_id": row["runtime_task_id"],
                        "task_name": row["name"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                logging.exception("LIBERO-Plus task failed: %s", row["name"])
                fatal_cuda = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                    "CUDA out of memory" in str(exc)
                )
                if (
                    fatal_cuda
                    or isinstance(exc, TimeoutError)
                    or not bool(cfg.EVALUATION.get("continue_on_error", False))
                    or errors_this_attempt
                    >= int(cfg.EVALUATION.get("max_task_errors", 10))
                ):
                    raise

    write_json_atomic(
        output_dir / "workers" / f"shard_{shard_index:02d}.json",
        {
            "schema": "rift.libero-plus-worker-result.v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "shard_index": shard_index,
            "num_shards": num_shards,
            "assigned": len(tasks),
            "completed_this_attempt": completed,
        },
    )


if __name__ == "__main__":
    evaluate_shard()

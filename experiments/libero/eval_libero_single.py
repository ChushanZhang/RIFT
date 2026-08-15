"""Evaluate one RIFT checkpoint on one task from an official LIBERO suite."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
from PIL import Image
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

from experiments.checkpoint_utils import load_rift_checkpoint_exact  # noqa: E402
from experiments.libero.action_ensembler import ActionEnsembler  # noqa: E402
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
from rift.datasets.lerobot.processors.rift_processor import (  # noqa: E402
    RIFTProcessor,
)
from rift.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from rift.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)
from rift.utils.pytorch_utils import set_global_seed  # noqa: E402


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = str(mixed_precision).strip().lower()
    mapping = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if precision not in mapping:
        raise ValueError(
            f"Unsupported mixed_precision={mixed_precision!r}; expected no, fp16, or bf16."
        )
    return mapping[precision]


def _resolve_eval_device(cfg: DictConfig) -> str:
    requested = str(cfg.EVALUATION.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA evaluation was requested, but torch.cuda.is_available() is false."
        )
    return requested


def _expand_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    candidates: list[Path] = []
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    if explicit:
        candidates.append(_expand_path(explicit))

    checkpoint = _expand_path(str(cfg.ckpt))
    candidates.append(checkpoint.parent / "dataset_stats.json")
    for parent in list(checkpoint.parents)[1:4]:
        candidates.append(parent / "dataset_stats.json")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen and candidate.is_file():
            return candidate
        seen.add(candidate)
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate dataset_stats.json. Keep it beside the checkpoint or pass "
        f"EVALUATION.dataset_stats_path=/path/to/dataset_stats.json. Tried: {tried}"
    )


def _load_model_checkpoint(model: torch.nn.Module, checkpoint: str | Path) -> None:
    checkpoint = _expand_path(checkpoint).resolve()
    load_rift_checkpoint_exact(model, checkpoint)
    logging.info("Loaded checkpoint: %s", checkpoint)


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    source_width, source_height = pil_image.size
    scale = max(width / source_width, height / source_height)
    resized = pil_image.resize(
        (round(source_width * scale), round(source_height * scale)),
        resample=Image.BILINEAR,
    )
    resized_width, resized_height = resized.size
    left = max((resized_width - width) // 2, 0)
    top = max((resized_height - height) // 2, 0)
    return np.asarray(
        resized.crop((left, top, left + width, top + height)), dtype=np.uint8
    )


def _extract_sim_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: RIFTProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("LIBERO evaluation expects one merged state field.")
    state_key = state_meta[0]["key"]
    batch = {
        "state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}
    }
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    return batch["state"][state_key]


def _meta_hw(meta: dict[str, Any], camera_index: int) -> tuple[int, int]:
    shape = meta["shape"]
    if len(shape) != 3:
        raise ValueError(
            f"shape_meta.images[{camera_index}].shape must be [C,H,W], got {shape}."
        )
    return int(shape[1]), int(shape[2])


def _obs_to_model_input(
    obs: dict[str, Any],
    cfg: DictConfig,
    processor: RIFTProcessor,
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
    images = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    num_cameras = int(processor.num_output_cameras)
    if num_cameras != 2 or len(image_meta) < 2:
        raise ValueError(
            "The released LIBERO policy requires agent-view and wrist-view cameras."
        )

    primary_height, primary_width = _meta_hw(image_meta[0], 0)
    wrist_height, wrist_width = _meta_hw(image_meta[1], 1)
    primary = _center_crop_resize(
        images["image"], width=primary_width, height=primary_height
    )
    wrist = _center_crop_resize(
        images["wrist_image"], width=wrist_width, height=wrist_height
    )
    concatenation = str(cfg.data.train.get("concat_multi_camera", "horizontal"))
    if concatenation == "horizontal":
        rgb = np.concatenate([primary, wrist], axis=1)
    elif concatenation == "vertical":
        rgb = np.concatenate([primary, wrist], axis=0)
    else:
        raise ValueError(f"Unsupported concat_multi_camera={concatenation!r}.")

    expected_height, expected_width = map(int, cfg.data.train.video_size)
    if rgb.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            "LIBERO image geometry does not match the policy recipe: "
            f"got {rgb.shape[:2]}, expected {(expected_height, expected_width)}."
        )

    image_tensor = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    image_tensor = image_tensor * (2.0 / 255.0) - 1.0
    proprio = _normalize_proprio(_extract_sim_state(obs), processor)
    return image_tensor, proprio, images


def _denormalize_action(
    action: torch.Tensor,
    processor: RIFTProcessor,
) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action [B,T,D], got {tuple(action.shape)}.")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("LIBERO evaluation expects one merged action field.")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    return normalizer.backward(action.float().cpu()).numpy()


def _num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(
        cfg.data.train.action_video_freq_ratio
    ) + 1


def _predict_action_chunk(
    obs: dict[str, Any],
    task_description: str,
    model: torch.nn.Module,
    processor: RIFTProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    model_device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    image, proprio, images = _obs_to_model_input(
        obs,
        cfg,
        processor,
        device=model_device,
        dtype=model.torch_dtype,
    )
    prompt = DEFAULT_PROMPT.format(task=task_description)
    with torch.inference_mode():
        prediction = model.infer_action(
            prompt=prompt,
            input_image=image,
            action_horizon=int(action_horizon),
            num_video_frames=_num_video_frames(cfg),
            negative_prompt=str(cfg.EVALUATION.get("negative_prompt", "")),
            text_cfg_scale=float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
            num_inference_steps=int(cfg.EVALUATION.num_inference_steps),
            proprio=proprio,
            sigma_shift=(
                None
                if cfg.EVALUATION.get("sigma_shift") is None
                else float(cfg.EVALUATION.sigma_shift)
            ),
            seed=None if cfg.get("seed") is None else int(cfg.seed),
            rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
            tiled=bool(cfg.EVALUATION.get("tiled", False)),
        )

    action = _denormalize_action(prediction["action"], processor)[0]
    action[..., -1] = action[..., -1] * 2.0 - 1.0
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", True)):
        action[..., -1] = np.where(action[..., -1] >= 0.0, 1.0, -1.0)
    return action, images


def _get_max_steps(task_suite_name: str) -> int:
    maximum_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    try:
        return maximum_steps[task_suite_name]
    except KeyError as exc:
        raise ValueError(f"Unknown LIBERO task suite: {task_suite_name}") from exc


def run_single_episode(
    env: Any,
    initial_state: Any,
    task_description: str,
    model: torch.nn.Module,
    processor: RIFTProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    model_device: str,
) -> tuple[bool, list[dict[str, np.ndarray]]]:
    maximum_steps = _get_max_steps(str(cfg.EVALUATION.task_suite_name))
    wait_steps = int(cfg.EVALUATION.get("num_steps_wait", 30))
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 10))
    if replan_steps <= 0 or replan_steps > action_horizon:
        raise ValueError(
            f"replan_steps must be in [1, {action_horizon}], got {replan_steps}."
        )

    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(wait_steps):
        obs, _, done, _ = env.step(get_libero_dummy_action())
        if bool(done):
            return True, []

    ensembler = ActionEnsembler() if cfg.EVALUATION.use_action_ensembler else None
    pending_actions: list[list[float]] = []
    rollout_images: list[dict[str, np.ndarray]] = []
    success = False

    progress = tqdm(total=maximum_steps, desc=f"Episode {episode_idx + 1}")
    try:
        for step in range(maximum_steps):
            if not pending_actions:
                action_chunk, images = _predict_action_chunk(
                    obs,
                    task_description,
                    model,
                    processor,
                    cfg,
                    action_horizon=action_horizon,
                    model_device=model_device,
                )
                rollout_images.append(images)
                if ensembler is None:
                    pending_actions = action_chunk[:replan_steps].tolist()
                else:
                    ensembler.add_actions(action_chunk, step)
                    pending_actions = [
                        ensembler.get_action(timestamp).tolist()
                        for timestamp in range(step, step + replan_steps)
                    ]
            else:
                rollout_images.append(get_libero_image(obs))

            obs, _, done, _ = env.step(pending_actions.pop(0))
            progress.update(1)
            if bool(done):
                success = True
                break
    finally:
        progress.close()
    return success, rollout_images


def run_single_task(
    task: Any,
    initial_states: list[Any],
    model: torch.nn.Module,
    processor: RIFTProcessor,
    cfg: DictConfig,
    video_dir: Path,
    *,
    action_horizon: int,
    model_device: str,
) -> dict[str, Any]:
    env, task_description = get_libero_env(
        task, LIBERO_ENV_RESOLUTION, int(cfg.get("seed", 42))
    )
    results: dict[str, Any] = {
        "successes": 0,
        "success_episodes": [],
        "failure_episodes": [],
        "task_description": task_description,
    }
    try:
        for trial_idx in range(int(cfg.EVALUATION.num_trials)):
            success, rollout_images = run_single_episode(
                env,
                initial_states[trial_idx % len(initial_states)],
                task_description,
                model,
                processor,
                cfg,
                trial_idx,
                action_horizon=action_horizon,
                model_device=model_device,
            )
            key = "success_episodes" if success else "failure_episodes"
            results[key].append(trial_idx)
            results["successes"] += int(success)
            if bool(cfg.EVALUATION.get("save_video", True)) and rollout_images:
                save_rollout_video(
                    video_dir,
                    rollout_images,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    success,
                    task_description,
                )
    finally:
        env.close()
    return results


def _get_benchmark(task_suite_name: str):
    try:
        from libero.libero import benchmark
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LIBERO is required for benchmark evaluation. Install the official "
            "LIBERO environment before running this script."
        ) from exc
    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown LIBERO task suite: {task_suite_name}")
    return benchmark_dict[task_suite_name]()


def _get_task_initial_states(task_suite: Any, task_id: int) -> list[Any]:
    """Load trusted official LIBERO init states with PyTorch 2.6+.

    LIBERO currently calls ``torch.load`` without an explicit ``weights_only``
    value. PyTorch 2.6 changed that default to ``True``, while the official
    init-state files contain NumPy arrays. Keep unsafe pickle loading narrowly
    scoped to the benchmark asset selected by LIBERO itself.
    """
    try:
        from libero.libero import get_libero_path
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LIBERO is required for benchmark evaluation. Install the official "
            "LIBERO environment before running this script."
        ) from exc

    task = task_suite.get_task(task_id)
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    if not init_states_path.is_file():
        raise FileNotFoundError(f"LIBERO init-state file not found: {init_states_path}")
    states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    return list(states)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero")
def eval_single_process(cfg: DictConfig) -> dict[str, Any]:
    started = time.time()
    if cfg.ckpt is None:
        raise ValueError("Pass ckpt=/path/to/rift_step021700.pt.")
    if int(cfg.EVALUATION.get("env_num", 1)) != 1:
        raise ValueError(
            "eval_libero_single.py supports env_num=1; use run_libero_manager.py "
            "for task-level multi-GPU parallelism."
        )
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    stats_path = _resolve_dataset_stats_path(cfg)
    processor: RIFTProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))

    action_horizon = int(
        cfg.EVALUATION.get("action_horizon") or (int(cfg.data.train.num_frames) - 1)
    )
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}.")

    suite_name = str(cfg.EVALUATION.task_suite_name)
    task_id = int(cfg.EVALUATION.task_id)
    task_suite = _get_benchmark(suite_name)
    if task_id < 0 or task_id >= int(task_suite.n_tasks):
        raise ValueError(
            f"task_id={task_id} is outside {suite_name} task range [0, {task_suite.n_tasks})."
        )
    task = task_suite.get_task(task_id)
    initial_states = _get_task_initial_states(task_suite, task_id)
    if not initial_states:
        raise RuntimeError(
            f"No initial states available for {suite_name} task {task_id}."
        )

    output_root = _expand_path(str(cfg.EVALUATION.output_dir))
    suite_dir = output_root / suite_name
    video_dir = suite_dir / "videos"
    suite_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.EVALUATION.get("save_video", True)):
        video_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "task_suite": suite_name,
        "task_id": task_id,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results.update(
        run_single_task(
            task,
            initial_states,
            model,
            processor,
            cfg,
            video_dir,
            action_horizon=action_horizon,
            model_device=model_device,
        )
    )
    results["duration"] = time.time() - started

    result_path = suite_dir / f"gpu{cfg.gpu_id}_task{task_id}_results.json"
    result_path.write_text(
        json.dumps(results, indent=2, cls=NumpyEncoder) + "\n", encoding="utf-8"
    )
    print(
        f"{suite_name} task {task_id}: {results['successes']}/"
        f"{results['total_episodes']} successes ({results['duration']:.1f}s)"
    )
    return results


if __name__ == "__main__":
    eval_single_process()

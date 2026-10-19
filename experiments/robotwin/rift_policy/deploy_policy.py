"""RIFT adapter for RoboTwin's policy interface."""

from __future__ import annotations

import inspect
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

from experiments.checkpoint_utils import load_rift_checkpoint_exact  # noqa: E402
from rift.datasets.lerobot.processors.rift_processor import (  # noqa: E402
    RIFTProcessor,
)
from rift.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from rift.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "none", "null"}
    )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> int | None:
    return None if _is_none_like(value) else int(value)


def _parse_optional_float(value: Any) -> float | None:
    return None if _is_none_like(value) else float(value)


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = str(mixed_precision).strip().lower()
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(
        f"Unsupported mixed_precision: {mixed_precision}. Expected one of: no, fp16, bf16."
    )


def _resolve_sim_cfg_name(sim_cfg_path: str | None, sim_cfg_name: str | None) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            return cfg_path.relative_to(configs_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
    return "sim_robotwin.yaml" if _is_none_like(sim_cfg_name) else str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: str | None,
    sim_cfg_name: str | None,
    sim_task: str | None,
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path, sim_cfg_name)
    overrides = [] if _is_none_like(sim_task) else [f"task={sim_task}"]
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str((PROJECT_ROOT / "configs").resolve()),
    ):
        return compose(config_name=config_name, overrides=overrides)


def _resolve_dataset_stats_path(dataset_stats_path: str | None) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required for RoboTwin evaluation."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset stats file not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.asarray(
        pil_image.resize(size_wh, resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


class RIFTRoboTwinPolicy:
    def __init__(
        self,
        *,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: float | None,
        seed: int | None,
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
    ) -> None:
        model_cfg_copy = OmegaConf.create(
            OmegaConf.to_container(model_cfg, resolve=True)
        )
        model_cfg_copy.load_text_encoder = True
        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        load_rift_checkpoint_exact(self.model, checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: RIFTProcessor = instantiate(processor_cfg).eval()
        stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = max(1, min(int(replan_steps), self.action_horizon))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self.num_video_frames = int(num_video_frames)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        logger.info(
            "Initialized RIFT RoboTwin policy | checkpoint=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]
        state_batch = {
            "state": {
                state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            }
        }
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action [B, T, D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected one merged action key in shape_meta['action'].")
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        return normalizer.backward(action.float().cpu()).numpy()

    def _build_image_tensor(self, observation: dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        image = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
        image_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.model.device, dtype=self.model.torch_dtype)
        )
        return image_tensor * (2.0 / 255.0) - 1.0

    def _infer_action_chunk(
        self,
        observation: dict[str, Any],
        instruction: str,
    ) -> np.ndarray:
        image_tensor = self._build_image_tensor(observation)
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state)
        infer_kwargs = {
            "prompt": DEFAULT_PROMPT.format(task=instruction),
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = self.num_video_frames

        infer_start = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            prediction = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_start
        return self._denormalize_action(prediction["action"])[0]

    def _fill_action_queue(self, observation: dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation, instruction)
        for action in action_chunk[: self.replan_steps]:
            self.pending_actions.append(np.asarray(action, dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env: Any, observation: dict[str, Any] | None) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError("Observation is required at each replanning step.")
            self._fill_action_queue(observation, task_env.get_instruction())
        if not self.pending_actions:
            raise RuntimeError("RIFT returned an empty action chunk.")

        action = self.pending_actions.popleft()
        sim_start = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_start
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

    def get_timing_rollout(self) -> dict[str, float]:
        return {key: float(value) for key, value in self._timing_rollout.items()}

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: dict[str, Any] | None) -> dict[str, Any] | None:
    return observation


def get_model(usr_args: dict[str, Any]) -> RIFTRoboTwinPolicy:
    cfg = _compose_sim_cfg(
        sim_cfg_path=usr_args.get("sim_cfg_path"),
        sim_cfg_name=usr_args.get("sim_cfg_name"),
        sim_task=usr_args.get("sim_task"),
    )
    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; using CPU.")
        device = "cpu"
    model_dtype = _mixed_precision_to_model_dtype(
        str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    )
    dataset_stats_path = _resolve_dataset_stats_path(usr_args.get("dataset_stats_path"))

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        action_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
    if action_horizon is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 24))
    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", 10))
    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(
        usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0))
    )
    negative_prompt = str(
        usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", ""))
    )
    rand_device = str(
        usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu"))
    )
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )

    num_frames = int(cfg.data.train.num_frames)
    frame_ratio = int(cfg.data.train.action_video_freq_ratio)
    return RIFTRoboTwinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(num_frames - 1) // frame_ratio + 1,
    )


def eval(
    task_env: Any,
    model: RIFTRoboTwinPolicy,
    observation: dict[str, Any] | None,
) -> None:
    model.step(task_env, encode_obs(observation))


def reset_model(model: RIFTRoboTwinPolicy) -> None:
    model.reset()

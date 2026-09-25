"""Simulator and video helpers for standalone LIBERO evaluation."""

from __future__ import annotations

import math
from pathlib import Path
import re
import time
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw


LIBERO_ENV_RESOLUTION = 256
_DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_libero_env(task: Any, resolution: int, seed: int):
    """Create the official single-process off-screen LIBERO environment."""
    try:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LIBERO is required for benchmark evaluation. Install the official "
            "LIBERO environment before running this script."
        ) from exc

    task_bddl_file = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=int(resolution),
        camera_widths=int(resolution),
    )
    env.seed(int(seed))
    return env, task.language


def get_libero_dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def get_libero_image(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Rotate simulator camera frames to match the released training data."""
    return {
        "image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "wrist_image": np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1, ::-1]
        ),
    }


def _frame_to_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        views = []
        for name, value in frame.items():
            array = np.asarray(value, dtype=np.uint8)
            image = Image.fromarray(array)
            ImageDraw.Draw(image).text((8, 8), str(name), fill=(255, 255, 255))
            views.append(np.asarray(image))
        array = np.concatenate(views, axis=1)
    elif isinstance(frame, Image.Image):
        array = np.asarray(frame.convert("RGB"))
    else:
        array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Video frame must be HWC RGB, got {array.shape}.")
    return np.ascontiguousarray(array)


def _write_mp4(frames: list[Any], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("Cannot save an empty rollout video.")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [_frame_to_array(frame) for frame in frames]
    height, width = arrays[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for array in arrays:
            if array.shape[:2] != (height, width):
                array = np.asarray(
                    Image.fromarray(array).resize((width, height), Image.BILINEAR)
                )
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def save_rollout_video(
    rollout_dir: str | Path,
    rollout_images: list[Any],
    idx: str | int,
    success: bool,
    task_description: str,
    *,
    fps: int = 24,
) -> Path:
    task_slug = re.sub(r"[^a-z0-9]+", "_", task_description.lower()).strip("_")[:50]
    path = Path(rollout_dir) / (
        f"{_DATE_TIME}--episode={idx}--success={bool(success)}--task={task_slug}.mp4"
    )
    _write_mp4(rollout_images, path, fps=fps)
    return path


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert an ``(x, y, z, w)`` quaternion to axis-angle coordinates."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / denominator).astype(np.float32)


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    action[..., -1] *= -1.0
    return action

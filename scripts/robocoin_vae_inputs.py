"""Prepare RoboCOIN video windows without loading policy models or action stats.

An episode is decoded once, in order, into float32 camera canvases. The original
RIFT dataset performs the final composition and normalization. Validation takes
an independent route through the training decoder and the full 33-frame image
processor, then compares the resulting nine-frame windows byte for byte.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


VIDEO_OFFSETS = tuple(range(0, 33, 4))
CANVAS_SHAPE = (3, 384, 320)
HEAD_CAMERA = "observation.images.cam_head_left_rgb"
HIGH_CAMERA = "observation.images.cam_high_left_rgb"
HIGH_MONO_CAMERA = "observation.images.cam_high_rgb"
WRIST_CAMERAS = (
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)


def _json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def discover_episodes(
    root: str | Path,
    *,
    camera_keys_by_dataset: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Discover a v2.1 dataset or immediate child datasets, including symlinks.

    Each returned, JSON-serializable record contains dataset, path,
    episode_index, length, fps, camera_keys, video_paths and parquet_path.
    Camera overrides use dataset directory names and head/left/right order.
    """
    root = Path(root).expanduser().absolute()
    if not root.is_dir():
        raise FileNotFoundError(root)
    datasets = ([root] if (root / "meta/info.json").is_file() else
                sorted(path for path in root.iterdir()
                       if path.is_dir() and (path / "meta/info.json").is_file()))
    if not datasets:
        raise ValueError(f"No LeRobot datasets found under {root}")
    overrides = dict(camera_keys_by_dataset or {})
    unknown_overrides = set(overrides) - {path.name for path in datasets}
    if unknown_overrides:
        raise ValueError(f"Unknown dataset camera overrides: {sorted(unknown_overrides)}")

    result = []
    seen_paths: set[Path] = set()
    for dataset in datasets:
        resolved = dataset.resolve()
        if resolved in seen_paths:
            raise ValueError(f"Dataset appears more than once through symlinks: {dataset}")
        seen_paths.add(resolved)
        meta_paths = {name: dataset / "meta" / name
                      for name in ("info.json", "episodes.jsonl", "tasks.jsonl")}
        meta_sha256 = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for name, path in meta_paths.items()}
        info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
        fps = float(info["fps"])
        if fps != 30.0:
            raise ValueError(f"Expected RoboCOIN fps=30, got {fps} in {dataset}")
        features = info["features"]
        if dataset.name in overrides:
            camera_keys = list(overrides[dataset.name])
        else:
            heads = [key for key in (HEAD_CAMERA, HIGH_CAMERA, HIGH_MONO_CAMERA) if key in features]
            if len(heads) != 1:
                raise ValueError(
                    f"Ambiguous or missing head camera in {dataset}: {heads}; "
                    "provide camera_keys_by_dataset in head/left/right order."
                )
            camera_keys = [heads[0], *WRIST_CAMERAS]
        if len(camera_keys) != 3 or len(set(camera_keys)) != 3:
            raise ValueError(f"Exactly three distinct cameras are required: {camera_keys}")
        for key in camera_keys:
            if features.get(key, {}).get("dtype") != "video":
                raise ValueError(f"Missing video feature {key!r} in {dataset}")

        chunks_size = int(info["chunks_size"])
        if chunks_size <= 0:
            raise ValueError(f"Invalid chunks_size in {dataset}: {chunks_size}")
        episodes = _json_lines(dataset / "meta/episodes.jsonl")
        if len(episodes) != int(info["total_episodes"]):
            raise ValueError(f"Episode count differs from meta/info.json in {dataset}")
        if sum(int(item["length"]) for item in episodes) != int(info["total_frames"]):
            raise ValueError(f"Episode lengths differ from total_frames in {dataset}")
        seen_indices: set[int] = set()
        for item in sorted(episodes, key=lambda value: int(value["episode_index"])):
            index, length = int(item["episode_index"]), int(item["length"])
            if index < 0 or index in seen_indices or length <= 0:
                raise ValueError(f"Invalid or duplicate episode metadata in {dataset}: {item}")
            seen_indices.add(index)
            fields = {"episode_index": index, "episode_chunk": index // chunks_size}
            parquet = dataset / info["data_path"].format(**fields)
            videos = {key: dataset / info["video_path"].format(video_key=key, **fields)
                      for key in camera_keys}
            for path in [parquet, *videos.values()]:
                if not path.is_file():
                    raise FileNotFoundError(path)
            result.append({
                "dataset": dataset.name, "path": str(dataset),
                "episode_index": index, "length": length, "fps": fps,
                "camera_keys": camera_keys, "video_paths": {k: str(v) for k, v in videos.items()},
                "parquet_path": str(parquet), "tolerance_s": 1e-4,
                "meta_paths": {name: str(path) for name, path in meta_paths.items()},
                "meta_sha256": meta_sha256,
            })
    return result


def _episode_timestamps(episode: Mapping[str, Any]) -> torch.Tensor:
    import pyarrow.parquet as pq

    table = pq.read_table(episode["parquet_path"], columns=["timestamp"])
    timestamps = torch.tensor(table["timestamp"].to_pylist(), dtype=torch.float32)
    if (len(timestamps) != int(episode["length"])
            or not torch.isfinite(timestamps).all()
            or (len(timestamps) > 1 and not (timestamps[1:] > timestamps[:-1]).all())):
        raise ValueError(f"Invalid episode timestamps: {episode['parquet_path']}")
    return timestamps


@contextmanager
def _worker_threads(count=1):
    """Set image-transform threads; reference validation uses DataLoader's one."""
    previous = torch.get_num_threads()
    try:
        if previous != count:
            torch.set_num_threads(count)
        yield
    finally:
        if previous != count:
            torch.set_num_threads(previous)


def _dataset_from_sample(sample: dict[str, Any], *, subsample: bool):
    from rift.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
    from rift.datasets.lerobot.robot_video_dataset import RobotVideoDataset

    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.lerobot_dataset = [sample]
    dataset.max_padding_retry = 0
    dataset.skip_padding_as_possible = False
    frames = sample["pixel_values"].shape[1]
    dataset.video_sample_indices = list(VIDEO_OFFSETS) if subsample else list(range(frames))
    dataset.sample_video_before_transforms = False
    dataset.concat_multi_camera = "robotwin"
    dataset.override_instruction = ""
    dataset.resize_transform = ResizeSmallestSideAspectPreserving({"img_w": 320, "img_h": 384})
    dataset.crop_transform = CenterCrop({"img_w": 320, "img_h": 384})
    dataset.normalize_transform = Normalize({"mean": 0.5, "std": 0.5})
    dataset._get_cached_text_context = lambda _prompt: (torch.zeros(1, 1), torch.ones(1, dtype=torch.bool))
    return dataset


def _sample_for_images(pixel_values: torch.Tensor) -> dict[str, Any]:
    frames = pixel_values.shape[1]
    return {
        "pixel_values": pixel_values, "instruction": "",
        "image_is_pad": torch.zeros(frames, dtype=torch.bool),
        "action": torch.zeros(frames - 1, 1),
        "action_is_pad": torch.zeros(frames - 1, dtype=torch.bool),
        "proprio": torch.zeros(frames, 1),
        "proprio_is_pad": torch.zeros(frames, dtype=torch.bool),
    }


def _compose_canvas(pixel_values: torch.Tensor) -> torch.Tensor:
    """Use the training dataset's composition, including its final resize."""
    frames = pixel_values.shape[1]
    if frames == 1:
        # _get requires two frames; its spatial transforms do not mix time.
        pixel_values = pixel_values.repeat(1, 2, 1, 1, 1)
    dataset = _dataset_from_sample(_sample_for_images(pixel_values), subsample=False)
    return dataset._get(0)["video"].permute(1, 0, 2, 3)[:frames].contiguous()


def prepare_episode(episode: Mapping[str, Any], *, chunk_size: int = 32) -> torch.Tensor:
    """Sequentially decode three MP4s; retain only resized episode canvases.

    Returns contiguous CPU float32 [N,3,384,320]. Raw high-resolution frames
    live for one camera chunk only. Every decoded timestamp must match its
    parquet row within the same tolerance used by the training video loader.
    """
    import av
    from torchvision.transforms import Resize
    from rift.datasets.lerobot.transforms.image import ToTensor

    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    length = int(episode["length"])
    timestamps = _episode_timestamps(episode)
    tolerance = float(episode.get("tolerance_s", 1e-4))
    camera_keys = list(episode["camera_keys"])
    if len(camera_keys) != 3:
        raise ValueError("Expected head, left wrist and right wrist cameras")
    output = torch.empty((length, *CANVAS_SHAPE), dtype=torch.float32)
    to_tensor, resize = ToTensor(), Resize([224, 224])
    with _worker_threads(4), ExitStack() as stack:
        decoders = {}
        for key in camera_keys:
            container = stack.enter_context(av.open(episode["video_paths"][key]))
            decoders[key] = iter(container.decode(video=0))
        for start in range(0, length, chunk_size):
            end = min(length, start + chunk_size)
            resized = []
            for key in camera_keys:
                raw = []
                for index in range(start, end):
                    try:
                        frame = next(decoders[key])
                    except StopIteration as exc:
                        raise ValueError(f"Video ends at frame {index}, expected {length}: {episode['video_paths'][key]}") from exc
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError(f"Video frame has no timestamp: {key} frame {index}")
                    pts = torch.tensor(float(frame.pts * frame.time_base), dtype=torch.float32)
                    if not abs(float(pts - timestamps[index])) < tolerance:
                        raise ValueError(f"Video/parquet timestamp mismatch: {key} frame {index}, pts={pts.item()}, expected={timestamps[index].item()}")
                    # Match torchvision's PyAV VideoReader RGB conversion.
                    raw.append(torch.as_tensor(frame.to_rgb().to_ndarray()).permute(2, 0, 1))
                frames = torch.stack(raw)
                del raw, frame
                # Preserve decoder float conversion and BaseLerobotDataset's uint8 round trip.
                frames = (frames.to(torch.float32) / 255.0 * 255.0).to(torch.uint8)
                resized.append(resize(to_tensor(frames)))
                del frames
            output[start:end] = _compose_canvas(torch.stack(resized))
        for key, decoder in decoders.items():
            if next(decoder, None) is not None:
                raise ValueError(f"Video contains more than {length} frames: {episode['video_paths'][key]}")
    return output


def make_window(canvas: torch.Tensor, start: int) -> torch.Tensor:
    """Return [3,9,384,320], repeating the final episode frame for padding."""
    if (canvas.device.type != "cpu" or canvas.dtype != torch.float32
            or canvas.ndim != 4 or tuple(canvas.shape[1:]) != CANVAS_SHAPE):
        raise ValueError("Expected CPU float32 canvas [N,3,384,320]")
    if not 0 <= start < len(canvas):
        raise IndexError(f"Window start {start} is outside episode length {len(canvas)}")
    indices = [min(start + offset, len(canvas) - 1) for offset in VIDEO_OFFSETS]
    return canvas[indices].permute(1, 0, 2, 3).contiguous()


class _Passthrough:
    def forward(self, value):
        return value


def _original_window(episode: Mapping[str, Any], start: int, timestamps: torch.Tensor, backend: str):
    from omegaconf import OmegaConf
    from torchvision.transforms import Resize
    from rift.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    from rift.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
    from rift.datasets.lerobot.processors.rift_processor import RIFTProcessor
    from rift.datasets.lerobot.transforms.image import ToTensor

    length = int(episode["length"])
    indices = [min(start + offset, length - 1) for offset in range(33)]
    padded = torch.tensor([start + offset >= length for offset in range(33)])
    query = timestamps[indices].tolist()
    images = {}
    for key in episode["camera_keys"]:
        decoded = decode_video_frames(episode["video_paths"][key], query,
                                      float(episode.get("tolerance_s", 1e-4)), backend=backend)
        meta = {"key": key, "lerobot_key": key, "raw_shape": list(decoded.shape[1:])}
        images[key] = BaseLerobotDataset._get_image(None, meta, {key: decoded})

    # Exercise the real image processor while passing unrelated modalities through.
    processor = RIFTProcessor.__new__(RIFTProcessor)
    processor.shape_meta = OmegaConf.create({
        "images": [{"key": key, "shape": [3, 224, 224]} for key in episode["camera_keys"]],
        "action": [], "state": [],
    })
    processor.num_obs_steps = 33
    processor.num_output_cameras = 3
    processor.image_sample_indices = None
    processor.action_output_dim = processor.proprio_output_dim = 1
    processor.train_transforms = [ToTensor(), Resize([224, 224])]
    processor.val_transforms = processor.train_transforms
    processor._is_train = True
    processor.action_state_transforms = []
    processor.delta_action_dim_mask = None
    processor._normalizer = processor.action_state_merger = _Passthrough()
    processor.augment_instruction = lambda _data: ""
    sample = processor.preprocess({
        "images": images, "image_is_pad": padded, "idx": start,
        "action": torch.zeros(32, 1), "action_is_pad": padded[:-1],
        "action_dim_is_pad": torch.zeros(1, dtype=torch.bool),
        "state": torch.zeros(33, 1), "state_is_pad": padded,
        "state_dim_is_pad": torch.zeros(1, dtype=torch.bool),
    })
    dataset = _dataset_from_sample(sample, subsample=True)
    return dataset._get(0)


def validate_episode_inputs(
    episode: Mapping[str, Any],
    canvas: torch.Tensor,
    starts: Sequence[int] | None = None,
    *,
    backend: str | None = None,
) -> dict[str, Any]:
    """Compare against original decoding/33-frame transforms; fail on any byte."""
    from rift.datasets.lerobot.lerobot.datasets.video_utils import get_safe_default_codec

    length = int(episode["length"])
    if len(canvas) != length:
        raise ValueError(f"Canvas length {len(canvas)} differs from episode length {length}")
    starts = list(dict.fromkeys([0, length // 2, length - 1] if starts is None else starts))
    if not starts:
        raise ValueError("Validation requires at least one window")
    backend = backend or get_safe_default_codec()
    timestamps = _episode_timestamps(episode)
    windows = []
    with _worker_threads():
        for start in starts:
            optimized = make_window(canvas, start)
            original = _original_window(episode, start, timestamps, backend)
            expected = original["video"].contiguous()
            if expected.shape != optimized.shape or expected.dtype != optimized.dtype:
                raise AssertionError(f"Window shape/dtype mismatch for {episode['dataset']}:{episode['episode_index']}:{start}")
            different = expected.view(torch.uint8) != optimized.view(torch.uint8)
            byte_count = int(different.sum().item())
            if byte_count:
                max_diff = float((expected - optimized).abs().max().item())
                raise AssertionError(f"Video input parity failed for {episode['dataset']}:{episode['episode_index']}:{start}: different_bytes={byte_count}, max_abs_diff={max_diff}")
            expected_padding = [start + offset >= length for offset in VIDEO_OFFSETS]
            if original["image_is_pad"].tolist() != expected_padding:
                raise AssertionError(f"Window padding mismatch at {start}")
            windows.append({
                "start": int(start), "shape": list(expected.shape), "dtype": str(expected.dtype),
                "different_bytes": 0, "image_is_pad": expected_padding,
                "sha256": hashlib.sha256(expected.view(torch.uint8).numpy().tobytes()).hexdigest(),
            })
    return {"dataset": episode["dataset"], "episode_index": int(episode["episode_index"]),
            "length": length, "backend": backend, "bitwise_equal": True, "windows": windows}

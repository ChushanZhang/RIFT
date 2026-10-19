"""Decode each existing LIBERO episode once, preserving training RGB windows."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import torch

from rift.utils.libero_indexed_vae_cache import META_FILES, SOURCE_CAMERAS


VIDEO_OFFSETS = tuple(range(0, 33, 4))
CANVAS_SHAPE = (3, 224, 448)


def discover_episodes(dataset_directories):
    episodes = []
    seen = set()
    for directory in map(Path, dataset_directories):
        directory = directory.resolve(strict=True)
        if directory in seen:
            raise ValueError(f"Repeated dataset: {directory}")
        seen.add(directory)
        metadata = {name: directory / "meta" / name for name in META_FILES}
        info = json.loads(metadata["info.json"].read_text())
        if info["fps"] != 20 or any(info["features"].get(camera, {}).get("dtype") != "video"
                                    for camera in SOURCE_CAMERAS):
            raise ValueError(f"Expected LIBERO two-camera video at 20 fps: {directory}")
        inventory = [json.loads(line) for line in metadata["episodes.jsonl"].read_text().splitlines() if line]
        if (len(inventory) != info["total_episodes"]
                or sum(row["length"] for row in inventory) != info["total_frames"]
                or len({row["episode_index"] for row in inventory}) != len(inventory)):
            raise ValueError(f"Incomplete or duplicate episode inventory: {directory}")
        metadata_checksums = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in metadata.items()}
        for row in sorted(inventory, key=lambda item: item["episode_index"]):
            index, length = int(row["episode_index"]), int(row["length"])
            if index < 0 or length <= 0:
                raise ValueError(f"Invalid episode: {row}")
            fields = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
            videos = {camera: str(directory / info["video_path"].format(video_key=camera, **fields))
                      for camera in SOURCE_CAMERAS}
            parquet = directory / info["data_path"].format(**fields)
            for path in [parquet, *map(Path, videos.values())]:
                if not path.is_file():
                    raise FileNotFoundError(path)
            episodes.append({"dataset": directory.name, "path": str(directory),
                             "episode_index": index, "length": length, "fps": 20,
                             "camera_keys": list(SOURCE_CAMERAS), "video_paths": videos,
                             "parquet_path": str(parquet), "tolerance_s": 1e-4,
                             "meta_paths": {name: str(path) for name, path in metadata.items()},
                             "meta_sha256": metadata_checksums})
    if not episodes:
        raise ValueError("No episodes found")
    return episodes


def _canvas_dataset():
    from rift.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
    from rift.datasets.lerobot.robot_video_dataset import RobotVideoDataset

    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.sample_video_before_transforms = True
    dataset.concat_multi_camera = "horizontal"
    dataset.resize_transform = ResizeSmallestSideAspectPreserving({"img_w": 448, "img_h": 224})
    dataset.crop_transform = CenterCrop({"img_w": 448, "img_h": 224})
    dataset.normalize_transform = Normalize({"mean": 0.5, "std": 0.5})
    return dataset


def prepare_episode(episode, chunk_size=16):
    """Return CPU float32 [frames,3,224,448] through the real canvas transform."""
    import av
    import pyarrow.parquet as parquet
    from torchvision.transforms import Resize
    from rift.datasets.lerobot.transforms.image import ToTensor

    table = parquet.read_table(episode["parquet_path"], columns=["timestamp", "frame_index", "episode_index"])
    length = episode["length"]
    timestamps = torch.tensor(table["timestamp"].to_pylist(), dtype=torch.float32)
    if (len(table) != length or table["frame_index"].to_pylist() != list(range(length))
            or set(table["episode_index"].to_pylist()) != {episode["episode_index"]}
            or not torch.allclose(timestamps, torch.arange(length) / 20, rtol=0, atol=1e-4)):
        raise ValueError(f"Episode frame/timestamp mapping differs: {episode['parquet_path']}")
    dataset = _canvas_dataset()
    to_tensor, resize = ToTensor(), Resize([224, 224])
    output = torch.empty((length, *CANVAS_SHAPE), dtype=torch.float32)
    with ExitStack() as stack:
        decoders = {camera: iter(stack.enter_context(av.open(path)).decode(video=0))
                    for camera, path in episode["video_paths"].items()}
        for start in range(0, length, chunk_size):
            end = min(length, start + chunk_size)
            cameras = []
            for camera in SOURCE_CAMERAS:
                raw = []
                for index in range(start, end):
                    frame = next(decoders[camera], None)
                    if frame is None or frame.pts is None or frame.time_base is None:
                        raise ValueError(f"Missing frame/timestamp: {camera}:{index}")
                    timestamp = torch.tensor(float(frame.pts * frame.time_base), dtype=torch.float32)
                    if abs(float(timestamp - timestamps[index])) >= episode["tolerance_s"]:
                        raise ValueError(f"Video timestamp differs: {camera}:{index}")
                    raw.append(torch.as_tensor(frame.to_rgb().to_ndarray()).permute(2, 0, 1))
                # Preserve the training decoder's float -> uint8 round trip.
                pixels = (torch.stack(raw).float() / 255.0 * 255.0).to(torch.uint8)
                cameras.append(resize(to_tensor(pixels)))
            output[start:end] = dataset._get_video({"pixel_values": torch.stack(cameras)}).permute(1, 0, 2, 3)
        if any(next(decoder, None) is not None for decoder in decoders.values()):
            raise ValueError("Video contains more frames than the episode metadata")
    return output


def make_window(canvas, start):
    if canvas.dtype != torch.float32 or tuple(canvas.shape[1:]) != CANVAS_SHAPE:
        raise ValueError("Expected float32 [frames,3,224,448] canvas")
    if not 0 <= start < len(canvas):
        raise IndexError(start)
    indices = [min(start + offset, len(canvas) - 1) for offset in VIDEO_OFFSETS]
    return canvas[indices].permute(1, 0, 2, 3).contiguous()


def validate_episode_pixels(episode, canvas):
    """Compare three windows to the original decoder and image processing path."""
    from torchvision.transforms import Resize
    from rift.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
    from rift.datasets.lerobot.transforms.image import ToTensor

    starts = sorted({0, len(canvas) // 2, len(canvas) - 1})
    dataset = _canvas_dataset()
    for start in starts:
        query = [min(start + offset, len(canvas) - 1) / 20 for offset in VIDEO_OFFSETS]
        cameras = []
        for camera in SOURCE_CAMERAS:
            frames = decode_video_frames(episode["video_paths"][camera], query, 1e-4, backend="pyav")
            cameras.append(Resize([224, 224])(ToTensor()((frames * 255.0).to(torch.uint8))))
        reference = dataset._get_video({"pixel_values": torch.stack(cameras)})
        actual = make_window(canvas, start)
        if not torch.equal(reference.contiguous().view(torch.uint8), actual.view(torch.uint8)):
            raise RuntimeError(f"Training pixel parity failed: {episode['dataset']}:{episode['episode_index']}:{start}")
    return {"dataset": episode["dataset"], "episode_index": episode["episode_index"],
            "starts": starts, "pixels_bitwise_equal": True}

"""Read the audited RoboCOIN window index without decoding video during training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
import torch

from .vae_latent_cache import identity_hash, tensor_digest


LATENT_SHAPE = (48, 3, 24, 20)
CAMERAS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _read_json(path):
    return json.loads(path.read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class IndexedVaeLatentReader:
    """Bind task/episode/frame indices to immutable, previously verified pixels.

    Startup validates every episode index, source video fingerprint and current
    frame/timestamp mapping. Action/state unit conversions do not affect pixels.
    Each access checks the latent payload checksum; missing entries are errors.
    """

    def __init__(self, index_dir, cache_dir, dataset_dirs, num_frames, video_size,
                 video_sample_indices, concat_multi_camera, processor):
        from torchvision.transforms import InterpolationMode, Resize
        from rift.datasets.lerobot.transforms.image import ToTensor

        expected = {
            "num_frames": 33, "video_offsets": list(range(0, 33, 4)),
            "per_camera_resize": [224, 224], "canvas": [384, 320],
            "normalization": {"mean": 0.5, "std": 0.5},
            "window_policy": "each frame, repeat final frame at episode boundary",
        }
        _require(num_frames == 33 and list(video_sample_indices) == expected["video_offsets"]
                 and list(video_size) == expected["canvas"] and concat_multi_camera == "robotwin",
                 "Indexed RoboCOIN latents require 33 observations, stride 4 and robotwin 384x320.")
        _require([meta["key"] for meta in processor.shape_meta["images"]] == list(CAMERAS)
                 and all(list(meta["shape"]) == [3, 224, 224]
                         for meta in processor.shape_meta["images"])
                 and processor.num_output_cameras == 3 and processor.num_obs_steps == 33,
                 "Indexed latents require head/left wrist/right wrist camera order.")
        for transforms in (processor.train_transforms, processor.val_transforms):
            if isinstance(transforms, dict):
                _require(set(transforms) == set(CAMERAS), "Image transform camera keys differ.")
            pipelines = transforms.values() if isinstance(transforms, dict) else [transforms]
            for pipeline in pipelines:
                _require(pipeline is not None and len(pipeline) == 2
                         and type(pipeline[0]) is ToTensor and type(pipeline[1]) is Resize,
                         "Indexed latents require exactly ToTensor followed by Resize.")
                resize = pipeline[1]
                _require(list(resize.size) == [224, 224]
                         and resize.interpolation == InterpolationMode.BILINEAR
                         and resize.antialias is True and resize.max_size is None,
                         "Image resize settings differ from the precomputed latents.")

        run = Path(index_dir)
        report = _read_json(run / "validation.json")
        _require(report.get("status") == "passed" and report.get("all_starts_covered_once")
                 and report.get("all_indices_sha256_verified"),
                 "Indexed loading requires a completed, validated VAE precompute run.")
        self.namespace = report["namespace"]
        _require(isinstance(self.namespace, str) and HEX_DIGEST.fullmatch(self.namespace),
                 "Invalid VAE cache namespace.")
        self.directory = Path(cache_dir) / self.namespace
        identity = _read_json(self.directory / "identity.json")
        _require(identity_hash(identity) == self.namespace and identity == report["identity"],
                 "VAE cache identity differs from the validated precompute run.")
        _require(report["latent_shape"] == list(LATENT_SHAPE)
                 and report["latent_dtype"] == "torch.bfloat16", "Unexpected latent contract.")
        settings = _read_json(run / "settings.json")
        for key, value in expected.items():
            _require(settings.get(key) == value and report["preprocessing"].get(key) == value,
                     f"Precomputed image settings differ: {key}")
        inventory = _read_json(run / "episodes.json")
        episodes = {(entry["dataset"], entry["episode_index"]): entry for entry in inventory}
        _require(len(episodes) == len(inventory) == report["episode_count"]
                 and sum(entry["length"] for entry in inventory) == report["window_count"],
                 "Precompute episode inventory is incomplete or duplicated.")
        fingerprints = _read_json(run / "input_fingerprints.json")
        self.dataset_names = [Path(path).name for path in dataset_dirs]
        _require(len(set(self.dataset_names)) == len(self.dataset_names), "Duplicate dataset names.")
        self._keys = {}
        for dataset_index, directory in enumerate(dataset_dirs):
            directory = Path(directory)
            info = _read_json(directory / "meta/info.json")
            current_episodes = [json.loads(line) for line in
                                (directory / "meta/episodes.jsonl").read_text().splitlines() if line]
            expected_episodes = {episode for name, episode in episodes if name == directory.name}
            _require(len(current_episodes) == len(expected_episodes) == info["total_episodes"]
                     and {entry["episode_index"] for entry in current_episodes} == expected_episodes,
                     f"Episode coverage differs from the cached dataset: {directory}")
            _require(sum(entry["length"] for entry in current_episodes) == info["total_frames"],
                     f"Dataset frame count is inconsistent: {directory}")
            for entry in current_episodes:
                episode_index = entry["episode_index"]
                source = episodes[directory.name, episode_index]
                _require(entry["length"] == source["length"] and info["fps"] == source["fps"] == 30,
                         f"Episode length or fps changed: {directory.name}/{episode_index}")
                fields = {"episode_index": episode_index,
                          "episode_chunk": episode_index // info["chunks_size"]}
                for camera, source_camera in zip(CAMERAS, source["camera_keys"], strict=True):
                    video = directory / info["video_path"].format(
                        video_key=f"observation.images.{camera}", **fields)
                    original = Path(source["video_paths"][source_camera])
                    _require(video.resolve(strict=True) == original.resolve(strict=True),
                             f"Camera source or ordering changed: {video}")
                    stat = video.stat()
                    _require(fingerprints[str(original)] == {"size": stat.st_size,
                                                            "mtime_ns": stat.st_mtime_ns},
                             f"Video changed since VAE precomputation: {video}")
                self._validate_frames(directory / info["data_path"].format(**fields), source)
                index_path = run / "indices" / directory.name / f"episode_{episode_index:06d}.jsonl"
                done = _read_json(index_path.with_suffix(".done.json"))
                raw = index_path.read_bytes()
                _require(done.get("namespace") == self.namespace
                         and done.get("dataset") == directory.name
                         and done.get("episode_index") == episode_index
                         and done.get("windows") == source["length"]
                         and done.get("index_sha256") == hashlib.sha256(raw).hexdigest(),
                         f"Incomplete or corrupted VAE index: {index_path}")
                keys = []
                for start_frame, line in enumerate(raw.splitlines()):
                    row = json.loads(line)
                    key = row.get("key")
                    _require(row.get("start_frame") == start_frame and isinstance(key, str)
                             and HEX_DIGEST.fullmatch(key), f"Invalid VAE index row: {index_path}")
                    keys.append(bytes.fromhex(key))
                _require(len(keys) == source["length"], f"Missing VAE window indices: {index_path}")
                # Compact immutable bytes keep forked DataLoader workers inexpensive.
                self._keys[dataset_index, episode_index] = b"".join(keys)

    @staticmethod
    def _validate_frames(parquet, source):
        columns = ["episode_index", "frame_index", "timestamp"]
        table = pq.read_table(parquet, columns=columns)
        original = pq.read_table(source["parquet_path"], columns=columns)
        length = source["length"]
        _require(len(table) == length and table.equals(original),
                 f"Training view frame/timestamp mapping differs from source: {parquet}")
        _require(np.array_equal(table["frame_index"].to_numpy(), np.arange(length))
                 and np.all(table["episode_index"].to_numpy() == source["episode_index"])
                 and np.allclose(table["timestamp"].to_numpy(), np.arange(length) / source["fps"],
                                 rtol=0, atol=source["tolerance_s"]),
                 f"Indexed latents require consecutive episode frames and original timestamps: {parquet}")

    def key(self, dataset_index, episode_index, start_frame):
        try:
            keys = self._keys[int(dataset_index), int(episode_index)]
        except KeyError as exc:
            raise KeyError(f"No VAE index for dataset={dataset_index}, episode={episode_index}") from exc
        start_frame = int(start_frame)
        if not 0 <= start_frame < len(keys) // 32:
            raise IndexError(f"VAE window start frame out of range: {start_frame}")
        return keys[start_frame * 32:(start_frame + 1) * 32].hex()

    def load(self, dataset_index, episode_index, start_frame):
        key = self.key(dataset_index, episode_index, start_frame)
        path = self.directory / key[:2] / f"{key}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        latent = payload["latent"]
        if (payload["namespace"] != self.namespace or payload["key"] != key
                or latent.dtype != torch.bfloat16 or tuple(latent.shape) != LATENT_SHAPE
                or payload["sha256"] != tensor_digest(latent) or not torch.isfinite(latent).all()):
            raise RuntimeError(f"Invalid or corrupted indexed VAE latent: {path}")
        return latent

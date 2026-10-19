"""Read audited LIBERO two-camera window latents without decoding video."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
import torch

from .vae_latent_cache import identity_hash


CONTRACT = "libero_2cam224_v1"
PACKED_FORMAT = "libero_episode_bf16_v1"
LATENT_SHAPE = (48, 3, 14, 28)
CAMERAS = ("image", "wrist_image")
SOURCE_CAMERAS = tuple(f"observation.images.{camera}" for camera in CAMERAS)
META_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl")
HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
PREPROCESSING = {
    "contract": CONTRACT,
    "num_frames": 33,
    "video_offsets": list(range(0, 33, 4)),
    "per_camera_resize": [224, 224],
    "canvas": [224, 448],
    "normalization": {"mean": 0.5, "std": 0.5},
    "window_policy": "each frame, repeat final frame at episode boundary",
}


def _read_json(path):
    return json.loads(path.read_text())


CONTENT_FINGERPRINTS = "input_content_sha256.json"
logger = logging.getLogger(__name__)
_drift_reported = False


def _report_drift_once(path, expected, actual):
    global _drift_reported
    if not _drift_reported:
        _drift_reported = True
        logger.warning("Input mtime differs from VAE precomputation on this node while size and SHA-256 match "
                       "(first: %s; expected=%s, actual=%s); such inputs are accepted by content hash.",
                       path, expected, actual)


def _read_optional_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_content_fingerprints(index_dir):
    """Add SHA-256 content hashes next to the size/mtime fingerprints of an index run.

    Run on a node where every size/mtime fingerprint still matches (the index builder's view), so that
    the hashes describe the audited inputs; readers then accept a file with a drifted mtime only when its
    content hash matches. Returns the number of hashed files.
    """
    run = Path(index_dir)
    fingerprints = _read_json(run / "input_fingerprints.json")
    hashes = {}
    for path, expected in fingerprints.items():
        stat = Path(path).stat()
        _require(expected == {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                 f"Cannot hash a drifted input; run where fingerprints match: {path}")
        hashes[path] = _sha256_file(path)
    target = run / CONTENT_FINGERPRINTS
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(hashes, indent=1, sort_keys=True))
    temporary.replace(target)
    return len(hashes)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_episode_index(cache_dir):
    """Resolve the published LIBERO episode index for default training."""
    if cache_dir is None:
        return None
    root = Path(cache_dir).expanduser().resolve()
    ready_path = root / "ready.json"
    if not ready_path.is_file():
        raise ValueError(f"LIBERO episode cache is not ready: {ready_path}. "
                         "Run scripts/precompute_libero_vae.py and the training-loader audit, "
                         "or set vae_cache_dir=null for online VAE encoding.")
    ready = _read_json(ready_path)
    _require(ready.get("status") == "passed" and ready.get("contract") == CONTRACT
             and ready.get("format") == PACKED_FORMAT,
             f"LIBERO episode readiness is missing or invalid: {ready_path}")
    _require(Path(ready["cache_dir"]).resolve() == root,
             f"LIBERO ready.json names a different cache directory: {ready_path}")
    index = Path(ready["index_dir"])
    training = _read_json(index / "training_validation.json")
    _require(training.get("status") == "passed" and training.get("namespace") == ready.get("namespace")
             and training.get("window_count") == ready.get("window_count"),
             f"LIBERO training-loader audit is missing or differs: {index}")
    return str(index)


def _read_packed_manifest(directory, namespace, expected_episodes):
    """Bind a published pack inventory to the already validated window indices.

    Every indexed episode must have exactly one matching safetensors file.
    """
    manifest_path = directory / "episodes" / "manifest.json"
    _require(manifest_path.is_file(),
             f"Missing LIBERO safetensors manifest: {manifest_path}. "
             "Run scripts/precompute_libero_vae.py encode and finalize first.")
    manifest = _read_json(manifest_path)
    _require(isinstance(manifest, dict) and manifest.get("format") == PACKED_FORMAT
             and manifest.get("namespace") == namespace,
             f"Invalid packed LIBERO VAE manifest: {manifest_path}")
    rows = manifest.get("episodes")
    _require(isinstance(rows, list) and len(rows) == len(expected_episodes),
             f"Packed LIBERO VAE episode coverage differs: {manifest_path}")
    packed = {}
    for row in rows:
        _require(isinstance(row, dict), f"Invalid packed LIBERO VAE episode entry: {manifest_path}")
        dataset = row.get("dataset")
        episode_index = row.get("episode_index")
        _require(isinstance(dataset, str) and dataset not in ("", ".", "..")
                 and Path(dataset).name == dataset and type(episode_index) is int
                 and episode_index >= 0,
                 f"Invalid packed LIBERO VAE episode identity: {manifest_path}")
        identity = (dataset, episode_index)
        _require(identity in expected_episodes and identity not in packed,
                 f"Unexpected or duplicate packed LIBERO VAE episode: {identity}")
        dataset_index, windows, index_sha256 = expected_episodes[identity]
        relative = f"episodes/{dataset}/episode_{episode_index:06d}.safetensors"
        checksum = row.get("file_sha256")
        _require(type(row.get("windows")) is int and row["windows"] == windows
                 and row.get("index_sha256") == index_sha256
                 and row.get("path") == relative and isinstance(checksum, str)
                 and HEX_DIGEST.fullmatch(checksum),
                 f"Packed LIBERO VAE episode differs from validated index: {identity}")
        path = directory / relative
        _require(path.is_file(), f"Missing packed LIBERO VAE episode: {path}")
        packed[dataset_index, episode_index] = (path, dataset, index_sha256, windows)
    _require(len(packed) == len(expected_episodes),
             f"Packed LIBERO VAE episode coverage differs: {manifest_path}")
    return packed


class LiberoIndexedVaeLatentReader:
    """Bind dataset/episode/start indices to validated frozen VAE outputs.

    All starts, including windows padded at an episode boundary, are required.
    Training still loads the original state/action rows and kinematic targets;
    only deterministic RGB processing and VAE encoding are replaced.
    """

    def __init__(self, index_dir, cache_dir, dataset_dirs, num_frames, video_size,
                 video_sample_indices, concat_multi_camera, processor):
        from torchvision.transforms import InterpolationMode, Resize
        from rift.datasets.lerobot.transforms.image import ToTensor

        _require(num_frames == 33
                 and list(video_sample_indices) == PREPROCESSING["video_offsets"]
                 and list(video_size) == PREPROCESSING["canvas"]
                 and concat_multi_camera == "horizontal",
                 "Indexed LIBERO latents require 33 observations, stride 4 and horizontal 224x448.")
        _require([meta["key"] for meta in processor.shape_meta["images"]] == list(CAMERAS)
                 and all(list(meta["shape"]) == [3, 224, 224]
                         for meta in processor.shape_meta["images"])
                 and processor.num_output_cameras == 2 and processor.num_obs_steps == 33,
                 "Indexed LIBERO latents require image/wrist_image camera order.")
        for transforms in (processor.train_transforms, processor.val_transforms):
            if isinstance(transforms, dict):
                _require(set(transforms) == set(CAMERAS), "Image transform camera keys differ.")
            pipelines = transforms.values() if isinstance(transforms, dict) else [transforms]
            for pipeline in pipelines:
                _require(pipeline is not None and len(pipeline) == 2
                         and type(pipeline[0]) is ToTensor and type(pipeline[1]) is Resize,
                         "Indexed LIBERO latents require exactly ToTensor followed by Resize.")
                resize = pipeline[1]
                _require(list(resize.size) == [224, 224]
                         and resize.interpolation == InterpolationMode.BILINEAR
                         and resize.antialias is True and resize.max_size is None,
                         "Image resize settings differ from the precomputed LIBERO latents.")

        run = Path(index_dir)
        report = _read_json(run / "validation.json")
        _require(report.get("status") == "passed" and report.get("all_starts_covered_once")
                 and report.get("all_indices_sha256_verified"),
                 "Indexed loading requires a completed, validated LIBERO VAE precompute run.")
        self.namespace = report["namespace"]
        _require(isinstance(self.namespace, str) and HEX_DIGEST.fullmatch(self.namespace),
                 "Invalid VAE cache namespace.")
        self.directory = Path(cache_dir) / self.namespace
        identity = _read_json(self.directory / "identity.json")
        _require(identity_hash(identity) == self.namespace and identity == report["identity"],
                 "VAE cache identity differs from the validated precompute run.")
        _require(report["latent_shape"] == list(LATENT_SHAPE)
                 and report["latent_dtype"] == "torch.bfloat16", "Unexpected LIBERO latent contract.")
        settings = _read_json(run / "settings.json")
        for key, value in PREPROCESSING.items():
            _require(settings.get(key) == value and report["preprocessing"].get(key) == value,
                     f"Precomputed LIBERO image settings differ: {key}")

        inventory = _read_json(run / "episodes.json")
        episodes = {(entry["dataset"], entry["episode_index"]): entry for entry in inventory}
        _require(len(episodes) == len(inventory) == report["episode_count"]
                 and sum(entry["length"] for entry in inventory) == report["window_count"],
                 "Precompute episode inventory is incomplete or duplicated.")
        fingerprints = _read_json(run / "input_fingerprints.json")
        content_hashes = _read_optional_json(run / CONTENT_FINGERPRINTS)
        # A training view may have another directory name while its metadata,
        # action rows and videos resolve to the same audited source files.
        # Bind by those files, never by a directory-name substitution.
        self.dataset_names = []
        for directory in dataset_dirs:
            directory = Path(directory)
            current_meta = {name: (directory / "meta" / name).resolve(strict=True)
                            for name in META_FILES}
            matches = {entry["dataset"] for entry in inventory
                       if all(current_meta[name] == Path(entry["meta_paths"][name]).resolve(strict=True)
                              for name in META_FILES)}
            _require(len(matches) == 1, f"Training metadata differs from the audited LIBERO source: {directory}")
            self.dataset_names.append(matches.pop())
        _require(len(set(self.dataset_names)) == len(self.dataset_names), "Duplicate dataset names.")
        _require(set(self.dataset_names) == {entry["dataset"] for entry in inventory},
                 "Training datasets differ from the validated LIBERO cache inventory.")
        self._keys = {}
        expected_packed = {}
        for dataset_index, directory in enumerate(dataset_dirs):
            directory = Path(directory)
            source_name = self.dataset_names[dataset_index]
            info = _read_json(directory / "meta/info.json")
            current_episodes = [json.loads(line) for line in
                                (directory / "meta/episodes.jsonl").read_text().splitlines() if line]
            expected_episodes = {episode for name, episode in episodes if name == source_name}
            _require(len(current_episodes) == len(expected_episodes) == info["total_episodes"]
                     and {entry["episode_index"] for entry in current_episodes} == expected_episodes,
                     f"Episode coverage differs from the cached dataset: {directory}")
            _require(sum(entry["length"] for entry in current_episodes) == info["total_frames"],
                     f"Dataset frame count is inconsistent: {directory}")
            _require(info["fps"] == 20
                     and all(info["features"].get(camera, {}).get("dtype") == "video"
                             for camera in SOURCE_CAMERAS),
                     f"Indexed LIBERO latents require fps=20 and both RGB cameras: {directory}")
            meta_hashes = {name: hashlib.sha256((directory / "meta" / name).read_bytes()).hexdigest()
                           for name in META_FILES}
            for entry in current_episodes:
                episode_index = entry["episode_index"]
                source = episodes[source_name, episode_index]
                _require(entry["length"] == source["length"] and source["fps"] == 20,
                         f"Episode length or fps changed: {directory.name}/{episode_index}")
                _require(source.get("meta_sha256") == meta_hashes
                         and set(source.get("meta_paths", {})) == set(META_FILES),
                         f"Dataset metadata changed since VAE precomputation: {directory}")
                for name in META_FILES:
                    _require((directory / "meta" / name).resolve(strict=True)
                             == Path(source["meta_paths"][name]).resolve(strict=True),
                             f"Dataset metadata source changed: {directory}/meta/{name}")
                _require(source["camera_keys"] == list(SOURCE_CAMERAS),
                         f"Camera ordering differs from LIBERO precomputation: {directory}")
                fields = {"episode_index": episode_index,
                          "episode_chunk": episode_index // info["chunks_size"]}
                for camera in SOURCE_CAMERAS:
                    video = directory / info["video_path"].format(video_key=camera, **fields)
                    original = Path(source["video_paths"][camera])
                    _require(video.resolve(strict=True) == original.resolve(strict=True),
                             f"Camera source changed: {video}")
                    self._validate_fingerprint(original, fingerprints, content_hashes)
                original_parquet = Path(source["parquet_path"])
                self._validate_fingerprint(original_parquet, fingerprints, content_hashes)
                self._validate_frames(directory / info["data_path"].format(**fields), source)

                index_path = run / "indices" / source_name / f"episode_{episode_index:06d}.jsonl"
                done = _read_json(index_path.with_suffix(".done.json"))
                raw = index_path.read_bytes()
                index_sha256 = hashlib.sha256(raw).hexdigest()
                _require(done.get("namespace") == self.namespace
                         and done.get("dataset") == source_name
                         and done.get("episode_index") == episode_index
                         and done.get("windows") == source["length"]
                         and done.get("index_sha256") == index_sha256,
                         f"Incomplete or corrupted LIBERO VAE index: {index_path}")
                keys = []
                for start_frame, line in enumerate(raw.splitlines()):
                    row = json.loads(line)
                    key = row.get("key")
                    _require(row.get("start_frame") == start_frame and isinstance(key, str)
                             and HEX_DIGEST.fullmatch(key), f"Invalid VAE index row: {index_path}")
                    keys.append(bytes.fromhex(key))
                _require(len(keys) == source["length"], f"Missing LIBERO VAE window indices: {index_path}")
                self._keys[dataset_index, episode_index] = b"".join(keys)
                expected_packed[source_name, episode_index] = (dataset_index, source["length"], index_sha256)
        self._packed = _read_packed_manifest(self.directory, self.namespace, expected_packed)

    @staticmethod
    def _validate_fingerprint(path, fingerprints, content_hashes=None):
        stat = path.stat()
        expected = fingerprints.get(str(path))
        actual = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if expected == actual:
            return
        # Compute nodes have reported a different mtime than the node that built the index for the same,
        # unmodified file (client-side Lustre metadata). When the index ships content hashes, a file whose
        # size matches is accepted on a matching SHA-256; anything else is still a changed input.
        if (expected is not None and expected.get("size") == stat.st_size and content_hashes
                and content_hashes.get(str(path)) == _sha256_file(path)):
            _report_drift_once(path, expected, actual)
            return
        raise ValueError(f"Input changed since VAE precomputation: {path}; "
                         f"expected={expected}, actual={actual}")

    @staticmethod
    def _validate_frames(parquet, source):
        columns = ["episode_index", "frame_index", "timestamp"]
        table = pq.read_table(parquet, columns=columns)
        original = pq.read_table(source["parquet_path"], columns=columns)
        length = source["length"]
        tolerance = source["tolerance_s"]
        _require(isinstance(tolerance, (int, float)) and 0 < tolerance <= 1e-4,
                 "LIBERO timestamp tolerance must be positive and at most 1e-4 seconds.")
        _require(len(table) == length and table.equals(original),
                 f"Training frame/timestamp mapping differs from source: {parquet}")
        _require(np.array_equal(table["frame_index"].to_numpy(), np.arange(length))
                 and np.all(table["episode_index"].to_numpy() == source["episode_index"])
                 and np.allclose(table["timestamp"].to_numpy(), np.arange(length) / source["fps"],
                                 rtol=0, atol=tolerance),
                 f"Indexed LIBERO latents require consecutive episode frames and original timestamps: {parquet}")

    def key(self, dataset_index, episode_index, start_frame):
        try:
            keys = self._keys[int(dataset_index), int(episode_index)]
        except KeyError as exc:
            raise KeyError(f"No LIBERO VAE index for dataset={dataset_index}, episode={episode_index}") from exc
        start_frame = int(start_frame)
        if not 0 <= start_frame < len(keys) // 32:
            raise IndexError(f"LIBERO VAE window start frame out of range: {start_frame}")
        return keys[start_frame * 32:(start_frame + 1) * 32].hex()

    def load(self, dataset_index, episode_index, start_frame):
        from .libero_episode_cache import EpisodeHeaderCache, load_window

        key = self.key(dataset_index, episode_index, start_frame)
        header_cache = getattr(self, "_packed_header_cache", None)
        if header_cache is None:
            header_cache = EpisodeHeaderCache()
            self._packed_header_cache = header_cache
        path, dataset, index_sha256, windows = self._packed[int(dataset_index), int(episode_index)]
        return load_window(path, int(start_frame), key=key, namespace=self.namespace,
                           dataset=dataset, episode_index=int(episode_index),
                           index_sha256=index_sha256, window_count=windows,
                           header_cache=header_cache)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LIBERO indexed VAE cache utilities")
    parser.add_argument("--write-content-hashes", metavar="INDEX_DIR",
                        help="write input_content_sha256.json for a validated precompute run")
    args = parser.parse_args()
    if args.write_content_hashes:
        print(f"hashed {write_content_fingerprints(args.write_content_hashes)} inputs")

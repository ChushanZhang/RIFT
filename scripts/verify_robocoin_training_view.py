"""Check real RIFT/FastWAM training samples and existing VAE keys on CPU."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fastwam_targets(value):
    if isinstance(value, dict):
        return {key: fastwam_targets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [fastwam_targets(item) for item in value]
    if isinstance(value, str) and value.startswith("rift."):
        return value.replace("rift.", "fastwam.", 1).replace(
            ".rift_processor.RIFTProcessor", ".fastwam_processor.FastWAMProcessor"
        ).replace(".relative_joint.RelativeJointTransform", ".relative_action.RelativeJointTransform")
    return value


def exact_sample(dataset, global_index, dataset_index, start_frame):
    """Keep the training transforms, but reject its fallback to random samples."""
    base = dataset.lerobot_dataset
    original_split = base._split_lerobot_sample

    def checked_split(sample):
        for key, expected in (("dataset_index", dataset_index), ("episode_index", 0),
                              ("frame_index", start_frame)):
            if sample[key].numel() != 1 or sample[key].item() != expected:
                raise ValueError(f"Training loader substituted the requested sample: {key}")
        return original_split(sample)

    base._split_lerobot_sample = checked_split
    try:
        return dataset._get(global_index)
    finally:
        base._split_lerobot_sample = original_split


def verify(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.rift_repo.resolve()))
    sys.path.insert(0, str((args.fastwam_repo / "src").resolve()))
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from rift.utils.vae_latent_cache import tensor_digest

    torch.set_num_threads(args.cpu_threads)
    config_path = args.config or args.rift_repo / "configs/data/robocoin_multitask.yaml"
    config = OmegaConf.create({"data": OmegaConf.load(config_path)})
    train = config.data.train
    if args.train_root:
        train.dataset_dirs = [str(args.train_root / Path(path).name) for path in train.dataset_dirs]
    if args.stats:
        train.pretrained_norm_stats = str(args.stats)
    if args.text_cache:
        train.text_embedding_cache_dir = str(args.text_cache)
    train.vae_cache_index_dir = None
    resolved = OmegaConf.to_container(train, resolve=True)
    if not resolved.get("pretrained_norm_stats") or resolved["skip_padding_as_possible"]:
        raise ValueError("Verification requires saved statistics and deterministic window starts")
    stats_path = Path(resolved["pretrained_norm_stats"])
    if args.output.resolve() == stats_path.resolve():
        raise ValueError("Output must not overwrite normalization statistics")
    stats = json.loads(stats_path.read_text())
    vae_validation_path = args.vae_run_dir / "validation.json"
    vae_validation = json.loads(vae_validation_path.read_text())
    if vae_validation["status"] != "passed":
        raise ValueError("The VAE precomputation must have passed validation")
    namespace = vae_validation["namespace"]
    if len(namespace) != 64 or any(char not in "0123456789abcdef" for char in namespace):
        raise ValueError("Invalid VAE namespace")
    namespace_dir = args.vae_cache_dir / namespace
    if not namespace_dir.is_dir():
        raise FileNotFoundError(namespace_dir)

    specifications = []
    frame_offset = episode_count = 0
    metadata_hashes = {}
    for dataset_index, directory in enumerate(resolved["dataset_dirs"]):
        root = Path(directory)
        info_path, episodes_path = root / "meta/info.json", root / "meta/episodes.jsonl"
        info = json.loads(info_path.read_text())
        view = info.get("robocoin_training_view", {})
        if view.get("version") != 2 or view.get("gripper_units") != "mm":
            raise ValueError(f"Expected a canonical mm view: {root}")
        episodes = sorted((json.loads(line) for line in episodes_path.read_text().splitlines()
                           if line.strip()), key=lambda episode: episode["episode_index"])
        if (len(episodes) != info["total_episodes"] or episodes[0]["episode_index"] != 0
                or sum(episode["length"] for episode in episodes) != info["total_frames"]):
            raise ValueError(f"Inconsistent episode metadata: {root}")
        index_path = args.vae_run_dir / "indices" / root.name / "episode_000000.jsonl"
        index_rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
        length = episodes[0]["length"]
        if [row["start_frame"] for row in index_rows] != list(range(length)):
            raise ValueError(f"VAE index does not cover the same first episode: {root}")
        for start in (0, length // 2, length - 1):
            specifications.append({"dataset": root.name, "dataset_index": dataset_index,
                                   "episode_index": 0, "start_frame": start,
                                   "global_index": frame_offset + start,
                                   "vae_key": index_rows[start]["key"]})
        metadata_hashes.update({str(path): sha256(path) for path in (info_path, episodes_path, index_path)})
        frame_offset += info["total_frames"]
        episode_count += info["total_episodes"]
    if len(specifications) != 9 or episode_count != 342 or frame_offset != 238864:
        raise ValueError("Expected the audited three-task, 342-episode, 238864-frame snapshot")
    if stats["num_episodes"] != episode_count or stats["num_transition"] != frame_offset:
        raise ValueError("Statistics do not cover the complete mixed dataset")
    if (vae_validation["episode_count"] != episode_count
            or vae_validation["window_count"] != frame_offset):
        raise ValueError("VAE cache coverage differs from the training views")

    fast_config = fastwam_targets(copy.deepcopy(resolved))
    fast_module = importlib.import_module("fastwam.datasets.lerobot.robot_video_dataset")
    accepted = inspect.signature(fast_module.RobotVideoDataset.__init__).parameters
    removed = []
    for option in ("sample_video_before_transforms", "text_memory_cache_size",
                   "vae_cache_index_dir", "vae_cache_dir"):
        if option in fast_config and option not in accepted:
            del fast_config[option]
            removed.append(option)
    shapes = {"video": (3, 9, 384, 320), "action": (32, 14), "proprio": (32, 14),
              "context": (128, 4096), "context_mask": (128,), "image_is_pad": (9,),
              "action_is_pad": (32,), "proprio_is_pad": (33,)}
    reports = []
    with tempfile.TemporaryDirectory(prefix="training-view-smoke-", dir=args.output.parent) as work:
        datasets = []
        for package, package_config in (("rift", resolved), ("fastwam", fast_config)):
            misc = importlib.import_module(f"{package}.utils.misc")
            misc.register_work_dir(Path(work) / package)
            dataset = instantiate(OmegaConf.create(package_config))
            if len(dataset) != frame_offset:
                raise ValueError(f"Wrong {package} training length: {len(dataset)}")
            datasets.append(dataset)
        for specification in specifications:
            outputs = [exact_sample(dataset, specification["global_index"],
                                    specification["dataset_index"], specification["start_frame"])
                       for dataset in datasets]
            a, b = outputs
            if set(a) != set(b) or a["prompt"] != b["prompt"]:
                raise ValueError(f"RIFT/FastWAM output structure or prompt mismatch: {specification}")
            for key, shape in shapes.items():
                for package, sample in zip(("rift", "fastwam"), outputs):
                    tensor = sample[key]
                    if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                            or tensor.device.type != "cpu" or not torch.isfinite(tensor).all()):
                        raise ValueError(f"Invalid {package} {key} tensor: {specification}")
                if a[key].dtype != b[key].dtype or not torch.equal(a[key], b[key]):
                    raise ValueError(f"RIFT/FastWAM {key} mismatch: {specification}")
            if a["video"].dtype != torch.float32 or a["context"].dtype != torch.bfloat16:
                raise ValueError("Unexpected video or T5 dtype")
            key = tensor_digest(a["video"])
            if key != specification["vae_key"]:
                raise ValueError(f"Training pixels differ from the original VAE cache: {specification}")
            latent_path = namespace_dir / key[:2] / f"{key}.pt"
            if not latent_path.is_file() or latent_path.stat().st_size <= 48 * 3 * 24 * 20 * 2:
                raise ValueError(f"Missing or truncated cached latent: {latent_path}")
            reports.append({**specification, "prompt": a["prompt"], "vae_cache_file": str(latent_path),
                            "rift_fastwam_bitwise_equal": True, "vae_pixel_key_equal": True,
                            "dtypes": {name: str(a[name].dtype) for name in shapes}})
            print(json.dumps({"status": "sample_passed", **specification}), flush=True)
    return {"status": "passed", "validated_at": datetime.now(timezone.utc).isoformat(),
            "episodes": episode_count, "frames": frame_offset, "sample_count": len(reports),
            "sample_policy": "First episode per task; first, middle, and final window",
            "config": str(config_path), "config_sha256": sha256(config_path),
            "resolved_data_config": resolved, "stats_sha256": sha256(stats_path),
            "metadata_sha256": metadata_hashes, "vae_namespace": namespace,
            "vae_validation_sha256": sha256(vae_validation_path),
            "fastwam_removed_unsupported_options": removed,
            "tensor_shapes": {key: list(shape) for key, shape in shapes.items()}, "samples": reports,
            "scope": "Real CPU dataset outputs, normalization/text parity and cached pixel keys; "
                     "no model loading, GPU encoding or FastWAM VAE cache integration."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rift-repo", type=Path, required=True)
    parser.add_argument("--fastwam-repo", type=Path, required=True)
    parser.add_argument("--vae-run-dir", type=Path, required=True)
    parser.add_argument("--vae-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--text-cache", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve previous verification")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = verify(args)
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                  "validated_at": datetime.now(timezone.utc).isoformat()}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "output": str(args.output),
                      "samples": report["sample_count"]}), flush=True)


if __name__ == "__main__":
    main()

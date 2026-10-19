"""Compute training normalization statistics for converted 14D RoboCOIN data.

Use the existing Galaxea processor and statistics implementation. Split flat
vectors into named fields without changing units. Joint statistics require the
canonical training views produced by prepare_robocoin_training_view.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from rift.datasets.lerobot.base_lerobot_dataset import (
    BaseLerobotDataset,
    sliding_window_with_replication,
)
from rift.datasets.lerobot.utils.normalizer import (
    load_dataset_stats_from_json,
    save_dataset_stats_to_json,
)


RAW_NAMES = [
    *[f"left_arm_joint_{i}_rad" for i in range(1, 7)],
    *[f"right_arm_joint_{i}_rad" for i in range(1, 7)],
    "left_gripper_open", "right_gripper_open",
]
FIELD_INDICES = {
    "left_arm": list(range(6)),
    "left_gripper": [12],
    "right_arm": list(range(6, 12)),
    "right_gripper": [13],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RoboCOINStatsDataset(BaseLerobotDataset):
    """Expose flat 14D records to the unchanged training statistics routine."""

    def __init__(self, dataset_dirs, shape_meta, action_horizon):
        self.source_files = []
        self.episode_lengths = []
        self.episode_ids = []
        self.dataset_counts = []
        self.training_views = []
        for directory in dataset_dirs:
            root = Path(directory)
            info_path, episodes_path = root / "meta/info.json", root / "meta/episodes.jsonl"
            info = json.loads(info_path.read_text())
            view = info.get("robocoin_training_view")
            if len(dataset_dirs) > 1 and (not view or (view.get("version"), view.get("gripper_units"))
                                         not in ((1, "opening_percent"), (2, "mm"))):
                raise ValueError("Joint statistics require canonical training views with declared gripper units")
            self.training_views.append(view)
            for key in ("action", "observation.state"):
                feature = info["features"][key]
                if (feature["shape"] != [14] or feature.get("names") != RAW_NAMES
                        or feature["dtype"] != "float32"):
                    raise ValueError(f"Unexpected 14D schema/order: {root}:{key}")
            episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
            episodes.sort(key=lambda item: item["episode_index"])
            if [ep["episode_index"] for ep in episodes] != list(range(info["total_episodes"])):
                raise ValueError(f"Missing or duplicate episodes: {root}")
            if sum(ep["length"] for ep in episodes) != info["total_frames"]:
                raise ValueError(f"Metadata frame count mismatch: {root}")
            self.source_files.extend([info_path, episodes_path])
            for ep in episodes:
                if ep["length"] < 2:
                    raise ValueError(f"Episode too short for unbiased variance: {root}:{ep['episode_index']}")
                self.episode_lengths.append(ep["length"])
                self.episode_ids.append(ep["episode_index"])
                self.source_files.append(root / info["data_path"].format(
                    episode_chunk=ep["episode_index"] // info["chunks_size"],
                    episode_index=ep["episode_index"],
                ))
            self.dataset_counts.append({"path": str(root.resolve()), "episodes": len(episodes),
                                        "frames": info["total_frames"], "fps": info["fps"]})
        if len({view["gripper_units"] for view in self.training_views if view}) > 1:
            raise ValueError("Cannot mix training views with different gripper units")
        self.source_hashes = {str(path): sha256(path) for path in self.source_files}
        packed_meta = {"images": [], "action": [{"key": "default", "raw_shape": 14, "shape": 14}],
                       "state": [{"key": "default", "raw_shape": 14, "shape": 14}]}
        super().__init__(dataset_dirs=dataset_dirs, shape_meta=packed_meta,
                         action_size=action_horizon, obs_size=action_horizon + 1,
                         val_set_proportion=0, is_training_set=True, global_sample_stride=1)
        self.shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        self.action_meta, self.state_meta = self.shape_meta["action"], self.shape_meta["state"]
        for kind, metas in (("action", self.action_meta), ("observation.state", self.state_meta)):
            for meta in metas:
                meta["lerobot_key"] = f"{kind}.{meta['key']}"
        if self.multi_dataset.num_frames != sum(self.episode_lengths):
            raise ValueError("Loaded frame count does not match audited metadata")

    def _split_lerobot_sample(self, sample):
        n = len(sample["action"])
        if not torch.equal(sample["frame_index"], torch.arange(n, dtype=sample["frame_index"].dtype)):
            raise ValueError("Episode frame_index must start at zero and be contiguous")
        if torch.unique(sample["episode_index"]).numel() != 1:
            raise ValueError("Parquet file contains multiple episode indices")
        for raw_key in ("action", "observation.state"):
            values = sample[raw_key]
            if values.shape != (n, 14) or values.dtype != torch.float32 or not torch.isfinite(values).all():
                raise ValueError(f"Invalid {raw_key} tensor: {values.shape}, {values.dtype}")
            for key, indices in FIELD_INDICES.items():
                named_key = f"{raw_key}.{key}"
                split = values[:, indices]
                if any(self.training_views):
                    named = sample.get(named_key)
                    if named is not None and named.ndim == 1:
                        named = named.unsqueeze(-1)
                    if named is None or not torch.equal(named, split):
                        raise ValueError(f"Training named field differs from packed statistics input: {named_key}")
                sample[named_key] = split
        return sample

    def _get_episode_data(self, episode_idx):
        sample = self.multi_dataset.get_episode_data(episode_idx)
        if not (sample["episode_index"] == self.episode_ids[episode_idx]).all():
            raise ValueError(f"Episode index does not match metadata: {episode_idx}")
        sample = self._split_lerobot_sample(sample)
        if len(sample["action"]) != self.episode_lengths[episode_idx]:
            raise ValueError(f"Episode length mismatch: {episode_idx}")
        # Match BaseLerobotDataset._get_episode_data after validating raw records.
        state = {meta["key"]: self._get_state(meta, sample).unsqueeze(1).float()
                 for meta in self.state_meta}
        action = {meta["key"]: sliding_window_with_replication(
            self._get_action(meta, sample), self.action_size).float() for meta in self.action_meta}
        return {"state": state, "action": action}


def validate_stats(stats, dataset, action_horizon):
    if stats["num_episodes"] != len(dataset.episode_lengths):
        raise ValueError("Statistics episode count mismatch")
    if stats["num_transition"] != sum(dataset.episode_lengths):
        raise ValueError("Statistics frame count mismatch")
    for kind in ("action", "state"):
        if set(stats[kind]) != set(FIELD_INDICES):
            raise ValueError(f"Unexpected {kind} statistics fields")
        steps = action_horizon if kind == "action" else 1
        for key, indices in FIELD_INDICES.items():
            for scope in ("stepwise", "global"):
                for metric in ("min", "max", "mean", "std", "q01", "q99"):
                    value = stats[kind][key][f"{scope}_{metric}"]
                    shape = (steps, len(indices)) if scope == "stepwise" else (len(indices),)
                    if tuple(value.shape) != shape or not torch.isfinite(value).all():
                        raise ValueError(f"Invalid statistic: {kind}.{key}.{scope}_{metric}")
                    if metric == "std" and (value < 0).any():
                        raise ValueError("Negative standard deviation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    output = args.output.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        parser.error("Output already exists; choose a new path to preserve previous statistics")
    torch.set_num_threads(args.cpu_threads)
    cfg = OmegaConf.create({"data": OmegaConf.load(PROJECT_ROOT / "configs/data/galaxea_indoor_cleaning.yaml")})
    processor = instantiate(cfg.data.train.processor)
    horizon = int(cfg.data.train.num_frames) - 1
    dataset_dirs = [str(Path(directory).resolve()) for directory in args.dataset_dir]
    dataset = RoboCOINStatsDataset(dataset_dirs, cfg.data.train.shape_meta, horizon)
    stats = dataset.get_dataset_stats(processor)
    validate_stats(stats, dataset, horizon)
    for path, expected in dataset.source_hashes.items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"Dataset changed while computing statistics: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".norm-stats-", suffix=".json", dir=output.parent)
    os.close(fd)
    try:
        save_dataset_stats_to_json(stats, temporary)
        restored = load_dataset_stats_from_json(temporary)
        validate_stats(restored, dataset, horizon)
        processor.set_normalizer_from_stats(restored)
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    manifest = {
        "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": dataset.dataset_counts, "statistics_sha256": sha256(output),
        "action_horizon": horizon, "observation_horizon": horizon + 1, "global_sample_stride": 1,
        "field_order": list(FIELD_INDICES), "raw_field_indices": FIELD_INDICES,
        "relative_action_fields": ["left_arm", "right_arm"],
        "gripper_units": (dataset.training_views[0]["gripper_units"] if all(dataset.training_views) else
                          "Unchanged from source; configure gripper normalization for these units"),
        "training_views": dataset.training_views,
        "aggregation": "BaseLerobotDataset.get_dataset_stats (episode equal weighting)",
        "source_sha256": dataset.source_hashes,
        "implementation_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in (
            Path(__file__).resolve(),
            PROJECT_ROOT / "rift/datasets/lerobot/base_lerobot_dataset.py",
            PROJECT_ROOT / "rift/datasets/lerobot/transforms/relative_joint.py",
            PROJECT_ROOT / "configs/data/galaxea_indoor_cleaning.yaml",
        )},
    }
    fd, temporary = tempfile.mkstemp(prefix=".norm-manifest-", suffix=".json", dir=output.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.link(temporary, manifest_path)
    finally:
        os.unlink(temporary)
    print(json.dumps({"status": "complete", "output": str(output),
                      "episodes": stats["num_episodes"], "frames": stats["num_transition"]}), flush=True)


if __name__ == "__main__":
    main()

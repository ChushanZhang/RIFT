"""Independently validate RoboCOIN training stats using NumPy and both loaders."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.dont_write_bytecode = True

FIELDS = (("left_arm", [0, 1, 2, 3, 4, 5]), ("left_gripper", [12]),
          ("right_arm", [6, 7, 8, 9, 10, 11]), ("right_gripper", [13]))
NAMES = ([f"left_arm_joint_{i}_rad" for i in range(1, 7)]
         + [f"right_arm_joint_{i}_rad" for i in range(1, 7)]
         + ["left_gripper_open", "right_gripper_open"])


def values(table, name):
    result = np.asarray(table[name].to_pylist(), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 14 or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: {result.shape}")
    return result


def arrays_to_batch(action, state, torch):
    return {"action": {field: torch.as_tensor(action[:, indices].copy(), dtype=torch.float32)
                       for field, indices in FIELDS},
            "state": {field: torch.as_tensor(state[:, indices].copy(), dtype=torch.float32)
                      for field, indices in FIELDS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--rift-repo", type=Path, required=True)
    parser.add_argument("--fastwam-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stats = json.loads(args.stats.read_text())
    horizon = 32
    moments = {kind: {metric: [] for metric in ("mean", "var", "min", "max")}
               for kind in ("action", "state")}
    episodes_count = frames_count = 0
    samples = []
    gripper_ranges = []
    training_views = []
    for root in args.dataset_dir:
        info = json.loads((root / "meta/info.json").read_text())
        view = info.get("robocoin_training_view")
        if len(args.dataset_dir) > 1 and (not view or (view.get("version"), view.get("gripper_units"))
                                        not in ((1, "opening_percent"), (2, "mm"))):
            raise ValueError("Joint statistics require canonical training views with declared gripper units")
        training_views.append(view)
        for name in ("action", "observation.state"):
            if info["features"][name]["shape"] != [14] or info["features"][name]["names"] != NAMES:
                raise ValueError(f"Wrong packed schema: {root}, {name}")
        episodes = sorted((json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()
                           if line.strip()), key=lambda episode: episode["episode_index"])
        if [episode["episode_index"] for episode in episodes] != list(range(info["total_episodes"])):
            raise ValueError(f"Invalid episode indices in {root}")
        dataset_frames = 0
        grip_min, grip_max = np.full(2, np.inf), np.full(2, -np.inf)
        for episode in episodes:
            episode_id, n = episode["episode_index"], episode["length"]
            path = root / info["data_path"].format(episode_index=episode_id,
                                                   episode_chunk=episode_id // info["chunks_size"])
            columns = ["action", "observation.state", "frame_index", "episode_index"]
            if view:
                columns += [f"{kind}.{field}" for kind in ("action", "observation.state")
                            for field, _ in FIELDS]
            table = pq.read_table(path, columns=columns)
            action, state = values(table, "action"), values(table, "observation.state")
            if view:
                for kind, packed in (("action", action), ("observation.state", state)):
                    for field, indices in FIELDS:
                        named = np.asarray(table[f"{kind}.{field}"].to_pylist())
                        if named.ndim == 1:
                            named = named[:, None]
                        np.testing.assert_array_equal(named, packed[:, indices])
            if n < 2 or len(action) != n or len(state) != n:
                raise ValueError(f"Invalid episode length: {path}")
            if not np.array_equal(table["frame_index"].to_numpy(), np.arange(n)):
                raise ValueError(f"Invalid frame_index: {path}")
            if not np.all(table["episode_index"].to_numpy() == episode_id):
                raise ValueError(f"Invalid episode_index: {path}")
            indices = np.minimum(np.arange(n)[:, None] + np.arange(horizon)[None, :], n - 1)
            future = action[indices].copy()
            future[:, :, :12] -= state[:, None, :12]
            for kind, current in (("action", future), ("state", state[:, None, :])):
                moments[kind]["mean"].append(current.mean(axis=0))
                moments[kind]["var"].append(current.var(axis=0, ddof=1))
                moments[kind]["min"].append(current.min(axis=0))
                moments[kind]["max"].append(current.max(axis=0))
            if episode_id in (0, len(episodes) // 2, len(episodes) - 1):
                for start in (0, n // 2, n - 1):
                    obs_indices = np.minimum(start + np.arange(horizon + 1), n - 1)
                    samples.append((future[start].copy(), state[obs_indices].copy()))
            grip_min = np.minimum(grip_min, action[:, 12:].min(axis=0))
            grip_max = np.maximum(grip_max, action[:, 12:].max(axis=0))
            dataset_frames += n
            episodes_count += 1
        if dataset_frames != info["total_frames"]:
            raise ValueError(f"Wrong metadata frame count: {root}")
        frames_count += dataset_frames
        gripper_ranges.append({"dataset": str(root), "action_min": grip_min.tolist(), "action_max": grip_max.tolist()})
    if len({view["gripper_units"] for view in training_views if view}) > 1:
        raise ValueError("Cannot mix training views with different gripper units")
    if stats["num_episodes"] != episodes_count or stats["num_transition"] != frames_count:
        raise ValueError("Wrong statistics coverage")

    comparison = {}
    for kind in ("action", "state"):
        means, variances = (np.stack(moments[kind][metric]) for metric in ("mean", "var"))
        step_mean = means.mean(axis=0)
        global_mean = means.mean(axis=(0, 1))
        expected = {
            "stepwise_mean": step_mean,
            "stepwise_std": np.sqrt((variances + (means - step_mean) ** 2).mean(axis=0)),
            "global_mean": global_mean,
            "global_std": np.sqrt((variances + (means - global_mean) ** 2).mean(axis=(0, 1))),
            "stepwise_min": np.stack(moments[kind]["min"]).min(axis=0),
            "stepwise_max": np.stack(moments[kind]["max"]).max(axis=0),
        }
        expected["global_min"] = expected["stepwise_min"].min(axis=0)
        expected["global_max"] = expected["stepwise_max"].max(axis=0)
        for field, indices in FIELDS:
            for metric, array in expected.items():
                wanted = array[..., indices]
                actual = np.asarray(stats[kind][field][metric], dtype=np.float64)
                if actual.shape != wanted.shape:
                    raise ValueError(f"Wrong shape: {kind}.{field}.{metric}: {actual.shape} != {wanted.shape}")
                np.testing.assert_allclose(actual, wanted, rtol=1e-4, atol=2e-5,
                                           err_msg=f"Independent NumPy mismatch: {kind}.{field}.{metric}")
                comparison[f"{kind}.{field}.{metric}"] = float(np.max(np.abs(actual - wanted)))

    sys.path.insert(0, str(args.rift_repo))
    sys.path.insert(0, str(args.fastwam_repo / "src"))
    import torch
    torch.set_num_threads(1)
    modules = [importlib.import_module(f"{package}.datasets.lerobot.utils.normalizer")
               for package in ("rift", "fastwam")]
    shape_meta = {kind: [{"key": field, "shape": len(indices), "raw_shape": len(indices)}
                        for field, indices in FIELDS] for kind in ("action", "state")}
    normalizers = [module.LinearNormalizer(shape_meta=shape_meta, use_stepwise_action_norm=True,
                                          default_mode="z-score", exception_mode={"action": {
                                              "left_gripper": "0/100", "right_gripper": "0/100"}},
                                          stats=module.load_dataset_stats_from_json(str(args.stats)))
                   for module in modules]
    max_normalizer_error = 0.0
    for action, state in samples:
        original = arrays_to_batch(action, state, torch)
        results = [normalizer.forward(copy.deepcopy(original)) for normalizer in normalizers]
        for kind in ("action", "state"):
            for field, _ in FIELDS:
                a, b = results[0][kind][field], results[1][kind][field]
                if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.equal(a, b):
                    raise ValueError(f"RIFT/FastWAM normalizer mismatch: {kind}.{field}")
                max_normalizer_error = max(max_normalizer_error, float(torch.max(torch.abs(a - b))))
    result = {"status": "passed", "stats": str(args.stats), "episodes": episodes_count,
              "frames": frames_count, "independent_backend": "NumPy float64, episode equal weight, ddof=1",
              "statistics_checked": len(comparison), "max_stats_absolute_error": max(comparison.values()),
              "stats_absolute_errors": comparison, "normalizer_samples": len(samples),
              "rift_fastwam_normalizer_max_error": max_normalizer_error,
              "gripper_ranges": gripper_ranges,
              "training_views": training_views,
              "unit_note": (f"Canonical {training_views[0]['gripper_units']} views; action grippers use the Galaxea 0/100 mode."
                            if all(training_views) else
                            "Raw units preserved. 0/100 checks normalizer parity only; configure training for the source units.")}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

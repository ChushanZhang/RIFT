"""Build a Galaxea-compatible multitask view without changing RoboCOIN sources.

Grippers retain the source action/state labels. Classification and test-tube
values are multiplied by 180/pi to undo the Galaxea converter's degree2rad;
basket values already use the target millimetre scale and remain unchanged.
The separate state scale annotation is audited, but does not set this conversion.
Videos are linked unchanged, so resize and camera composition remain the job of
the existing training processor.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DATASETS = {
    "Galaxea_R1_Lite_classify_object_four": 1.75,
    "Galaxea_R1_Lite_mix_red_yellow_large_test_tube": 1.8,
    "R1_Lite_stack_baskets": 100.0,
}
GRIPPER_SCALES = {
    "Galaxea_R1_Lite_classify_object_four": 180.0 / np.pi,
    "Galaxea_R1_Lite_mix_red_yellow_large_test_tube": 180.0 / np.pi,
    "R1_Lite_stack_baskets": 1.0,
}
CONVERTER_SOURCE = (
    "https://github.com/RogersPyke/robocoin-dataset/blob/"
    "ec951eb78c34aca04d160543a4b6d03408f0cad6/"
    "scripts/format_converters/tolerobot/configs/converter_config_galaxea_r1_lite_mcap.yaml"
)
RAW_NAMES = [
    *[f"left_arm_joint_{i}_rad" for i in range(1, 7)],
    *[f"right_arm_joint_{i}_rad" for i in range(1, 7)],
    "left_gripper_open", "right_gripper_open",
]
FIELD_INDICES = {
    "left_arm": list(range(6)), "left_gripper": [12],
    "right_arm": list(range(6, 12)), "right_gripper": [13],
}
SCALARS = {
    "timestamp": "float32", "frame_index": "int64", "episode_index": "int64",
    "index": "int64", "task_index": "int64",
}
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def vector(table, key, width):
    column = table[key].combine_chunks()
    if not (pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type)):
        raise ValueError(f"{key}: expected a list of float32, got {column.type}")
    if column.type.value_type != pa.float32() or column.null_count or column.values.null_count:
        raise ValueError(f"{key}: values must be non-null float32")
    values = np.asarray(column.to_pylist(), dtype=np.float32)
    if values.shape != (table.num_rows, width) or not np.isfinite(values).all():
        raise ValueError(f"{key}: expected finite [{table.num_rows}, {width}] values")
    return values


def feature_stats(values):
    values = np.asarray(values).reshape(len(values), -1).astype(np.float64)
    return {
        "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0, ddof=0).tolist(),
        "count": [len(values)],
    }


def video_identity(path):
    target = path.resolve(strict=True)
    stat = target.stat()
    if not target.is_file() or stat.st_size == 0:
        raise ValueError(f"Missing or empty source video: {path}")
    return {"target": str(target), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def safe_relative(template, **fields):
    path = Path(template.format(**fields))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe dataset path: {path}")
    return path


def rename_without_overwrite(source, destination):
    # CFS does not support renameat2(RENAME_NOREPLACE). Serialize publishers with
    # a persistent sibling lock, then use the filesystem's atomic rename.
    lock = destination.parent / f".{destination.name}.publish.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        os.rename(source, destination)


def build_dataset(source, destination, full_scale, source_hashes, video_sources):
    gripper_scale = GRIPPER_SCALES[destination.name]
    metadata = {name: source / "meta" / name for name in
                ("info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl")}
    for path in metadata.values():
        source_hashes[str(path)] = sha256(path)
    info = json.loads(metadata["info.json"].read_text())
    for key in ("action", "observation.state"):
        feature = info["features"][key]
        if (feature.get("dtype") != "float32" or feature.get("shape") != [14]
                or feature.get("names") != RAW_NAMES):
            raise ValueError(f"Unexpected 14D field order/schema: {source}:{key}")
    scale_feature = info["features"].get("gripper_open_scale_state", {})
    if scale_feature.get("shape") != [2] or scale_feature.get("dtype") != "float32":
        raise ValueError(f"Missing two-channel gripper state scale annotation: {source}")
    if float(info["fps"]) != 30 or int(info["chunks_size"]) <= 0:
        raise ValueError(f"Unexpected fps or chunk size: {source}")
    for key, dtype in SCALARS.items():
        feature = info["features"][key]
        if feature["dtype"] != dtype or feature["shape"] != [1]:
            raise ValueError(f"Unexpected scalar schema: {source}:{key}")

    episodes = sorted(read_jsonl(metadata["episodes.jsonl"]), key=lambda ep: ep["episode_index"])
    if [ep["episode_index"] for ep in episodes] != list(range(info["total_episodes"])):
        raise ValueError(f"Missing or duplicate episodes: {source}")
    if not episodes or any(ep["length"] < 2 for ep in episodes):
        raise ValueError(f"Empty dataset or episode too short for training statistics: {source}")
    if sum(ep["length"] for ep in episodes) != info["total_frames"]:
        raise ValueError(f"Metadata frame count mismatch: {source}")
    tasks = read_jsonl(metadata["tasks.jsonl"])
    task_ids = {task["task_index"] for task in tasks}
    if len(task_ids) != len(tasks) or len(tasks) != info["total_tasks"]:
        raise ValueError(f"Missing or duplicate task metadata: {source}")
    original_stats = read_jsonl(metadata["episodes_stats.jsonl"])
    stats_by_episode = {entry["episode_index"]: entry["stats"] for entry in original_stats}
    if len(original_stats) != len(episodes) or set(stats_by_episode) != set(range(len(episodes))):
        raise ValueError(f"Episode statistics coverage mismatch: {source}")

    heads = [key for key in ("observation.images.cam_head_left_rgb",
                            "observation.images.cam_high_left_rgb") if key in info["features"]]
    if len(heads) != 1:
        raise ValueError(f"Ambiguous or missing head camera: {source}")
    cameras = {
        "observation.images.head_rgb": heads[0],
        "observation.images.left_wrist_rgb": "observation.images.cam_left_wrist_rgb",
        "observation.images.right_wrist_rgb": "observation.images.cam_right_wrist_rgb",
    }
    features = {key: copy.deepcopy(info["features"][key]) for key in SCALARS}
    for raw_key in ("action", "observation.state"):
        features[raw_key] = copy.deepcopy(info["features"][raw_key])
        for name, indices in FIELD_INDICES.items():
            features[f"{raw_key}.{name}"] = {
                "dtype": "float32", "shape": [len(indices)],
                "names": [RAW_NAMES[index] for index in indices],
            }
    for target_key, source_key in cameras.items():
        if info["features"].get(source_key, {}).get("dtype") != "video":
            raise ValueError(f"Missing video camera: {source}:{source_key}")
        features[target_key] = copy.deepcopy(info["features"][source_key])
    conversion = {
        "version": 2, "gripper_units": "mm",
        "source_annotation_denominator": full_scale, "source_root": str(source),
        "gripper_scale": gripper_scale,
        "gripper_formula": "float32(float64(raw) * gripper_scale)",
        "conversion_basis": "User-selected inverse degree2rad for classification/test tubes; basket unchanged",
        "converter_source": CONVERTER_SOURCE,
        "provenance_limit": "Published converter confirms the mechanism; per-task export logs are unavailable",
        "gripper_source_indices": [12, 13], "clip_grippers": False,
        "action_source": "Original action column; gripper_open_scale_action is not substituted",
        "annotation_check": "clip(raw state / annotation_denominator, 0, 1) matches gripper_open_scale_state; not the mm conversion",
        "raw_names": RAW_NAMES, "named_field_indices": FIELD_INDICES,
        "camera_sources": cameras,
    }
    out_info = copy.deepcopy(info)
    out_info.update({
        "codebase_version": "v2.1", "features": features, "data_path": DATA_PATH,
        "video_path": VIDEO_PATH, "total_videos": len(episodes) * 3,
        "total_chunks": (len(episodes) - 1) // int(info["chunks_size"]) + 1,
        "splits": {"train": f"0:{len(episodes)}"}, "robocoin_training_view": conversion,
    })
    (destination / "meta").mkdir(parents=True)
    write_json(destination / "meta/info.json", out_info)
    for name in ("episodes.jsonl", "tasks.jsonl"):
        shutil.copyfile(metadata[name], destination / "meta" / name)

    total_frames = 0
    max_scale_error = 0.0
    output_stats = []
    output_hashes = {}
    for episode in episodes:
        index, length = episode["episode_index"], episode["length"]
        fields = {"episode_index": index, "episode_chunk": index // int(info["chunks_size"])}
        parquet = source / safe_relative(info["data_path"], **fields)
        source_hashes[str(parquet)] = sha256(parquet)
        table = pq.read_table(parquet, columns=[*SCALARS, "action", "observation.state",
                                               "gripper_open_scale_state"])
        if table.num_rows != length:
            raise ValueError(f"Episode length differs from metadata: {parquet}")
        scalar_values = {}
        for key, dtype in SCALARS.items():
            if table[key].type != pa.type_for_alias(dtype) or table[key].null_count:
                raise ValueError(f"Unexpected scalar values/schema: {parquet}:{key}")
            scalar_values[key] = table[key].to_numpy()
            if not np.isfinite(scalar_values[key]).all():
                raise ValueError(f"Non-finite scalar: {parquet}:{key}")
        if not np.array_equal(scalar_values["frame_index"], np.arange(length)):
            raise ValueError(f"Non-contiguous frame indices: {parquet}")
        if not np.all(scalar_values["episode_index"] == index):
            raise ValueError(f"Wrong episode indices: {parquet}")
        if not np.array_equal(scalar_values["index"], np.arange(total_frames, total_frames + length)):
            raise ValueError(f"Non-contiguous dataset indices: {parquet}")
        if not set(scalar_values["task_index"].tolist()).issubset(task_ids):
            raise ValueError(f"Unknown task index: {parquet}")

        raw = {key: vector(table, key, 14) for key in ("action", "observation.state")}
        annotation = vector(table, "gripper_open_scale_state", 2)
        scale_error = float(np.abs(np.clip(raw["observation.state"][:, 12:].astype(np.float64)
                                          / full_scale, 0, 1) - annotation).max())
        if scale_error >= 1e-6:
            raise ValueError(f"Gripper scale annotation mismatch ({scale_error}): {parquet}")
        max_scale_error = max(max_scale_error, scale_error)
        columns = {key: table[key] for key in SCALARS}
        expected = dict(scalar_values)
        for raw_key, values in raw.items():
            transformed = values.copy()
            transformed[:, 12:] = (values[:, 12:].astype(np.float64) * gripper_scale).astype(np.float32)
            if not np.array_equal(transformed[:, :12], values[:, :12]):
                raise AssertionError("Arm values changed during gripper conversion")
            columns[raw_key] = pa.array(transformed.tolist(), type=pa.list_(pa.float32(), 14))
            expected[raw_key] = transformed
            for name, indices in FIELD_INDICES.items():
                key = f"{raw_key}.{name}"
                values = transformed[:, indices]
                columns[key] = (pa.array(values[:, 0], type=pa.float32()) if len(indices) == 1 else
                                pa.array(values.tolist(), type=pa.list_(pa.float32(), len(indices))))
                expected[key] = values
        out_parquet = destination / safe_relative(DATA_PATH, **fields)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), out_parquet, compression="zstd")
        restored = pq.read_table(out_parquet)
        if set(restored.column_names) != set(expected) or restored.num_rows != length:
            raise AssertionError(f"Output schema/row mismatch: {out_parquet}")
        for key, values in expected.items():
            restored_values = (vector(restored, key, values.shape[1]) if values.ndim == 2
                               and values.shape[1] > 1 else restored[key].to_numpy().reshape(values.shape))
            if not np.array_equal(restored_values, values):
                raise AssertionError(f"Readback differs from source transformation: {out_parquet}:{key}")
            if key in SCALARS and not restored[key].equals(table[key]):
                raise AssertionError(f"Scalar Arrow representation changed: {out_parquet}:{key}")
        output_hashes[str(out_parquet.relative_to(destination))] = sha256(out_parquet)
        episode_stats = {key: feature_stats(value) for key, value in expected.items()}
        for target_key, source_key in cameras.items():
            source_video = source / safe_relative(info["video_path"], video_key=source_key, **fields)
            identity = video_identity(source_video)
            video_sources[str(source_video)] = identity
            output_video = destination / safe_relative(VIDEO_PATH, video_key=target_key, **fields)
            output_video.parent.mkdir(parents=True, exist_ok=True)
            output_video.symlink_to(identity["target"])
            if video_identity(output_video) != identity:
                raise AssertionError(f"Video link differs from source: {output_video}")
            video_stats = copy.deepcopy(stats_by_episode[index][source_key])
            for metric in ("min", "max", "mean", "std", "count"):
                value = np.asarray(video_stats[metric])
                shape = (1,) if metric == "count" else (3, 1, 1)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid source video statistic: {source}:{index}:{source_key}:{metric}")
            episode_stats[target_key] = video_stats
        output_stats.append({"episode_index": index, "stats": episode_stats})
        total_frames += length
        if (index + 1) % 25 == 0:
            print(json.dumps({"dataset": source.name, "episodes_verified": index + 1,
                              "episodes_total": len(episodes)}), flush=True)
    stats_path = destination / "meta/episodes_stats.jsonl"
    with stats_path.open("w") as stream:
        for entry in output_stats:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
    if read_jsonl(stats_path) != output_stats:
        raise AssertionError(f"Statistics readback mismatch: {stats_path}")
    for name in ("episodes.jsonl", "tasks.jsonl"):
        if sha256(destination / "meta" / name) != source_hashes[str(metadata[name])]:
            raise AssertionError(f"Copied metadata changed: {name}")
    if json.loads((destination / "meta/info.json").read_text()) != out_info:
        raise AssertionError("Info metadata readback mismatch")
    for path in sorted((destination / "meta").iterdir()):
        output_hashes[str(path.relative_to(destination))] = sha256(path)
    result = {
        "name": source.name, "source_root": str(source), "episodes": len(episodes),
        "frames": total_frames, "video_symlinks": 3 * len(episodes),
        "max_gripper_scale_annotation_error": max_scale_error,
        "conversion": conversion, "output_sha256": output_hashes,
    }
    print(json.dumps({"dataset": source.name, "status": "verified", "episodes": len(episodes),
                      "frames": total_frames, "max_scale_error": max_scale_error}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/cfsdata/chushan/RoboCOIN-downloads"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().absolute()
    if os.path.lexists(output_root):
        parser.error("Output exists; choose a new path to preserve previous training views")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root = output_root.parent.resolve() / output_root.name
    source_dirs = [(source_root / name).resolve(strict=True) for name in DATASETS]
    if any(output_root == source or source in output_root.parents for source in source_dirs):
        parser.error("Output must not be inside a source dataset")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-staging-", dir=output_root.parent))
    try:
        source_hashes, video_sources, results = {}, {}, []
        for source, (name, full_scale) in zip(source_dirs, DATASETS.items()):
            result = build_dataset(source, staging / name, full_scale, source_hashes, video_sources)
            result["name"] = name
            results.append(result)
        for path, expected in source_hashes.items():
            if sha256(Path(path)) != expected:
                raise RuntimeError(f"Source changed while preparing training view: {path}")
        for path, expected in video_sources.items():
            if video_identity(Path(path)) != expected:
                raise RuntimeError(f"Source video changed while preparing training view: {path}")
        manifest = {
            "status": "complete", "version": 2, "gripper_units": "mm",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root), "output_root": str(output_root), "datasets": results,
            "episodes": sum(result["episodes"] for result in results),
            "frames": sum(result["frames"] for result in results),
            "video_symlinks": sum(result["video_symlinks"] for result in results),
            "source_sha256": source_hashes, "source_video_identity": video_sources,
            "video_verification": "Symlink target, size and mtime_ns; videos were not re-encoded or rehashed",
            "implementation_sha256": sha256(Path(__file__).resolve()),
            "validation": {"all_output_arrays_read_back": True, "all_source_hashes_rechecked": True,
                           "arm_values_exact": True, "scalar_values_exact": True,
                           "flat_and_named_values_consistent": True, "episode_coverage_complete": True},
        }
        write_json(staging / "manifest.json", manifest)
        rename_without_overwrite(staging, output_root)
        print(json.dumps({"status": "complete", "output_root": str(output_root),
                          "episodes": manifest["episodes"], "frames": manifest["frames"],
                          "video_symlinks": manifest["video_symlinks"]}), flush=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
